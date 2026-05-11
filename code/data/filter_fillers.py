"""
Filler Filtering Pipeline
==========================
Pre-filter filler pool by checking for conflicts with ALL possible gold facts.

1. Generate all entity×value gold fact sentences
2. Hybrid search: BM25 + text-embedding-3-small (top-K most similar fillers per gold fact)
3. LLM judge (GPT-4o-mini): check A/B/C conflict types
4. Save filtered pool

Embeddings are cached to .npy files so they only need to be generated once.

Usage:
  # Generate and cache embeddings only
  python3 filter_fillers.py --domain all --embed-only

  # Test with 5 gold facts per domain
  python3 filter_fillers.py --domain all --sample 5

  # Full run
  python3 filter_fillers.py --domain all
"""

import json
import os
import sys
import argparse
import asyncio
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    from openai import OpenAI, AsyncOpenAI
except ImportError:
    print("pip install openai required")
    sys.exit(1)

# ============================================================
# Step 1: Generate all gold fact sentences
# ============================================================

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_fact_templates():
    """Load FACT_TEMPLATES from generate_gold_facts.py (sibling module)."""
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
    from generate_gold_facts import FACT_TEMPLATES
    return FACT_TEMPLATES


def generate_all_gold_facts(domain):
    """Generate all entity × value gold fact sentences for a domain."""
    fact_tpl = load_fact_templates()[domain]
    pool_file = {
        "personal_life": os.path.join(_HERE, "entity_pools", "entity_pool_personallife.json"),
        "software_project": os.path.join(_HERE, "entity_pools", "entity_pool_software.json"),
    }[domain]

    with open(pool_file) as f:
        pool = json.load(f)

    gold_facts = []
    for cat in pool["categories"]:
        for entity in cat["entities"]:
            eid = entity["id"]
            template = fact_tpl.get(eid)
            if not template:
                continue
            for value in entity["values"]:
                fact = template.format(value=value)
                gold_facts.append({
                    "entity": eid,
                    "value": value,
                    "fact": fact,
                    "category": cat["name"],
                })
    return gold_facts


# ============================================================
# Step 2: Embedding cache + Hybrid search
# ============================================================

EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100  # Keep small to stay under 300K token-per-request limit


import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")  # used by text-embedding-3-small

def count_tokens(text):
    """Exact token count using tiktoken."""
    return len(_enc.encode(text, disallowed_special=()))


def get_embeddings(client, texts, model=EMBED_MODEL):
    """Get embeddings in batches, return numpy array."""
    all_embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        response = client.embeddings.create(model=model, input=batch)
        batch_embs = [d.embedding for d in response.data]
        all_embeddings.extend(batch_embs)
        if i + EMBED_BATCH_SIZE < len(texts):
            print(f"    Embedded {i + len(batch)}/{len(texts)}...")
    return np.array(all_embeddings, dtype=np.float32)


def load_or_create_filler_embeddings(client, filler_texts, domain):
    """Load cached filler embeddings or create and save them."""
    cache_path = Path(f"filler_embeddings_{domain}.npy")
    meta_path = Path(f"filler_embeddings_{domain}_meta.json")

    if cache_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("count") == len(filler_texts):
            print(f"  Loading cached embeddings from {cache_path} ({meta['count']} fillers)")
            return np.load(cache_path)
        else:
            print(f"  Cache stale (cached={meta.get('count')}, current={len(filler_texts)}), regenerating...")

    print(f"  Generating embeddings for {len(filler_texts)} fillers...")
    t0 = time.time()
    embeddings = get_embeddings(client, filler_texts)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, shape={embeddings.shape}")

    np.save(cache_path, embeddings)
    with open(meta_path, "w") as f:
        json.dump({
            "count": len(filler_texts),
            "model": EMBED_MODEL,
            "domain": domain,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"  Saved to {cache_path}")

    return embeddings


def load_or_create_gold_embeddings(client, gold_facts, domain):
    """Load cached gold fact embeddings or create and save them."""
    cache_path = Path(f"gold_embeddings_{domain}.npy")
    meta_path = Path(f"gold_embeddings_{domain}_meta.json")

    gold_texts = [gf["fact"] for gf in gold_facts]

    if cache_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        if meta.get("count") == len(gold_texts):
            print(f"  Loading cached gold embeddings from {cache_path} ({meta['count']} facts)")
            return np.load(cache_path)
        else:
            print(f"  Gold cache stale, regenerating...")

    print(f"  Generating embeddings for {len(gold_texts)} gold facts...")
    t0 = time.time()
    embeddings = get_embeddings(client, gold_texts)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s, shape={embeddings.shape}")

    np.save(cache_path, embeddings)
    with open(meta_path, "w") as f:
        json.dump({
            "count": len(gold_texts),
            "model": EMBED_MODEL,
            "domain": domain,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)
    print(f"  Saved to {cache_path}")

    return embeddings


def tokenize(text):
    """Simple whitespace + lowercase tokenizer."""
    return text.lower().split()


def build_bm25_index(filler_texts):
    """Build BM25 index over filler texts."""
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("pip install rank-bm25")
        sys.exit(1)

    tokenized = [tokenize(t) for t in filler_texts]
    return BM25Okapi(tokenized)


def hybrid_search(bm25, filler_texts, filler_embeddings, query_text, query_embedding, top_k=10):
    """Hybrid BM25 + embedding search. Returns deduplicated top-K by union of both."""
    # BM25 scores
    bm25_scores = bm25.get_scores(tokenize(query_text))
    bm25_top = bm25_scores.argsort()[-top_k:][::-1]
    bm25_top = [i for i in bm25_top if bm25_scores[i] > 0]

    # Embedding cosine similarity
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-10)
    filler_norms = filler_embeddings / (np.linalg.norm(filler_embeddings, axis=1, keepdims=True) + 1e-10)
    cosine_scores = filler_norms @ query_norm
    embed_top = cosine_scores.argsort()[-top_k:][::-1]

    # Union (deduplicated), preserve order by combined rank
    seen = set()
    results = []
    for idx in list(bm25_top) + list(embed_top):
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        results.append({
            "index": idx,
            "bm25_score": float(bm25_scores[idx]),
            "cosine_score": float(cosine_scores[idx]),
            "text": filler_texts[idx],
        })

    # Sort by max of normalized scores
    bm25_max = max(bm25_scores[r["index"]] for r in results) if results else 1
    for r in results:
        r["combined"] = 0.5 * (r["bm25_score"] / (bm25_max + 1e-10)) + 0.5 * r["cosine_score"]
    results.sort(key=lambda r: r["combined"], reverse=True)

    return results[:top_k]


# ============================================================
# Step 3: LLM conflict judgment
# ============================================================

FILTER_PROMPT = """You are checking if a filler conversation conflicts with a known gold fact.

Gold fact: "{gold_fact}"

Filler conversation:
---
{filler}
---

Check these 3 conflict types:
A) CONTRADICTION: Does the filler directly contradict the gold fact?
B) ALTERNATIVE: Does the filler introduce a plausible alternative answer that could confuse a memory system?
C) ENTITY_CONFUSION: Does the filler mention the same entity/topic in a confusing way?

Answer in JSON format:
{{"A": true/false, "B": true/false, "C": true/false, "reason": "brief explanation or 'no conflict'"}}"""


async def check_conflict_async(aclient, semaphore, gold_fact, filler_text, model="gpt-4o-mini"):
    """Async LLM judge with semaphore for rate limiting."""
    prompt = FILTER_PROMPT.format(
        gold_fact=gold_fact["fact"],
        filler=filler_text,
    )
    async with semaphore:
        for attempt in range(3):
            try:
                response = await aclient.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    response_format={"type": "json_object"},
                    max_tokens=200,
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                if attempt < 2 and ("rate" in str(e).lower() or "429" in str(e)):
                    await asyncio.sleep(2 ** attempt)
                    continue
                return {"A": False, "B": False, "C": False, "reason": f"error: {e}"}


# ============================================================
# Step 4: Load filler pool
# ============================================================

def load_filler_pool(domain, path):
    """Load filler pool and extract user text. `path` points to the source JSON."""
    with open(path) as f:
        raw = json.load(f)

    fillers = []
    skipped = 0
    for i, conv in enumerate(raw):
        if isinstance(conv, list):
            # ShareGPT format: list of turns
            text = " ".join(t.get("content", "") for t in conv if isinstance(t, dict) and t.get("role") == "user")
            raw_turns = conv
        elif isinstance(conv, dict) and "conversation" in conv:
            # LongMemEval format
            text = " ".join(t.get("content", "") for t in conv["conversation"] if t.get("role") == "user")
            raw_turns = conv["conversation"]
        else:
            continue

        tok_est = count_tokens(text)
        if tok_est < 500 or tok_est > 5000:
            skipped += 1
            continue

        fid = conv.get("sid", f"filler_{i}") if isinstance(conv, dict) else f"filler_{i}"
        fillers.append({"id": fid, "user_text": text, "raw": raw_turns})

    if skipped:
        print(f"  Filler pool: skipped {skipped} (< 500 or > 5000 tok estimate)")
    return fillers


def get_full_conversation_text(filler):
    """Get full conversation text (all roles) for LLM judge."""
    raw = filler.get("raw", [])
    parts = []
    for turn in raw:
        if isinstance(turn, dict):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            parts.append(f"[{role}]: {content}")
    return "\n".join(parts) if parts else filler.get("user_text", "")


# ============================================================
# Step 5: Run pipeline
# ============================================================

def run_pipeline(domain, filler_path, top_k=10, judge_model="gpt-4o-mini",
                 sample_n=None, embed_only=False):
    """Run the full filtering pipeline for a domain."""
    print(f"\n{'='*60}")
    print(f"Filler Filtering: {domain}")
    print(f"{'='*60}")

    # Step 1: Generate gold facts
    gold_facts = generate_all_gold_facts(domain)
    print(f"Gold facts: {len(gold_facts)}")

    # Load filler pool
    fillers = load_filler_pool(domain, filler_path)
    filler_texts = [f["user_text"] for f in fillers]
    print(f"Filler pool: {len(fillers)}")

    # Step 2: Embeddings (always generate/cache full set)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY required")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    print("\n--- Embeddings ---")
    filler_embeddings = load_or_create_filler_embeddings(client, filler_texts, domain)
    gold_embeddings = load_or_create_gold_embeddings(client, gold_facts, domain)

    if embed_only:
        print("Embed-only mode: done.")
        return None

    # Step 3: Build BM25 index
    print("\nBuilding BM25 index...")
    bm25 = build_bm25_index(filler_texts)

    # Sample mode: pick N gold facts spread across entities
    if sample_n:
        # Pick evenly from different entities
        by_entity = defaultdict(list)
        for gf in gold_facts:
            by_entity[gf["entity"]].append(gf)
        sampled = []
        entity_keys = list(by_entity.keys())
        for i in range(sample_n):
            eid = entity_keys[i % len(entity_keys)]
            candidates = by_entity[eid]
            sampled.append(candidates[i // len(entity_keys) % len(candidates)])
        gold_facts_to_check = sampled
        # Get corresponding embeddings
        gold_fact_indices = []
        all_gold = generate_all_gold_facts(domain)
        for sf in gold_facts_to_check:
            for j, gf in enumerate(all_gold):
                if gf["entity"] == sf["entity"] and gf["value"] == sf["value"]:
                    gold_fact_indices.append(j)
                    break
        gold_embeddings_subset = gold_embeddings[gold_fact_indices]
        print(f"\nSample mode: checking {len(gold_facts_to_check)} gold facts")
    else:
        gold_facts_to_check = gold_facts
        gold_embeddings_subset = gold_embeddings

    # Step 4: Search (sync) then judge (async parallel)
    print(f"Checking {len(gold_facts_to_check)} gold facts × top-{top_k} fillers (hybrid search)...")
    t0 = time.time()

    # First: collect all search hits (fast, no API calls)
    search_tasks = []  # list of (gold_fact_idx, gold_fact, hits)
    for gi, gf in enumerate(gold_facts_to_check):
        query_emb = gold_embeddings_subset[gi]
        similar = hybrid_search(bm25, filler_texts, filler_embeddings,
                                gf["fact"], query_emb, top_k=top_k)
        search_tasks.append((gi, gf, similar))

    print(f"  Search done ({time.time()-t0:.1f}s). Now judging with async LLM calls...")

    # Flatten all (gold_fact, hit) pairs for async judging
    judge_items = []  # (gi, gf, hit, filler_full_text)
    for gi, gf, similar in search_tasks:
        for hit in similar:
            filler_idx = hit["index"]
            filler_full_text = get_full_conversation_text(fillers[filler_idx])
            if not filler_full_text.strip():
                continue
            judge_items.append((gi, gf, hit, filler_full_text))

    print(f"  Total judge calls: {len(judge_items)}")

    # Async judge all pairs
    aclient = AsyncOpenAI(api_key=api_key)

    async def judge_all():
        semaphore = asyncio.Semaphore(40)  # must create inside the running loop
        tasks = []
        for gi, gf, hit, filler_full_text in judge_items:
            task = check_conflict_async(aclient, semaphore, gf, filler_full_text, model=judge_model)
            tasks.append(task)
        return await asyncio.gather(*tasks)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    judge_results = loop.run_until_complete(judge_all())
    loop.close()

    elapsed_judge = time.time() - t0
    print(f"  Judging done ({elapsed_judge:.1f}s)")

    # Reassemble results by gold fact
    all_checks = []
    conflicts = []
    conflict_fillers = set()
    checked = 0

    # Group judge results back by gold fact index
    result_idx = 0
    for gi, gf, similar in search_tasks:
        gf_checks = []
        for hit in similar:
            filler_idx = hit["index"]
            filler_full_text = get_full_conversation_text(fillers[filler_idx])
            if not filler_full_text.strip():
                continue

            result = judge_results[result_idx]
            result_idx += 1
            checked += 1

            is_conflict = result.get("A") or result.get("B") or result.get("C")
            fid = fillers[filler_idx]["id"]

            check_entry = {
                "filler_id": fid,
                "filler_idx": filler_idx,
                "bm25_score": hit["bm25_score"],
                "cosine_score": hit["cosine_score"],
                "combined_score": hit["combined"],
                "conflict_A": result.get("A", False),
                "conflict_B": result.get("B", False),
                "conflict_C": result.get("C", False),
                "is_conflict": is_conflict,
                "reason": result.get("reason", ""),
                "filler_snippet": filler_full_text[:300],
            }
            gf_checks.append(check_entry)

            if is_conflict:
                conflict_fillers.add(fid)
                conflicts.append({
                    **check_entry,
                    "gold_entity": gf["entity"],
                    "gold_value": gf["value"],
                    "gold_fact": gf["fact"],
                })

        all_checks.append({
            "gold_entity": gf["entity"],
            "gold_value": gf["value"],
            "gold_fact": gf["fact"],
            "category": gf["category"],
            "hits": gf_checks,
            "conflict_count": sum(1 for c in gf_checks if c["is_conflict"]),
            "clean_count": sum(1 for c in gf_checks if not c["is_conflict"]),
        })

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. "
          f"Checked {checked} pairs, found {len(conflicts)} conflicts "
          f"({len(conflict_fillers)} unique fillers).")

    # Build result
    clean_filler_ids = [f["id"] for f in fillers if f["id"] not in conflict_fillers]

    result = {
        "domain": domain,
        "metadata": {
            "filter_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "judge_model": judge_model,
            "search_method": "hybrid_bm25_embedding",
            "embed_model": EMBED_MODEL,
            "top_k": top_k,
            "gold_facts_total": len(gold_facts),
            "gold_facts_checked": len(gold_facts_to_check),
            "sample_mode": sample_n is not None,
            "filler_pool_size": len(fillers),
        },
        "summary": {
            "total_checked": checked,
            "total_conflicts": len(conflicts),
            "unique_conflict_fillers": len(conflict_fillers),
            "clean_fillers": len(clean_filler_ids),
            "conflict_rate": f"{len(conflict_fillers)/len(fillers)*100:.1f}%",
            "by_type": {
                "A_contradiction": sum(1 for c in conflicts if c["conflict_A"]),
                "B_alternative": sum(1 for c in conflicts if c["conflict_B"]),
                "C_entity_confusion": sum(1 for c in conflicts if c["conflict_C"]),
            }
        },
        "per_fact_results": all_checks,
        "conflicts": conflicts,
        "removed_filler_ids": list(conflict_fillers),
        "clean_filler_count": len(clean_filler_ids),
    }

    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter filler pool for conflicts")
    parser.add_argument("--domain", type=str, default="all",
                        choices=["personal_life", "software_project", "all"])
    parser.add_argument("--filler-pl-path", type=str, default=None,
                        help="Path to PL filler pool JSON (LongMemEval haystack format). Required for personal_life.")
    parser.add_argument("--filler-sw-path", type=str, default=None,
                        help="Path to SW filler pool JSON (ShareGPT format). Required for software_project.")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sample", type=int, default=None,
                        help="Test with N gold facts per domain (spread across entities)")
    parser.add_argument("--embed-only", action="store_true",
                        help="Only generate and cache embeddings, skip conflict check")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output file (default: auto-named based on mode)")
    args = parser.parse_args()

    domains = ["personal_life", "software_project"] if args.domain == "all" else [args.domain]

    paths = {"personal_life": args.filler_pl_path, "software_project": args.filler_sw_path}
    for d in domains:
        if not paths[d]:
            sys.exit(f"Error: --filler-{d[:2]}-path is required for domain={d}")

    all_results = {}
    for domain in domains:
        result = run_pipeline(domain, paths[domain], top_k=args.top_k, judge_model=args.judge_model,
                              sample_n=args.sample, embed_only=args.embed_only)
        if result:
            all_results[domain] = result

    if not all_results:
        if args.embed_only:
            print("\nEmbeddings cached. Run without --embed-only to check conflicts.")
        sys.exit(0)

    # Print summary per domain
    for domain, result in all_results.items():
        s = result["summary"]
        print(f"\n--- {domain} Summary ---")
        print(f"  Pool: {result['metadata']['filler_pool_size']}")
        print(f"  Gold facts checked: {result['metadata']['gold_facts_checked']}/{result['metadata']['gold_facts_total']}")
        print(f"  Checked: {s['total_checked']}")
        print(f"  Conflicts: {s['unique_conflict_fillers']} ({s['conflict_rate']})")
        print(f"  Clean: {s['clean_fillers']}")
        print(f"  By type: A={s['by_type']['A_contradiction']} B={s['by_type']['B_alternative']} C={s['by_type']['C_entity_confusion']}")

        # Per-fact breakdown
        print(f"\n  Per-fact results:")
        for pf in result["per_fact_results"]:
            marker = "CONFLICT" if pf["conflict_count"] > 0 else "clean"
            print(f"    [{marker:8s}] {pf['gold_entity']:25s} "
                  f"val=\"{str(pf['gold_value'])[:35]}\" "
                  f"conflicts={pf['conflict_count']}/{pf['conflict_count']+pf['clean_count']}")
            if pf["conflict_count"] > 0:
                for hit in pf["hits"]:
                    if hit["is_conflict"]:
                        types = []
                        if hit["conflict_A"]: types.append("A")
                        if hit["conflict_B"]: types.append("B")
                        if hit["conflict_C"]: types.append("C")
                        print(f"      [{','.join(types)}] bm25={hit['bm25_score']:.2f} "
                              f"cos={hit['cosine_score']:.3f} | {hit['reason'][:80]}")

    # Save
    if args.output:
        out_path = args.output
    elif args.sample:
        out_path = f"filter_sample_{args.sample}.json"
    else:
        out_path = "filtered_fillers.json"

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
