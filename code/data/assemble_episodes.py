"""
Haystack Assembly
==================
Evidence sessions (from generate_conversations.py) +
Filler sessions (from filler pool) →
Final assembled episodes.

Input:
  - conversations/ folder (our evidence sessions)
  - filler pool JSON (domain-specific)

Output:
  - assembled/ folder (final episodes with filler interleaved)

Usage:
  # Personal life (default)
  python3 assemble_episodes.py -c conversations/ -o assembled/

  # Software project
  python3 assemble_episodes.py -c conversations/ -o assembled/ --domain software_project
"""

import json
import os
import sys
import argparse
import random
from datetime import datetime, timedelta

import tiktoken
_enc = tiktoken.get_encoding("cl100k_base")

# ============================================================
# Concept Keywords — for filler filtering (per domain)
# ============================================================
# Used to exclude filler sessions that contain keywords
# overlapping with the episode's evidence entities.

# Filler contamination filter: keywords that indicate the filler user
# is sharing personal/project facts that could conflict with evidence entities.
# Checked against USER messages only (not assistant responses).
FILLER_BLOCK_KEYWORDS = {
    "personal_life": [
        # Work/employer
        "i work at", "my company", "my job at", "my employer", "my workplace",
        "my office is", "i started at", "i work as", "my job title",
        "my work hours", "my shift", "i work from",
        # Residence
        "i live in", "i moved to", "my apartment", "my house", "my home is",
        "my neighborhood", "i live alone", "i live with", "my roommate",
        # Health
        "diagnosed with", "my condition", "my doctor", "my symptoms",
        "my illness", "my health", "my blood pressure",
        # Medication
        "medication", "prescription", "i take pills",
        # Relationships (standalone)
        "husband", "wife", "boyfriend", "girlfriend",
        "my partner", "my spouse", "engaged", "married", "dating", "divorced",
        # School
        "my school", "my university", "my college", "i'm studying at", "my classes at",
        # Activities/Exercise — standalone keywords (biggest contamination source)
        "meditation", "yoga", "gym", "workout", "exercise",
        "running", "jogging", "swimming", "cycling", "biking",
        "spin class", "pilates", "crossfit", "fitness",
        "my commute", "i drive to work", "i take the bus", "i bike to",
        # Appointments — standalone
        "dentist", "therapist", "chiropractor", "appointment",
        "checkup", "check-up",
        # Food/Diet — standalone
        "vegetarian", "vegan", "gluten-free", "dairy-free",
        "allergy", "allergic", "i don't eat", "my diet",
        # Sleep — standalone
        "my sleep", "i wake up at", "i go to bed", "my bedtime", "insomnia",
        # Pets — standalone
        "my dog", "my cat", "my pet", "puppy", "kitten",
        # Vehicle
        "my car", "i drive a",
        # Hobbies/Sports — standalone
        "my hobby", "in my free time",
        "guitar", "piano", "photography",
        "soccer", "tennis", "basketball", "volleyball", "golf",
        # Living/Finance
        "my insurance", "my health plan", "my coverage",
        "i'm saving for", "paying off", "my savings",
        "my vacation", "i'm traveling to", "planning a trip",
        "book club", "board game", "i joined a club",
        "i subscribe to", "my subscription",
        "i'm planning to buy", "saving up for",
        "my favorite restaurant",
        # Codes/IDs
        "my phone number", "my employee id", "my reservation",
        "my booking code", "my address is",
    ],
    "software_project": [
        # Possession-prefix
        "we use ", "we are using ", "we're using ", "we switched to ",
        "our framework", "our database", "our orm", "our test",
        "our auth", "our deploy", "our ci", "our docker",
        "my project uses", "i use ", "i'm using ",
        "we deploy on", "we host on", "we run on", "we built with",
        "our stack is", "our tech stack", "our setup is",
        "we migrated to ", "we moved to ",
        # Team/Process
        "our team lead", "our standup", "our sprint",
        "our meeting is", "our on-call", "our slack channel",
        "our repo is", "our branch", "our release",
        # Infra
        "our endpoint", "our api key", "our dsn",
        "our monitoring", "our staging",
        # Personal life keywords (SW fillers can also contain personal info)
        "meditation", "yoga", "gym", "workout", "exercise",
        "running", "jogging", "swimming", "cycling", "biking",
        "spin class", "pilates", "crossfit", "fitness",
        "dentist", "therapist", "chiropractor", "appointment",
        "vegetarian", "vegan", "gluten-free", "dairy-free",
        "allergy", "allergic",
        "husband", "wife", "boyfriend", "girlfriend",
        "my partner", "my spouse", "engaged", "married",
        "medication", "prescription",
        "my dog", "my cat", "my pet",
        "soccer", "tennis", "basketball", "volleyball", "golf",
        "guitar", "piano", "photography",
    ],
}



# ============================================================
# Filler Pool Loaders
# ============================================================

def load_filler_pool_longmemeval(path):
    """Load filler sessions from LongMemEval S format.
    Returns: list of {"sid": str, "conversation": list, "text_lower": str, "words": int}
    """
    print(f"Loading filler pool from {path}...")
    with open(path) as f:
        data = json.load(f)

    filler_pool = []
    seen_sids = set()

    for instance in data:
        evidence_ids = set(instance.get('answer_session_ids', []))
        for sid, sess in zip(instance['haystack_session_ids'], instance['haystack_sessions']):
            if sid not in evidence_ids and sid not in seen_sids:
                seen_sids.add(sid)
                # User messages only for filtering (avoids assistant content contamination)
                user_text = " ".join(t['content'].lower() for t in sess if t.get('role') == 'user')
                text = " ".join(t['content'].lower() for t in sess)
                tokens_est = _count_tokens(" ".join(t['content'] for t in sess))
                if tokens_est < 500:
                    continue  # skip short fillers
                filler_pool.append({
                    "sid": sid,
                    "conversation": sess,
                    "text_lower": text,
                    "user_text_lower": user_text,
                    "words": words
                })

    print(f"  Loaded {len(filler_pool)} unique filler sessions.")
    return filler_pool


def load_filler_pool_sharegpt(path):
    """Load filler sessions from ShareGPT coding format.
    Input: list of conversations, each is list of {"role": ..., "content": ...}
    Returns: same format as load_filler_pool_longmemeval
    """
    print(f"Loading filler pool from {path}...")
    with open(path) as f:
        data = json.load(f)

    filler_pool = []
    for i, conv in enumerate(data):
        text = " ".join(t.get('content', '').lower() for t in conv if isinstance(t, dict))
        user_text = " ".join(t.get('content', '').lower() for t in conv if isinstance(t, dict) and t.get('role') == 'user')
        words = sum(len(t.get('content', '').split()) for t in conv if isinstance(t, dict))
        filler_pool.append({
            "sid": f"sharegpt_{i}",
            "conversation": conv,
            "text_lower": text,
            "user_text_lower": user_text,
            "words": words
        })

    print(f"  Loaded {len(filler_pool)} filler sessions.")
    return filler_pool


def load_filler_pool(path, domain="personal_life"):
    """Load filler pool based on domain."""
    if domain == "personal_life":
        # Cleaned flat-list format (same as SW)
        return load_filler_pool_sharegpt(path)
    elif domain == "software_project":
        return load_filler_pool_sharegpt(path)
    else:
        raise ValueError(f"Unknown domain: {domain}")


# ============================================================
# Filler Filtering
# ============================================================

def get_evidence_entities(conversation_data):
    """Extract the list of evidence entities from an episode."""
    entities = set()
    for sess in conversation_data['sessions']:
        for gf in sess.get('gold_facts', []):
            entities.add(gf['entity'])
    return entities


CONTENT_BLOCK_WORDS = ["porn", "nude", "sex", "kill", "drug", "weapon", "suicide",
                       "violence", "abuse", "murder", "rape", "torture"]


def filter_fillers(filler_pool, evidence_entities, domain="personal_life"):
    """
    Exclude fillers that:
    1. Contain personal/project facts conflicting with evidence (semantic contamination)
    2. Contain unsafe content that triggers API content filters

    Checks USER messages only — assistant responses are safe.

    Returns: filtered filler list
    """
    block_keywords = FILLER_BLOCK_KEYWORDS.get(domain, [])
    if not block_keywords:
        return filler_pool

    filtered = []
    blocked = 0
    for filler in filler_pool:
        text = filler.get('text_lower', '')
        user_text = filler.get('user_text_lower', text)
        # Content safety filter (full text)
        if any(w in text for w in CONTENT_BLOCK_WORDS):
            blocked += 1
            continue
        # Semantic contamination filter (user text only)
        if any(kw in user_text for kw in block_keywords):
            blocked += 1
            continue
        filtered.append(filler)

    print(f"  Filler filter: {blocked} blocked, {len(filtered)} passed ({domain})")
    return filtered


# ============================================================
# Assemble episode
# ============================================================

def generate_timestamps(num_sessions, base_date=None):
    """
    Generate a list of timestamps for sessions.
    Format: "2023/04/10 (Mon) 17:50"
    Gap between sessions: 2 hours ~ 2 days (random)
    """
    if base_date is None:
        base_date = datetime(2023, 3, 1, 9, 0)

    timestamps = []
    current = base_date

    for i in range(num_sessions):
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_name = day_names[current.weekday()]
        ts = current.strftime(f"%Y/%m/%d ({day_name}) %H:%M")
        timestamps.append(ts)

        gap_hours = random.uniform(2, 48)
        current += timedelta(hours=gap_hours)

    return timestamps, current


def _count_tokens(text):
    """Exact token count using tiktoken."""
    return len(_enc.encode(text, disallowed_special=()))


def _estimate_tokens(filler):
    """Exact token count for a filler session using tiktoken."""
    text = " ".join(t['content'] for t in filler['conversation'])
    return _count_tokens(text)


def _sample_fillers_by_token_budget(available_fillers, num_gaps, target_tokens_per_gap):
    """Sample fillers per gap until each gap reaches the token budget."""
    random.shuffle(available_fillers)
    pool = list(available_fillers)  # mutable copy
    pool_idx = 0

    filler_groups = []
    for g in range(num_gaps):
        group = []
        group_tokens = 0
        while group_tokens < target_tokens_per_gap and pool_idx < len(pool):
            filler = pool[pool_idx]
            pool_idx += 1
            group.append(filler)
            group_tokens += _estimate_tokens(filler)
        filler_groups.append(group)

    return filler_groups


def _sample_fillers_by_total_budget(available_fillers, num_gaps, total_filler_budget,
                                     overshoot_tolerance=0.2):
    """Random-sample fillers into per-gap buckets, strictly budget-bounded.

    Each gap's target = total_budget / num_gaps. Multiple small fillers stack per gap
    until target is met. Each gap is capped at (1+tolerance)×target so oversized
    fillers (e.g., SW coding fillers with 100K+ tokens) can't blow the budget.

    Algorithm:
    1. Shuffle the pool (only source of randomness — no size bias).
    2. For each filler in shuffled order, find gaps that (a) still need fillers and
       (b) have room within the overshoot cap. Place in a random one.
    3. If a filler doesn't fit anywhere (too big), skip it.
    4. Stop once every gap has reached its target budget.
    """
    if num_gaps <= 0:
        return []
    per_gap_budget = total_filler_budget / num_gaps
    max_per_gap = per_gap_budget * (1 + overshoot_tolerance)

    pool = [(f, _estimate_tokens(f)) for f in available_fillers]
    random.shuffle(pool)

    filler_groups = [[] for _ in range(num_gaps)]
    group_tokens = [0.0] * num_gaps

    for filler, tok in pool:
        candidate_gaps = [
            g for g in range(num_gaps)
            if group_tokens[g] < per_gap_budget
            and group_tokens[g] + tok <= max_per_gap
        ]
        if not candidate_gaps:
            continue
        g = random.choice(candidate_gaps)
        filler_groups[g].append(filler)
        group_tokens[g] += tok

        if all(gt >= per_gap_budget for gt in group_tokens):
            break

    return filler_groups


def assemble_episode(conversation_data, filler_pool, num_filler_per_gap=8,
                     target_tokens_per_gap=None, total_tokens_budget=None,
                     seed=None, domain="personal_life"):
    """
    Evidence sessions + filler sessions → final assembled episode.

    Structure:
      [filler x N] [Fact Intro Part 1] [filler x N] [Fact Intro Part 2]
      [filler x N] --- before_questions --- [filler x N]
      [Change Event] [Delete Event] [filler x N] --- after_questions ---

    If target_tokens_per_gap is set, fills each gap to that token budget
    instead of using a fixed number of fillers per gap.
    """
    if seed is not None:
        random.seed(seed)

    ep_id = conversation_data['episode_id']

    evidence_entities = get_evidence_entities(conversation_data)
    available_fillers = filter_fillers(filler_pool, evidence_entities, domain=domain)

    # Build ordered evidence session list from conversation data.
    # Supports both old format (Part 1, Part 2) and new format (Part 1a, 1b, 1c, Part 2).
    EVIDENCE_ORDER = [
        "Fact Introduction (Part 1)",
        "Fact Introduction (Part 1a)",
        "Fact Introduction (Part 1b)",
        "Fact Introduction (Part 1c)",
        "Fact Introduction (Part 2)",
    ]
    # Before-questions go here (inserted between fact intro and change/delete)
    EVIDENCE_POST_BQ = ["Change+Delete Event"]

    evidence_map = {}
    for sess in conversation_data['sessions']:
        evidence_map[sess['type']] = sess

    # Collect evidence sessions in order (skip missing ones)
    pre_bq_evidence = [t for t in EVIDENCE_ORDER if t in evidence_map]
    post_bq_evidence = [t for t in EVIDENCE_POST_BQ if t in evidence_map]

    # Total gaps = (before each pre_bq evidence) + (between bq and post_bq) + (after post_bq)
    # = len(pre_bq) + 1 (before bq) + 1 (after post_bq) = len(pre_bq) + 2
    num_gaps = len(pre_bq_evidence) + 2

    # Re-distribute fillers for new gap count
    if total_tokens_budget is not None:
        # Total-budget mode: sample fillers = total_budget - evidence_tokens, round-robin
        evidence_tokens = sum(
            _count_tokens(" ".join(t['content'] for t in evidence_map[etype]['conversation']))
            for etype in pre_bq_evidence + post_bq_evidence
            if etype in evidence_map
        )
        filler_budget = max(0, total_tokens_budget - evidence_tokens)
        filler_groups = _sample_fillers_by_total_budget(
            available_fillers, num_gaps, filler_budget)
    elif target_tokens_per_gap is not None:
        filler_groups = _sample_fillers_by_token_budget(
            available_fillers, num_gaps, target_tokens_per_gap)
    else:
        total_fillers_needed = num_filler_per_gap * num_gaps
        if len(available_fillers) < total_fillers_needed:
            print(f"  Warning: Only {len(available_fillers)} fillers available, need {total_fillers_needed}")
            total_fillers_needed = len(available_fillers)
        sampled_fillers = random.sample(available_fillers, total_fillers_needed)
        filler_groups = []
        idx = 0
        for g in range(num_gaps):
            n = min(num_filler_per_gap, total_fillers_needed - idx)
            filler_groups.append(sampled_fillers[idx:idx+n])
            idx += n

    assembled_sessions = []
    evidence_session_indices = []

    # --- Pre-BQ evidence sessions with filler gaps ---
    for i, etype in enumerate(pre_bq_evidence):
        # Filler gap before this evidence session
        for filler in filler_groups[i]:
            assembled_sessions.append({
                "session_id": filler['sid'],
                "type": "filler",
                "conversation": filler['conversation']
            })

        # Evidence session
        sess = evidence_map[etype]
        evidence_session_indices.append(len(assembled_sessions))
        sid = f"evidence_{etype.lower().replace(' ', '_').replace('(', '').replace(')', '')}"
        assembled_sessions.append({
            "session_id": sid,
            "type": "evidence",
            "evidence_type": etype,
            "conversation": sess["conversation"],
            "gold_facts": sess.get("gold_facts", [])
        })

    # --- BEFORE QUESTIONS position ---
    # Filler gap between last pre-BQ evidence and before-questions
    bq_gap_idx = len(pre_bq_evidence)
    for filler in filler_groups[bq_gap_idx]:
        assembled_sessions.append({
            "session_id": filler['sid'],
            "type": "filler",
            "conversation": filler['conversation']
        })
    before_q_position = len(assembled_sessions)

    # --- Post-BQ evidence sessions (Change, Delete) ---
    for etype in post_bq_evidence:
        sess = evidence_map[etype]
        evidence_session_indices.append(len(assembled_sessions))
        sid = f"evidence_{etype.lower().replace(' ', '_')}"
        assembled_sessions.append({
            "session_id": sid,
            "type": "evidence",
            "evidence_type": etype,
            "conversation": sess["conversation"],
            "gold_facts": sess.get("gold_facts", [])
        })

    # Final filler gap (after post-BQ evidence, before after-questions)
    last_gap_idx = len(pre_bq_evidence) + 1
    for filler in filler_groups[last_gap_idx]:
        assembled_sessions.append({
            "session_id": filler['sid'],
            "type": "filler",
            "conversation": filler['conversation']
        })

    # --- AFTER QUESTIONS position ---
    after_q_position = len(assembled_sessions)

    # Generate timestamps
    timestamps, final_time = generate_timestamps(len(assembled_sessions))
    for i, sess in enumerate(assembled_sessions):
        sess['timestamp'] = timestamps[i]

    if before_q_position > 0:
        before_q_ts = timestamps[before_q_position - 1]
    else:
        before_q_ts = timestamps[0]

    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    after_q_time = final_time + timedelta(hours=random.uniform(2, 24))
    day_name = day_names[after_q_time.weekday()]
    after_q_ts = after_q_time.strftime(f"%Y/%m/%d ({day_name}) %H:%M")

    # Stats (tiktoken-based)
    total_tokens = sum(
        _count_tokens(" ".join(t['content'] for t in sess['conversation']))
        for sess in assembled_sessions
    )
    evidence_tokens_final = sum(
        _count_tokens(" ".join(t['content'] for t in sess['conversation']))
        for sess in assembled_sessions if sess['type'] == 'evidence'
    )
    filler_tokens = total_tokens - evidence_tokens_final

    # Replace timestamp placeholders in questions
    before_questions = conversation_data.get('before_questions', [])
    after_questions = conversation_data.get('after_questions', [])

    for q in after_questions:
        if '[TIMESTAMP_BEFORE_CHANGE]' in q.get('question', ''):
            q['question'] = q['question'].replace('[TIMESTAMP_BEFORE_CHANGE]', before_q_ts)
    for q in before_questions:
        if '[TIMESTAMP_BEFORE_CHANGE]' in q.get('question', ''):
            q['question'] = q['question'].replace('[TIMESTAMP_BEFORE_CHANGE]', before_q_ts)

    result = {
        "episode_id": ep_id,
        "domain": domain,
        "root": conversation_data['root'],
        "total_sessions": len(assembled_sessions),
        "evidence_sessions": len(evidence_session_indices),
        "filler_sessions": len(assembled_sessions) - len(evidence_session_indices),
        "total_tokens": total_tokens,
        "evidence_tokens": evidence_tokens_final,
        "filler_tokens": filler_tokens,
        "sessions": assembled_sessions,
        "evidence_session_indices": evidence_session_indices,
        "before_questions": {
            "timestamp": before_q_ts,
            "position_after_session": before_q_position - 1,
            "questions": before_questions
        },
        "after_questions": {
            "timestamp": after_q_ts,
            "position_after_session": after_q_position - 1,
            "questions": after_questions
        }
    }

    return result


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble episodes with filler sessions")
    parser.add_argument("-c", "--conversations_dir", type=str, default="conversations",
                        help="Directory with conversation JSON files (default: conversations/)")
    parser.add_argument("-f", "--filler_path", type=str, default=None,
                        help="Path to filler pool JSON (default: auto per domain)")
    parser.add_argument("-o", "--output_dir", type=str, default="assembled",
                        help="Output directory (default: assembled/)")
    parser.add_argument("-n", "--num_filler_per_gap", type=int, default=8,
                        help="Number of filler sessions per gap (default: 8)")
    parser.add_argument("-t", "--target_tokens", type=int, default=None,
                        help="Target filler tokens per gap (overrides -n). 7000 produces filler32k-style episodes, 25000 produces filler128k-style. Total per-episode tokens vary by domain.")
    parser.add_argument("-T", "--total_tokens", type=int, default=None,
                        help="Total episode token budget (overrides -t and -n). Filler budget = total - evidence tokens.")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--domain", type=str, default=None,
                        help="Domain override. If not set, read from conversation JSON.")
    args = parser.parse_args()

    random.seed(args.seed)

    # Find conversation files
    conv_files = sorted([
        f for f in os.listdir(args.conversations_dir)
        if f.startswith("conversations_") and f.endswith(".json")
    ])

    if not conv_files:
        print(f"No conversation files found in {args.conversations_dir}/")
        sys.exit(1)

    # Detect domain from first conversation file if not overridden
    first_conv_path = os.path.join(args.conversations_dir, conv_files[0])
    with open(first_conv_path) as f:
        first_conv = json.load(f)
    domain = args.domain or first_conv.get("domain", "personal_life")

    # Load filler pool
    if not args.filler_path:
        sys.exit("Error: -f/--filler_path is required (download from meme-benchmark/MEME-fillers)")
    filler_pool = load_filler_pool(args.filler_path, domain=domain)

    print(f"Domain: {domain}")
    print(f"Found {len(conv_files)} conversation files to assemble.\n")

    os.makedirs(args.output_dir, exist_ok=True)

    for conv_file in conv_files:
        conv_path = os.path.join(args.conversations_dir, conv_file)
        with open(conv_path) as f:
            conv_data = json.load(f)

        ep_id = conv_data['episode_id']

        result = assemble_episode(
            conv_data, filler_pool,
            num_filler_per_gap=args.num_filler_per_gap,
            target_tokens_per_gap=args.target_tokens,
            total_tokens_budget=args.total_tokens,
            seed=args.seed + ep_id,
            domain=domain
        )

        out_file = f"episode_{ep_id:03d}.json"
        out_path = os.path.join(args.output_dir, out_file)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"  Ep{ep_id:3d}: {result['total_sessions']} sessions "
              f"({result['evidence_sessions']} evidence + {result['filler_sessions']} filler) | "
              f"{result['total_tokens']} tokens (ev={result['evidence_tokens']} filler={result['filler_tokens']}) | "
              f"before_q={len(result['before_questions']['questions'])} "
              f"after_q={len(result['after_questions']['questions'])}")

    print(f"\nAssembly complete → {args.output_dir}/")
