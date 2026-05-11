"""
Gold Fact → Self-Chat Conversation Generator (v3)
==================================================
LongMemEval-style turn-by-turn self-chat conversation generation.

Pipeline:
  Step 1: Gold fact seed (first-person) → third-person statement conversion
  Step 2: Generate session via self-chat (user conveys third-person facts in their own words)
  Step 3: Annotation — LLM determines which turn each fact was conveyed in

Usage:
  export OPENAI_API_KEY=your_key_here
  python3 generate_conversations.py --input gold_facts/gold_facts_001.json
"""

import json
import sys
import os
import argparse
import time
import re
import random

try:
    from openai import OpenAI
except ImportError:
    print("pip install openai required")
    sys.exit(1)


# ============================================================
# Step 1: Gold fact seed (first-person) → third-person conversion
# ============================================================

STEP1_SYSTEM_PROMPTS = {
    "personal_life": """You are rewriting first-person fact statements into third-person statements about "the user".

You will receive a list of first-person fact seeds.
Rewrite each into a natural, grammatically correct third-person sentence about "the user".

STRICT RULES:
- Convert first-person ("I", "my", "we", "our") to third-person ("the user", "the user's", "they", "their")
- Fix verb conjugation: "I work" → "The user works", "I have" → "The user has"
- Fix grammar issues: articles (a/an/the), prepositions, verb forms, capitalization
- Fix semantic mismatches: e.g., "watching a show called a podcast" → "listening to a podcast called..."
- Keep ALL proper nouns, numbers, codes, and entity values EXACTLY as given
  (e.g., "Velthari Studio", "10-to-7 fixed", "RSV-09831-THETA" must appear verbatim)
- Keep dependency/causal relationships intact
- Each output must be a single self-contained sentence
- Do NOT add information not present in the seed
- Do NOT merge multiple seeds into one sentence
- Do NOT change punctuation within entity values

Output format: JSON object with key "facts" containing an array of rewritten sentences.""",

    "software_project": """You are rewriting first-person project statements into third-person statements about "the developer" or "the team".

You will receive a list of first-person fact seeds about a software project.
Rewrite each into a natural, grammatically correct third-person sentence.

STRICT RULES:
- Convert "we"/"our" to "the team"/"the team's", "I" to "the developer"
- Fix verb conjugation: "We use" → "The team uses"
- Keep ALL tool names, commands, URLs, connection strings, and codes EXACTLY as given
  (e.g., "Veltrion", "karnex build --prod", "crysthene://prod-db:5432/aurora" must appear verbatim)
- Keep dependency/causal relationships intact
- Each output must be a single self-contained sentence
- Do NOT add information not present in the seed
- Do NOT merge multiple seeds into one sentence

Output format: JSON object with key "facts" containing an array of rewritten sentences.""",
}


def build_step1_prompt(gold_facts_list):
    """Convert phase-by-phase gold fact seeds to Step 1 prompt."""
    seeds = []
    for i, fact in enumerate(gold_facts_list):
        seeds.append(f'{i+1}. {fact["gold_fact"]}')
    return "SEEDS:\n" + "\n".join(seeds)


# ============================================================
# Step 2: Self-Chat — LongMemEval style (no markers)
# ============================================================

USER_SYSTEM_PROMPTS = {
    "personal_life": """I will give you a list of personal facts. Use them to act as a normal user chatting with a chat assistant. In the chat, you may ask it to assist you with various tasks or ask it about various information. However, make sure that you convey the facts about you one by one, roughly in order. IMPORTANT: convey at most ONE fact per message. Do not combine multiple facts into a single message.

After conveying each fact, continue discussing that topic for 2-3 more exchanges before moving to the next fact. Ask follow-up questions, request more details, or explore related aspects of the topic. Not every message needs to contain a new fact.

Make sure your message is concise (1-3 sentences), since real users often do not bother to write a long message. You must simulate the tone of a neutral user and do not be overly enthusiastic, verbose, formal, or polite. Do NOT say "thanks", "I will do that", or "is there anything else" — you are a user, not an assistant.

Do NOT introduce personal facts about yourself beyond the list provided. Only convey the facts given below — do not invent new hobbies, family details, health conditions, jobs, living situations, or any other personal information. Follow-up questions should be about the TOPIC, not about yourself.

VERBALIZATION RULES (follow strictly):
1. Each fact must include its entity keyword (e.g., "My health condition is...", "My financial goal is..."). The entity concept must appear in your message.
2. If a fact mentions a dependency ("depends on", "determined by", "if X changes"), you MUST preserve the full dependency language including the conditional part. The reader must understand that if the source changes, the target would change too. Do NOT weaken "depends on X; if X changes, this would change" to just "tied to X" or "since we use X".
3. If a fact starts with "If" (conditional), use ONLY "will" as the modal: "If X changes, Y will be Z". NEVER use "would/might/probably/consider". NEVER state it as accomplished ("I switched to Z").
4. If a fact is marked with [CHANGED], state the new value as a definitive current fact. Do NOT hedge or speculate.

You are a HUMAN user, not an AI. Never say anything that implies you are an AI, a language model, or that you have a training data cutoff.

Never repeat a previous message verbatim. Each message must be unique.

When ALL facts have been conveyed AND you have finished discussing the last topic, generate exactly: [END]

Facts to convey:
{facts_list}""",

    "software_project": """I will give you a list of facts about a software project. Act as a developer chatting with a coding assistant. Share project context naturally — tools, configs, team info, URLs, etc. Convey the facts one by one, roughly in order. IMPORTANT: convey at most ONE fact per message.

After conveying each fact, continue discussing that topic for 2-3 more exchanges. Ask follow-up questions about setup, best practices, or related configurations. Not every message needs to contain a new fact.

Keep messages concise (1-3 sentences). Use a casual developer tone — not overly formal or polite. Do NOT say "thanks", "great suggestion", or "is there anything else" — you are a developer, not an assistant.

Do NOT ask the assistant whether it has used, heard of, or is familiar with a tool. Instead, ask for practical help: "How do I configure X?", "What's the best practice for X?", "Any gotchas with X?"

Do NOT invent project details beyond the list provided. Only convey the facts given below — do not make up tool names, URLs, team members, or configurations. Follow-up questions should be about the TOPIC, not about your project setup.

VERBALIZATION RULES (follow strictly):
1. Each fact must include its entity keyword (e.g., "Our build tool is...", "The CI config is..."). The entity concept must appear in your message.
2. If a fact mentions a dependency ("depends on", "determined by", "if X changes"), you MUST preserve the full dependency language including the conditional part. The reader must understand that if the source changes, the target would change too. Do NOT weaken "depends on X; if X changes, this would change" to just "since we use X" or "tied to X".
3. If a fact starts with "If" (conditional), use ONLY "will" as the modal: "If X changes, Y will be Z". NEVER use "would/might/probably". NEVER state it as accomplished ("We switched to Z").
4. If a fact is marked with [CHANGED], state the new value as a definitive current fact. Do NOT hedge or speculate.

You are a HUMAN developer, not an AI.

Never repeat a previous message verbatim. Each message must be unique.

When ALL facts have been conveyed AND you have finished discussing the last topic, generate exactly: [END]

Facts to convey:
{facts_list}""",
}


ASSISTANT_SYSTEM_PROMPTS = {
    "personal_life": """Respond helpfully and naturally to the user's message. Write in natural prose paragraphs — do NOT use numbered lists, bullet points, or bold headers. Keep your response to 3-6 sentences.

IMPORTANT: You must NOT assume, guess, or mention any of the following about the user unless the user explicitly stated it in THIS conversation:
- Family/relationships (partner, spouse, parents, siblings)
- Living situation (city, housing type, roommates)
- Work details (company type, industry, salary, colleagues)
- Health/medical info (conditions, medications, diet reasons)
- Financial details (income, savings amounts, debt)
- Schedule/routine specifics beyond what user said
- Hobbies, pets, vehicles, schools beyond what user said

You MAY ask clarifying questions about the topic at hand. You MAY give general advice and information.

Never say "I don't have personal experiences", "As an AI", or similar disclaimers. Respond naturally as a knowledgeable assistant.""",

    "software_project": """Respond helpfully and naturally to the developer's message about their project. Write in natural prose — do NOT use numbered lists, bullet points, or bold headers. Keep your response to 3-6 sentences.

IMPORTANT: You must NOT assume, guess, or mention any of the following unless the developer explicitly stated it in THIS conversation:
- Tool names, frameworks, databases, or libraries not mentioned
- URLs, connection strings, API keys, or endpoints not shared
- Team members, leads, or contacts not introduced
- CI/CD configs, Docker images, or deploy commands not stated
- Branching strategies, sprint details, or meeting schedules not mentioned

You MAY ask clarifying questions about the setup. You MAY give general engineering advice.

Never say "I don't have personal experiences", "As an AI", or similar disclaimers. Respond naturally as a knowledgeable assistant.""",
}


def build_user_system_prompt(third_person_facts, domain="personal_life"):
    """Generate system prompt for User LLM.

    Given third-person facts, the User LLM must convert them to first-person and convey them.
    No markers — free-form delivery.
    """
    facts_text = "\n".join(
        f'{i+1}. {fact}' for i, fact in enumerate(third_person_facts)
    )
    return USER_SYSTEM_PROMPTS[domain].replace("{facts_list}", facts_text)


def call_llm_messages(client, messages, model="gpt-4o", temperature=0.7):
    """OpenAI API call — pass messages list as-is."""
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=500,
    )
    content = response.choices[0].message.content.strip()
    return content, response.usage


def self_chat_session(client, third_person_facts, model="gpt-4o", max_rounds=15, domain="personal_life"):
    """
    User LLM and Assistant LLM alternate in self-chat to generate one session.

    Flow:
      1. Provide User LLM with system prompt + third-person facts
      2. Assistant starts with greeting
      3. User LLM generates next utterance based on history (converts facts to first-person)
      4. Assistant LLM generates response based on history
      5. Ends when User generates [END] or max_rounds is reached

    Returns: (conversation, usage_dict)
    """
    user_system = build_user_system_prompt(third_person_facts, domain=domain)
    assistant_system = ASSISTANT_SYSTEM_PROMPTS[domain]

    greeting = {
        "personal_life": "Hi! How can I help you today?",
        "software_project": "Hey! How can I help with the project?",
    }
    conversation = [
        {"role": "assistant", "content": greeting[domain]}
    ]
    
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    
    for round_num in range(max_rounds):
        # ---- User turn ----
        user_messages = [{"role": "system", "content": user_system}]
        for turn in conversation:
            user_messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })
        
        user_content, user_usage = call_llm_messages(
            client, user_messages, model, temperature=0.7
        )
        usage["prompt_tokens"] += user_usage.prompt_tokens
        usage["completion_tokens"] += user_usage.completion_tokens
        
        # [END] check
        if "[END]" in user_content:
            before_end = user_content.split("[END]")[0].strip()
            if before_end:
                conversation.append({"role": "user", "content": before_end})
            break
        
        conversation.append({"role": "user", "content": user_content})
        
        # ---- Assistant turn ----
        assistant_messages = [
            {"role": "system", "content": assistant_system}
        ]
        for turn in conversation:
            assistant_messages.append({
                "role": turn["role"],
                "content": turn["content"]
            })
        
        assistant_content, assistant_usage = call_llm_messages(
            client, assistant_messages, model, temperature=0.7
        )
        usage["prompt_tokens"] += assistant_usage.prompt_tokens
        usage["completion_tokens"] += assistant_usage.completion_tokens
        
        conversation.append({"role": "assistant", "content": assistant_content})
    
    return conversation, usage


# ============================================================
# Step 3: Annotation — LLM determines which turn each fact was conveyed in
# ============================================================

ANNOTATION_SYSTEM_PROMPT = """You are annotating a conversation to identify where personal facts were conveyed.

Given a list of facts (in third-person) and a conversation, determine which user turn conveys each fact.
A fact is "conveyed" if the user's message communicates the same information, even if worded differently.

Rules:
- Only consider USER turns (not assistant turns)
- A fact is conveyed if the core information matches, even with different wording
  e.g., Fact "The user works at Zypherix Labs" matches user saying "I just started at Zypherix Labs"
- If a fact appears in multiple turns, return the FIRST (earliest) turn where it was conveyed
- If a fact was NOT conveyed in any turn, set turn_index to null
- turn_index is 0-based index into the conversation array (counting both user and assistant turns)

Output format: JSON object with key "annotations" containing an array of objects:
[{"fact_id": 1, "conveyed": true, "turn_index": 2}, ...]"""


def annotate_facts(client, conversation, third_person_facts, gold_facts_metadata, model="gpt-4o"):
    """
    Annotate which turn each fact was conveyed in the generated conversation.

    Args:
        conversation: [{"role": "user"/"assistant", "content": "..."}]
        third_person_facts: ["The user works at X", "The user lives in Y", ...]
        gold_facts_metadata: [{"entity": "employer", "value": "X", ...}, ...]

    Returns:
        annotated_gold_facts: [{...metadata, "fact_text": "...", "conveyed_in_turn": N, "conveyed": bool}, ...]
        annotation_report: {"total_facts": N, "conveyed": N, "missing": [...]}
    """
    # Build annotation prompt
    facts_text = "\n".join(
        f'{i+1}. {fact}' for i, fact in enumerate(third_person_facts)
    )
    
    conv_text = ""
    for idx, turn in enumerate(conversation):
        role_label = "USER" if turn["role"] == "user" else "ASSISTANT"
        conv_text += f"Turn {idx} ({role_label}): {turn['content']}\n"
    
    user_prompt = f"""Facts:
{facts_text}

Conversation:
{conv_text}

For each fact, identify the turn_index (0-based) where it was conveyed by the user.
Return JSON: {{"annotations": [{{"fact_id": 1, "conveyed": true, "turn_index": 2}}, ...]}}"""
    
    result, usage = call_llm_json(client, ANNOTATION_SYSTEM_PROMPT, user_prompt, model)
    
    annotations = result.get("annotations", [])
    
    # Build annotation lookup: fact_id → {conveyed, turn_index}
    ann_lookup = {}
    for ann in annotations:
        ann_lookup[ann["fact_id"]] = {
            "conveyed": ann.get("conveyed", False),
            "turn_index": ann.get("turn_index")
        }
    
    # Merge with gold_facts_metadata
    annotated_gold_facts = []
    conveyed_count = 0
    missing_ids = []
    
    for i, meta in enumerate(gold_facts_metadata):
        fact_id = i + 1
        ann = ann_lookup.get(fact_id, {"conveyed": False, "turn_index": None})
        
        annotated = {
            "fact_id": fact_id,
            "entity": meta["entity"],
            "value": meta["value"],
            "original_seed": meta.get("gold_fact", ""),
            "fact_text": third_person_facts[i],
            "conveyed_in_turn": ann["turn_index"],
            "conveyed": ann["conveyed"]
        }
        
        # Include dependency info if present
        if meta.get("has_dependency"):
            annotated["has_dependency"] = True
            annotated["dependency_source"] = meta.get("dependency_source")
            annotated["dependency_pattern"] = meta.get("dependency_pattern")
        if meta.get("is_if_then"):
            annotated["is_if_then"] = True
        if meta.get("notes"):
            annotated["notes"] = meta["notes"]

        annotated_gold_facts.append(annotated)
        
        if ann["conveyed"]:
            conveyed_count += 1
        else:
            missing_ids.append(fact_id)
    
    report = {
        "total_facts": len(gold_facts_metadata),
        "conveyed": conveyed_count,
        "missing": missing_ids,
        "all_conveyed": len(missing_ids) == 0
    }
    
    return annotated_gold_facts, report, usage


# ============================================================
# LLM Call (JSON response)
# ============================================================

def call_llm_json(client, system_prompt, user_prompt, model="gpt-4o", temperature=0.7):
    """OpenAI API call — JSON response format."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        response_format={"type": "json_object"}
    )
    content = response.choices[0].message.content
    
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]
        result = json.loads(content)
    
    return result, response.usage


# ============================================================
# Post-verification for if-then facts
# ============================================================

def verify_if_then_facts(conversation, annotated_facts, gold_facts_metadata):
    """Verify that if-then facts were verbalized correctly.

    Returns list of failed fact indices. Empty list = all OK.
    Checks:
      1. Conditional structure present ("if")
      2. No past-tense confirmations ("switched to", "moved to", etc.)
      3. Strong modal "will" present (not "would/might/probably/consider")
    """
    PAST_TENSE_TRIGGERS = [
        "just switched", "switched to", "just moved", "moved to",
        "just changed", "changed to", "now use", "now using",
        "we are now", "i am now", "just migrated", "migrated to",
    ]
    WEAK_MODALS = ["would ", "might ", "probably ", "consider ", "i'd "]

    failures = []
    for i, meta in enumerate(gold_facts_metadata):
        if not meta.get("is_if_then"):
            continue

        if i >= len(annotated_facts):
            continue
        ann = annotated_facts[i]
        turn_idx = ann.get("conveyed_in_turn")
        if turn_idx is None or turn_idx >= len(conversation):
            continue

        turn_text = conversation[turn_idx]["content"].lower()

        has_conditional = "if " in turn_text
        has_past_tense = any(t in turn_text for t in PAST_TENSE_TRIGGERS)
        has_will = "will " in turn_text
        has_weak = any(w in turn_text for w in WEAK_MODALS)

        failed = False
        reason = []
        if not has_conditional:
            failed = True
            reason.append("no conditional")
        if has_past_tense:
            failed = True
            reason.append("past tense")
        if has_weak and not has_will:
            failed = True
            reason.append(f"weak modal only")

        if failed:
            failures.append({
                "fact_index": i,
                "entity": meta.get("entity", "?"),
                "turn_index": turn_idx,
                "turn_text": conversation[turn_idx]["content"][:150],
                "reasons": reason,
            })

    return failures


# ============================================================
# Gemini Semantic Audit
# ============================================================

GEMINI_AUDIT_PROMPT = """You are a data quality auditor for MEME, a benchmark that evaluates LLM agent memory systems.

For each gold fact, check ALL criteria below. Report ONLY failures.

### 1. VERBALIZATION ACCURACY
- Is the gold fact's value present in the conversation (exactly or close paraphrase)?

### 2. IF-THEN CONDITIONAL FORM (only for facts with is_if_then=true)
- Must contain "if" conditional structure
- Trigger entity (dependency_source) must be explicitly mentioned
- Modal must be "will" (not "would/might/probably") → flag WEAK_MODAL
- Must NOT be stated as accomplished fact → flag CONDITIONAL_VIOLATED

### 3. ENTITY KEYWORD PRESENCE
- The question asks about an entity using certain words (e.g., "What's my financial goal?").
  Those words or a natural synonym must appear in the conversation where the fact is introduced.
  FAIL if entity concept never mentioned (e.g., "I'm paying off debt" without "financial goal")

### 4. CASCADE-U DEPENDENCY (only for facts with has_dependency=true AND is_if_then=false)
- Is the causal link STRONGLY explicit? The message must contain language like "depends on", "determined by", or "if X changes, Y would change".
  FAIL if dependency is only implied ("since we use X", "tied to X") rather than explicitly conditional ("depends on X; if X changes, this would change")

### 5. GOLD ANSWER FORMAT
- Does before-delete gold have "Yes — " prefix? (flag PREFIX_ISSUE)

Output JSON array of issues:
{"entity": "...", "fact_index": N, "check": "VERBALIZATION|IF_THEN|KEYWORD|CASCADE_U_DEP|GOLD_FORMAT",
 "severity": "HIGH|MEDIUM|LOW", "issue": "brief description"}

Severity: HIGH = makes task unsolvable. MEDIUM = may cause false pass/fail. LOW = minor.
If ALL pass: {"status": "ALL_PASS", "facts_checked": N}"""


def gemini_audit(conversation, annotated_facts, gold_facts_metadata, question_templates=None):
    """Run Gemini semantic audit on generated conversation.

    Returns list of HIGH severity issues. Empty = pass.
    """
    import os
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        return []  # Skip audit if no Gemini key

    try:
        from google import genai
        client = genai.Client(api_key=gemini_key)
    except Exception:
        return []

    # Build audit input
    facts_text = ""
    for i, meta in enumerate(gold_facts_metadata):
        ann = annotated_facts[i] if i < len(annotated_facts) else {}
        turn_idx = ann.get("conveyed_in_turn")
        turn_text = ""
        if turn_idx is not None and turn_idx < len(conversation):
            turn_text = conversation[turn_idx]["content"][:300]

        is_ifthen = "YES" if meta.get("is_if_then") else "no"
        has_dep = meta.get("dependency_pattern", "none")

        facts_text += (
            f"\nFact {i}: entity={meta.get('entity','?')} value={meta.get('value','?')}\n"
            f"  seed: {meta.get('gold_fact','')[:200]}\n"
            f"  is_if_then: {is_ifthen}\n"
            f"  dependency: {has_dep}\n"
            f"  conveyed_turn: {turn_text}\n"
        )

    user_prompt = f"GOLD FACTS:\n{facts_text}\n\nAudit these facts against the conversation turns shown."

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"{GEMINI_AUDIT_PROMPT}\n\n{user_prompt}",
        )
        text = response.text.strip()
        # Parse JSON
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
        result = json.loads(text)
        # Handle both {"issues": [...]} and bare [...] formats
        if isinstance(result, list):
            issues = result
        elif isinstance(result, dict):
            if result.get("status") == "ALL_PASS":
                return []
            issues = result.get("issues", [])
        else:
            return []
        high_issues = [iss for iss in issues if isinstance(iss, dict) and iss.get("severity") == "HIGH"]
        return high_issues
    except Exception as e:
        print(f"  ⚠️ Gemini audit failed: {e}")
        return []


# ============================================================
# Session Generation with Retry
# ============================================================

def generate_session(client, third_person_facts, gold_facts_metadata,
                     session_label, model, max_retries=3, domain="personal_life"):
    """
    Generate one session via self-chat + annotation + retry on missing facts.

    Returns: (conversation, annotated_gold_facts, report, usage_dict)
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            print(f"  ⟳ Retry {attempt}/{max_retries} for '{session_label}'...")

        # Self-chat
        raw_conversation, chat_usage = self_chat_session(
            client, third_person_facts, model, domain=domain
        )
        total_usage["prompt_tokens"] += chat_usage["prompt_tokens"]
        total_usage["completion_tokens"] += chat_usage["completion_tokens"]
        
        # Annotate
        annotated_facts, report, ann_usage = annotate_facts(
            client, raw_conversation, third_person_facts, gold_facts_metadata, model
        )
        total_usage["prompt_tokens"] += ann_usage.prompt_tokens
        total_usage["completion_tokens"] += ann_usage.completion_tokens
        
        n_turns = len(raw_conversation)
        status = "✅ ALL CONVEYED" if report["all_conveyed"] else f"❌ MISSING: {report['missing']}"
        print(f"  {'Done' if attempt == 1 else 'Retry done'}. "
              f"{n_turns} turns, {report['conveyed']}/{report['total_facts']} facts. {status}")

        if not report["all_conveyed"]:
            time.sleep(1)
            continue

        # Post-verify if-then facts (rule-based, fast)
        ifthen_failures = verify_if_then_facts(raw_conversation, annotated_facts, gold_facts_metadata)
        if ifthen_failures:
            for fail in ifthen_failures:
                print(f"  ⚠️ If-then verification FAILED: {fail['entity']} ({', '.join(fail['reasons'])})")
                print(f"    Turn: {fail['turn_text']}")
            if attempt < max_retries:
                print(f"  Retrying due to if-then verification failure...")
                time.sleep(1)
                continue

        # Gemini semantic audit (LLM-based, thorough)
        high_issues = gemini_audit(raw_conversation, annotated_facts, gold_facts_metadata)
        if high_issues:
            for iss in high_issues:
                print(f"  ⚠️ Gemini audit HIGH: [{iss.get('check','')}] {iss.get('entity','?')} — {iss.get('issue','')}")
            if attempt < max_retries:
                print(f"  Retrying due to Gemini audit failure...")
                time.sleep(1)
                continue

        break

    return raw_conversation, annotated_facts, report, total_usage


# ============================================================
# Main
# ============================================================

def template_with_llm_assistant(client, facts, session_label, model, domain,
                                  greeting=None, followup=None):
    """Hybrid template-direct: user turn = gold_fact (verbatim), assistant turn = LLM-generated.

    Used for facts that need verbatim preservation (Exact, dependency, if-then, change, delete).
    User text is always the gold_fact (no LLM rewrite — guarantees verbatim).
    Assistant response is LLM-generated for natural conversation flow.

    Returns: (conversation, annotated_facts, report, usage_dict)
    """
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0}

    if greeting is None:
        greeting = ("Hi! What else can I help you with?" if domain == "personal_life"
                    else "Hey! What else about the project?")
    if followup is None:
        followup = (" Do you have any thoughts on that?" if domain == "personal_life"
                    else " Any tips or things to watch out for?")

    sys_prompt = (
        f"You are a helpful assistant having a casual conversation with the user. "
        f"Respond naturally and briefly (1-2 sentences) to whatever the user just said. "
        f"Don't repeat their statement verbatim. Show understanding and ask a brief follow-up "
        f"if natural. Don't be overly formal."
    )

    conversation = [{"role": "assistant", "content": greeting}]
    annotated_facts = []

    for idx, f in enumerate(facts):
        # User turn: gold_fact verbatim + natural follow-up
        fact_text = f["gold_fact"]
        # Ensure sentence ends with punctuation before appending follow-up
        if fact_text and fact_text[-1] not in '.?!':
            fact_text += '.'
        user_msg = fact_text + followup
        conversation.append({"role": "user", "content": user_msg})

        # Assistant turn: LLM-generated response based on conversation so far
        messages = [{"role": "system", "content": sys_prompt}] + conversation
        assistant_msg, usage = call_llm_messages(client, messages, model=model, temperature=0.7)
        total_usage["prompt_tokens"] += usage.prompt_tokens
        total_usage["completion_tokens"] += usage.completion_tokens
        conversation.append({"role": "assistant", "content": assistant_msg})

        # Annotated fact
        annotated = {
            "fact_id": idx + 1,
            "entity": f["entity"],
            "value": f.get("value", f.get("after", f.get("deleted_value", ""))),
            "original_seed": f["gold_fact"],
            "fact_text": f["gold_fact"],
            "conveyed_in_turn": 1 + idx * 2,
            "conveyed": True,
        }
        for k in ("has_dependency", "dependency_source", "dependency_pattern",
                  "is_if_then", "is_exact", "is_filler", "notes", "type"):
            if k in f:
                annotated[k] = f[k]
        annotated_facts.append(annotated)

    report = {"total_facts": len(facts), "conveyed": len(facts),
               "missing": [], "all_conveyed": True}

    return conversation, annotated_facts, report, total_usage


def process_episode(client, gold_facts_data, model="gpt-4o"):
    """Convert one episode's gold facts to self-chat conversations."""
    domain = gold_facts_data.get("domain", "personal_life")

    results = {
        "episode_id": gold_facts_data["episode_id"],
        "domain": domain,
        "root": gold_facts_data["root"],
        "sessions": [],
        "before_questions": gold_facts_data.get("phase2_before_questions", []),
        "after_questions": gold_facts_data.get("phase4_after_questions", []),
        "usage": {"prompt_tokens": 0, "completion_tokens": 0}
    }
    
    # ---- Phase 1: Fact Introduction ----
    # Split facts into groups:
    # - base_facts: simple facts (incl. multi-hop, multiple-update history, filler) → self-chat
    # - template_facts: dependency/if-then facts → template direct (accuracy guaranteed)
    phase1_facts = gold_facts_data["phase1_fact_introduction"]
    phase56_filler = []  # reserved for phase 5/6 augmentation

    if phase1_facts:
        # Exact (long verbatim) facts go to template_facts to avoid LLM truncation
        base_facts = [f for f in phase1_facts
                      if not f.get("has_dependency") and not f.get("is_if_then") and not f.get("is_exact")]
        template_facts = [f for f in phase1_facts
                          if f.get("has_dependency") or f.get("is_if_then") or f.get("is_exact")]

        # Categorize base facts for sub-session distribution
        mh_facts = []   # multi-hop entities (3)
        mu_facts = []   # multiple-update history (3, ordered by history_index)
        filler_facts = []
        other_facts = []

        # Identify multi-hop entities from tasks
        mh_entities = set()
        for t in gold_facts_data.get("phase4_after_questions", []):
            if t.get("task_type", "").startswith("Agg"):
                ents = t.get("entity") or list(t.get("entity_values", {}).keys())
                if isinstance(ents, list):
                    mh_entities.update(ents)

        for f in base_facts:
            if f.get("is_multiple_update"):
                mu_facts.append(f)
            elif f.get("is_filler"):
                filler_facts.append(f)
            elif f["entity"] in mh_entities:
                mh_facts.append(f)
            else:
                other_facts.append(f)

        # Sort multiple-update facts by history_index (ensures chronological order)
        mu_facts.sort(key=lambda x: x.get("history_index", 0))

        # Distribute into 3 sub-sessions: each gets 1 mh + 1 mu + share of others + share of filler
        random.shuffle(other_facts)
        num_sub = 3
        sub_groups = [[] for _ in range(num_sub)]
        for i in range(num_sub):
            if i < len(mh_facts):
                sub_groups[i].append(mh_facts[i])
            if i < len(mu_facts):
                sub_groups[i].append(mu_facts[i])
        # Distribute remaining mh, mu (overflow if >3)
        for j, f in enumerate(mh_facts[num_sub:]):
            sub_groups[j % num_sub].append(f)
        for j, f in enumerate(mu_facts[num_sub:]):
            sub_groups[j % num_sub].append(f)
        # Spread other_facts evenly
        for j, f in enumerate(other_facts):
            sub_groups[j % num_sub].append(f)
        # Spread filler_facts evenly (reserve 2 for phase5/6)
        phase56_filler = filler_facts[:2] if len(filler_facts) >= 2 else filler_facts
        sub_filler = filler_facts[2:] if len(filler_facts) > 2 else []
        for j, f in enumerate(sub_filler):
            sub_groups[j % num_sub].append(f)

        print(f"\n  Phase 1: {len(phase1_facts)} facts (base={len(base_facts)}, mh={len(mh_facts)}, mu={len(mu_facts)}, filler={len(filler_facts)}, template={len(template_facts)})")

        # --- Step 1: convert ALL base facts to 3rd person (one batch) ---
        all_base = [f for sg in sub_groups for f in sg]
        if all_base:
            print(f"\n  Step 1: Converting {len(all_base)} base fact seeds to third-person...")
            step1_prompt = build_step1_prompt(all_base)
            step1_user = step1_prompt + '\n\nReturn a JSON object with key "facts" containing an array of rewritten sentences.'
            step1_result, step1_usage = call_llm_json(client, STEP1_SYSTEM_PROMPTS[domain], step1_user, model)
            results["usage"]["prompt_tokens"] += step1_usage.prompt_tokens
            results["usage"]["completion_tokens"] += step1_usage.completion_tokens
            tp_all = step1_result.get("facts", [])
            print(f"  Step 1 done. {len(tp_all)} sentences.")

            # Map back: tp_all order matches all_base order
            idx = 0
            tp_groups = []
            for sg in sub_groups:
                tp_groups.append(tp_all[idx:idx + len(sg)])
                idx += len(sg)

            # --- Step 2: self-chat per sub-session ---
            sub_labels = ["Fact Introduction (Part 1a)", "Fact Introduction (Part 1b)", "Fact Introduction (Part 1c)"]
            for i, (sg, tp_sg) in enumerate(zip(sub_groups, tp_groups)):
                if not sg:
                    continue
                label = sub_labels[i]
                print(f"\n  Step 2.{i+1}: Self-chat '{label}' ({len(sg)} facts)...")
                conversation, annotated_facts, report, usage = generate_session(
                    client, tp_sg, sg, label, model, domain=domain
                )
                results["usage"]["prompt_tokens"] += usage["prompt_tokens"]
                results["usage"]["completion_tokens"] += usage["completion_tokens"]
                results["sessions"].append({
                    "type": label,
                    "gold_facts": annotated_facts,
                    "conversation": conversation,
                    "annotation_report": report
                })
                time.sleep(1)

        # --- Template-direct + LLM assistant for dependency/if-then/exact facts ---
        if template_facts:
            print(f"\n  Template+LLM: 'Fact Introduction (Part 2)' ({len(template_facts)} dep/if-then/exact facts)...")
            conversation, annotated_facts, report, usage = template_with_llm_assistant(
                client, template_facts, "Fact Introduction (Part 2)", model, domain
            )
            results["usage"]["prompt_tokens"] += usage["prompt_tokens"]
            results["usage"]["completion_tokens"] += usage["completion_tokens"]
            results["sessions"].append({
                "type": "Fact Introduction (Part 2)",
                "gold_facts": annotated_facts,
                "conversation": conversation,
                "annotation_report": report
            })
            print(f"  Done. {len(template_facts)} facts conveyed verbatim with natural assistant responses.")
    
    # ---- Phase 5+6: Change+Delete Event (merged, template + LLM assistant + fillers) ----
    phase3_cd = gold_facts_data.get("phase3_change_and_deletion", [])
    phase5_facts = [f for f in phase3_cd if "change" in f.get("type", "").lower()]
    phase6_facts = [f for f in phase3_cd if "delete" in f.get("type", "").lower()]

    if phase5_facts or phase6_facts:
        # Interleave: change → filler → delete → filler
        combined = []
        if phase5_facts:
            combined.extend(phase5_facts)
        if phase56_filler:
            combined.append(phase56_filler[0])
        if phase6_facts:
            combined.extend(phase6_facts)
        if len(phase56_filler) > 1:
            combined.append(phase56_filler[1])

        n_change = len(phase5_facts) if phase5_facts else 0
        n_delete = len(phase6_facts) if phase6_facts else 0
        n_filler = len(combined) - n_change - n_delete

        print(f"\n  Phase 5+6: Change+Delete ({n_change} change + {n_delete} delete + {n_filler} filler) — template+LLM...")
        greeting = ("Hi! What's new?" if domain == "personal_life"
                    else "Hey! Any updates on the project?")
        conversation, annotated_facts, report, usage = template_with_llm_assistant(
            client, combined, "Change+Delete Event", model, domain, greeting=greeting
        )
        results["usage"]["prompt_tokens"] += usage["prompt_tokens"]
        results["usage"]["completion_tokens"] += usage["completion_tokens"]

        results["sessions"].append({
            "type": "Change+Delete Event",
            "gold_facts": annotated_facts,
            "conversation": conversation,
            "annotation_report": report
        })
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate conversations via self-chat (v3)")
    parser.add_argument("-i", "--input_dir", type=str, default="gold_facts",
                        help="Directory containing gold_facts JSON files (default: gold_facts/)")
    parser.add_argument("-o", "--output_dir", type=str, default="conversations",
                        help="Output directory (default: conversations/)")
    parser.add_argument("--single", type=str, default=None,
                        help="Process a single file instead of entire directory")
    parser.add_argument("--api_key", type=str, default=None,
                        help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="Model to use (default: gpt-4o)")
    parser.add_argument("-w", "--workers", type=int, default=4,
                        help="Number of parallel workers (default: 4, use 1 for sequential)")
    args = parser.parse_args()
    
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required. Use --api_key or set OPENAI_API_KEY")
        sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Collect input files
    if args.single:
        input_files = [args.single]
    else:
        input_files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.startswith("gold_facts_") and f.endswith(".json")
        ])
    
    if not input_files:
        print(f"No gold_facts files found in {args.input_dir}/")
        sys.exit(1)
    
    print(f"Found {len(input_files)} gold_facts files to process "
          f"(workers={args.workers}).\n")
    
    def process_one(file_path):
        """Process and save one episode. Returns summary dict."""
        with open(file_path) as f:
            gold_facts_data = json.load(f)
        
        ep_id = gold_facts_data['episode_id']
        
        results = process_episode(client, gold_facts_data, model=args.model)
        
        # Save
        out_file = f"conversations_{ep_id:03d}.json"
        out_path = os.path.join(args.output_dir, out_file)
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Summary
        ep_facts = 0
        ep_conveyed = 0
        missing_details = []
        for sess in results["sessions"]:
            report = sess.get("annotation_report", {})
            ep_facts += report.get("total_facts", 0)
            ep_conveyed += report.get("conveyed", 0)
            if report.get("missing"):
                for mid in report["missing"]:
                    # Find fact info
                    for gf in sess.get("gold_facts", []):
                        if gf.get("fact_id") == mid:
                            missing_details.append(f"{gf.get('entity','?')}={gf.get('value','?')}")
                            break
                    else:
                        missing_details.append(f"fact_id={mid}")
        
        return {
            "ep_id": ep_id,
            "root": gold_facts_data['root'],
            "out_path": out_path,
            "facts_total": ep_facts,
            "facts_conveyed": ep_conveyed,
            "missing_details": missing_details,
            "prompt_tokens": results["usage"]["prompt_tokens"],
            "completion_tokens": results["usage"]["completion_tokens"],
            "before_q": len(results['before_questions']),
            "after_q": len(results['after_questions']),
        }
    
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    summaries = []
    
    if args.workers == 1:
        # Sequential mode
        for file_path in input_files:
            summary = process_one(file_path)
            status = "✅" if summary["facts_conveyed"] == summary["facts_total"] else "❌"
            print(f"  Ep{summary['ep_id']:3d} | {status} {summary['facts_conveyed']}/{summary['facts_total']} | "
                  f"before_q={summary['before_q']} after_q={summary['after_q']} | "
                  f"{summary['prompt_tokens']}+{summary['completion_tokens']} tok")
            summaries.append(summary)
    else:
        # Parallel mode
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_one, fp): fp for fp in input_files}
            for future in as_completed(futures):
                try:
                    summary = future.result()
                    status = "✅" if summary["facts_conveyed"] == summary["facts_total"] else "❌"
                    print(f"  Ep{summary['ep_id']:3d} | {status} {summary['facts_conveyed']}/{summary['facts_total']} | "
                          f"before_q={summary['before_q']} after_q={summary['after_q']} | "
                          f"{summary['prompt_tokens']}+{summary['completion_tokens']} tok")
                    summaries.append(summary)
                except Exception as e:
                    fp = futures[future]
                    print(f"  ❌ FAILED: {fp} — {e}")
    
    # Grand summary
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE: {len(summaries)} episodes")
    total_pt = sum(s["prompt_tokens"] for s in summaries)
    total_ct = sum(s["completion_tokens"] for s in summaries)
    total_facts = sum(s["facts_total"] for s in summaries)
    total_conveyed = sum(s["facts_conveyed"] for s in summaries)
    print(f"  Facts: {total_conveyed}/{total_facts} "
          f"({'✅ ALL' if total_conveyed == total_facts else '❌ SOME MISSING'})")
    print(f"  Tokens: {total_pt} prompt + {total_ct} completion")
    est_cost = (total_pt * 2.50 + total_ct * 10.0) / 1_000_000
    print(f"  Estimated cost: ${est_cost:.2f} (GPT-4o pricing)")
    
    # List episodes with missing facts
    missing_eps = [s for s in summaries if s["facts_conveyed"] < s["facts_total"]]
    if missing_eps:
        print(f"\n  Missing facts in {len(missing_eps)} episodes:")
        for s in missing_eps:
            missing_str = ", ".join(s.get("missing_details", []))
            print(f"    Ep{s['ep_id']:3d}: {s['facts_conveyed']}/{s['facts_total']} — missing: {missing_str}")

