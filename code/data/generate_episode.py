"""
=============================================================================
Step 3: Subgraph Sampling + Value Assignment + Task Assignment (v2)
=============================================================================

Changes from v1:
  - 2-hop cascade: Task A/B now assignable to 2-hop targets (not just 1-hop)
  - Agg: Task F uses 3 entities, can cross chain/orphan boundaries
  - Exact recall: New orphan entities added (phone_number, address, etc.)
  - No-overlap: E, F, G targets are mutually exclusive
  - RETAIN removed entirely

Pipeline position:
  1. Entity pool definition (entity_pool.json)       ← done
  2. Dependency map definition (dependency_map.json)  ← done
  3. Subgraph sampling + value assignment             ← THIS FILE
  4. Session structure assembly
  5. Convert to NL
  6. Quality validation

Usage:
  python3 generate_episode.py
"""

import json
import random
from copy import deepcopy


# ============================================================
# Data Loading
# ============================================================

def load_data(domain="personal_life"):
    """Load entity pool and dependency map for the given domain."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    if domain == "personal_life":
        pool_file = os.path.join(here, "entity_pools", "entity_pool_personallife.json")
        dep_file = os.path.join(here, "dependency_maps", "dependency_map_personallife.json")
    elif domain == "software_project":
        pool_file = os.path.join(here, "entity_pools", "entity_pool_software.json")
        dep_file = os.path.join(here, "dependency_maps", "dependency_map_software.json")
    else:
        raise ValueError(f"Unknown domain: {domain}")

    with open(pool_file, 'r') as f:
        pool = json.load(f)
    with open(dep_file, 'r') as f:
        dep_map = json.load(f)
    return pool, dep_map


def build_entity_lookup(pool):
    """
    Convert nested structure of entity_pool.json to flat dict.
    { "employer": {definition, role, values, category}, ... }
    """
    lookup = {}
    for cat in pool['categories']:
        for e in cat['entities']:
            entry = {
                'definition': e['definition'],
                'role': e['role'],
                'values': e['values'],
                'category': cat['name']
            }
            if e.get('exact_recall'):
                entry['exact_recall'] = True
            lookup[e['id']] = entry
    return lookup


def build_edge_lookup(dep_map):
    """
    dependency_map.json → two structures:
    1) adj: source → [target1, target2, ...]
    2) edge_details: (source, target) → {pattern, declaration_example}
    """
    adj = {}
    edge_details = {}
    for edge in dep_map['edges']:
        src, tgt = edge['source'], edge['target']
        if src not in adj:
            adj[src] = []
        adj[src].append(tgt)
        edge_details[(src, tgt)] = {
            'pattern': edge['pattern'],
            'declaration_example': edge['declaration_example']
        }
    return adj, edge_details


# ============================================================
# Constants & Rules
# ============================================================

# Values that represent "none/absence".
# Excluded from Task C (update) before/after.
# "A → B" requires both to be real values. none → B is create, A → none is delete.
NULL_VALUES = {
    "none", "no pet", "not currently enrolled", "not currently learning anything",
    "none currently", "no major purchase planned",
    "no insurance", "freelance", "remote (no commute)",
    "home gym"
}


def get_active_values(entity_lookup, entity_id):
    """Return only real values, excluding NULL_VALUES."""
    return [v for v in entity_lookup[entity_id]['values']
            if v.lower() not in {nv.lower() for nv in NULL_VALUES}]


def get_exact_recall_ids(entity_lookup):
    """Return entity IDs marked as exact_recall in entity_pool."""
    return [eid for eid, info in entity_lookup.items() if info.get('exact_recall')]

# ============================================================
# Agg Triple Definitions (Task F) — per domain
# ============================================================
# Each triple has 3 semantically related entities.
# The question does NOT name the entities — agent must recall them from memory.

MULTIHOP_TRIPLES = {
    "personal_life": [
        {
            "triple": ["hobby", "sports", "club_membership"],
            "question": "What do I do in my free time? List everything you know.",
            "gold_answer_template": "The user's hobby is {hobby}, plays {sports}, and is part of {club_membership}."
        },
        {
            "triple": ["commute_method", "commute_duration", "work_location"],
            "question": "Tell me everything you know about my commute.",
            "gold_answer_template": "The user commutes by {commute_method} for {commute_duration} to {work_location}."
        },
        {
            "triple": ["diet_preference", "dietary_restriction", "food_allergy"],
            "question": "What do you know about my diet? Tell me everything.",
            "gold_answer_template": "The user follows a {diet_preference} diet, is on a {dietary_restriction} restriction, and is allergic to {food_allergy}."
        },
        {
            "triple": ["employer", "job_title", "work_schedule"],
            "question": "Tell me everything you know about my work situation.",
            "gold_answer_template": "The user works at {employer} as a {job_title} with {work_schedule} hours."
        },
        {
            "triple": ["residence_location", "housing_type", "living_arrangement"],
            "question": "What do you know about my living situation? Tell me everything.",
            "gold_answer_template": "The user lives in {residence_location} in a {housing_type}, {living_arrangement}."
        },
        {
            "triple": ["health_condition", "medication", "exercise_routine"],
            "question": "What do you know about my health and fitness? Tell me everything.",
            "gold_answer_template": "The user has {health_condition}, takes {medication}, and does {exercise_routine}."
        },
        {
            "triple": ["travel_plan", "upcoming_event", "family_event"],
            "question": "What do I have coming up? List everything you know.",
            "gold_answer_template": "The user is planning {travel_plan}, has {upcoming_event} coming up, and has {family_event}."
        },
        {
            "triple": ["work_schedule", "regular_appointment", "social_gathering"],
            "question": "What are my regular weekly commitments? List everything.",
            "gold_answer_template": "The user works {work_schedule}, has {regular_appointment}, and attends {social_gathering}."
        },
        {
            "triple": ["financial_goal", "planned_purchase", "insurance"],
            "question": "What do you know about my financial situation? Tell me everything.",
            "gold_answer_template": "The user is {financial_goal}, planning to buy {planned_purchase}, and has {insurance}."
        },
    ],
    # Triples use only orphan entities so they are always available in f_pool.
    "software_project": [
        {
            "triple": ["meeting_day", "slack_channel", "standup_format"],
            "question": "How does our team coordinate? Tell me everything you know.",
            "gold_answer_template": "Team meets {meeting_day}, uses {slack_channel} for chat, and standups are {standup_format}."
        },
        {
            "triple": ["branch_strategy", "default_branch", "release_cadence"],
            "question": "What do you know about our release process? Tell me everything.",
            "gold_answer_template": "We use {branch_strategy} branching on {default_branch}, releasing {release_cadence}."
        },
        {
            "triple": ["design_system", "test_coverage_target", "code_review_policy"],
            "question": "What are our code quality standards? List everything you know.",
            "gold_answer_template": "We use {design_system}, target {test_coverage_target} test coverage, and require {code_review_policy} for reviews."
        },
        {
            "triple": ["oncall_rotation", "alert_channel", "incident_runbook_url"],
            "question": "What do you know about our on-call setup? Tell me everything.",
            "gold_answer_template": "On-call is {oncall_rotation}, alerts go to {alert_channel}, and runbook is at {incident_runbook_url}."
        },
        {
            "triple": ["documentation_url", "onboarding_guide", "changelog_location"],
            "question": "Where can I find project documentation? List everything you know.",
            "gold_answer_template": "Docs are at {documentation_url}, onboarding guide at {onboarding_guide}, and changelog at {changelog_location}."
        },
    ],
}

# ============================================================
# Question Templates — per domain
# ============================================================
# Natural question for each entity. Used in generate_tasks().
# Every entity that can appear as a task target MUST have a template.

QUESTION_TEMPLATES = {
    "personal_life": {
        "residence_location": "Where do I live?",
        "housing_type": "What type of home do I live in?",
        "living_arrangement": "Who do I live with?",
        "commute_method": "How do I get to work?",
        "commute_duration": "How long is my commute?",
        "vehicle": "What car do I drive?",
        "employer": "Where do I work?",
        "job_title": "What is my job title?",
        "work_location": "Where is my office?",
        "work_schedule": "What's my work schedule?",
        "work_project": "What project am I working on?",
        "health_condition": "What health condition do I have?",
        "medication": "What medication am I taking?",
        "exercise_routine": "What's my exercise routine?",
        "fitness_facility": "Where do I work out?",
        "sleep_pattern": "What's my sleep schedule?",
        "diet_preference": "What's my diet?",
        "dietary_restriction": "What dietary restrictions do I have?",
        "favorite_restaurant": "What's my favorite restaurant?",
        "food_allergy": "What food allergies do I have?",
        "relationship_status": "What's my relationship status?",
        "partner": "What's my partner's name?",
        "family_event": "What family event do I have coming up?",
        "financial_goal": "What's my current financial goal?",
        "insurance": "What insurance do I have?",
        "subscription_service": "What am I subscribed to?",
        "planned_purchase": "What big purchase am I planning?",
        "travel_plan": "What trip am I planning?",
        "regular_appointment": "What regular appointments do I have?",
        "upcoming_event": "What event do I have coming up?",
        "hobby": "What's my hobby?",
        "sports": "What sport do I play?",
        "club_membership": "What club am I part of?",
        "media_consumption": "What media am I consuming?",
        "social_gathering": "What's my regular social meetup?",
        "pet": "Do I have a pet?",
        "school": "Where am I studying?",
        "skill_acquisition": "What am I currently learning?",
        "life_philosophy": "Recite my life philosophy verbatim.",
    },
    "software_project": {
        "framework": "What framework are we using?",
        "build_tool": "What's our build tool?",
        "build_command": "What's the build command?",
        "test_framework": "What testing framework do we use?",
        "test_command": "What's the test command?",
        "project_structure": "What's our project structure?",
        "dev_server_command": "How do I run the dev server?",
        "database": "What database are we using?",
        "orm_library": "What ORM do we use?",
        "model_syntax": "What's the model syntax?",
        "migration_tool": "What migration tool do we use?",
        "connection_string": "What's the database connection string?",
        "backup_schedule": "What's our backup schedule?",
        "deploy_target": "Where do we deploy?",
        "ci_config": "What's our CI configuration?",
        "deploy_command": "What's the deploy command?",
        "monitoring_url": "What's our monitoring URL?",
        "staging_url": "What's our staging URL?",
        "docker_image": "What Docker image do we use?",
        "dockerfile_path": "Where's the Dockerfile?",
        "log_drain_endpoint": "What's our log drain endpoint?",
        "team_lead": "Who's the team lead?",
        "code_reviewer": "Who reviews our code?",
        "escalation_contact": "Who's the escalation contact?",
        "approval_authority": "Who has approval authority?",
        "weekly_report_recipient": "Who receives the weekly report?",
        "auth_provider": "What auth provider do we use?",
        "auth_method": "What authentication method do we use?",
        "middleware_config": "What's the auth middleware config?",
        "token_format": "What token format do we use?",
        "login_endpoint": "What's the login endpoint?",
        "user_session_ttl": "What's the session TTL?",
        "error_log": "Recite the exact error log I shared.",
        "sprint_deadline": "When's the sprint deadline?",
        "secret_manager": "What secret manager do we use?",
        "package_manager": "What package manager do we use?",
        "branch_strategy": "What's our branching strategy?",
        "design_system": "What design system do we use?",
        "meeting_day": "When's our team meeting?",
        "slack_channel": "What's our team Slack channel?",
        "standup_format": "What's our standup format?",
        "default_branch": "What's our default branch?",
        "release_cadence": "How often do we release?",
        "test_coverage_target": "What's our test coverage target?",
        "code_review_policy": "What's our code review policy?",
        "oncall_rotation": "What's our on-call rotation?",
        "alert_channel": "What's our alert channel?",
        "incident_runbook_url": "Where's the incident runbook?",
        "documentation_url": "Where are our docs?",
        "onboarding_guide": "Where's the onboarding guide?",
        "changelog_location": "Where's the changelog?",
    },
}

# ============================================================
# Multiple-Update Templates — per domain
# ============================================================
# Questions asking for the FULL HISTORY of an entity that changed multiple times.
# Used for the new Tr task (replaces Tracking-before/after).
# Each episode picks ONE entity from the candidates and creates 3 ordered values.

MULTIPLE_UPDATE_CANDIDATES = {
    "personal_life": ["media_consumption", "partner", "subscription_service", "vehicle"],
    "software_project": ["sprint_deadline", "secret_manager", "package_manager"],
}

MULTIPLE_UPDATE_TEMPLATES = {
    "personal_life": {
        "media_consumption": "List all media I've consumed over time, from earliest to latest.",
        "partner": "List all partners I've had over time, from earliest to latest.",
        "subscription_service": "List all subscription services I've had over time, from earliest to latest.",
        "vehicle": "List all vehicles I've had over time, from earliest to latest.",
    },
    "software_project": {
        "sprint_deadline": "List all sprint deadlines we've had, from earliest to latest.",
        "secret_manager": "List all secret managers we've used over time, from earliest to latest.",
        "package_manager": "List all package managers we've used over time, from earliest to latest.",
    },
}

# ============================================================
# Derived Values — software domain only
# ============================================================
# 2-hop leaf entities whose values must be consistent with their parent.
# e.g., build_tool="Karnex Bundler" → build_command must be "karnex build --*"
# Key: (parent_entity, child_entity) → {parent_value: [child_values]}

DERIVED_VALUES = {
    ("build_tool", "build_command"): {
        "Karnex Bundler": ["karnex build --prod", "karnex build --dev", "karnex build --analyze"],
        "Tarvex CLI": ["tarvex compile --release", "tarvex compile --debug", "tarvex compile --watch"],
        "Pyronis Build": ["pyronis pack --optimize", "pyronis pack --quick", "pyronis pack --verbose"],
        "Zelthra Pack": ["zelthra pack --minify", "zelthra pack --sourcemap"],
        "Quorix Forge": ["quorix forge --target prod", "quorix forge --target dev"],
        "Dranith Compiler": ["dranith compile --emit bundle", "dranith compile --emit modules"],
    },
    ("test_framework", "test_command"): {
        "Karnex Test": ["karnex test --coverage", "karnex test --watch", "karnex test --ci"],
        "Tarvex Verify": ["tarvex verify --all", "tarvex verify --unit", "tarvex verify --integration"],
        "Pyronis Check": ["pyronis check --full", "pyronis check --quick", "pyronis check --snapshot"],
        "Zelthra Assert": ["zelthra assert --suite all", "zelthra assert --suite unit"],
        "Quorix Spec": ["quorix spec --run all", "quorix spec --run smoke"],
        "Dranith Probe": ["dranith probe --mode full", "dranith probe --mode fast"],
    },
    ("orm_library", "model_syntax"): {
        "Crysthene ORM": ["Crysthene.define('Model', {fields})", "Crysthene.table('Model', columns)"],
        "Vorathis ODM": ["Vorathis.schema('Model', {fields})", "Vorathis.collection('Model', spec)"],
        "Thalnex Query": ["Thalnex.document('Model', {fields})", "Thalnex.graph('Model', edges)"],
        "Pyravar Mapper": ["Pyravar.entity('Model', attrs)", "Pyravar.record('Model', schema)"],
        "Zyndra Access": ["Zyndra.model('Model', {fields})", "Zyndra.struct('Model', types)"],
        "Keldaris Link": ["Keldaris.object('Model', props)", "Keldaris.node('Model', shape)"],
    },
    ("ci_config", "deploy_command"): {
        "narvex-ci.yml": ["narvex deploy --env prod", "narvex deploy --env staging"],
        "zephyra-pipeline.json": ["zephyra push --target prod", "zephyra push --target preview"],
        "korinth-deploy.yaml": ["korinth release --prod", "korinth release --canary"],
        "thandrel-flow.toml": ["thandrel ship --channel stable", "thandrel ship --channel beta"],
        "velturis-ci.json": ["velturis deploy --region us-east", "velturis deploy --region ap-northeast"],
        "orinthal-build.yml": ["orinthal publish --tier production", "orinthal publish --tier sandbox"],
    },
    ("docker_image", "dockerfile_path"): {
        "narvex-registry/aurora:latest": ["deploy/Dockerfile.narvex", "infra/narvex/Dockerfile"],
        "zephyra-hub/aurora:prod": ["deploy/Dockerfile.zephyra", "infra/zephyra/Dockerfile"],
        "korinth-repo/aurora:stable": ["deploy/Dockerfile.korinth", "infra/korinth/Dockerfile"],
        "thandrel-images/aurora:release": ["deploy/Dockerfile.thandrel", "infra/thandrel/Dockerfile"],
        "velturis-cr/aurora:main": ["deploy/Dockerfile.velturis", "infra/velturis/Dockerfile"],
        "orinthal-registry/aurora:deploy": ["deploy/Dockerfile.orinthal", "infra/orinthal/Dockerfile"],
    },
    ("auth_method", "middleware_config"): {
        "OAuth 2.0 + PKCE": ["verithos-oauth-middleware v2.1", "verithos-pkce-guard v1.4"],
        "JWT Bearer": ["kryptal-jwt-verify v3.0", "kryptal-bearer-check v2.2"],
        "Session Cookie + CSRF": ["zenthos-session-middleware v1.8", "zenthos-csrf-guard v1.1"],
        "API Key + HMAC": ["phalanx-hmac-validator v2.0", "phalanx-key-guard v1.3"],
        "mTLS Client Cert": ["oathkeeper-mtls-proxy v4.1", "oathkeeper-cert-check v3.2"],
        "SAML 2.0 SSO": ["cerberix-saml-handler v2.5", "cerberix-sso-bridge v1.9"],
    },
}


# Consistency rules per domain.
# Format: (condition_entity, condition_values, constrained_entity, allowed_values)
# "If condition_entity is one of condition_values, constrained_entity must be one of allowed_values"
CONSISTENCY_RULES = {
    "personal_life": [
        # --- Relationship ---
        ("relationship_status", ["single", "divorced", "separated"], "partner", ["none"]),
        ("relationship_status", ["single", "dating"], "living_arrangement",
         ["alone", "with roommate", "with parents", "with sibling"]),
        ("relationship_status", ["single", "divorced", "separated"], "living_arrangement",
         ["alone", "with roommate", "with parents", "with sibling"]),
        # --- Housing ---
        ("housing_type", ["dormitory"], "living_arrangement", ["alone", "with roommate"]),
        ("housing_type", ["studio apartment"], "living_arrangement",
         ["alone", "with roommate", "with partner"]),
        # --- Work/commute ---
        ("work_location", ["remote"], "commute_method", ["remote (no commute)"]),
        ("work_location", ["remote"], "commute_duration", ["10 min"]),
        ("vehicle", ["none"], "commute_method",
         ["subway", "bus", "bike", "walking", "subway + bus transfer", "commuter rail", "remote (no commute)"]),
        ("commute_method", ["remote (no commute)"], "commute_duration", ["10 min"]),
        ("commute_method", ["walking"], "commute_duration",
         ["10 min", "15 min", "20 min", "30 min", "40 min"]),
        # --- Health (upstream rules first: health→exercise before exercise→facility) ---
        ("health_condition", ["none"], "medication",
         ["multivitamin", "none", "Velnorath", "Xyravine"]),
        ("health_condition", ["herniated disc"], "exercise_routine",
         ["yoga 2x/week", "swimming 2x/week", "pilates 3x/week",
          "walking daily", "none"]),
        ("health_condition", ["tendinitis"], "exercise_routine",
         ["yoga 2x/week", "swimming 2x/week", "pilates 3x/week",
          "walking daily", "cycling weekends", "none"]),
        ("health_condition", ["seasonal allergies"], "dietary_restriction",
         ["dairy-free", "nut-free", "gluten-free", "no caffeine",
          "no alcohol", "none"]),
        # --- Exercise → facility (downstream, after health→exercise) ---
        ("exercise_routine", ["none"], "fitness_facility", ["none", "home gym"]),
        ("exercise_routine", ["swimming 2x/week"], "fitness_facility",
         ["Crysthene Pool", "none", "home gym"]),
        ("exercise_routine", ["running 3x/week", "walking daily", "cycling weekends"],
         "fitness_facility",
         ["Zorathel Peak Gym", "Velthari Studio", "Pyravar CrossFit",
          "Kaelstrom Fitness", "Thandrel YMCA", "none", "home gym"]),
        # --- Dependency value constraints ---
        ("relationship_status", ["single"], "family_event",
         ["parent's birthday", "sibling's wedding", "family reunion",
          "Thanksgiving gathering", "none currently"]),
        ("relationship_status", ["dating"], "family_event",
         ["parent's birthday", "sibling's wedding", "family reunion",
          "Thanksgiving gathering", "none currently"]),
        ("relationship_status", ["separated", "divorced"], "family_event",
         ["parent's birthday", "family reunion", "Thanksgiving gathering",
          "child's school event", "none currently"]),
        ("relationship_status", ["separated", "divorced"], "financial_goal",
         ["saving for house down payment", "paying off student loans",
          "building emergency fund", "maxing out retirement fund",
          "paying off credit card debt", "saving for kids' education fund"]),
        # health→exercise and health→dietary rules moved to upstream section above
        ("school", ["online bootcamp (coding)"], "commute_method",
         ["remote (no commute)"]),
    ],
    # Software uses fictitious names — no real-world compatibility constraints.
    # 2-hop value coherence is handled by DERIVED_VALUES, not consistency rules.
    "software_project": [],
}


def check_consistency(assigned_values, domain="personal_life"):
    """Check and fix logical consistency of assigned values.
    Rules are applied in order — upstream rules (health→exercise) must come
    before downstream rules (exercise→fitness) in CONSISTENCY_RULES.
    Runs two passes to handle cascading changes.
    """
    rules = CONSISTENCY_RULES.get(domain, [])
    for _pass in range(2):
        for cond_entity, cond_vals, constrained_entity, allowed_vals in rules:
            if cond_entity in assigned_values and constrained_entity in assigned_values:
                if assigned_values[cond_entity] in cond_vals:
                    if assigned_values[constrained_entity] not in allowed_vals:
                        assigned_values[constrained_entity] = random.choice(allowed_vals)

    # Personal life only: same city for home and work → short commute
    if domain == "personal_life":
        if "residence_location" in assigned_values and "work_location" in assigned_values:
            if assigned_values["residence_location"] == assigned_values["work_location"]:
                if "commute_duration" in assigned_values:
                    assigned_values["commute_duration"] = random.choice(["10 min", "15 min"])
                if "commute_method" in assigned_values:
                    assigned_values["commute_method"] = random.choice(["walking", "bike", "bus"])

    return assigned_values


# ============================================================
# Step 3-1: Root Selection
# ============================================================

def select_root(entity_lookup, adj, exclude_roots=None):
    """
    Select one root entity.
    Conditions: active values >= 2, not in exclude_roots.
    """
    roots = [
        eid for eid, info in entity_lookup.items()
        if info['role'] == 'root'
        and len(get_active_values(entity_lookup, eid)) >= 2
    ]
    if exclude_roots:
        roots = [r for r in roots if r not in exclude_roots]
    return random.choice(roots)


# ============================================================
# Step 3-2: Dependency Chain Construction (v2: includes 2-hop)
# ============================================================

def build_chain(root, adj, entity_lookup,
                num_task_a_1hop=1, num_task_b_1hop=1,
                num_task_a_2hop=1, num_task_b_2hop=1):
    """
    Build dependency chain from root + assign Task A/B.

    v3 change: path conflict resolution.
    When the same entity is reachable via both 1-hop and 2-hop,
    only **one path is activated** within an episode.

    Approach:
      1) Collect root's 1-hop downstream
      2) Collect 2-hop downstream from middle nodes among 1-hop entities
      3) Remove entities from 2-hop that already exist in 1-hop (path conflict resolution)
         → Those entities are reached via 1-hop only
         → Since different roots are selected per episode, 2-hop paths may be used in other episodes
      4) Assign Task A/B from deduplicated pool

    Structure example (root=employer):
      employer (root, Task C)
        ├── job_title (1-hop, Task A)
        ├── work_location (1-hop, middle)
        │     ├── commute_method (2-hop, Task A) ← not in 1-hop, OK
        │     └── commute_duration (2-hop, Task B) ← not in 1-hop, OK
        ├── insurance (1-hop, Task B)
        └── work_project (1-hop, no task)
    """
    if root not in adj:
        return {
            "task_a_targets": [], "task_b_targets": [],
            "all_chain_entities": [root], "all_downstream": []
        }

    downstream = adj[root][:]
    random.shuffle(downstream)

    chain_entities = [root]
    one_hop_targets = []
    one_hop_entity_ids = set()  # Track 1-hop entity ids (for path conflict check)

    # Step 1: Collect 1-hop targets
    for target in downstream:
        one_hop_targets.append({"entity": target, "hop": 1, "source": root})
        chain_entities.append(target)
        one_hop_entity_ids.add(target)

    # Step 2: Collect 2-hop targets (exclude entities already in 1-hop)
    two_hop_targets = []
    for target in downstream:
        if target in adj:
            for t2 in adj[target]:
                if t2 not in one_hop_entity_ids and t2 not in chain_entities:
                    # This entity is unreachable via 1-hop → only 2-hop path exists → OK
                    two_hop_targets.append({"entity": t2, "hop": 2, "source": target})
                    chain_entities.append(t2)
                # else: already reachable via 1-hop → 2-hop path deactivated in this episode

    # Step 3: Assign Task A/B (1-hop)
    random.shuffle(one_hop_targets)
    task_a_targets = one_hop_targets[:num_task_a_1hop]
    remaining_1h = one_hop_targets[num_task_a_1hop:]
    task_b_targets = remaining_1h[:num_task_b_1hop]

    # Step 4: Assign Task A/B (2-hop, only when available)
    if two_hop_targets:
        random.shuffle(two_hop_targets)
        task_a_2h = two_hop_targets[:num_task_a_2hop]
        remaining_2h = two_hop_targets[num_task_a_2hop:]
        task_b_2h = remaining_2h[:num_task_b_2hop]
        task_a_targets += task_a_2h
        task_b_targets += task_b_2h

    return {
        "task_a_targets": task_a_targets,
        "task_b_targets": task_b_targets,
        "all_chain_entities": chain_entities,
        "all_downstream": one_hop_targets + two_hop_targets
    }


# ============================================================
# Step 3-3: Independent Entity Assignment (Task E, F, G)
# ============================================================

def assign_independent_entities(entity_lookup, chain_entities, task_assigned_entities, domain="personal_life"):
    """
    Select Task E, F, G target entities outside the chain.
    Assignment order: E → F → G (no overlap).

    Task E: 1 entity from DELETABLE outside chain
    Task F: 3 entities from chain + orphan mix (cross-chain multi-hop)
            Excludes entities already assigned to Task A/B/C/D
    Task G: 1 EXACT_RECALL entity

    Used entities accumulate in exclude pool → no overlap between tasks.
    """
    used = set(task_assigned_entities)
    exact_recall_ids = set(get_exact_recall_ids(entity_lookup))

    # --- Task E: delete ---
    available_delete = [
        e for e in entity_lookup
        if e not in used
        and len(get_active_values(entity_lookup, e)) >= 1
    ]
    task_e_target = None
    if available_delete:
        task_e_target = random.choice(available_delete)
        used.add(task_e_target)

    # --- Task F: multi-hop 3-entity combination ---
    # Match MULTIHOP_TRIPLES from chain entities + orphans
    # Exclude those in used, exact recall entities
    # Also exclude entities that may only have "none" values (e.g., partner=none forced by consistency)
    ENTITIES_WITH_NULL_RISK = {"partner"}  # Entities where "none" may be forced by consistency rules
    orphans = [eid for eid, info in entity_lookup.items()
               if info['role'] == 'orphan'
               and eid not in exact_recall_ids
               and eid not in ENTITIES_WITH_NULL_RISK]
    f_pool = list(set(chain_entities + orphans) - used)

    # Only those with active values
    f_pool = [e for e in f_pool if len(get_active_values(entity_lookup, e)) >= 1]
    f_pool_set = set(f_pool)

    # Find matching triple from MULTIHOP_TRIPLES
    task_f_triple = None
    task_f_match = None
    eligible_triples = [
        mt for mt in MULTIHOP_TRIPLES[domain]
        if set(mt["triple"]).issubset(f_pool_set)
    ]
    if eligible_triples:
        task_f_match = random.choice(eligible_triples)
        task_f_triple = task_f_match["triple"]
        used.update(task_f_triple)

    # --- Task G: exact recall ---
    # Select 1 from entities marked exact_recall in entity_pool
    exact_ids = list(exact_recall_ids - used)
    task_g_target = random.choice(exact_ids) if exact_ids else None
    used.add(task_g_target)

    # --- Task H: multiple-update ---
    # Select 1 entity from MULTIPLE_UPDATE_CANDIDATES that's not in used
    mu_candidates = [e for e in MULTIPLE_UPDATE_CANDIDATES[domain] if e not in used]
    # Need >= 3 distinct values for history
    mu_candidates = [e for e in mu_candidates if len(get_active_values(entity_lookup, e)) >= 3]
    task_h_target = random.choice(mu_candidates) if mu_candidates else None
    if task_h_target:
        used.add(task_h_target)

    return {
        "task_e_target": task_e_target,
        "task_f_triple": task_f_triple,
        "task_f_match": task_f_match,
        "task_g_target": task_g_target,
        "task_h_target": task_h_target,
        "used_entities": used
    }


# ============================================================
# Step 3-4: Filler Entity Selection
# ============================================================

def select_filler(entity_lookup, chain_entities, used_entities, num_filler=5):
    """
    Select orphan entities for filler.
    Sample 5 after excluding chain, Task E/F/G used entities, and exact recall entities.
    """
    exact_recall_ids = set(get_exact_recall_ids(entity_lookup))
    orphans = [eid for eid, info in entity_lookup.items()
               if info['role'] == 'orphan' and eid not in exact_recall_ids]
    available = [o for o in orphans if o not in chain_entities and o not in used_entities]
    return random.sample(available, min(num_filler, len(available)))


# ============================================================
# Step 3-5: Value Assignment
# ============================================================

def assign_values(entity_lookup, chain, independent, filler_entities, root, domain="personal_life"):
    """
    Assign concrete values to all entities.

    Assignment rules:
      Root (Task C): before + after (both active)
      Task A targets: before + after (both active)
      Task B targets: before only (after = None → uncertain)
      Remaining chain: before only (world state)
      Task E: before = active, after = "deleted"
      Task G: 1 value from exact recall entity
      Task F: all 3 entities before = active
      Filler: before only (no change)
    """
    entities = {}

    # Ordered value entities: before→after must follow this order
    # Per-domain ordering constraints for root entity before/after pairs.
    ORDERED_VALUES = {
        "personal_life": {
            "relationship_status": {
                "chains": [
                    ["single", "dating", "in a relationship", "engaged", "married"],
                    ["married", "separated", "divorced"],
                ]
            },
        },
        "software_project": {},
    }
    domain_ordered = ORDERED_VALUES.get(domain, {})

    def sample_ordered_pair(values, order_spec):
        """Sample before/after pair respecting direction constraint.

        order_spec can be:
          - list: single ordered chain
          - dict with "chains": multiple allowed chains
        """
        if isinstance(order_spec, dict) and "chains" in order_spec:
            # Multi-chain: collect all valid (before, after) pairs
            valid_pairs = []
            for chain in order_spec["chains"]:
                chain_vals = [v for v in chain if v in values]
                for i in range(len(chain_vals)):
                    for j in range(i + 1, len(chain_vals)):
                        valid_pairs.append((chain_vals[i], chain_vals[j]))

            if valid_pairs:
                return random.choice(valid_pairs)
            return random.sample(values, 2)
        else:
            # Single chain
            ordered_list = order_spec
            ordered_vals = [v for v in ordered_list if v in values]

            if len(ordered_vals) >= 2:
                idxs = sorted(random.sample(range(len(ordered_vals)), 2))
                return ordered_vals[idxs[0]], ordered_vals[idxs[1]]
            else:
                return random.sample(values, 2)

    # ---- Group 1: Chain entities ----

    # Root: Task C (direct update)
    root_active = get_active_values(entity_lookup, root)
    assert len(root_active) >= 2, f"Root '{root}' needs >= 2 active values"
    if root in domain_ordered:
        before, after = sample_ordered_pair(root_active, domain_ordered[root])
    else:
        before, after = random.sample(root_active, 2)
    entities[root] = {"before": before, "after": after, "task": "C"}

    # Assign middle nodes first (needed before 2-hop leaf assignment).
    # 2-hop Task A middle nodes get before + after values for if-then chain.
    middle_nodes_needing_after = set()
    for t in chain['task_a_targets']:
        if t['hop'] == 2:
            middle_nodes_needing_after.add(t['source'])

    for eid in chain['all_chain_entities']:
        if eid not in entities:
            av = get_active_values(entity_lookup, eid)
            if eid in middle_nodes_needing_after:
                if len(av) >= 2:
                    b, a = random.sample(av, 2)
                else:
                    b, a = random.sample(entity_lookup[eid]['values'], 2)
                entities[eid] = {
                    "before": b, "after": a,
                    "task": None,
                    "is_2hop_middle": True
                }
            else:
                entities[eid] = {
                    "before": random.choice(av) if av else random.choice(entity_lookup[eid]['values']),
                    "after": None, "task": None
                }

    # Helper: pick value from DERIVED_VALUES if available (software 2-hop),
    # otherwise fall back to random from active values.
    def _pick_derived(parent_eid, child_eid, parent_value, fallback_values):
        key = (parent_eid, child_eid)
        if key in DERIVED_VALUES and parent_value in DERIVED_VALUES[key]:
            return random.choice(DERIVED_VALUES[key][parent_value])
        return random.choice(fallback_values)

    # Task A targets (Cas): before + after values
    for t in chain['task_a_targets']:
        eid = t['entity']
        source_eid = t['source']
        av = get_active_values(entity_lookup, eid)

        if t['hop'] == 2 and source_eid in entities:
            # 2-hop: derive values from parent
            parent_before = entities[source_eid]['before']
            parent_after = entities[source_eid].get('after') or parent_before
            b = _pick_derived(source_eid, eid, parent_before, av if av else entity_lookup[eid]['values'])
            a = _pick_derived(source_eid, eid, parent_after, av if av else entity_lookup[eid]['values'])
            # Ensure before != after
            if b == a and len(av) >= 2:
                candidates = [v for v in av if v != b]
                if candidates:
                    a = random.choice(candidates)
        else:
            # 1-hop or no derivation: random
            if len(av) >= 2:
                b, a = random.sample(av, 2)
            else:
                b, a = random.sample(entity_lookup[eid]['values'], 2)

        entities[eid] = {
            "before": b, "after": a, "task": "A",
            "cascade_source": t['source'], "hop": t['hop']
        }

    # Task B targets (Abs): before only
    for t in chain['task_b_targets']:
        eid = t['entity']
        source_eid = t['source']
        av = get_active_values(entity_lookup, eid)

        if t['hop'] == 2 and source_eid in entities:
            parent_before = entities[source_eid]['before']
            b = _pick_derived(source_eid, eid, parent_before, av if av else entity_lookup[eid]['values'])
        else:
            b = random.choice(av) if av else random.choice(entity_lookup[eid]['values'])

        # Keep existing 'after' if set by middle node assignment
        existing_after = entities.get(eid, {}).get("after")
        entities[eid] = {
            "before": b, "after": existing_after, "task": "B",
            "cascade_source": t['source'], "hop": t['hop']
        }

    # ---- Group 2: Independent entities ----

    # Task E: delete
    if independent['task_e_target']:
        eid = independent['task_e_target']
        vals = entity_lookup[eid]['values']
        valid = [v for v in vals if v.lower() not in {nv.lower() for nv in NULL_VALUES}]
        if valid:
            entities[eid] = {"before": random.choice(valid), "after": "deleted", "task": "E"}

    # Task G: exact recall (marked in entity_pool with exact_recall=true)
    if independent['task_g_target']:
        eid = independent['task_g_target']
        entities[eid] = {
            "before": random.choice(entity_lookup[eid]['values']),
            "after": None, "task": "G"
        }

    # Task H: multiple-update (3 ordered history values for an independent leaf)
    if independent.get('task_h_target'):
        eid = independent['task_h_target']
        all_values = entity_lookup[eid]['values']  # original pool order
        av = get_active_values(entity_lookup, eid)
        if len(av) >= 3:
            picked = random.sample(av, 3)
        else:
            picked = random.choices(av, k=3)
        # Sort by original pool order (preserves chronological/logical ordering)
        pool_order = {v: i for i, v in enumerate(all_values)}
        history = sorted(picked, key=lambda v: pool_order.get(v, 0))
        entities[eid] = {
            "before": history[0], "after": history[-1], "task": "H",
            "history": history
        }

    # Task F: multi-hop triple
    if independent['task_f_triple']:
        for eid in independent['task_f_triple']:
            if eid not in entities:
                if eid in entity_lookup:
                    av = get_active_values(entity_lookup, eid)
                    entities[eid] = {
                        "before": random.choice(av) if av else random.choice(entity_lookup[eid]['values']),
                        "after": None, "task": "F"
                    }

    # ---- Group 3: Filler ----
    for eid in filler_entities:
        if eid not in entities:
            vals = entity_lookup[eid]['values']
            entities[eid] = {"before": random.choice(vals), "after": None, "task": "filler"}

    # ---- Consistency check ----
    # Apply to before_state. If consistency changes a Task A/B before value,
    # ensure after value is also re-sampled to remain different.
    before_state = {eid: info['before'] for eid, info in entities.items()}
    before_state = check_consistency(before_state, domain=domain)
    for eid in entities:
        old_before = entities[eid]['before']
        new_before = before_state[eid]
        entities[eid]['before'] = new_before
        # If before changed and entity has after (Task A), ensure before != after
        if old_before != new_before and entities[eid].get('after') and entities[eid]['after'] != "deleted":
            if entities[eid]['after'] == new_before:
                av = get_active_values(entity_lookup, eid)
                alternatives = [v for v in av if v != new_before]
                if alternatives:
                    entities[eid]['after'] = random.choice(alternatives)

    after_state = {}
    for eid, info in entities.items():
        if info['after'] and info['after'] != "deleted":
            after_state[eid] = info['after']
        elif info['after'] is None:
            after_state[eid] = info['before']
    after_state = check_consistency(after_state, domain=domain)
    for eid in entities:
        if entities[eid]['after'] and entities[eid]['after'] != "deleted":
            entities[eid]['after'] = after_state[eid]

    return entities


# ============================================================
# Step 3-6: Task & Question Generation
# ============================================================

def generate_tasks(entities, root, chain, independent, domain="personal_life"):
    """
    Generate task list.

    Tasks per episode:
      Required: Tr(1) + Cas 1-hop(1) + Abs 1-hop(1) + Deletion(1) + Agg(1) + Exact(1) = 6
      Optional: Cas 2-hop(1) + Abs 2-hop(1) = +2 (when middle node exists)
      Total: 6 ~ 8
    """
    tasks = []
    qtpl = QUESTION_TEMPLATES[domain]  # raises KeyError if domain missing
    mu_tpl = MULTIPLE_UPDATE_TEMPLATES[domain]

    # ---------------------------------------------------------------
    # entity_values convention (used by judge for entity-level eval):
    #
    #   entity_values = {entity_id: gold_value}
    #
    #   The meaning of gold_value depends on task_type:
    #     Tr    : list of 3 ordered history values — answer must contain ALL in correct order
    #     Cas          : after value — answer SHOULD contain this
    #     Abs          : before value — answer should say this is NOW UNCERTAIN
    #     Deletion           : before value — answer should NOT contain this (it was deleted)
    #     Agg          : current values (after if changed, else before) — answer should contain ALL
    #     Exact              : before value — answer must match this VERBATIM
    #
    #   W-check uses entity_values to verify memory storage:
    #     Most tasks       : check if gold_value IS in memory
    #     Tr  : check if ALL 3 history values are in memory
    #     Deletion         : check if gold_value is ABSENT from memory
    #
    #   target_entities is always a list for consistency.
    # ---------------------------------------------------------------

    # Tr: 3 ordered history values for an independent leaf entity
    if independent.get('task_h_target') and independent['task_h_target'] in entities:
        eid = independent['task_h_target']
        history = entities[eid]['history']  # list of 3 values
        tasks.append({
            "type": "Tr",
            "target_entities": [eid],
            "entity_values": {eid: history},
            "question_template": mu_tpl[eid],
            "gold_answer": ", ".join(history),
            "notes": "Entity changed 3 times; answer must list all values in chronological order."
        })

    # Cas (Cascade): gold = after value (answer should contain the cascaded new value)
    for t in chain['task_a_targets']:
        eid = t['entity']
        tasks.append({
            "type": "Cas",
            "hop": t['hop'],
            "target_entities": [eid],
            "entity_values": {eid: entities[eid]['after']},
            "cascade_source": t['source'],
            "question_template": qtpl[eid],
            "gold_answer": entities[eid]['after'],
            "notes": f"Cascade from {t['source']} (hop={t['hop']}). Replacement declared."
        })

    # Abs (Absence): gold = before value (answer should indicate this value is now uncertain)
    for t in chain['task_b_targets']:
        eid = t['entity']
        tasks.append({
            "type": "Abs",
            "hop": t['hop'],
            "target_entities": [eid],
            "entity_values": {eid: entities[eid]['before']},
            "cascade_source": t['source'],
            "question_template": qtpl[eid],
            "gold_answer": f"Uncertain — previously '{entities[eid]['before']}', but {t['source']} changed",
            "notes": f"Cascade from {t['source']} (hop={t['hop']}). No replacement declared."
        })

    # Deletion: gold = before value (answer should NOT contain this; it was deleted)
    if independent['task_e_target'] and independent['task_e_target'] in entities:
        eid = independent['task_e_target']
        tasks.append({
            "type": "Del",
            "target_entities": [eid],
            "entity_values": {eid: entities[eid]['before']},
            "question_template": qtpl[eid],
            "gold_answer": "No — explicitly removed",
            "notes": "Explicit deletion."
        })

    # Agg: gold = current values (after if changed, before otherwise)
    if independent['task_f_triple'] and independent.get('task_f_match'):
        match = independent['task_f_match']
        triple = independent['task_f_triple']

        # Build entity_values: use after value if entity changed, else before
        entity_values = {}
        for eid in triple:
            after = entities[eid].get('after')
            if after and after != "deleted":
                entity_values[eid] = after
            else:
                entity_values[eid] = entities[eid]['before']

        try:
            gold_answer = match["gold_answer_template"].format(**entity_values)
        except KeyError:
            parts = [f"{e}={entity_values[e]}" for e in triple]
            gold_answer = "Combine: " + ", ".join(parts)

        tasks.append({
            "type": "Agg",
            "target_entities": triple,
            "entity_values": entity_values,
            "question_template": match["question"],
            "gold_answer": gold_answer,
            "notes": "3-entity aggregation question."
        })

    # Exact: gold = before value (answer must match this verbatim)
    if independent['task_g_target'] and independent['task_g_target'] in entities:
        eid = independent['task_g_target']
        tasks.append({
            "type": "ER",
            "target_entities": [eid],
            "entity_values": {eid: entities[eid]['before']},
            "question_template": qtpl[eid],
            "gold_answer": entities[eid]['before'],
            "notes": "Verbatim recall. Must match exactly."
        })

    return tasks


def _build_dep_edges(chain, root, edge_details):
    """Build dependency_edges_used list, including root→middle edges for 2-hop."""
    edges = [
        {
            "source": t.get('source', root),
            "target": t['entity'],
            "hop": t['hop'],
            "pattern": edge_details.get(
                (t.get('source', root), t['entity']), {}
            ).get('pattern', 'unknown')
        }
        for t in chain.get('task_a_targets', []) + chain.get('task_b_targets', [])
    ]

    # Add root→middle edges for 2-hop middle nodes (both Task A and Task B).
    # Without this, dep_lookup in generate_gold_facts.py omits the middle's
    # dependency on root, so the verbalized fact for the 2-hop middle has no
    # "depends on root" clause — breaking the chain that Cascade questions test.
    seen_middle = set()
    for t in chain.get('task_a_targets', []) + chain.get('task_b_targets', []):
        if t['hop'] == 2:
            middle = t['source']
            if (root, middle) not in seen_middle:
                seen_middle.add((root, middle))
                edges.append({
                    "source": root,
                    "target": middle,
                    "hop": 1,
                    "pattern": edge_details.get(
                        (root, middle), {}
                    ).get('pattern', 'unknown'),
                    "is_2hop_middle": True
                })

    return edges


# ============================================================
# Main: Generate One Episode
# ============================================================

REQUIRED_TASK_TYPES = {"Tr", "Cas", "Abs",
                       "Del", "Agg", "ER"}

MAX_EPISODE_RETRIES = 20


def generate_episode(pool, dep_map, episode_id=1, exclude_roots=None, domain="personal_life"):
    """
    Generate one episode.

    Flow:
      3-1. Select root
      3-2. Build chain (1-hop + 2-hop) + assign Task A/B
      3-3. Assign independent entities (E → F → G, no overlap)
      3-4. Select fillers (orphans not used in E/F/G)
      3-5. Assign values + consistency check
      3-6. Generate tasks & questions
      3-7. Validate all required task types present; retry if not
    """
    entity_lookup = build_entity_lookup(pool)
    adj, edge_details = build_edge_lookup(dep_map)

    for attempt in range(MAX_EPISODE_RETRIES):
        # 3-1
        root = select_root(entity_lookup, adj, exclude_roots)

        # 3-2: Include 2-hop cascade
        chain = build_chain(root, adj, entity_lookup,
                            num_task_a_1hop=1, num_task_b_1hop=1,
                            num_task_a_2hop=1, num_task_b_2hop=1)

        # 3-3: Assign in order E → F → G (no overlap)
        # Pass entities already assigned to Task A/B/C/D to exclude from F pool
        task_assigned = {root}  # Update
        for t in chain['task_a_targets']:
            task_assigned.add(t['entity'])
        for t in chain['task_b_targets']:
            task_assigned.add(t['entity'])
        independent = assign_independent_entities(
            entity_lookup, chain['all_chain_entities'], task_assigned, domain=domain)

        # 3-4: Select 5 fillers from remaining orphans
        filler_entities = select_filler(
            entity_lookup,
            chain['all_chain_entities'],
            independent['used_entities'],
            num_filler=5
        )

        # 3-5
        entities = assign_values(entity_lookup, chain, independent, filler_entities, root, domain=domain)

        # 3-6
        tasks = generate_tasks(entities, root, chain, independent, domain=domain)

        # 3-7: Validate required task types
        task_types_present = {t['type'].split(' (')[0] for t in tasks}
        missing = REQUIRED_TASK_TYPES - task_types_present
        if not missing:
            break
        print(f"  Ep{episode_id} attempt {attempt+1}: missing {missing}, retrying...")
    else:
        raise RuntimeError(
            f"Episode {episode_id}: failed to generate all required tasks after "
            f"{MAX_EPISODE_RETRIES} attempts. Missing: {missing}"
        )

    # Episode skeleton
    episode = {
        "episode_id": episode_id,
        "domain": domain,
        "root": root,
        "root_change": {
            "before": entities[root]['before'],
            "after": entities[root]['after']
        },
        "chain_entities": chain['all_chain_entities'],
        "filler_entities": filler_entities,
        "entities": entities,
        "tasks": tasks,
        "dependency_edges_used": _build_dep_edges(chain, root, edge_details),
        "has_2hop": any(t['hop'] == 2 for t in
                        chain.get('task_a_targets', []) + chain.get('task_b_targets', []))
    }

    return episode


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate episode skeletons")
    parser.add_argument("-n", "--num_episodes", type=int, default=6,
                        help="Number of episodes to generate (default: 6)")
    parser.add_argument("-s", "--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("-o", "--output_dir", type=str, default="episodes",
                        help="Output directory (default: episodes/)")
    parser.add_argument("--domain", type=str, default="personal_life",
                        choices=["personal_life", "software_project"],
                        help="Domain (default: personal_life)")
    args = parser.parse_args()

    pool, dep_map = load_data(args.domain)
    random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    all_episodes = []
    for i in range(args.num_episodes):
        episode = generate_episode(pool, dep_map, episode_id=i + 1, domain=args.domain)
        all_episodes.append(episode)

        # Save individual episodes
        path = os.path.join(args.output_dir, f"episode_{i+1:03d}.json")
        with open(path, 'w') as f:
            json.dump(episode, f, indent=2, ensure_ascii=False)

    # Collect and save all episodes
    all_path = os.path.join(args.output_dir, "all_episodes.json")
    with open(all_path, 'w') as f:
        json.dump(all_episodes, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"Generated {len(all_episodes)} episodes → {args.output_dir}/")
    print(f"  Individual: episode_001.json ~ episode_{len(all_episodes):03d}.json")
    print(f"  Combined:   all_episodes.json")
    print()
    for ep in all_episodes:
        h2 = "2hop" if ep['has_2hop'] else "1hop"
        print(f"  Ep{ep['episode_id']:3d}: root={ep['root']:25s} "
              f"({ep['root_change']['before']} → {ep['root_change']['after']}) "
              f"| {h2} | tasks={len(ep['tasks'])}")

