"""
Gold Fact Seed Generator
========================
Episode skeleton → gold fact template seeds

Input: episode JSON (from generate_episode.py)
Output: phase-by-phase gold fact seed list

Usage:
  python3 generate_gold_facts.py -i episodes/
"""

import json
import sys

# ============================================================
# Fact Templates: entity_id → first-person statement
# ============================================================

FACT_TEMPLATES = {
    # Each template must contain the entity concept keyword that appears in
    # QUESTION_TEMPLATES, so the agent can match question → stored fact
    # without semantic inference.
    "personal_life": {
        "employer": "I work at {value}",
        "job_title": "My job title is {value}",
        "work_location": "Our office is in {value}",
        "work_schedule": "My work hours are {value}",
        "work_project": "I'm currently working on {value}",
        "insurance": "I have {value} for insurance",
        "residence_location": "I live in {value}",
        "housing_type": "My home is a {value}",
        "living_arrangement": "I live {value}",
        "commute_method": "I take the {value} to work",
        "commute_duration": "My commute is about {value}",
        "vehicle": "I drive a {value}",
        "health_condition": "My health condition is {value}",
        "medication": "I take a medication called {value}",
        "exercise_routine": "My exercise routine is {value}",
        "fitness_facility": "I go to {value} for workouts",
        "sleep_pattern": "I usually sleep {value}",
        "diet_preference": "My diet is {value}",
        "dietary_restriction": "My dietary restriction is {value}",
        "favorite_restaurant": "My favorite restaurant is {value}",
        "food_allergy": "I have a food allergy to {value}",
        "relationship_status": "My relationship status is {value}",
        "partner": "My partner's name is {value}",
        "family_event": "We have {value} coming up",
        "financial_goal": "My financial goal is {value}",
        "travel_plan": "I'm planning a {value}",
        "regular_appointment": "My regular appointment is {value}",
        "school": "I'm studying at {value}",
        "skill_acquisition": "I'm learning {value}",
        "hobby": "My hobby is {value}",
        "sports": "I play {value}",
        "club_membership": "I'm part of a {value}",
        "media_consumption": "My current media is {value}",
        "social_gathering": "My regular social meetup is {value}",
        "subscription_service": "I'm subscribed to {value}",
        "planned_purchase": "I'm planning a big purchase — {value}",
        "upcoming_event": "I have {value} coming up",
        "pet": "I have a pet — {value}",
        "life_philosophy": "My life philosophy goes like this — please remember it word for word: \"{value}\"",
    },
    "software_project": {
        "framework": "We're using {value} as our framework",
        "build_tool": "Our build tool is {value}",
        "build_command": "The build command is {value}",
        "test_framework": "We use {value} for testing",
        "test_command": "The test command is {value}",
        "project_structure": "Our project follows a {value}",
        "dev_server_command": "To run the dev server, use {value}",
        "database": "We're using {value} as our database",
        "orm_library": "Our ORM is {value}",
        "model_syntax": "Models are defined with {value}",
        "migration_tool": "We use {value} for migrations",
        "connection_string": "The database connection string is {value}",
        "backup_schedule": "Database backups run {value}",
        "deploy_target": "We deploy to {value}",
        "ci_config": "Our CI config is {value}",
        "deploy_command": "The deploy command is {value}",
        "monitoring_url": "Monitoring dashboard is at {value}",
        "staging_url": "Staging is at {value}",
        "docker_image": "Our Docker image is {value}",
        "dockerfile_path": "The Dockerfile is at {value}",
        "log_drain_endpoint": "Logs drain to {value}",
        "team_lead": "Our team lead is {value}",
        "code_reviewer": "Code reviews go to {value}",
        "escalation_contact": "For escalations, contact {value}",
        "approval_authority": "The approval authority is {value}",
        "weekly_report_recipient": "The weekly report recipient is {value}",
        "auth_provider": "We use {value} for authentication",
        "auth_method": "Our authentication method is {value}",
        "middleware_config": "The auth middleware is {value}",
        "token_format": "We use {value} token format",
        "login_endpoint": "The login endpoint is {value}",
        "user_session_ttl": "Session TTL is {value}",
        "sprint_deadline": "The sprint deadline is {value}",
        "secret_manager": "Our secret manager is {value}",
        "package_manager": "Our package manager is {value}",
        "branch_strategy": "We use {value} branching",
        "design_system": "Our design system is {value}",
        "meeting_day": "Team meeting is {value}",
        "slack_channel": "Our team channel is {value}",
        "standup_format": "Our standup format is {value}",
        "default_branch": "Our default branch is {value}",
        "release_cadence": "We release {value}",
        "test_coverage_target": "Our test coverage target is {value}",
        "code_review_policy": "Our code review policy is {value}",
        "oncall_rotation": "Our on-call rotation is {value}",
        "alert_channel": "Production alerts go to {value}",
        "incident_runbook_url": "The incident runbook is at {value}",
        "documentation_url": "Project docs are at {value}",
        "onboarding_guide": "The onboarding guide is at {value}",
        "changelog_location": "Changelog is at {value}",
        "error_log": "Here's an exact error log I want you to remember word for word:\n\"{value}\"",
    },
}

# ============================================================
# Dependency Templates: pattern → sentence expansion
# ============================================================

DEPENDENCY_TEMPLATES = {
    # Each template must make the dependency STRONGLY explicit — the reader must
    # understand that if the source entity changes, the target would change too.
    # Use "depends on" / "determined by" + "if X changes, Y would change".
    "personal_life": {
        "proximity": "{target_fact} — this depends on where {source_entity_phrase}; if I move, this would change",
        "infrastructure": "{target_fact} — this depends on the fact that {source_entity_phrase}; if that changes, this would change too",
        "company_policy": "{target_fact} — this is through my employer ({source_entity_phrase}); if my employer changes, this would change",
        "medical_causation": "{target_fact} — this is determined by my {source_value}; if my health condition changes, this would likely change too",
        "life_event": "{target_fact} — this depends on the fact that {source_entity_phrase}; if that changes, this would change",
        "priority_shift": "{target_fact} — this is determined by the fact that {source_entity_phrase}; if that changes, my priorities would shift",
        "activity_facility": "{target_fact} — this is for my {source_value}; if my routine changes, I'd need a different facility",
        "distance": "{target_fact} — this depends on where {source_entity_phrase}; if that changes, the distance would change",
        "schedule_conflict": "{target_fact} — this depends on the fact that {source_entity_phrase}; if that changes, my schedule would change",
        "curriculum": "{target_fact} as part of my program — if my school changes, this would change",
        "accommodation": "{target_fact} — this depends on my living situation; if that changes, this would change",
        "preference": "{target_fact} — this fits my {source_value} diet; if my diet changes, this would change too",
    },
    "software_project": {
        "tech_compatibility": "{target_fact} — this is determined by which framework we use ({source_entity_phrase}); if we switch frameworks, this would change",
        "data_layer": "{target_fact} — this depends on our database ({source_entity_phrase}); if we change databases, this would change",
        "infra_coupling": "{target_fact} — this is determined by our infrastructure ({source_entity_phrase}); if that changes, this would change",
        "team_assignment": "{target_fact} — {source_entity_phrase}; if the team lead changes, this assignment would change",
        "auth_coupling": "{target_fact} — this is determined by our auth provider ({source_entity_phrase}); if we switch providers, this would change",
        "derived_config": "{target_fact} — this is determined by the fact that {source_entity_phrase}; if that changes, this would change",
    },
}

# ============================================================
# Source Phrases: source entity → natural phrase
# ============================================================

SOURCE_PHRASES = {
    "personal_life": {
        "employer": "I work at {value}",
        "work_location": "our office is in {value}",
        "residence_location": "I live in {value}",
        "health_condition": "my {value}",
        "exercise_routine": "my {value} routine",
        "relationship_status": "I'm {value}",
        "school": "I'm studying at {value}",
        "living_arrangement": "I live {value}",
        "diet_preference": "I eat {value}",
        "pet": "I have {value}",
    },
    "software_project": {
        "framework": "we use {value}",
        "database": "our database is {value}",
        "deploy_target": "we deploy on {value}",
        "team_lead": "{value} assigned it",
        "auth_provider": "we use {value}",
        "build_tool": "we use {value}",
        "test_framework": "we use {value}",
        "orm_library": "we use {value}",
        "ci_config": "our CI is {value}",
        "docker_image": "our image is {value}",
        "auth_method": "we use {value}",
    },
}

# ============================================================
# Change Event Templates
# ============================================================

CHANGE_TEMPLATES = {
    "personal_life": {
        "employer": "I just accepted a new position at {after}",
        "residence_location": "I just moved to {after}",
        "health_condition": "My {before} has been treated and resolved. But I was recently diagnosed with {after}",
        "relationship_status": "I'm {after} now",
        "school": "I just started {after}",
    },
    "software_project": {
        "framework": "We just migrated to a new framework, {after}",
        "database": "We switched our database to {after}",
        "deploy_target": "We moved our deployment to {after}",
        "team_lead": "{after} is our new team lead",
        "auth_provider": "We switched our auth provider to {after}",
    },
}

# ============================================================
# If-Then Future Value Templates (for Cas)
# ============================================================
# Format: (source_entity, target_entity) → template
# Used in Fact Introduction to pre-declare what changes IF the source changes.

IF_THEN_TEMPLATES = {
    # Rules: 1) Trigger = explicit source entity name  2) Modal = "will" not "would probably"
    "personal_life": {
        # residence_location →
        ("residence_location", "commute_method"): "If I change my residence location, I will switch to {after_value} for commuting",
        ("residence_location", "commute_duration"): "If I change my residence, my commute will be around {after_value}",
        ("residence_location", "favorite_restaurant"): "If I change my residence, my favorite restaurant will be {after_value}",
        ("residence_location", "fitness_facility"): "If I change my residence location, I will work out at {after_value}",
        ("residence_location", "regular_appointment"): "If I change my residence, my regular appointment will be at {after_value}",
        # employer →
        ("employer", "job_title"): "If I change my employer, my job title will be {after_value}",
        ("employer", "work_location"): "If I change my employer, my office will be in {after_value}",
        ("employer", "work_schedule"): "If I change my employer, my work schedule will be {after_value}",
        ("employer", "work_project"): "If I change my employer, I will work on {after_value}",
        ("employer", "insurance"): "If I change my employer, my insurance will be {after_value}",
        # health_condition →
        ("health_condition", "medication"): "If my health condition changes, my medication will switch to {after_value}",
        ("health_condition", "dietary_restriction"): "If my health condition changes, my dietary restriction will be {after_value}",
        ("health_condition", "exercise_routine"): "If my health condition changes, my exercise routine will be {after_value}",
        ("health_condition", "sleep_pattern"): "If my health condition changes, my sleep pattern will shift to {after_value}",
        ("health_condition", "regular_appointment"): "If my health condition changes, my regular appointment will be {after_value}",
        # relationship_status →
        ("relationship_status", "living_arrangement"): "If my relationship status changes, I will live {after_value}",
        ("relationship_status", "housing_type"): "If my relationship status changes, my home will be a {after_value}",
        ("relationship_status", "financial_goal"): "If my relationship status changes, my financial goal will be {after_value}",
        ("relationship_status", "travel_plan"): "If my relationship status changes, I will plan {after_value}",
        ("relationship_status", "family_event"): "If my relationship status changes, we will plan {after_value}",
        # school →
        ("school", "skill_acquisition"): "If I change schools, I will start learning {after_value}",
        ("school", "work_schedule"): "If I change schools, my work schedule will adjust to {after_value}",
        ("school", "financial_goal"): "If I change schools, my financial goal will be {after_value}",
        ("school", "commute_method"): "If I change schools, I will take {after_value} to get there",
        # --- 2-hop: middle → leaf ---
        ("work_location", "commute_method"): "If my work location changes, I will switch to {after_value} for commuting",
        ("work_location", "commute_duration"): "If my work location changes, my commute will be around {after_value}",
        ("work_location", "favorite_restaurant"): "If my work location changes, my favorite restaurant will be {after_value}",
        ("work_location", "fitness_facility"): "If my work location changes, I will work out at {after_value}",
        ("exercise_routine", "fitness_facility"): "If my exercise routine changes, I will work out at {after_value}",
        ("diet_preference", "favorite_restaurant"): "If my diet changes, my favorite restaurant will be {after_value}",
        ("living_arrangement", "housing_type"): "If my living arrangement changes, my home will be a {after_value}",
        ("living_arrangement", "financial_goal"): "If my living arrangement changes, my financial goal will be {after_value}",
    },
    "software_project": {
        # framework →
        ("framework", "build_tool"): "If we switch frameworks, our build tool will be {after_value}",
        ("framework", "test_framework"): "If we switch frameworks, our test framework will be {after_value}",
        ("framework", "project_structure"): "If we switch frameworks, our project structure will be {after_value}",
        ("framework", "dev_server_command"): "If we switch frameworks, the dev server command will be {after_value}",
        # database →
        ("database", "orm_library"): "If we change databases, our ORM will be {after_value}",
        ("database", "migration_tool"): "If we change databases, our migration tool will be {after_value}",
        ("database", "connection_string"): "If we change databases, the connection string will be {after_value}",
        ("database", "backup_schedule"): "If we change databases, the backup schedule will be {after_value}",
        # deploy_target →
        ("deploy_target", "ci_config"): "If we change deploy targets, the CI config will be {after_value}",
        ("deploy_target", "monitoring_url"): "If we change deploy targets, monitoring will be at {after_value}",
        ("deploy_target", "staging_url"): "If we change deploy targets, staging will be at {after_value}",
        ("deploy_target", "docker_image"): "If we change deploy targets, the Docker image will be {after_value}",
        ("deploy_target", "log_drain_endpoint"): "If we change deploy targets, logs will drain to {after_value}",
        # team_lead →
        ("team_lead", "code_reviewer"): "If the team lead changes, the code reviewer will be {after_value}",
        ("team_lead", "escalation_contact"): "If the team lead changes, the escalation contact will be {after_value}",
        ("team_lead", "approval_authority"): "If the team lead changes, the approval authority will be {after_value}",
        ("team_lead", "weekly_report_recipient"): "If the team lead changes, the weekly report recipient will be {after_value}",
        # auth_provider →
        ("auth_provider", "auth_method"): "If we switch auth providers, our authentication method will be {after_value}",
        ("auth_provider", "token_format"): "If we switch auth providers, the token format will be {after_value}",
        ("auth_provider", "login_endpoint"): "If we switch auth providers, the login endpoint will be {after_value}",
        ("auth_provider", "user_session_ttl"): "If we switch auth providers, the session TTL will be {after_value}",
        # --- 2-hop: middle → leaf ---
        ("build_tool", "build_command"): "If we change the build tool, the build command will be {after_value}",
        ("test_framework", "test_command"): "If we change the test framework, the test command will be {after_value}",
        ("orm_library", "model_syntax"): "If we change the ORM library, the model syntax will be {after_value}",
        ("ci_config", "deploy_command"): "If the CI config changes, the deploy command will be {after_value}",
        ("docker_image", "dockerfile_path"): "If we change the Docker image, the Dockerfile will be at {after_value}",
        ("auth_method", "middleware_config"): "If we change the auth method, the middleware config will be {after_value}",
    },
}

# ---- Startup validation: every if-then template must mention source entity ----
for _domain in IF_THEN_TEMPLATES:
    for (_src, _tgt), _tpl in IF_THEN_TEMPLATES[_domain].items():
        _tpl_lower = _tpl.lower()
        _src_words = _src.replace('_', ' ')
        if _src_words not in _tpl_lower:
            _found = any(w in _tpl_lower for w in _src_words.split() if len(w) > 3)
            assert _found, f"IF_THEN_TEMPLATES[{_domain}][({_src},{_tgt})] missing source keyword '{_src_words}': {_tpl}"

# ============================================================
# Delete Templates
# ============================================================

DELETE_PATTERNS = {
    "my_is":       "My {entity_name} is {value}. Please remove that from your memory.",
    "i_live_in":   "I live in {value}. Please forget that.",
    "i_live_with": "I live {value}. Please forget that.",
    "i_work_at":   "I work at {value}. Please remove that from your memory.",
    "i_work_on":   "I'm working on {value}. Please forget that.",
    "i_verb":      "I {verb} {value}. Please remove that from your memory.",
    "i_part_of":   "I'm part of {value}. Please forget that.",
    "i_go_to":     "I go to {value}. Please forget that.",
    "i_have":      "I have {value}. Please remove that from your memory.",
    "we_use":      "We use {value} as our {entity_name}. Please remove that from your memory.",
    "our_is":      "Our {entity_name} is {value}. Please remove that from your memory.",
    "the_url":     "The {entity_name} is {value}. Please delete that from your memory.",
    "person_role": "{value} is our {entity_name}. Please remove that from your memory.",
}

DELETE_ENTITY_MAP = {
    "personal_life": {
        # my_is
        "job_title": "my_is", "health_condition": "my_is", "sleep_pattern": "my_is",
        "insurance": "my_is", "financial_goal": "my_is", "housing_type": "my_is",
        "relationship_status": "my_is", "work_schedule": "my_is", "medication": "my_is",
        "planned_purchase": "my_is",
        "subscription_service": "my_is", "upcoming_event": "my_is",
        "family_event": "my_is", "regular_appointment": "my_is",
        "partner": "my_is", "travel_plan": "my_is",
        # i_live_in
        "residence_location": "i_live_in",
        # i_live_with
        "living_arrangement": "i_live_with",
        # i_work_at
        "employer": "i_work_at", "work_location": "i_work_at",
        # i_work_on
        "work_project": "i_work_on", "skill_acquisition": "i_work_on",
        # i_verb (with verb)
        "hobby": "my_is", "sports": ("i_verb", "play"),
        "exercise_routine": "my_is", "commute_method": ("i_verb", "take the"),
        "diet_preference": "my_is", "media_consumption": "my_is",
        # i_part_of
        "club_membership": "i_part_of", "social_gathering": "i_part_of",
        # i_go_to
        "fitness_facility": "i_go_to", "school": "i_go_to",
        "favorite_restaurant": "i_go_to",
        # i_have
        "food_allergy": "my_is", "pet": "i_have", "vehicle": "i_have",
        "dietary_restriction": "i_have",
    },
    "software_project": {
        # we_use
        "framework": "we_use", "database": "we_use", "build_tool": "we_use",
        "orm_library": "we_use", "test_framework": "we_use",
        "auth_provider": "we_use", "ci_config": "we_use",
        "docker_image": "we_use", "deploy_target": "we_use",
        "design_system": "we_use", "migration_tool": "we_use",
        "secret_manager": "we_use", "package_manager": "we_use",
        # our_is
        "auth_method": "our_is", "token_format": "our_is",
        "middleware_config": "our_is", "project_structure": "our_is",
        "default_branch": "our_is", "branch_strategy": "our_is",
        "deploy_command": "our_is",
        "build_command": "our_is", "test_command": "our_is",
        "dev_server_command": "our_is", "model_syntax": "our_is",
        "test_coverage_target": "our_is", "release_cadence": "our_is",
        "standup_format": "our_is", "oncall_rotation": "our_is",
        "code_review_policy": "our_is", "user_session_ttl": "our_is",
        "meeting_day": "our_is", "sprint_deadline": "our_is",
        "backup_schedule": "our_is", "slack_channel": "our_is",
        "alert_channel": "our_is",
        # the_url
        "documentation_url": "the_url",
        "monitoring_url": "the_url", "staging_url": "the_url",
        "login_endpoint": "the_url", "log_drain_endpoint": "the_url",
        "incident_runbook_url": "the_url", "onboarding_guide": "the_url",
        "changelog_location": "the_url",
        "connection_string": "the_url",
        "dockerfile_path": "the_url",
        # person_role
        "team_lead": "person_role", "code_reviewer": "person_role",
        "escalation_contact": "person_role", "approval_authority": "person_role",
        "weekly_report_recipient": "person_role",
    },
}

# ============================================================
# NULL values per domain — values that should not appear as gold facts
# ============================================================

NULL_VALUES = {
    "personal_life": {
        "none", "no pet", "not currently enrolled", "not currently learning anything",
        "none currently", "no major purchase planned",
        "no insurance", "freelance", "remote (no commute)", "home gym"
    },
    "software_project": set(),  # Software entities don't have null-like values
}


# ============================================================
# Main: Generate gold facts from episode skeleton
# ============================================================

def generate_gold_facts(episode):
    """
    Episode skeleton → phase-by-phase gold fact seeds.
    Domain is read from episode['domain'].

    Returns dict with:
      phase1_facts: for fact introduction
      phase2_before_questions: for before questions
      phase3_change_and_deletion: for change + delete events
      phase4_after_questions: for after questions
    """
    entities = episode['entities']
    root = episode['root']
    tasks = episode['tasks']
    dep_edges = episode.get('dependency_edges_used', [])
    domain = episode.get('domain', 'personal_life')

    # Select domain-specific templates
    fact_tpl = FACT_TEMPLATES[domain]
    dep_tpl = DEPENDENCY_TEMPLATES[domain]
    src_phrases = SOURCE_PHRASES[domain]
    change_tpl = CHANGE_TEMPLATES[domain]
    ifthen_tpl = IF_THEN_TEMPLATES[domain]
    del_map = DELETE_ENTITY_MAP[domain]
    null_vals = NULL_VALUES[domain]

    # Build dependency lookup: target_entity → {source, pattern}
    dep_lookup = {}
    for edge in dep_edges:
        dep_lookup[edge['target']] = {
            'source': edge['source'],
            'pattern': edge['pattern']
        }

    # ============================================================
    # Phase 1: Fact Introduction
    # ============================================================
    phase1_facts = []

    for eid, info in entities.items():
        if info['before'] is None:
            continue
        if info['before'].lower() in {nv.lower() for nv in null_vals}:
            continue

        # Filler: emit a single fact marked is_filler (not used for evaluation,
        # but verbalized in self-chat to make conversation natural)
        if info['task'] == 'filler':
            value = info['before']
            template = fact_tpl.get(eid, f"My {eid.replace('_', ' ')} is {{value}}")
            phase1_facts.append({
                "entity": eid,
                "value": value,
                "gold_fact": template.format(value=value),
                "is_filler": True,
                "has_dependency": False,
            })
            continue

        # Tr: emit 3 ordered history facts, one per timepoint
        # (NOT just `before` — all 3 values must be verbalized)
        if info['task'] == 'H' and 'history' in info:
            history = info['history']
            template = fact_tpl.get(eid, f"My {eid.replace('_', ' ')} is {{value}}")
            for i, val in enumerate(history):
                # Add temporal markers to make order natural in conversation
                if i == 0:
                    prefix = ""
                elif i == len(history) - 1:
                    prefix = "Now I've switched again — "
                else:
                    prefix = "Update: "
                fact_text = prefix + template.format(value=val)
                phase1_facts.append({
                    "entity": eid,
                    "value": val,
                    "gold_fact": fact_text,
                    "has_dependency": False,
                    "is_multiple_update": True,
                    "history_index": i,
                    "notes": f"Tr step {i+1}/{len(history)}"
                })
            continue

        value = info['before']

        # Base fact from template
        template = fact_tpl.get(eid, f"My {eid.replace('_', ' ')} is {{value}}")
        base_fact = template.format(value=value)

        # If dependency exists, expand with dependency template
        # Skip dependency expansion if source value is null/absent
        if eid in dep_lookup:
            dep = dep_lookup[eid]
            source_eid = dep['source']
            pattern = dep['pattern']
            source_value = entities.get(source_eid, {}).get('before', source_eid)

            source_is_null = source_value.lower() in {nv.lower() for nv in null_vals}

            if source_is_null:
                # Source is null — use base fact without dependency
                phase1_facts.append({
                    "entity": eid,
                    "value": value,
                    "gold_fact": base_fact,
                    "has_dependency": False
                })
                continue

            source_template = src_phrases.get(source_eid, "{value}")
            source_phrase = source_template.format(value=source_value)

            dep_template = dep_tpl.get(pattern, "{target_fact}")
            combined = dep_template.format(
                target_fact=base_fact,
                source_entity_phrase=source_phrase,
                source_value=source_value
            )
            phase1_facts.append({
                "entity": eid,
                "value": value,
                "gold_fact": combined,
                "has_dependency": True,
                "dependency_source": source_eid,
                "dependency_pattern": pattern
            })
        else:
            fact_entry = {
                "entity": eid,
                "value": value,
                "gold_fact": base_fact,
                "has_dependency": False
            }
            # Mark Exact (Task G) entity facts so they bypass self-chat
            # (long verbatim values would get truncated by LLM).
            if info.get('task') == 'G':
                fact_entry["is_exact"] = True
            phase1_facts.append(fact_entry)

    # Add 2-hop MIDDLE if-then FIRST, so it gets its own turn in self-chat.
    # e.g., "If we switch frameworks, our test framework will be Pyronis Check"
    # Must come before leaf if-thens to avoid sharing a turn.
    middle_eids = set()
    for t in episode.get("tasks", []):
        if t.get("type") == "Cas":
            middle_eids.add(t.get("cascade_source", ""))

    for eid in middle_eids:
        info = entities.get(eid, {})
        if not info.get('after'):
            continue
        after_val = info['after']
        source_eid = root

        template_key = (source_eid, eid)
        if template_key in ifthen_tpl:
            if_then_fact = ifthen_tpl[template_key].format(after_value=after_val)
        else:
            if_then_fact = f"If my {source_eid.replace('_', ' ')} changes, my {eid.replace('_', ' ')} will be {after_val}"

        phase1_facts.append({
            "entity": eid,
            "value": after_val,
            "gold_fact": if_then_fact,
            "has_dependency": True,
            "dependency_source": source_eid,
            "is_if_then": True,
            "notes": "2-hop middle node: future value pre-declaration"
        })

    # Add Cas LEAF if-then declarations (after middle if-thens)
    # Track (source, target) pairs to avoid duplicates with middle if-thens
    emitted_ifthen_pairs = {(root, eid) for eid in middle_eids}

    for task in tasks:
        task_type = task['type'].split(' (')[0]
        if task_type == 'Cas':
            eid = task['target_entities'][0]
            after_val = entities[eid]['after']

            source_eid = None
            for edge in dep_edges:
                if edge['target'] == eid:
                    source_eid = edge['source']
                    break
            if not source_eid:
                source_eid = root

            template_key = (source_eid, eid)
            if template_key in emitted_ifthen_pairs:
                continue  # already emitted by middle if-then loop
            emitted_ifthen_pairs.add(template_key)
            if template_key in ifthen_tpl:
                if_then_fact = ifthen_tpl[template_key].format(after_value=after_val)
            else:
                if_then_fact = f"If my {source_eid.replace('_', ' ')} changes, my {eid.replace('_', ' ')} will be {after_val}"

            phase1_facts.append({
                "entity": eid,
                "value": after_val,
                "gold_fact": if_then_fact,
                "has_dependency": True,
                "dependency_source": source_eid,
                "is_if_then": True,
                "notes": "Cas: future value pre-declaration"
            })

    # ============================================================
    # Phase 2: Before-questions
    # ============================================================
    phase2_questions = []

    for task in tasks:
        task_type = task['type'].split(' (')[0]

        if task_type in ['Cas', 'Abs']:
            eid = task['target_entities'][0]
            value = entities[eid]['before']
            q = {
                "task_type": task['type'],
                "entity": [eid],
                "entity_values": {eid: value},
                "question": task['question_template'],
                "expected_answer": value,
            }
            if 'hop' in task:
                q["hop"] = task['hop']
            phase2_questions.append(q)
        elif task_type == 'Tr':
            # Tr question only makes sense after all 3 history values
            # are verbalized — skip in before phase
            pass
        elif task_type == 'Del':
            eid = task['target_entities'][0]
            value = entities[eid]['before']
            phase2_questions.append({
                "task_type": "Del",
                "entity": [eid],
                "entity_values": {eid: value},
                "question": task['question_template'],
                "expected_answer": value
            })
        elif task_type == 'ER':
            eid = task['target_entities'][0]
            value = entities[eid]['before']
            phase2_questions.append({
                "task_type": "ER",
                "entity": [eid],
                "entity_values": {eid: value},
                "question": task['question_template'],
                "expected_answer": value
            })

    # ============================================================
    # Phase 3: Change + Deletion Events (merged)
    # ============================================================
    phase3_cd_facts = []

    root_after = entities[root]['after']
    change_template = change_tpl.get(root, "My {entity} changed to {after}")
    root_change_fact = change_template.format(
        before=entities[root]['before'],
        after=root_after,
        entity=root.replace('_', ' ')
    )
    phase3_cd_facts.append({
        "entity": root,
        "type": "root_change",
        "gold_fact": root_change_fact,
        "value": root_after,
        "before": entities[root]['before'],
        "after": root_after
    })

    for task in tasks:
        if task['type'] == 'Del':
            eid = task['target_entities'][0]
            value = entities[eid]['before']
            mapping = del_map.get(eid, "my_is")
            if isinstance(mapping, tuple):
                pattern_key, verb = mapping
                del_fact = DELETE_PATTERNS[pattern_key].format(
                    entity_name=eid.replace('_', ' '), value=value, verb=verb)
            else:
                del_fact = DELETE_PATTERNS[mapping].format(
                    entity_name=eid.replace('_', ' '), value=value)
            phase3_cd_facts.append({
                "entity": eid,
                "type": "delete",
                "gold_fact": del_fact,
                "value": value,
                "deleted_value": value
            })

    # ============================================================
    # Phase 4: After-questions
    # ============================================================
    phase4_questions = []

    for task in tasks:
        q = {
            "task_type": task['type'],
            "entity": task['target_entities'],
            "entity_values": task['entity_values'],
            "question": task['question_template'],
            "gold_answer": task['gold_answer'],
        }
        if 'hop' in task:
            q["hop"] = task['hop']
        phase4_questions.append(q)

    return {
        "episode_id": episode['episode_id'],
        "domain": domain,
        "root": root,
        "root_change": episode['root_change'],
        "phase1_fact_introduction": phase1_facts,
        "phase2_before_questions": phase2_questions,
        "phase3_change_and_deletion": phase3_cd_facts,
        "phase4_after_questions": phase4_questions
    }


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate gold facts from episode skeletons")
    parser.add_argument("-i", "--input_dir", type=str, default="episodes",
                        help="Directory containing episode JSON files (default: episodes/)")
    parser.add_argument("-o", "--output_dir", type=str, default="gold_facts",
                        help="Output directory (default: gold_facts/)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    episode_files = sorted([
        f for f in os.listdir(args.input_dir)
        if f.startswith("episode_") and f.endswith(".json")
    ])

    if not episode_files:
        print(f"No episode files found in {args.input_dir}/")
        sys.exit(1)

    all_gold_facts = []
    for ep_file in episode_files:
        with open(os.path.join(args.input_dir, ep_file)) as f:
            episode = json.load(f)

        result = generate_gold_facts(episode)
        all_gold_facts.append(result)

        out_file = ep_file.replace("episode_", "gold_facts_")
        out_path = os.path.join(args.output_dir, out_file)
        with open(out_path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

    all_path = os.path.join(args.output_dir, "all_gold_facts.json")
    with open(all_path, 'w') as f:
        json.dump(all_gold_facts, f, indent=2, ensure_ascii=False)

    print(f"Generated gold facts for {len(all_gold_facts)} episodes → {args.output_dir}/")
    print(f"  Individual: gold_facts_001.json ~ gold_facts_{len(all_gold_facts):03d}.json")
    print(f"  Combined:   all_gold_facts.json")
    print()
    for gf in all_gold_facts:
        p1 = len(gf['phase1_fact_introduction'])
        p2 = len(gf['phase2_before_questions'])
        p3_cd = gf['phase3_change_and_deletion']
        p3_change = len([f for f in p3_cd if "change" in f.get("type", "").lower()])
        p3_delete = len([f for f in p3_cd if "delete" in f.get("type", "").lower()])
        p4 = len(gf['phase4_after_questions'])
        total = p1 + len(p3_cd)
        print(f"  Ep{gf['episode_id']:3d}: domain={gf['domain']} root={gf['root']:25s} "
              f"| facts={total} (intro={p1} change={p3_change} delete={p3_delete}) "
              f"| before_q={p2} after_q={p4}")
