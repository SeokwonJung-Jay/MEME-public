"""
Verify assembled episodes.
Run locally after assemble_episodes.py.

Usage:
  python3 verify_assembled.py -d assembled/
"""

import json
import os
import argparse
import sys


def verify_episode(filepath):
    """Verify one episode. Returns (ep_id, issues_list, stats_dict)."""
    with open(filepath) as f:
        data = json.load(f)
    
    ep_id = data['episode_id']
    issues = []
    
    sessions = data['sessions']
    
    # 1. Verify session order: check if evidence is in correct position
    evidence_types_in_order = []
    for i, sess in enumerate(sessions):
        if sess['type'] == 'evidence':
            evidence_types_in_order.append(sess['evidence_type'])

    expected_order = [
        "Fact Introduction (Part 1a)", "Fact Introduction (Part 1b)",
        "Fact Introduction (Part 1c)", "Fact Introduction (Part 2)",
        "Change+Delete Event",
    ]
    expected_present = [t for t in expected_order if t in evidence_types_in_order]
    if evidence_types_in_order != expected_present:
        issues.append(f"Evidence order wrong: {evidence_types_in_order}")

    # 2. Verify timestamps are in ascending order
    timestamps = [sess['timestamp'] for sess in sessions]
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i-1]:
            issues.append(f"Timestamp not ascending at session {i}: {timestamps[i-1]} >= {timestamps[i]}")
            break

    # 3. Verify before/after question positions wrap the Change+Delete Event
    before_q = data.get('before_questions', {})
    after_q = data.get('after_questions', {})
    before_pos = before_q.get('position_after_session', -1)
    after_pos = after_q.get('position_after_session', -1)

    change_delete_idx = None
    for i, sess in enumerate(sessions):
        if sess.get('evidence_type') == 'Change+Delete Event':
            change_delete_idx = i
            break

    if change_delete_idx is not None:
        if before_pos >= change_delete_idx:
            issues.append(f"Before-questions (pos={before_pos}) not before Change+Delete Event (idx={change_delete_idx})")
        if after_pos < change_delete_idx:
            issues.append(f"After-questions (pos={after_pos}) not after Change+Delete Event (idx={change_delete_idx})")
    
    # 4. Evidence sessions have gold_facts
    for i, sess in enumerate(sessions):
        if sess['type'] == 'evidence':
            if not sess.get('gold_facts'):
                issues.append(f"Evidence session {i} ({sess.get('evidence_type')}) has no gold_facts")
    
    # 5. Filler sessions don't have gold_facts
    for i, sess in enumerate(sessions):
        if sess['type'] == 'filler':
            if sess.get('gold_facts'):
                issues.append(f"Filler session {i} has gold_facts (shouldn't)")
    
    # 6. No duplicate filler session IDs within episode
    filler_sids = [sess['session_id'] for sess in sessions if sess['type'] == 'filler']
    if len(filler_sids) != len(set(filler_sids)):
        dup_count = len(filler_sids) - len(set(filler_sids))
        issues.append(f"Duplicate filler session IDs: {dup_count}")
    
    # 7. Question counts
    before_q_count = len(before_q.get('questions', []))
    after_q_count = len(after_q.get('questions', []))
    
    if before_q_count == 0:
        issues.append("No before-questions")
    if after_q_count == 0:
        issues.append("No after-questions")
    
    # Stats
    total_tok = data.get('total_tokens', 0)
    ev_tok = data.get('evidence_tokens', 0)
    stats = {
        "ep_id": ep_id,
        "root": data.get('root', '?'),
        "total_sessions": data['total_sessions'],
        "evidence_sessions": data['evidence_sessions'],
        "filler_sessions": data['filler_sessions'],
        "total_tokens": total_tok,
        "evidence_tokens": ev_tok,
        "evidence_pct": ev_tok / max(1, total_tok) * 100,
        "before_q": before_q_count,
        "after_q": after_q_count,
    }
    
    return ep_id, issues, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify assembled episodes")
    parser.add_argument("-d", "--dir", type=str, default="assembled",
                        help="Directory with assembled episode JSON files")
    args = parser.parse_args()
    
    files = sorted([
        os.path.join(args.dir, f) for f in os.listdir(args.dir)
        if f.startswith("episode_") and f.endswith(".json")
    ])
    
    if not files:
        print(f"No episode files found in {args.dir}/")
        sys.exit(1)
    
    print(f"Verifying {len(files)} assembled episodes...\n")
    
    all_stats = []
    total_issues = 0
    
    for filepath in files:
        ep_id, issues, stats = verify_episode(filepath)
        all_stats.append(stats)
        
        status = "OK" if not issues else "ISSUE"
        print(f"  Ep{ep_id:3d} | {status} | {stats['total_sessions']} sess | "
              f"{stats['total_tokens']} tok | "
              f"evidence {stats['evidence_pct']:.1f}% | "
              f"before_q={stats['before_q']} after_q={stats['after_q']} | "
              f"root={stats['root']}")
        
        if issues:
            for issue in issues:
                print(f"         - {issue}")
            total_issues += len(issues)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(files)} episodes verified")
    print(f"  Issues: {total_issues}{' (FIX NEEDED)' if total_issues else ''}")

    tokens = [s['total_tokens'] for s in all_stats]
    print(f"  Tokens: min={min(tokens)} max={max(tokens)} avg={sum(tokens)//len(tokens)}")
    
    ev_pcts = [s['evidence_pct'] for s in all_stats]
    print(f"  Evidence %: min={min(ev_pcts):.1f}% max={max(ev_pcts):.1f}% avg={sum(ev_pcts)/len(ev_pcts):.1f}%")
    
    sess_counts = [s['total_sessions'] for s in all_stats]
    print(f"  Sessions: min={min(sess_counts)} max={max(sess_counts)} avg={sum(sess_counts)//len(sess_counts)}")
    
    roots = {}
    for s in all_stats:
        roots[s['root']] = roots.get(s['root'], 0) + 1
    print(f"  Root distribution: {dict(sorted(roots.items()))}")
