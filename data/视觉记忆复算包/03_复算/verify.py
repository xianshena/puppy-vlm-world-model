"""Recompute episode outcomes, paired scores and model costs without a GPU."""
import csv
import gzip
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import re
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_原始数据"
REPORTS = ROOT / "01_我要看的内容"
SUMMARY = "single_action_compaction"
MEMORY = ("records", "full_history", SUMMARY)
POLICIES = ("current", "recent", "keyframes", *MEMORY)
CONDITIONS = ("stay", "seen", "unseen", "counterfactual")


def choose(raw):
    match = re.fullmatch(r"\s*([ABC])[.!]?\s*", raw, flags=re.I)
    return match[1].upper() if match else None


def score_episode(e, groups, invalid, roles):
    replies, observed = e["calls"], e["history"]
    first = {}
    seen_calls = set()
    for name in replies:
        if name.startswith("perception:"):
            seen_calls.add(name)
            roles["perception"] += 1
    for h in observed:
        assert replies["perception:" + h["frame"]]["raw"].strip() == h["perceived_blue"]
    assert len(observed) == 12
    for policy, steps in e["steps"].items():
        actions, writes = [], []
        if policy == SUMMARY:
            for t in range(len(observed)):
                tag = f"initial_compaction:{t}"
                writes.append(replies[tag])
                seen_calls.add(tag)
                roles["initial_write"] += 1
        choices = []
        for i, step in enumerate(steps):
            tag = f"{policy}:action:{i}"
            r = replies[tag]
            action = choose(r["raw"])
            choices.append(action)
            assert action == step["action"]
            assert bool(step.get("found")) == (action == e["target"])
            if action is None:
                invalid[policy][r["raw"]] += 1
                assert "frame" not in step
            else:
                assert replies["perception:" + step["frame"]]["raw"].strip() == step["observed"]
                if policy == SUMMARY:
                    name = f"{policy}:write:{i}"
                    writes.append(replies[name])
                    seen_calls.add(name)
                    roles["post_write"] += 1
            actions.append(r)
            seen_calls.add(tag)
            roles["decision"] += 1
        assert 1 <= len(choices) <= 3
        assert all(c and c != e["target"] for c in choices[:-1])
        assert len(choices) == 3 or not choices[-1] or choices[-1] == e["target"]
        first[policy] = choices[0]
        billed = actions + writes
        condition = e["id"].split("-")[1]
        groups[condition, policy].append({
            "success": e["target"] in choices, "first_correct": choices[0] == e["target"],
            "steps": len(choices), "inspection_misses": sum(c is not None and c != e["target"] for c in choices),
            "invalid_action": None in choices, "max_action_prompt_tokens": max(r["input_tokens"] for r in actions),
            "total_model_tokens": sum(r["input_tokens"] + r["output_tokens"] for r in billed),
            "total_model_seconds": sum(r["seconds"] for r in billed),
            "decision_images": sum(len(r["images"]) for r in actions),
            "token_usage_complete": all(isinstance(r["input_tokens"], int) and isinstance(r["output_tokens"], int)
                                        for r in billed),
        })
    assert seen_calls == set(replies), (e["id"], seen_calls ^ set(replies))
    return first


def main():
    manifest = json.loads((DATA / "来源.json").read_bytes())
    binary = (DATA / "正式回合.jsonl.gz").read_bytes()
    assert hashlib.sha256(binary).hexdigest() == manifest["data_gzip_sha256"]
    text = gzip.decompress(binary)
    assert hashlib.sha256(text).hexdigest() == manifest["data_uncompressed_sha256"]
    groups, layouts, invalid = defaultdict(list), defaultdict(dict), defaultdict(Counter)
    roles, cf, changed, matched = Counter(), Counter(), Counter(), Counter()
    for line in text.splitlines():
        episode = json.loads(line)
        layout, cond = episode["id"].split("-")
        assert cond in CONDITIONS and cond not in layouts[layout]
        assert set(episode["steps"]) == set(MEMORY if cond == "counterfactual" else POLICIES)
        first = score_episode(episode, groups, invalid, roles)
        layouts[layout][cond] = (first, episode["target"], episode["history"], episode["final_view_sha256"])
    assert set(map(int, layouts)) == set(manifest["formal_layouts"]) == set(range(1000, 1211))
    for variants in layouts.values():
        assert set(variants) == set(CONDITIONS)
        old, swapped, hidden = [variants[v] for v in ("stay", "counterfactual", "unseen")]
        assert old[2] == hidden[2] and old[1] != hidden[1]
        assert old[3] == swapped[3] and old[1] != swapped[1]
        assert [x["view"] for x in old[2]] == [x["view"] for x in swapped[2]]
        for p in MEMORY:
            cf[p] += old[0][p] == old[1] and swapped[0][p] == swapped[1]
            changed[p] += bool(old[0][p] and swapped[0][p] and old[0][p] != swapped[0][p])
        for p in POLICIES:
            matched[p] += old[0][p] == hidden[0][p]

    with (REPORTS / "分条件实测.csv").open(encoding="utf-8-sig", newline="") as handle:
        official = list(csv.DictReader(handle))
    assert len(official) == len(groups) == 21
    assert {(r["condition"], r["policy"]) for r in official} == set(groups)
    for summary in official:
        trials = groups[summary["condition"], summary["policy"]]
        assert len(trials) == int(summary["n"])
        for metric in trials[0]:
            assert math.isclose(mean(x[metric] for x in trials), float(summary[metric]),
                                rel_tol=1e-12, abs_tol=1e-10), (summary["condition"], summary["policy"], metric)
    with (REPORTS / "反事实配对实测.csv").open(encoding="utf-8-sig", newline="") as handle:
        expected_pairs = list(csv.DictReader(handle))
    assert len(expected_pairs) == len(MEMORY) and {row["policy"] for row in expected_pairs} == set(MEMORY)
    for row in expected_pairs:
        policy = row["policy"]
        assert len(layouts) == int(row["paired_layouts"])
        assert cf[policy] == int(row["both_first_correct"])
        assert changed[policy] == int(row["valid_first_action_changed"])
    result = {
        "layouts": len(layouts), "episodes": sum(len(v) for v in layouts.values()),
        "policy_trials": sum(len(v) for v in groups.values()),
        "verified_groups": len(groups), "requests": sum(roles.values()), "roles": dict(roles),
        "illegal_raw_replies": dict(invalid), "counterfactual_pairs": len(layouts) * len(MEMORY),
        "counterfactual_first_both_correct": dict(cf), "counterfactual_first_changed_valid": dict(changed),
        "matched_first_action_in_stay_unseen": dict(matched),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
