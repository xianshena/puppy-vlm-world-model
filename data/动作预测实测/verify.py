"""Recompute results from saved predictions with no CUDA or model imports."""
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
from random import Random
from statistics import mean

ROOT = Path(__file__).resolve().parent


def summarize(mapping):
    return [{"group": list(k), "n": len(rows), "mean": mean(rows)}
            for k, rows in sorted(mapping.items()) if rows]


def intervals(values):
    if not values:
        return None
    rng = Random(20261119)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(3000))
    return {"layouts": len(values), "rate_difference": mean(values),
            "interval95": [draws[74], draws[2924]]}


def binary_stats(samples):
    bins = []
    for i in range(10):
        batch = [(p, y) for p, y in samples if min(9, int(10*p)) == i]
        if batch:
            bins.append({"bin": i, "n": len(batch), "prediction": mean(p for p, _ in batch),
                         "actual": mean(y for _, y in batch)})
    return {"n": len(samples), "brier": mean((p-y)**2 for p, y in samples),
            "nll": mean(-math.log(max(1e-7, p if y else 1-p)) for p, y in samples), "bins": bins}


def hit(p, y):
    return int((p >= .5) == bool(y))


def weights_report():
    weights = {}
    for p in sorted((ROOT/"weights").glob("*/curve.json")):
        d = json.loads(p.read_text())
        weights[p.parent.name] = {"epochs": len(d["epochs"]), "updates": sum(r["updates"] for r in d["epochs"]),
                                 "device": d["device"], "parameters": d["params"], "seed_hash": d["initial_sha"],
                                 "order": [e["order_sha"] for e in d["epochs"]],
                                 "best_epoch": min(d["epochs"], key=lambda r: r["validation_nll"])["epoch"]}
    matches = {}
    for seed in (17, 29, 43):
        a,b = (weights.get(f"{tag}-{seed}") for tag in ("action", "no-action"))
        if a and b:
            matches[str(seed)] = a["seed_hash"]==b["seed_hash"] and a["order"]==b["order"]
    return weights, matches


def interface():
    rows = list(jsonl(ROOT/"interface.jsonl"))
    def clean(a):
        return a.strip().rstrip(".!")
    scores = {}
    for mode in ("old", "new"):
        legal = [r for r in rows if clean(r[mode]) in ("A","B","C")]
        correct = sum(clean(r[mode]) == r["target"] for r in legal)
        scores[mode] = {"n":len(rows),"legal":len(legal),"correct":correct,
                        "correct_given_legal":correct/len(legal) if legal else None}
    scores["h_to_correct"] = sum(clean(r["old"])=="H" and clean(r["new"])==r["target"] for r in rows)
    return scores


def forecast(directory):
    fields, pairs, hidden, masses, rollouts = (defaultdict(list) for _ in range(5))
    pairing = {}
    example = []
    for p in sorted(directory.glob("*-forks.jsonl.gz")):
        name = p.name.removesuffix("-forks.jsonl.gz")
        groups = defaultdict(list)
        for r in read_gz(p):
            w = r["world_after"]
            assert r["truth"][1:3] == [int(w[0]==w[1] and not w[2]),int(w[2])]
            groups[(r["layout"],r["style"],r["pair"])].append(r)
        for (layout,style,pair), rows in groups.items():
            assert all(rows[0]["frames"]==r["frames"] and rows[0]["history_actions"]==r["history_actions"] for r in rows)
            if pair=="hidden":
                assert len(rows)==12
                if not example:
                    example = [r for r in rows if r["action"]=="B"]
                masses[(name,style)].append(sum(r["predicted"][4] for r in rows if r["branch"]==0))
                for act, prior in zip("ABC",(.25,.5,.25)):
                    a = [r for r in rows if r["action"]==act]
                    assert len(a)==4 and mean(r["truth"][1] for r in a)==prior
                    assert max(r["predicted"][4] for r in a)-min(r["predicted"][4] for r in a)<1e-5
                    for row in a:
                        hidden[(name,style,act)].append((row["predicted"][4],row["truth"][1]))
                        hidden[("prior",style,act)].append((prior,row["truth"][1]))
                continue
            assert len(rows)==2 and rows[0]["action"]!=rows[1]["action"]
            for mode in ("predicted","oracle_observation","shuffled","persistence"):
                scores = []
                for r in rows:
                    for i,label in enumerate(("ground","held")):
                        ok = hit(r[mode][4+i],r["truth"][1+i])
                        fields[(name,style,pair,mode,label)].append(ok)
                        scores.append(ok)
                score = int(all(scores))
                pairs[(name,style,pair,mode)].append(score)
                if mode=="predicted":
                    pairing[name,layout,style,pair] = score
    for p in sorted(directory.glob("*-sequence.jsonl.gz")):
        name = p.name.removesuffix("-sequence.jsonl.gz")
        for r in read_gz(p):
            for call in r["predictions"]:
                for mode in ("open_loop","observed_refresh","shuffled"):
                    if mode in call:
                        for i,label in enumerate(("ground","held")):
                            rollouts[(name,r["style"],str(call["horizon"]),mode,label)].append(
                                hit(call[mode][4+i],call["truth"][1+i]))
    diffs = defaultdict(list)
    for (name,layout,style,pair),score in pairing.items():
        partner = ("no-"+name,layout,style,pair)
        if name.startswith("action-") and style==0 and pair in ("pick","place") and partner in pairing:
            diffs[layout].append(score-pairing[partner])
    effect = intervals([mean(v) for _,v in sorted(diffs.items()) if len(v)==6])
    if effect:
        effect["meets_threshold"] = effect["rate_difference"]>=.1 and effect["interval95"][0]>0
    return {"paired_action_effect":effect, "fork_pairs":summarize(pairs),"fork_fields":summarize(fields),
            "hidden":[{"group":list(k),**binary_stats(v)} for k,v in sorted(hidden.items())],
            "hidden_probability_sum":summarize(masses),"rollout":summarize(rollouts)}, example


def closed(directory):
    scores, lengths, latency = (defaultdict(list) for _ in range(3))
    results, examples = {}, {}
    for p in sorted(directory.glob("closed-*.jsonl.gz")):
        for r in read_gz(p):
            steps = r["steps"]
            assert 1<=len(steps)<=8 and all(a["after"]==b["before"] for a,b in zip(steps,steps[1:]))
            w = steps[-1]["after"]
            ok = w[1]==r["goal"] and not w[2]
            group = (r["seed"],r["style"],r["strategy"])
            scores[group].append(int(ok))
            lengths[group].append(len(steps))
            latency[group].extend(x["decision_seconds"] for x in steps)
            results[(r["layout"],)+group] = int(ok)
            if r["strategy"]=="planner":
                examples.setdefault("成功送球" if ok else "规划未成功",r)
    comparisons = {}
    for other in ("shuffled","greedy","fixed"):
        paired = defaultdict(list)
        for (layout,seed,style,strategy),ok in results.items():
            key = (layout,seed,style,other)
            if style==0 and strategy=="planner" and key in results:
                paired[layout].append(ok-results[key])
        comparisons[other] = intervals([mean(v) for _,v in sorted(paired.items()) if len(v)==3])
    return {"success":summarize(scores),"steps":summarize(lengths),
            "decision_seconds":summarize(latency),"paired_comparisons":comparisons},examples


def perception():
    cache = {r["frame"]:r for r in jsonl(ROOT/"perception.jsonl")}
    sample = ROOT/"vision-audit-sample.json"
    values = json.loads(sample.read_text()) if sample.is_file() else []
    matrix = [Counter() for _ in range(4)]
    errors = []
    for r in values:
        if r["frame"] in cache:
            actual = cache[r["frame"]]["codes"]
            expected = r["truth"]
            for counter, t, p in zip(matrix,expected,actual):
                counter[t+" -> "+p]+=1
            if expected!=actual and len(errors)<12:
                errors.append(dict(r,read=actual))
    return {"unique_frames":len(cache),"audit_sample":len(values),
            "confusion":dict(zip(("place","ground","held","feedback"),map(dict,matrix))),"errors":errors}


def jsonl(path):
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            yield from (json.loads(line) for line in handle)
    else:
        yield from read_gz(path.with_suffix(path.suffix + ".gz"))


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def compare(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key, value in actual.items():
            compare(value, expected[key])
    elif isinstance(actual, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected):
            compare(a, b)
    elif isinstance(actual, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def verify_manifest():
    entries = json.loads((ROOT/"来源.json").read_text(encoding="utf-8"))["files"]
    repo = ROOT.parents[1]
    for name, expected in entries.items():
        path = (repo/name).resolve()
        assert path.is_relative_to(repo) and path.is_file(), name
        data = path.read_bytes()
        assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"], name
    return len(entries)


def verify_benchmark(expected):
    evidence = json.loads((ROOT/"verified-throughput-progress.json").read_text(encoding="utf-8"))
    assert len(evidence["cases"]) == 48
    volumes = defaultdict(list)
    outcomes = []
    for run in evidence["runs"]:
        frames = len(run["results"])
        assert frames == 48
        volumes[str(run["lanes"])].append(frames/run["seconds"])
        outcomes.append([r["codes"] for r in run["results"]])
    assert [row["lanes"] for row in evidence["runs"]] == [1, 2, 2, 1]
    assert (all(v == outcomes[0] for v in outcomes[1:])) == expected["parsed_outputs_identical"]
    compare({key: mean(times) for key, times in volumes.items()}, expected["per_mode_mean_images_per_second"])


def validate_closed_ledger(directory):
    raw = read_gz(directory/"closed-all.jsonl.gz")
    keys = set()
    ledger = {}
    for row in jsonl(ROOT/"closed-forecast-ledger.jsonl"):
        index = tuple(row[x] for x in ("layout", "style", "seed", "strategy", "step"))
        assert index not in ledger
        ledger[index] = row
    replayed = 0
    for row in raw:
        index = tuple(row[x] for x in ("layout", "style", "seed", "strategy"))
        assert index not in keys
        keys.add(index)
        frame = row["start_frames"][-1]
        for k, step in enumerate(row["steps"]):
            action = step["action"]
            view, ball, held, _ = step["before"]
            if action in ("A", "B", "C"):
                state = [action, ball, held, "X"]
            elif action == "pick":
                state = [view, "", True, "S"] if view == ball and not held else [view, ball, held, "F"]
            elif action == "place":
                state = [view, view, False, "S"] if held and view in ("A", "B", "C") else [view, ball, held, "F"]
            else:
                assert action == "wait"
                state = [view, ball, held, "X"]
            assert state == step["after"]
            if row["strategy"] in ("planner", "shuffled"):
                record = ledger.pop(index + (k,))
                assert record["action"] == action and record["before_frame"] == frame
                assert record["next_probabilities"] == step["next_probabilities"]
                assert record["scores"] == step["scores"]
            frame = step["frame"]
            replayed += 1
    assert not ledger
    return len(keys), replayed


def main():
    files = verify_manifest()
    metrics = json.loads((ROOT/"metrics.json").read_text(encoding="utf-8"))
    verify_benchmark(metrics["benchmark"])
    weights, paired_order = weights_report()
    compare(weights, metrics["weights"])
    compare(paired_order, metrics["matched_initialization_and_order"])
    compare(interface(), metrics["interface"])
    compare(perception(), metrics["perception"])
    directory = ROOT/"evaluation"
    compare(forecast(directory)[0], metrics["predictions"])
    compare(closed(directory)[0], metrics["closed"])
    trials, steps = validate_closed_ledger(directory)
    print(f"PASS: {files} archived files, {trials} closed-loop trials, {steps} checked actions.")
    print("Benchmarks, training curves, visual audit, forks, rollouts and planning scores agree.")


if __name__ == "__main__":
    main()
