"""Extract the formal episodes from the saved experiment, using Python 3.9+."""
import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "02_原始数据"
REPORTS = ROOT / "01_我要看的内容"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path, help="project containing results and learning-20261118")
    args = parser.parse_args()
    original = args.experiment / "results"
    receipt = json.loads((original / "archive-receipt.json").read_bytes())["files"]
    DATA.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    def get(name):
        content = (original / name).read_bytes()
        if digest(content) != receipt[name]["sha256"] or len(content) != receipt[name]["bytes"]:
            raise ValueError(f"Source verification failed for {name}")
        return content

    plan = json.loads(get("plan.json"))
    ids = plan["evaluation_layouts"]
    formal = set(ids)
    calls = defaultdict(dict)
    excluded = 0
    for line in get("requests.jsonl").splitlines():
        request = json.loads(line)
        episode, tag = request["tag"].split(":", 1)
        if int(episode.split("-")[0]) not in formal:
            assert int(episode.split("-")[0]) in (0, 1)
            excluded += 1
            continue
        if tag in calls[episode]:
            raise ValueError(f"Duplicate tag {episode}:{tag}")
        usage = request["usage"]
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
        calls[episode][tag] = {
            "prompt": request["prompt"], "images": request["images"], "raw": request["raw"],
            "input_tokens": usage["prompt_tokens"], "output_tokens": usage["completion_tokens"],
            "seconds": request["latency_s"],
        }
    output = DATA / "正式回合.jsonl.gz"
    logical_hash = hashlib.sha256()
    cases = 0
    angles = ("stay", "seen", "unseen", "counterfactual")
    with output.open("wb") as handle, gzip.GzipFile(filename="", fileobj=handle, mode="wb", mtime=0) as zipped:
        for layout in ids:
            for variant in angles:
                episode = f"{layout:04d}-{variant}"
                source_file = f"episodes/{episode}.json"
                row = json.loads(get(source_file))
                assert row["scenario"]["id"] == episode
                steps = {}
                for policy, trial in row["policies"].items():
                    steps[policy] = []
                    for i, s in enumerate(trial["steps"]):
                        assert s["step"] == i and s["raw"] == calls[episode][f"{policy}:action:{i}"]["raw"]
                        steps[policy].append({k: s[k] for k in ("action", "found", "observed", "frame") if k in s})
                history = row["observation_history"]
                for marker, actual in zip(history, row["scenario"]["history"]):
                    assert marker["tick"] == actual["tick"] and marker["view"] == actual["view"]
                record = {
                    "id": episode, "target": row["scenario"]["target"], "initial": row["scenario"]["initial"],
                    "history": history, "records": row["records"], "keyframes": row["retrieved_indices"],
                    "source_episode_sha256": receipt[source_file]["sha256"],
                    "final_view_sha256": receipt["frames/" + history[-1]["frame"]]["sha256"],
                    "steps": steps, "calls": calls.pop(episode),
                }
                line = (json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
                zipped.write(line)
                logical_hash.update(line)
                cases += 1
    assert not calls
    (REPORTS / "分条件实测.csv").write_bytes(get("metrics.csv"))
    pairs = json.loads(get("summary.json"))["counterfactual"]
    with (REPORTS / "反事实配对实测.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["policy", "paired_layouts", "both_first_correct", "valid_first_action_changed"])
        for policy in ("records", "full_history", "single_action_compaction"):
            group = [p for p in pairs if p["policy"] == policy]
            writer.writerow([policy, len(group), sum(p["both_correct"] for p in group),
                             sum(p["changed"] for p in group)])
    casebook = args.experiment / "learning-20261118" / "examples" / "小狗找球_四局案例.html"
    shutil.copyfile(casebook, REPORTS / casebook.name)
    selected = ("requests.jsonl", "plan.json", "metrics.csv", "summary.json", "experiment.py", "world.py")
    for name in selected:
        get(name)
    model = json.loads(get("manifest.json"))
    provenance = {
        "formal_layouts": ids, "formal_episodes": cases, "pilot_requests_omitted": excluded,
        "model": model["model"], "projector": model["projector"],
        "model_sha256": model["model_files"][model["model"]]["sha256"],
        "projector_sha256": model["model_files"][model["projector"]]["sha256"],
        "data_gzip_sha256": digest(output.read_bytes()), "data_uncompressed_sha256": logical_hash.hexdigest(),
        "original_source_files": {name: receipt[name] for name in selected},
    }
    (DATA / "来源.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{cases} episodes, {excluded} pilot requests omitted, {output.stat().st_size} compressed bytes.")


if __name__ == "__main__":
    main()
