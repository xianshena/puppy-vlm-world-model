"""Validate and inspect the recorded development set with Python's standard library."""
import gzip
import hashlib
import json
import tarfile
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_table(name, digests):
    with gzip.open(ROOT / (name + ".gz"), "rb") as stream:
        data = stream.read()
    assert hashlib.sha256(data).hexdigest() == digests[name], name
    rows = [json.loads(line) for line in data.splitlines()]
    by_id = {row["id"]: row for row in rows}
    assert len(rows) == len(by_id)
    return by_id


def main():
    summary = json.loads((ROOT / "summary.json").read_text(encoding="utf-8"))
    inputs, targets, provenance = [
        load_table(name, summary["sha256"])
        for name in ("inputs.jsonl", "targets.jsonl", "provenance.jsonl")
    ]
    assert inputs.keys() == targets.keys() == provenance.keys()
    images = set()
    by_episode = defaultdict(set)
    grouped = defaultdict(set)
    for key, features in inputs.items():
        assert set(features) == {"id", "history", "inspect"}
        action = features["inspect"]
        history = features["history"]
        assert action in ("A", "B", "C") and len(history) == 12
        assert all(set(frame) == {"view", "image"} for frame in history)
        episode = provenance[key]["episode"]
        assert action not in by_episode[episode]
        by_episode[episode].add(action)
        signature = (tuple((frame["view"], frame["image"]) for frame in history), action)
        grouped[signature].add(targets[key]["blue_visible"])
        images.update(frame["image"] for frame in history)
        images.add(targets[key]["next_image"])

    frame_dir = ROOT.parent / "图像"
    frame_map = json.loads((frame_dir / "正式帧索引.json").read_text(encoding="utf-8"))
    expected = {f"images/{value}.jpg" for value in frame_map.values()}
    seen = set()
    with tarfile.open(frame_dir / "正式图像.tar.gz", "r|gz") as archive:
        for entry in archive:
            assert entry.isfile() and entry.name in expected and entry.name not in seen
            data = archive.extractfile(entry).read()
            digest = hashlib.sha256(data).hexdigest()
            assert entry.name == f"images/{digest}.jpg"
            seen.add(entry.name)
    assert expected == seen and images <= seen and len(images) == summary["distinct_images"]
    assert len(inputs) == summary["distinct_episode_action_rows"]
    assert len(by_episode) == summary["episodes"]
    assert len({p["layout"] for p in provenance.values()}) == summary["layouts"]
    observed = {str(n): count for n, count in Counter(map(len, by_episode.values())).items()}
    assert observed == summary["episodes_by_number_of_distinct_observed_first_actions"]
    assert len(grouped) == summary["distinct_observed_history_and_action_signatures"]
    both = sum(len(results) == 2 for results in grouped.values())
    assert both == summary["identical_observed_history_and_action_signatures_with_both_outcomes"]
    print(f"PASS: {len(inputs)} observations, {len(images)} JPEGs, {len(grouped)} distinct inputs, "
          f"{both} inputs with both possible outcomes.")


if __name__ == "__main__":
    main()
