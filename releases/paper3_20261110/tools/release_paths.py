"""Repository-relative paths and byte checks; no scientific-model changes."""
from pathlib import Path
import hashlib
import json

RELEASE = Path(__file__).resolve().parents[1]
REPOSITORY = RELEASE.parents[1]
CODE = RELEASE / "code"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scene_directory(_original_root, n):
    """Only adapt the directory lookup; use the original Scene and PHY code."""
    registry = load_json(RELEASE / "provenance/traffic_files.json")
    entry = registry["scenarios"][str(int(n))]
    directory = (REPOSITORY / entry["repository_relative_directory"]).resolve()
    if not directory.is_relative_to(REPOSITORY):
        raise ValueError("traffic directory is outside the repository")
    for name, expected in entry["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"repository traffic hash mismatch: n{n}/{name}")
    return directory


def verify_release():
    frozen = load_json(RELEASE / "provenance/source_files.json")
    manifest = load_json(RELEASE / "provenance/distribution_sources.json")
    if digest(RELEASE / manifest["frozen_source_manifest"]) != manifest["frozen_source_manifest_sha256"]:
        raise ValueError("historical 78-file source manifest changed")
    if digest(RELEASE / manifest["historical_baseline_manifest"]) != manifest["historical_baseline_manifest_sha256"]:
        raise ValueError("historical baseline manifest changed")
    original = {item["path"]: item for item in frozen["files"]}
    expected = {item["path"] for item in manifest["files"]}
    removed = {item["path"] for item in manifest["removed_from_distribution"]}
    if expected & removed or expected | removed != set(original):
        raise ValueError("distribution does not account for the full frozen snapshot")
    for item in manifest["files"] + manifest["removed_from_distribution"]:
        if item != original[item["path"]]:
            raise ValueError(f"distribution/source provenance differs: {item['path']}")
    actual = {p.relative_to(RELEASE).as_posix() for p in CODE.rglob("*.py")}
    if actual != expected:
        raise ValueError("source file set differs from the declared distribution subset")
    for item in manifest["files"] + [frozen["fixture"]]:
        path = RELEASE / item["path"]
        if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"release file hash mismatch: {item['path']}")
    aggregate = hashlib.sha256()
    for path in sorted(CODE.rglob("*.py")):
        aggregate.update(path.relative_to(RELEASE).as_posix().encode())
        aggregate.update(path.read_bytes())
    if aggregate.hexdigest() != manifest["distribution_aggregate_sha256"]:
        raise ValueError("distribution aggregate source hash mismatch")
    registry = load_json(RELEASE / "provenance/traffic_files.json")
    input_manifest = load_json(REPOSITORY / "data/INPUT_MANIFEST.json")
    input_files = {item["path"]: item for item in input_manifest["files"]}
    registry_files = {}
    for n, entry in registry["scenarios"].items():
        scene_directory(None, int(n))
        for name, expected_hash in entry["files"].items():
            relative = entry["repository_relative_directory"] + "/" + name
            registry_files[relative] = expected_hash
    if set(input_files) != set(registry_files):
        raise ValueError("input manifest does not match all 12 registered traffic files")
    for name, expected_hash in registry_files.items():
        item, path = input_files[name], REPOSITORY / name
        if item["sha256"] != expected_hash or path.stat().st_size != item["bytes"]:
            raise ValueError(f"input manifest differs: {name}")
    for case in ("10A", "10B", "50B"):
        settings = load_json(RELEASE / "configs" / f"{case}.json")
        source = load_json(RELEASE / "provenance" / f"replay_manifest_{case}.json")
        source_config = next(x["config"] for x in source["inputs"] if x["group"] == case)
        if settings["saved_config"] != source_config:
            raise ValueError(f"case settings differ from the saved manifest: {case}")
        if settings["traffic"] != registry["scenarios"][str(settings["vehicles"])]:
            raise ValueError(f"case traffic directory or hashes differ: {case}")
        if digest(RELEASE / "provenance" / f"replay_manifest_{case}.json") != settings["history"]["replay_manifest_sha256"]:
            raise ValueError(f"source manifest changed: {case}")
        for name, expected_hash in source["source_hashes"].items():
            if digest(CODE / name.replace("\\", "/")) != expected_hash:
                raise ValueError(f"frozen scientific source changed: {name}")
        scene_directory(None, settings["vehicles"])
    return {"source_files": manifest["distribution_count"], "source_bytes": manifest["distribution_bytes"],
            "frozen_source_files": frozen["count"], "omitted_development_files": len(removed),
            "traffic_files": len(input_files), "distribution_sha256": aggregate.hexdigest(),
            "cases": ["10A", "10B", "50B"], "status": "verified"}
