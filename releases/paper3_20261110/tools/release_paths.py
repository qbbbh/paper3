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
    directory = REPOSITORY / entry["repository_relative_directory"]
    for name, expected in entry["files"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"repository traffic hash mismatch: n{n}/{name}")
    return directory


def verify_release():
    manifest = load_json(RELEASE / "provenance/source_files.json")
    expected = {item["path"] for item in manifest["files"]}
    actual = {p.relative_to(RELEASE).as_posix() for p in CODE.rglob("*.py")}
    if actual != expected:
        raise ValueError("source file set differs from the byte-preserved snapshot")
    for item in manifest["files"] + [manifest["fixture"]]:
        path = RELEASE / item["path"]
        if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"release file hash mismatch: {item['path']}")
    baseline = load_json(RELEASE / "provenance/baseline_source_files.json")
    for item in baseline["files"]:
        path = RELEASE / item["path"]
        if digest(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
            raise ValueError(f"historical baseline source changed: {item['path']}")
    for case in ("10A", "10B", "50B"):
        settings = load_json(RELEASE / "configs" / f"{case}.json")
        source = load_json(RELEASE / "provenance" / f"replay_manifest_{case}.json")
        source_config = next(x["config"] for x in source["inputs"] if x["group"] == case)
        if settings["saved_config"] != source_config:
            raise ValueError(f"case settings differ from the saved manifest: {case}")
        if digest(RELEASE / "provenance" / f"replay_manifest_{case}.json") != settings["history"]["replay_manifest_sha256"]:
            raise ValueError(f"source manifest changed: {case}")
        for name, expected_hash in source["source_hashes"].items():
            if digest(CODE / name.replace("\\", "/")) != expected_hash:
                raise ValueError(f"frozen scientific source changed: {name}")
        scene_directory(None, settings["vehicles"])
    return {"source_files": manifest["count"], "source_bytes": manifest["bytes"],
            "baseline_files": baseline["count"], "baseline_bytes": baseline["bytes"],
            "cases": ["10A", "10B", "50B"], "status": "verified"}
