"""Freeze disjoint traffic partitions and reject data leakage before a run."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

REQUIRED_FILES = {"mobility.npz", "buildings.npz", "scenario.json", "audit.json"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(piece)
    return digest.hexdigest()


def freeze_datasets(root, splits, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("frozen manifest already exists; choose a new version")
    entries, identities = [], set()
    for split, directories in splits.items():
        if split not in ("train", "validation", "test"):
            raise ValueError("unknown split")
        for directory in directories:
            directory = Path(directory).resolve()
            if not directory.is_relative_to(root):
                raise ValueError("dataset must remain within the declared root")
            scenario = json.loads((directory / "scenario.json").read_text(encoding="utf-8"))
            audit = json.loads((directory / "audit.json").read_text(encoding="utf-8"))
            seed, n = int(scenario["seed"]), int(scenario["vehicles"])
            identity = (seed, n)
            if identity in identities:
                raise ValueError("traffic seed/scale reused across partitions")
            identities.add(identity)
            if int(audit.get("vehicles", -1)) != n:
                raise ValueError("scenario/audit vehicle counts disagree")
            for key in ("collision_count", "teleport_count", "building_overlap_count", "arrived_count"):
                if audit.get(key) != 0:
                    raise ValueError(f"traffic audit failed: {directory} {key}")
            with np.load(directory / "mobility.npz", allow_pickle=False) as data:
                if (data["center_xy_m"].shape != (1001, n, 2)
                        or data["heading_rad"].shape != (1001, n)
                        or data["speed_mps"].shape != (1001, n)
                        or data["time_s"].shape != (1001,)
                        or not np.allclose(data["time_s"], np.arange(1001)*.1, rtol=0, atol=1e-9)
                        or len(set(map(str, data["vehicle_ids"]))) != n
                        or not np.isfinite(data["center_xy_m"]).all()
                        or not np.isfinite(data["heading_rad"]).all()
                        or not np.isfinite(data["speed_mps"]).all()):
                    raise ValueError("invalid traffic identity, dimensions or sample grid")
            files = {name: sha256(directory / name)
                     for name in ("mobility.npz", "buildings.npz", "scenario.json", "audit.json")}
            if any(e["files"]["mobility.npz"] == files["mobility.npz"] for e in entries):
                raise ValueError("identical mobility reused across partitions")
            entries.append(dict(split=split, traffic_seed=seed, n=n,
                                directory=directory.relative_to(root).as_posix(), files=files))
    split_seeds = {split: {e["traffic_seed"] for e in entries if e["split"] == split}
                   for split in splits}
    for a in split_seeds:
        for b in split_seeds:
            if a != b and split_seeds[a] & split_seeds[b]:
                raise ValueError("traffic seeds must be disjoint even across scale")
    value = dict(schema_version=1, root_relative_to_manifest="../..", entries=entries,
                 purpose="independent traffic partitions; short runs remain development validation")
    # The root path is derived, so the registry also works after unpacking elsewhere.
    import os
    value["root_relative_to_manifest"] = Path(os.path.relpath(root, output.parent)).as_posix()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return value


def resolve_dataset(manifest, split, n):
    path = Path(manifest).resolve()
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or not isinstance(value.get("entries"), list) or not value["entries"]:
        raise ValueError("invalid dataset manifest schema")
    entries = value["entries"]
    seeds = {}
    identities, mobility_splits = set(), {}
    for item in entries:
        if (item.get("split") not in ("train", "validation", "test")
                or not isinstance(item.get("n"), int) or item["n"] < 2 or item["n"] % 2
                or not isinstance(item.get("traffic_seed"), int)):
            raise ValueError("invalid dataset identity")
        identity = (item["split"], item["n"])
        if identity in identities:
            raise ValueError("duplicate dataset split/scale")
        identities.add(identity)
        files = item.get("files", {})
        if not isinstance(files, dict) or set(files) != REQUIRED_FILES:
            raise ValueError("manifest must hash exactly the four required files")
        if any(not isinstance(d, str) or not re.fullmatch(r"[0-9a-f]{64}", d) for d in files.values()):
            raise ValueError("invalid SHA256 digest")
        other = mobility_splits.setdefault(files["mobility.npz"], item["split"])
        if other != item["split"]:
            raise ValueError("identical mobility data across partitions")
        old = seeds.setdefault(int(item["traffic_seed"]), item["split"])
        if old != item["split"]:
            raise ValueError("overlapping train/validation/test traffic seeds")
    chosen = [e for e in entries if e["split"] == split and e["n"] == n]
    if len(chosen) != 1:
        raise ValueError("require exactly one frozen scene for requested split/scale")
    item = chosen[0]
    root = (path.parent / value["root_relative_to_manifest"]).resolve()
    directory = (root / item["directory"]).resolve()
    if not directory.is_relative_to(root):
        raise ValueError("unsafe dataset path")
    for name, expected in item["files"].items():
        target = (directory / name).resolve()
        if not target.is_relative_to(directory):
            raise ValueError("unsafe dataset file path")
        if sha256(target) != expected:
            raise ValueError(f"frozen traffic file changed: {name}")
    return directory, item, sha256(path)
