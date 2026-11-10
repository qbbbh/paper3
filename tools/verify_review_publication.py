"""Read-only public review bundle checks. Does not run scientific simulations."""
from pathlib import Path
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    errors = []
    checked = 0
    snapshot = json.loads((ROOT / "SNAPSHOT_SHA256.json").read_text(encoding="utf-8"))
    derived_checked, derived_not_generated = 0, []
    for name, expected in snapshot.get("derived_legacy_files", {}).items():
        p = ROOT / name
        if p.is_file():
            derived_checked += 1
            if digest(p) != expected:
                errors.append("derived legacy mismatch:" + name)
        else:
            derived_not_generated.append(name)
    sources = ROOT / "manuscripts/paper3_20261111/SOURCE_MANIFEST.json"
    own = json.loads(sources.read_text(encoding="utf-8"))
    for item in own["files"]:
        p = ROOT / item["published_path"]
        checked += 1
        if not p.is_file() or digest(p) != item["published_sha256"]:
            errors.append("manuscript:" + item["published_path"])
    source_pub = ROOT / "Paper5修改版/SOURCE_MANIFEST.json"
    if not source_pub.is_file():
        errors.append("Paper5 source manifest missing")
    required = ["README.md", "START_HERE.md", "MODEL_SPEC.md", "PROJECT_STATUS.md", "REVIEW_BRIEF.md",
                "RUNBOOK.md", "PUBLICATION_SCOPE.md", "Paper5修改版/README.md",
                "releases/paper3_20261110/README.md", "evidence/paper3_20261110/README.md",
                "evidence/paper3_20261110/DATA_CATALOG.md", "manuscripts/paper3_20261111/README.md"]
    for rel in required:
        if not (ROOT / rel).is_file():
            errors.append("entry missing:" + rel)
    # Raw copied author files and historical documents deliberately retain source
    # paths. Validate navigation, the normalized manuscript view and evidence docs.
    navigation = [ROOT / p for p in required if p.endswith(".md")]
    navigation.extend((ROOT / "releases/paper3_20261110").rglob("README.md"))
    navigation.extend((ROOT / "evidence/paper3_20261110").rglob("*.md"))
    navigation.extend((ROOT / "manuscripts/paper3_20261111").rglob("*.md"))
    navigation.append(ROOT / "Paper5修改版/LINK_INDEX.md")
    link_errors = []
    abs_sources = 0
    expression = re.compile(r"!?\[[^\]\n]*\]\((<?)([^)\n]+?)(>?)\)")
    for p in sorted(set(navigation)):
        if not p.is_file():
            continue
        for match in expression.finditer(p.read_text(encoding="utf-8-sig")):
            text = match.group(2).strip().strip("<>")
            if re.match(r"^[a-z]+://", text, re.I) or text.startswith(("#", "mailto:", "data:")):
                continue
            if re.match(r"^[A-Z]:[\\/]", text, re.I):
                abs_sources += 1
                continue
            target = unquote(text.split("#", 1)[0].split("?", 1)[0])
            if not target:
                continue
            resolved = (p.parent / target).resolve()
            if not resolved.is_relative_to(ROOT) or not resolved.exists():
                link_errors.append(dict(file=p.relative_to(ROOT).as_posix(), target=text))
    errors.extend("link:" + x["file"] + " -> " + x["target"] for x in link_errors)
    forbidden = [p.relative_to(ROOT).as_posix() for area in (ROOT / "releases/paper3_20261110", ROOT / "Paper5修改版", ROOT / "manuscripts/paper3_20261111", ROOT / "evidence/paper3_20261110")
                 for p in area.rglob("*") if p.is_file() and (p.name == ".env" or p.suffix in {".pkl", ".pt", ".pth", ".pyc"} or p.name == "restart.pkl.gz")]
    errors.extend("excluded file:" + p for p in forbidden)
    result = dict(success=not errors, source_view_files_checked=checked,
                  derived_legacy_checked=derived_checked,
                  derived_legacy_not_generated=derived_not_generated,
                  navigation_files=len(set(navigation)), source_absolute_links=abs_sources,
                  source_absolute_note="retained only as provenance when documented; not executable cloud paths",
                  errors=errors)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
