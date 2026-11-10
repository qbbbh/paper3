"""Add byte-preserving gzip copies of original results, without running research code."""
import gzip
import json
from pathlib import Path
import shutil
import build_package as b


def main():
    b.manifest=b.read(b.OUT/"SOURCE_MANIFEST.json")
    b.source_ids={r["local_path"]:r["id"] for r in b.manifest["sources"]}
    full=b.read(b.SRC/"analysis/full_summary.json")
    index={"scope":"Original formal result files, BEFORE 20261110 guarded power refinement", "runs":[]}
    for group in b.GROUPS:
        for i, item in enumerate(full["groups"][group]["historical_provenance"]):
            src=Path(item["source"])
            name=group if group!="50B" else ("50B_main" if i==0 else "50B_supplement")
            dest=b.OUT/"baseline_records"/name;dest.mkdir(parents=True,exist_ok=True)
            row=dict(group=group,name=name,local_source=str(src),files=[])
            data=b.read(src/"summary.json")
            row["methods"]={m:dict(planned=sum(r["method"]==m for r in data["blocks"]),success=sum(r["method"]==m and r["success"] for r in data["blocks"])) for m in sorted({r["method"] for r in data["blocks"]})}
            for filename in ("results.csv","summary.json"):
                out=dest/(filename+".gz")
                with (src/filename).open("rb") as fin, out.open("wb") as fout:
                    with gzip.GzipFile(filename="",fileobj=fout,mode="wb",mtime=0,compresslevel=6) as gz:
                        shutil.copyfileobj(fin,gz,1024*1024)
                sid=b.source(src/filename,out,"Lossless gzip; decompressed bytes exactly equal original source")
                row["files"].append(dict(path=out.relative_to(b.OUT).as_posix(),source_id=sid,
                    uncompressed_sha256=b.sha(src/filename),compressed_sha256=b.sha(out)))
                assert out.stat().st_size<50_000_000
            index["runs"].append(row)
    old=b.SRC.parent/"跨规模结果分析_20261110"
    dest=b.OUT/"baseline_records/pre_guard_cross_scale"
    for src in old.iterdir():
        if src.is_file() and src.suffix in {".md",".json",".csv",".py"}:b.copy_map[src]=dest/src.name
    for src,out in list(b.copy_map.items()):
        if src.parent!=old:continue
        out.parent.mkdir(parents=True,exist_ok=True)
        if src.suffix==".md":
            body=b.rewrite_markdown(src.read_text(encoding="utf-8"),out)
            body="<!-- PUBLICATION NOTE: Original pre-guard results; not current guarded-result conclusions. -->\n\n"+body
            out.write_text(body,encoding="utf-8")
            b.source(src,out,"Pre-guard label and local link rewriting; numerical text unchanged")
        else:shutil.copyfile(src,out);b.source(src,out)
    b.write(b.OUT/"baseline_records/INDEX.json",index)
    b.write(b.OUT/"SOURCE_MANIFEST.json",b.manifest)
    print(json.dumps({r["name"]:r["methods"] for r in index["runs"]},ensure_ascii=False))


if __name__=="__main__":main()
