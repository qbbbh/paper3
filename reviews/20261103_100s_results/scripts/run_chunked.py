"""分块运行器：绕过"参考解失败导致整条 run 崩溃"的问题，**不修改 v5 源码**。

背景：`run_allocation.py:649` 在 `reference.eta_ref is None` 时抛 `TypeError`
（`budget = (1-args.delta) * reference.eta_ref`），整条 run 崩溃且输出全丢
（结果只在结尾统一写盘）。已确认种子 `2026102801` 的 **block 588** 触发。

策略：把 999 块切成若干段，**逐段独立进程运行**；
- 崩掉的段最多损失该段的块数；
- 失败块单独记录（按 MODEL_SPEC「求解失败块必须单列」）；
- 全部段跑完后按块合并 `results.csv`，供 `analyze_100s.py` 聚合。

**不改源码、不改配置语义**：每段仍是同一 `code/`、同一 `source_sha256`，
只是 `--start-block/--blocks` 不同。

用法::

    python -B -I run_chunked.py --channel-seed 2026102801 --label R2 ^
        --ranges "2-587,589-1000" --out-dir runs/R2_v5A_d010_100s_valseed ^
        --extra "--delta 0.10 --methods reference maxmin literature --maxmin-power-refine --maxmin-refine-solver nm"
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V5_CODE = HERE.parents[1] / "开发_10车机制定型_v5_20261101" / "code"
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # pragma: no cover
        pass


def parse_ranges(spec):
    out = []
    for piece in spec.split(","):
        piece = piece.strip()
        if not piece:
            continue
        a, b = piece.split("-")
        out.append((int(a), int(b)))
    return out


def base_cmd(python, seed, extra):
    cmd = [python, "-B", "-I", str(V5_CODE / "run_allocation.py"),
           "--n", "10", "--phase", "evaluate", "--split", "development",
           "--draws", "32",
           "--channel-seed", str(seed),
           "--forecast-seed", "2026102501", "--learning-seed", "2026102601",
           "--csi", "estimated", "--forecast-information", "trajectory",
           "--power-solver", "continuous", "--reference-search", "joint",
           "--joint-rounds", "8",
           "--qos-metric", "chance", "--qos-chance-inflation", "1.35",
           "--qos-fallback-mean", "--qos-chance-rounds", "2",
           "--threads", "2"]
    return cmd + extra.split()


def merge_csvs(chunks, out_path):
    header = None
    rows = 0
    with open(out_path, "w", encoding="utf-8", newline="") as dst:
        for chunk in chunks:
            path = os.path.join(chunk, "results.csv")
            if not os.path.isfile(path):
                continue
            with open(path, encoding="utf-8-sig", newline="") as src:
                first = src.readline()
                if header is None:
                    header = first
                    dst.write(header)
                for line in src:
                    if line.strip():
                        dst.write(line)
                        rows += 1
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="分块运行并合并")
    ap.add_argument("--channel-seed", type=int, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--ranges", required=True, help='例如 "2-587,589-1000"')
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--extra", default="--delta 0.10 --methods reference maxmin literature "
                                      "--maxmin-power-refine --maxmin-refine-solver nm")
    ap.add_argument("--python", default=r"D:\anaconda\envs\pymarl\python.exe")
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "chunk_logs"
    log_dir.mkdir(exist_ok=True)

    ranges = parse_ranges(args.ranges)
    chunks, status = [], []
    t_all = time.perf_counter()
    for a, b in ranges:
        nblocks = b - a + 1
        chunk_dir = out_dir / f"chunk_{a}_{b}"
        cmd = base_cmd(args.python, args.channel_seed, args.extra) + \
              ["--start-block", str(a), "--blocks", str(nblocks),
               "--out", str(chunk_dir)]
        log_path = log_dir / f"chunk_{a}_{b}.log"
        print(f"[{time.strftime('%H:%M:%S')}] 段 {a}-{b}（{nblocks} 块）→ {chunk_dir.name}")
        t0 = time.perf_counter()
        with open(log_path, "w", encoding="utf-8", errors="replace") as log:
            proc = subprocess.run(cmd, cwd=str(V5_CODE), stdout=log,
                                  stderr=subprocess.STDOUT)
        dt = time.perf_counter() - t0
        ok = proc.returncode == 0 and (chunk_dir / "results.csv").is_file()
        status.append({"range": f"{a}-{b}", "blocks": nblocks, "returncode": proc.returncode,
                       "ok": ok, "elapsed_s": round(dt, 1),
                       "log": str(log_path)})
        print(f"    rc={proc.returncode} ok={ok} 耗时 {dt/60:.1f} min")
        if ok:
            chunks.append(str(chunk_dir))
        else:
            tail = ""
            try:
                tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-6:])
            except Exception:
                pass
            print(f"    ⚠ 该段失败，最后几行:\n{tail}")

    merged = out_dir / "results.csv"
    rows = merge_csvs(chunks, merged) if chunks else 0
    meta = {"label": args.label, "channel_seed": args.channel_seed,
            "ranges": args.ranges, "segments": status,
            "merged_csv": str(merged), "merged_rows": rows,
            "total_elapsed_s": round(time.perf_counter() - t_all, 1),
            "note": "分块运行以规避 run_allocation.py:649 在 reference.eta_ref=None 时的崩溃；"
                    "未修改源码；失败段/块在此单列。"}
    (out_dir / "chunk_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                             encoding="utf-8")
    print(f"\n合并 {len(chunks)}/{len(ranges)} 段，{rows} 行 → {merged}")
    print(f"元数据 {out_dir / 'chunk_meta.json'}")
    return 0 if len(chunks) == len(ranges) else 2


if __name__ == "__main__":
    sys.exit(main())
