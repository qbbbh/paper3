"""Reproducible short independent train/validation/test acceptance (not convergence)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
import time


def run_suite(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("suite output must be fresh")
    args.out.mkdir(parents=True, exist_ok=True)
    entry = Path(__file__).with_name("run_allocation.py")

    def scale(n):
        outputs = []
        for index, split in enumerate(("train", "validation", "test")):
            out = args.out / f"{split}_n{n}"
            command = [sys.executable, "-B", "-I", str(entry), "--n", str(n), "--delta", str(args.delta),
                       "--dataset-manifest", str(args.dataset_manifest.resolve()), "--split", split,
                       "--phase", "train" if split == "train" else "evaluate", "--blocks",
                       str(args.train_blocks if split == "train" else args.eval_blocks),
                       "--draws", str(args.draws), "--channel-seed", str(2026102401+index),
                       "--forecast-seed", str(2026102500+1000*index), "--out", str(out)]
            if args.encoder:
                command.extend(["--encoder", args.encoder])
            command.extend(["--training-labels", args.training_labels])
            if args.credit:
                # Off by default: full enumeration is expensive at n=100.
                command.extend(["--credit", "--credit-max-evals", str(args.credit_max_evals)])
            if split != "train":
                command.extend(["--checkpoint", str(args.out / f"train_n{n}" / "model.pt")])
            started = time.perf_counter()
            with (args.out / f"{split}_n{n}.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace")
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line.rstrip(), flush=True)
                code = process.wait()
            if code:
                raise RuntimeError(f"n{n} {split} failed; see saved log")
            outputs.append(dict(n=n, split=split, directory=out.name, elapsed_s=time.perf_counter()-started,
                                returncode=code))
        return outputs

    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [result for group in pool.map(scale, (50, 100)) for result in group]
    audit_command = [sys.executable, "-B", "-I", str(Path(__file__).with_name("audit_allocation_outputs.py")),
                     *[str(args.out / item["directory"]) for item in runs]]
    audit = subprocess.run(audit_command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    (args.out / "audit.log").write_text(audit.stdout+audit.stderr, encoding="utf-8")
    report = dict(scope="short development acceptance, not training convergence", delta=args.delta,
                  runs=runs, audit_returncode=audit.returncode,
                  audit=json.loads(audit.stdout) if audit.returncode == 0 else None)
    (args.out / "suite_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit.returncode:
        raise RuntimeError("independent output audit failed; see audit.log")
    print(f"Completed and independently audited {len(runs)} runs", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--train-blocks", type=int, default=2)
    parser.add_argument("--eval-blocks", type=int, default=1)
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--encoder", choices=("mlp", "hetero_graph", "hetero_graph_v2"),
                        help="default: per-scale automatic (100 -> hetero_graph_v2)")
    parser.add_argument("--credit", action="store_true",
                        help="also write the forecast-only credit diagnostic (expensive at n=100)")
    parser.add_argument("--credit-max-evals", type=int, default=32)
    parser.add_argument("--training-labels", choices=("proposal", "executed"), default="proposal")
    run_suite(parser.parse_args())
