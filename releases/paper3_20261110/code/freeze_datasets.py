"""Freeze the generated independent seeds; no implicit train/test duplication."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p3learn.data import freeze_datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traffic-root", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--validation-seed", type=int, required=True)
    parser.add_argument("--test-seed", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    partitions = {name: [args.traffic_root / f"seed{seed}" / f"n{n}" for n in (50, 100)]
                  for name, seed in (("train", args.train_seed), ("validation", args.validation_seed),
                                     ("test", args.test_seed))}
    freeze_datasets(root, partitions, args.out)
    print(f"Frozen {args.out}")
