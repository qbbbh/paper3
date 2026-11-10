"""Preview or explicitly run a frozen numerical case using repository data.

This is a path/configuration adapter. All scientific work is performed by the
unchanged code/run_allocation.py. The default is preview, not simulation.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
from release_paths import CODE, RELEASE, load_json, scene_directory, verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("10A", "10B", "50B"), required=True)
    parser.add_argument("--out", type=Path, help="fresh output directory; required with --execute")
    parser.add_argument("--blocks", type=int,
                        help="optional prefix length for a smoke run; does not alter solver budgets")
    parser.add_argument("--execute", action="store_true", help="actually run; may take many hours")
    task = parser.parse_args()
    verify_release()
    settings = load_json(RELEASE / "configs" / f"{task.case}.json")
    cfg = settings["saved_config"]
    if task.execute and task.out is None:
        parser.error("--execute requires an explicit fresh --out directory")
    if task.blocks is not None and not 1 <= task.blocks <= cfg["blocks"]:
        parser.error(f"--blocks must be between 1 and {cfg['blocks']}")
    sys.path.insert(0, str(CODE))
    import run_allocation as runner
    output = task.out or RELEASE / ".runtime" / f"{task.case}_preview_only"
    options = runner.parser().parse_args([
        "--n", str(settings["vehicles"]), "--delta", str(cfg["delta"]),
        "--out", str(output)])
    for action in runner.parser()._actions:
        if action.dest in cfg:
            setattr(options, action.dest, cfg[action.dest])
    options.maxmin_refine_solver = settings["release_override"]["maxmin_refine_solver"]
    options.vlc_sinr = cfg["physical"]["vlc"]["sinr_model"]
    options.vlc_noise = cfg["physical"]["vlc"]["noise_model"]
    options.blocks = task.blocks if task.blocks is not None else cfg["blocks"]
    if options.phase != "evaluate" or "policy" in options.methods:
        raise ValueError("this release entry is numerical evaluation only")
    physical = runner.FoundationConfig()
    physical = physical.with_(vlc=replace(physical.vlc, sinr_model=options.vlc_sinr,
                                         noise_model=options.vlc_noise))
    if options.estimator == "complex_ar1":
        physical = physical.with_(estimation=replace(physical.estimation, rf_state="complex_ar1"))
    if physical.to_dict() != cfg["physical"]:
        raise ValueError("entry defaults do not reproduce the saved physical configuration")
    directory = scene_directory(None, options.n)
    runner.find_scene_root = scene_directory
    preview = dict(case=task.case, execute=task.execute, traffic=str(directory),
                   solver=options.maxmin_refine_solver, options=vars(options),
                   note="Same saved-scene inputs; not a new independent seed or fresh replay result.")
    print(json.dumps(preview, ensure_ascii=True, indent=2, default=str), flush=True)
    if not task.execute:
        return 0
    try:
        runner.run(options)
    except BaseException as error:
        runner.record_failure(options.out, error)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
