"""Verify this release; optionally run small, portable tests without replay data."""
import argparse
import json
import sys

sys.dont_write_bytecode = True
from release_paths import CODE, RELEASE, scene_directory, verify_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tests", action="store_true",
                        help="run the selected local invariant and interface tests")
    args = parser.parse_args()
    print(json.dumps(verify_release(), ensure_ascii=True), flush=True)
    if not args.tests:
        return 0
    sys.path.insert(0, str(CODE))
    # The old geometry test expects a sibling traffic folder. Only its path
    # resolver is adapted; the scene bytes, model, fixture and test are unchanged.
    import p3base.scene
    p3base.scene.find_scene_root = scene_directory
    import pytest
    tests = ["test_guarded_refine.py", "test_bottleneck_gradients.py",
             "test_bottleneck_power.py", "test_replay_saved.py",
             "test_block.py", "test_geometry_snapshot.py"]
    result = pytest.main(["-q", "-p", "no:cacheprovider",
                          *[str(CODE / "tests" / name) for name in tests]])
    verify_release()
    return int(result)


if __name__ == "__main__":
    raise SystemExit(main())
