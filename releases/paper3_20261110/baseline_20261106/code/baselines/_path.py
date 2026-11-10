"""把冻结主算法源码加入 sys.path，本目录只放基线适配，不改动冻结树。"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
FROZEN_CODE = PKG.resolve()


def ensure_frozen_on_path() -> Path:
    path = str(FROZEN_CODE)
    if path not in sys.path:
        sys.path.insert(0, path)
    if not FROZEN_CODE.is_dir():
        raise FileNotFoundError(f"frozen code missing: {FROZEN_CODE}")
    return FROZEN_CODE


METHOD_ABOAGYE = "aboagye_matching_qt"
METHOD_ZAPPONE = "zappone_mmee_fp_fixed"
