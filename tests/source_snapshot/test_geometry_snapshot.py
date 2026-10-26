import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from p3base import channel
from p3base.config import FoundationConfig
from p3base.scene import Scene,find_scene_root


def test_pruning_matches_original_snapshots():
    root=Path(__file__).resolve().parents[2]
    with np.load(root/'reference/geometry_expected.npz',allow_pickle=False) as z:
        for n in (10,50,100):
            scene=Scene(find_scene_root(root,n));cfg=FoundationConfig()
            actual=channel.vlc_geometry(scene,scene.ports(state=scene.at(.3)),cfg.vlc)
            for key in ('gain','candidate','d_port_m'):
                np.testing.assert_array_equal(actual[key],z[f'n{n}_{key}'])
