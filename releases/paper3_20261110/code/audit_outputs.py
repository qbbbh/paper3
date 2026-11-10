"""Independently recompute saved validation CSV from its fine-sample NPZ."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def audit(directory):
    checked=0;blocks=0
    for path in sorted(Path(directory).glob('n*_*.csv')):
        with path.open(encoding='utf-8-sig',newline='') as f:
            rows=list(csv.DictReader(f))
        with np.load(path.with_name(path.stem+'_fine.npz'),allow_pickle=False) as fine:
            for b in sorted({int(r['block']) for r in rows}):
                subset=[r for r in rows if int(r['block'])==b]
                n=int(subset[0]['n'])
                assert len(subset)==n//2
                ids=[int(r[key]) for r in subset for key in ('tx','rx')]
                assert sorted(ids)==list(range(n))
                rate=fine[f'b{b}_rate_bps'];power=fine[f'b{b}_powers_w']
                assert len(rate)==100
                actual=np.array([float(r['actual_rate_bps']) for r in subset])
                np.testing.assert_allclose(actual,rate.mean(0),rtol=1e-12)
                np.testing.assert_allclose([float(r['delivered_bits']) for r in subset],rate.sum(0)*.001,rtol=1e-12)
                np.testing.assert_allclose([float(r['energy_j']) for r in subset],power*.1,rtol=1e-12)
                np.testing.assert_allclose([float(r['network_ee_bit_per_j']) for r in subset],rate.sum()*.001/(power.sum()*.1),rtol=1e-12)
                np.testing.assert_allclose(fine[f'b{b}_sinr'],fine[f'b{b}_signal']/(fine[f'b{b}_interference']+fine[f'b{b}_noise']),rtol=1e-12)
                assert len({r['action_sha256'] for r in subset})==1
                assert all(float(r['power_w'])>=.001 and float(r['duration_s'])==.1 for r in subset)
                blocks+=1;checked+=len(subset)
    return dict(blocks=blocks,link_blocks=checked,checks_passed=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('directory',type=Path)
    args=ap.parse_args()
    print(json.dumps(audit(args.directory)))
