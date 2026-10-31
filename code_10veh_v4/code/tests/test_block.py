import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pytest
from p3base.config import FoundationConfig
from p3base import block, link_eval, contract


def fixture():
    cfg = FoundationConfig()
    links = [(0, 1, 'RF'), (2, 3, 'RF'), (4, 5, 'VLC')]
    p = np.array([.1, .2, .3])
    ports = dict(tx=np.zeros((6, 6), int), rx=np.zeros((6, 6), int))
    return cfg, links, p, ports


def test_vector_physics_matches_scalar():
    cfg, links, p, ports = fixture()
    rng = np.random.default_rng(33)
    for modes in [('RF', 'RF', 'VLC'), ('VLC', 'VLC', 'RF'), ('VLC',)*3, ('RF',)*3]:
        edges = [(i, j, mode) for (i, j, _), mode in zip(links, modes)]
        rf = rng.uniform(0, 1e-9, (4, 6, 6))
        vlc = rng.uniform(0, 1e-5, (4, 6, 6, 2, 2))
        ports = dict(tx=rng.integers(0, 2, (6, 6)), rx=rng.integers(0, 2, (6, 6)))
        for isi in (True, False):
            actual = block.sample_metrics(edges, p, ports, rf, vlc, cfg, isi)
            for s in range(4):
                ref = link_eval.evaluate(edges, p, ports, rf[s], vlc[s], cfg, isi)
                for key in ('sinr', 'signal', 'interference', 'noise'):
                    np.testing.assert_allclose(actual[key][s], ref[key], rtol=1e-12, atol=0)
                np.testing.assert_allclose(actual['rate_bps'][s], ref['link_rates_bps'], rtol=1e-12)


def test_two_manual_blocks_not_catchup():
    cfg, links, p, ports = fixture()
    block.validate_action(links, p, ports, 6, cfg)
    # Block 1 has a weak pair; block 2 cannot retrospectively repair it.
    a = block.summarize_block(links, p, dict(rate_bps=np.tile([1e6, 2e6, 3e6], (100, 1))), .001, 6, 1.5e6)
    b = block.summarize_block(links, p, dict(rate_bps=np.tile([9e6, 8e6, 7e6], (100, 1))), .001, 6, 1.5e6)
    assert not a['all_pairs_qos_met'] and b['all_pairs_qos_met']
    assert a['link_delivered_bits'].sum() == 600000
    np.testing.assert_allclose(a['link_energy_j'].sum(), .06)
    np.testing.assert_allclose(a['network_ee_bit_per_j'], 10e6)
    assert a['participation_rate_bps'].sum() == 2 * a['mean_rate_bps'].sum()
    np.testing.assert_allclose(a['pair_ee_bit_per_j'], [10e6]*3)


def test_mean_pair_ee_is_the_average_of_link_ees_not_the_network_ratio():
    # Different powers make the two definitions numerically different, so a
    # regression back to sum(rate)/sum(power) cannot pass this test.
    cfg, links, _, ports = fixture()
    p = np.array([.1, .2, .3])
    rates = np.tile([1e6, 1e6, 1e6], (100, 1))
    out = block.summarize_block(links, p, dict(rate_bps=rates), .001, 6)
    link_ee = rates[0] / p
    np.testing.assert_allclose(out['pair_ee_bit_per_j'], link_ee)
    assert out['mean_pair_ee_bit_per_j'] == pytest.approx(link_ee.mean())
    assert out['min_pair_ee_bit_per_j'] == pytest.approx(link_ee.min())
    assert out['network_ee_bit_per_j'] == pytest.approx(rates[0].sum() / p.sum())
    assert out['mean_pair_ee_bit_per_j'] != pytest.approx(out['network_ee_bit_per_j'])


def test_invalid_plans_are_rejected():
    cfg, links, p, ports = fixture()
    for edges, powers in [(links[:-1], p), (links, p[:-1]), (links, [0, .2, .3]),
                          ([(0, 1, 'BAD'), *links[1:]], p), ([(0, 1, 'RF'), (1, 3, 'RF'), links[2]], p)]:
        with pytest.raises((ValueError, AssertionError)):
            block.validate_action(edges, powers, ports, 6, cfg)
        with pytest.raises((ValueError, AssertionError)):
            contract.check_block_plan(edges, powers, 6, cfg)


def test_rate_mean_is_not_rate_of_mean_sinr():
    cfg, links, p, ports = fixture()
    links = [(i, j, 'RF') for i, j, _ in links]
    g = np.zeros((100, 6, 6))
    g[:50, 0, 1] = cfg.rf.noise_w / p[0]
    g[50:, 0, 1] = 100 * cfg.rf.noise_w / p[0]
    f = block.sample_metrics(links, p, ports, g, np.zeros((100, 6, 6, 2, 2)), cfg)
    mean = block.summarize_block(links, p, f, .001, 6)['mean_rate_bps'][0]
    wrong = cfg.rf.bandwidth_hz * np.log2(1 + f['sinr'][:, 0].mean()/cfg.rf.gap)
    assert mean < wrong


def test_rf_gain_is_power_gain():
    cfg = FoundationConfig()
    g = 1e-9
    result = block.sample_metrics([(0, 1, 'RF')], [.1], dict(tx=np.zeros((2,2),int),rx=np.zeros((2,2),int)),
                                  np.array([[0,g],[g,0]]), np.zeros((2,2,2,2)), cfg)
    # v4：RF 信号是 eta_PA * p * g，p 是电输入功率。
    np.testing.assert_allclose(result['signal'], [cfg.rf.pa_efficiency * .1 * g], atol=0)
