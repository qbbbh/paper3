"""Independent read-only audit of committed replay arrays; imports no solver.

Recompute rates from saved signal/interference/noise, EE from rate and electric
input plus circuits, calibrated service from prediction samples, and compare
old outputs against their original saved files. No optimization is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024*1024), b''):
            digest.update(part)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def compare(a, b, label, errors):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError(label + ': shape or nonfinite value')
    relative = float(np.max(np.abs(a-b)/np.maximum(np.abs(b), 1e-12), initial=0.0))
    errors[label] = max(errors.get(label, 0.0), relative)
    if not np.allclose(a, b, rtol=1e-10, atol=1e-7):
        raise ValueError(f'{label}: mismatch, maximum relative error {relative}')


def action_equal(tx, rx, mode, p, tp, rp, other):
    def entries(atx, arx, amode, ap, atp, arp):
        return sorted((int(i), int(j), str(m), float(power),
                       int(atp[i, j]) if m == 'VLC' else 0,
                       int(arp[i, j]) if m == 'VLC' else 0)
                      for i, j, m, power in zip(atx, arx, amode, ap))
    # Original storage holds selected ports as vectors rather than matrices.
    otx, orx, omode, op, otp, orp = other
    selected = sorted((int(i), int(j), str(m), float(power),
                       int(t) if m == 'VLC' else 0, int(r) if m == 'VLC' else 0)
                      for i, j, m, power, t, r in zip(otx, orx, omode, op, otp, orp))
    return entries(tx, rx, mode, p, tp, rp) == selected


def audit_block(folder, manifest, source, source_records, errors):
    commit = read_json(folder/'commit.json')
    for name, expected in commit.items():
        if sha(folder/name) != expected:
            raise ValueError('new output checksum mismatch: '+name)
    rec = read_json(folder/'record.json')
    k, method = rec['block'], manifest['method']
    original = source_records[(k, method)]
    if rec['status'] == 'original_failure_preserved':
        if original['success']:
            raise ValueError('success converted to original failure')
        return dict(group=rec['group'], block=k, failed=True)
    if rec['status'] != 'evaluated' or not original['success']:
        raise ValueError('expected an evaluated old-success block')
    config = next(x['config'] for x in manifest['inputs'] if x['group'] == rec['group'])
    physical = config['physical']
    old_folder = source/'block_checkpoints'/f'block_{k:06d}'
    old_commit = read_json(old_folder/'commit.json')
    if sha(old_folder/'fine_samples.npz') != old_commit['sha256']['fine_samples.npz']:
        raise ValueError('original array checksum mismatch')
    with np.load(folder/'arrays.npz', allow_pickle=False) as saved, np.load(
            old_folder/'fine_samples.npz', allow_pickle=False) as before:
        tx, rx, modes = [saved[x] for x in ('tx', 'rx', 'mode')]
        n = len(tx)*2
        if sorted(np.r_[tx, rx].tolist()) != list(range(n)):
            raise ValueError('not a complete one-role pairing')
        for x in ('tx', 'rx', 'mode'):
            if not np.array_equal(saved[x], before[f'{method}_b{k}_{x}']):
                raise ValueError('old topology or mode changed')
        if not set(modes).issubset({'RF', 'VLC'}):
            raise ValueError('unknown communication mode')
        old_p, new_p = saved['old_power_w'], saved['new_power_w']
        lo = np.array([physical[m.lower()]['power_min_w'] for m in modes])
        hi = np.array([physical[m.lower()]['power_max_w'] for m in modes])
        pc = np.array([physical[m.lower()]['circuit_power_w'] for m in modes])
        for p in (old_p, new_p):
            if not np.isfinite(p).all() or np.any(p < lo) or np.any(p > hi):
                raise ValueError('power outside original physical range')
        if not np.array_equal(old_p, before[f'{method}_b{k}_power_w']):
            raise ValueError('old input powers differ')
        for name, old_name in [('reference', 'reference'), ('old', method)]:
            compare(saved[name+'_samples_bps'], before[f'{old_name}_b{k}_forecast_samples_bps'],
                    name+'_saved_predictions', errors)
            compare(saved[name+'_rate_bps'], before[f'{old_name}_b{k}_rate_bps'],
                    name+'_saved_true_rates', errors)
        reference_action = tuple(before[f'reference_b{k}_'+f] for f in
                                 ('tx', 'rx', 'mode', 'power_w', 'tx_port', 'rx_port'))
        for name, p in [('old', old_p), ('new', new_p)]:
            if action_equal(tx, rx, modes, p, saved['tx_port'], saved['rx_port'],
                            reference_action) != rec[name+'_returned_initial_action']:
                raise ValueError(name+' initial-action identity flag mismatch')
        if rec['original_initial_mean_fallback_flag'] != original['chance_fallback']:
            raise ValueError('initial mean fallback history changed')
        result = dict(group=rec['group'], block=k, failed=False, actual={}, predicted={})
        for name, p, mode in [('old', old_p, modes), ('new', new_p, modes),
                             ('reference', reference_action[3], reference_action[2])]:
            overhead = np.array([physical[m.lower()]['circuit_power_w'] for m in mode])
            bw = np.array([physical[m.lower()]['bandwidth_hz'] for m in mode])
            gap = np.array([10**(physical[m.lower()]['gap_db']/10) for m in mode])
            s, i, noise = [saved[name+'_'+x] for x in ('signal', 'interference', 'noise')]
            sinr = s/(i+noise)
            compare(sinr, saved[name+'_sinr'], name+'_sinr', errors)
            rates = bw*np.log1p(sinr/gap)/np.log(2.)
            compare(rates, saved[name+'_rate_bps'], name+'_rates_from_sinr', errors)
            rate = rates.mean(axis=0)
            ee = rate/(p+overhead)
            actual = rec['actual'][name]
            compare(ee, actual['pair_ee_bit_per_j'], name+'_individual_ee', errors)
            compare(ee.mean(), actual['mean_pair_ee_bit_per_j'], name+'_mean_ee', errors)
            compare(ee.min(), actual['min_pair_ee_bit_per_j'], name+'_min_ee', errors)
            compare((p+overhead).sum(), actual['total_electrical_power_w'], name+'_total_cost', errors)
            passed = rate >= config['q_eval_bps']
            if not np.array_equal(passed, actual['pair_qos_met']):
                raise ValueError(name+' actual link service flags mismatch')
            if bool(passed.all()) != actual['all_pairs_qos_met']:
                raise ValueError(name+' actual cycle service flag mismatch')
            result['actual'][name] = dict(mean=float(ee.mean()), minimum=float(ee.min()),
                                          all_pass=bool(passed.all()), bad_links=int((~passed).sum()))
        for name, p in [('old', old_p), ('new', new_p), ('reference', reference_action[3])]:
            smp = saved[name+'_samples_bps']
            if smp.shape != (config['draws'], len(p)):
                raise ValueError('conditional sample count changed')
            mean = smp.mean(axis=0)
            lower = np.quantile(smp, .05, axis=0)
            if config['qos_metric'] == 'chance':
                safe_mean = np.maximum(mean, 1e-9)
                service = safe_mean*(np.maximum(lower, 1e-9)/safe_mean)**config['qos_chance_inflation']
            else:
                service = mean
            pred_mode = reference_action[2] if name == 'reference' else modes
            costs = p+np.array([physical[m.lower()]['circuit_power_w'] for m in pred_mode])
            pee = mean/costs
            reference = rec['reference_prediction']['mean_ee']
            budget = (1-config['delta'])*reference
            reported = rec[name+'_prediction']
            compare(pee.mean(), reported['mean_ee'], name+'_predicted_mean_ee', errors)
            compare(pee.min(), reported['min_ee'], name+'_predicted_min_ee', errors)
            compare(service.min(), reported['quantile_min_rate'], name+'_calibrated_quantile', errors)
            feasible = bool(pee.mean() >= budget-1e-7 and np.all(service >= config['q_min_bps']-1e-7))
            result['predicted'][name] = dict(mean=float(pee.mean()), minimum=float(pee.min()),
                                             feasible=feasible)
        unchanged = np.array_equal(old_p, new_p)
        if unchanged != rec['unchanged']:
            raise ValueError('unchanged flag mismatch')
        if not unchanged and not result['predicted']['new']['feasible']:
            raise ValueError('new changed action violates original constraints')
        if result['predicted']['new']['minimum'] < result['predicted']['old']['minimum']-1e-6:
            raise ValueError('predicted minimum EE decreased')
        compare(rec['original_complete_decision_s'], original['solve_time_s'], 'old_timing', errors)
        compare(rec['estimated_complete_decision_s'], original['solve_time_s']+rec['refine_time_s'],
                'estimated_timing', errors)
        result['unchanged'] = bool(unchanged)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    errors, results, failures, missing, seen = {}, [], [], [], set()
    originals = {}
    methods = set()
    for run in args.runs:
        manifest = read_json(run/'manifest.json')
        methods.add(manifest['method'])
        if len(methods) > 1:
            raise ValueError('audit one method at a time; do not pool distinct ablations')
        code = run.parent.parent/'code'
        for name, expected in manifest['source_hashes'].items():
            if sha(code/name) != expected:
                raise ValueError('frozen replay source changed: '+name)
        for group in manifest['groups']:
            info = next(x for x in manifest['inputs'] if x['group'] == group)
            source = Path(info['source'])
            if source not in originals:
                if sha(source/'summary.json') != info['summary_sha256']:
                    raise ValueError('original summary changed')
                data = read_json(source/'summary.json')
                originals[source] = (data['config'], {(int(r['block']), r['method']):r for r in data['blocks']})
            if info['config'] != originals[source][0]:
                raise ValueError('replay physical or evaluation configuration changed')
            for k in manifest['blocks']:
                identity = (group, k, manifest['method'])
                if identity in seen:
                    raise ValueError('duplicate block across shards: '+str(identity))
                seen.add(identity)
                folder = run/group/f'block_{k:06d}'
                if not (folder/'commit.json').exists():
                    missing.append(dict(group=group, block=k, run=str(run)))
                    continue
                try:
                    results.append(audit_block(folder, manifest, source, originals[source][1], errors))
                except Exception as exc:
                    failures.append(dict(group=group, block=k, error=f'{type(exc).__name__}: {exc}'))
    groups = {}
    for group in sorted({r['group'] for r in results}):
        rr = [r for r in results if r['group']==group]
        valid = [r for r in rr if not r['failed']]
        groups[group] = dict(committed_blocks=len(rr), original_failed=sum(r['failed'] for r in rr),
                             evaluated_blocks=len(valid))
        if valid:
            for name in ('old','new','reference'):
                groups[group][name] = dict(
                    mean_EE_MbitJ=float(np.mean([r['actual'][name]['mean'] for r in valid]))/1e6,
                    min_EE_MbitJ=float(np.mean([r['actual'][name]['minimum'] for r in valid]))/1e6,
                    all_link_service_pass=sum(r['actual'][name]['all_pass'] for r in valid),
                    violating_links=sum(r['actual'][name]['bad_links'] for r in valid))
    report = dict(status='passed' if not failures and (not missing or args.allow_partial) else 'failed',
                  partial=bool(missing), planned_blocks=len(seen), verified_blocks=len(results),
                  missing_count=len(missing), failures=failures, maximum_relative_errors=errors,
                  groups=groups, inputs=[str(p) for p in args.runs])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('maximum_relative_errors','inputs')}, ensure_ascii=True))
    return 0 if report['status']=='passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
