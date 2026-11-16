"""Record one frozen R1 boundary failure; optionally reuse the verified 50A final handoff fix."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='1'
import argparse,inspect,json,traceback,difflib
from pathlib import Path
import numpy as np
import run_delay_diagnostic as d
from p3base import block
from p3learn import allocation

HERE=Path(__file__).resolve().parent
EVENTS=[]
ORIGINAL_VALIDATE=block.validate_action

def offenses(links,powers,cfg):
    lo=np.array([getattr(cfg,m.lower()).power_min_w for _,_,m in links]);hi=np.array([getattr(cfg,m.lower()).power_max_w for _,_,m in links])
    p=np.asarray(powers);idx=np.flatnonzero((p<lo)|(p>hi)|~np.isfinite(p))
    return [{'index':int(j),'link':list(links[j]),'power_w':float(p[j]),'power_repr':repr(float(p[j])),
        'power_hex':float(p[j]).hex(),'lower_w':float(lo[j]),'upper_w':float(hi[j]),
        'below_lower_w':float(max(0.,lo[j]-p[j])),'above_upper_w':float(max(0.,p[j]-hi[j]))} for j in idx]

def validate(links,powers,ports,n,cfg,*args,**kwargs):
    bad=offenses(links,powers,cfg)
    if bad:EVENTS.append({'kind':'strict_validation_failure','offenses':bad,'stack':traceback.format_stack(limit=12)})
    return ORIGINAL_VALIDATE(links,powers,ports,n,cfg,*args,**kwargs)

def project(links,powers,cfg):
    # Same narrow interface and <=1e-12 W condition as the verified 50A correction.
    bad=offenses(links,powers,cfg)
    if not bad:return powers
    lo=np.array([getattr(cfg,m.lower()).power_min_w for _,_,m in links]);hi=np.array([getattr(cfg,m.lower()).power_max_w for _,_,m in links])
    p=np.asarray(powers)
    if not (np.isfinite(p).all() and np.all(p>=lo-1e-12) and np.all(p<=hi+1e-12)):
        raise ValueError('Final power boundary violation exceeds 1e-12 W tolerance')
    EVENTS.append({'kind':'final_refinement_projection','offenses':bad,'stack':traceback.format_stack(limit=8)})
    return np.clip(p,lo,hi)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repair',action='store_true');args=ap.parse_args()
    dest=HERE/'boundary_261'/('repaired' if args.repair else 'original');dest.mkdir(parents=True,exist_ok=True)
    d.RUNS=dest;block.validate_action=validate;allocation.validate_action=validate
    if args.repair:
        prior=json.loads((HERE/'boundary_261/original/diagnostic.json').read_text(encoding='utf-8'))
        assert prior['same_final_handoff'] and prior['within_solver_tolerance'] and prior['original_failure_reproduced']
        src=inspect.getsource(allocation.pure_ee_reference)
        old='''                refined = Action(list(best[3].links),
                                 _power_solve(list(best[3].links), belief_gains, checked_ports,
                                              cfg, power_solver, solver_log=solver_log, q_min_bps=q_min_bps),
                                 best[3].ports)'''
        new='''                refined_powers = _power_solve(list(best[3].links), belief_gains, checked_ports,
                                              cfg, power_solver, solver_log=solver_log, q_min_bps=q_min_bps)
                refined_powers = _strict_final_power_box(best[3].links, refined_powers, cfg)
                refined = Action(list(best[3].links), refined_powers, best[3].ports)'''
        assert src.count(old)==1
        patched=src.replace(old,new);allocation.__dict__['_strict_final_power_box']=project
        (dest/'isolated_final_handoff.patch').write_text(''.join(difflib.unified_diff(src.splitlines(True),patched.splitlines(True),fromfile='frozen',tofile='same_50A_boundary_fix')),encoding='utf-8')
        exec(compile(patched,str(dest/'isolated_final_handoff.py'),'exec'),allocation.__dict__)
        d.pure_ee_reference=allocation.pure_ee_reference
    case=d.rs.SavedCase('10B');case.summary.pop('blocks',None)
    r=d.calculate(case,261,5)
    stack=''.join(''.join(e.get('stack',[])) for e in EVENTS)
    report={'repair':args.repair,'events':EVENTS,'success':r['success'],'reason':r.get('reason'),
        'same_final_handoff':(('pure_ee_reference' in stack and 'refined_forecast = _forecast(evaluator, refined)' in stack)
            or (args.repair and any(e['kind']=='final_refinement_projection' for e in EVENTS))),
        'within_solver_tolerance':all(max(o['below_lower_w'],o['above_upper_w'])<=1e-12 for e in EVENTS for o in e['offenses']),
        'original_failure_reproduced':not r['success'] and 'reference_power_search_failure' in r.get('reason','')}
    (dest/'diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
