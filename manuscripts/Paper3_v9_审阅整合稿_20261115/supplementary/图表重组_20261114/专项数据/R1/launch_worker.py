"""Hidden, duplicate-protected R1 worker launch after input/source checks."""
from pathlib import Path
import os,sys,json,hashlib,subprocess,argparse
import psutil
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CODE=ROOT/'代码/公平功率求解改进_20261110/code'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--shard',type=int,required=True);parser.add_argument('--workers',type=int,default=2)
    args=parser.parse_args()
    manifest=json.loads((HERE/'prepare_manifest.json').read_text(encoding='utf-8'))
    assert manifest['prepared_through_block']==999,'Causal input preparation not finished'
    assert manifest['script_sha256']==sha(HERE/'run_delay_diagnostic.py'),'Driver differs from prepared-input manifest'
    for name,digest in manifest['formal_code_hashes'].items():assert sha(CODE/name)==digest,('Formal source changed',name)
    for k in manifest['blocks']:
        p=HERE/'inputs'/f'block_{k:06d}.pkl.gz'
        c=json.loads(p.with_suffix('.commit.json').read_text(encoding='utf-8'))
        assert sha(p)==c['sha256']
    for proc in psutil.process_iter(['pid','cmdline']):
        cmd=proc.info['cmdline'] or []
        if any('run_delay_diagnostic.py' in c for c in cmd) and 'worker' in cmd:
            idx=cmd.index('--shard') if '--shard' in cmd else -1
            if idx<0 or cmd[idx+1]==str(args.shard):raise SystemExit('Matching worker already running: '+str(proc.pid))
    free=psutil.virtual_memory().available/2**30
    if free<1.5:raise SystemExit('Wait for physical-memory headroom; available GiB='+str(free))
    log=(HERE/f'worker_{args.shard}_stdout.log').open('a',encoding='utf-8')
    err=(HERE/f'worker_{args.shard}_stderr.log').open('a',encoding='utf-8')
    env=os.environ.copy();env['PYTHONIOENCODING']='utf-8'
    proc=subprocess.Popen([sys.executable,'-u',str(HERE/'run_delay_diagnostic.py'),'worker','--shard',str(args.shard),'--workers',str(args.workers)],
        cwd=str(HERE),stdout=log,stderr=err,env=env,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
    record={'pid':proc.pid,'shard':args.shard,'workers':args.workers,'available_gib_at_launch':free,'driver_sha256':sha(HERE/'run_delay_diagnostic.py')}
    (HERE/f'worker_{args.shard}_pid.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
