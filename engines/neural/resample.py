#!/usr/bin/env python3
"""Resumable CPU post-processing of frozen score sequences; no training."""
import argparse,hashlib,os,signal,time,platform
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
import run,design
from bootstrap import bootstrap_test
from inference import statistic

SPEC=dict(draws=999,block_multipliers=[1,2,4],seed=2026111407)
STOP=False

def stop(*_):
    global STOP
    STOP=True

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def destination(out,path):return out/'bootstrap'/path.parent.name/path.name

def check_saved(out,path):
    dest=destination(out,path)
    if not dest.exists():return False
    x=run.read(dest)
    if x['spec']!=SPEC or x['input_sha256']!=digest(path):raise RuntimeError('Bootstrap checkpoint/input mismatch: '+str(dest))
    original=run.read(path)
    if [(r['cell_id'],r['replication']) for r in x['rows']]!=[(r['cell_id'],r['replication']) for r in original]:raise RuntimeError('Bootstrap replication mismatch')
    return True

def compute_record(r,c):
    lag=design.meta(c)['bandwidth'];extras={}
    for method in ('raw','orthogonal'):
        values=np.asarray(r['series']['scores'][method])
        for multiplier in SPEC['block_multipliers']:
            suffix={1:'_short',2:'',4:'_long'}[multiplier]
            # Same indices for both methods and matched paths across alternatives.
            seed=np.random.SeedSequence([SPEC['seed'],c['n'],r['replication'],multiplier])
            extras[method+'_bootstrap'+suffix]=bootstrap_test(values,r['theta'],lag,seed,
                draws=SPEC['draws'],block_length=multiplier*(lag+1))
        for suffix,length in (('half',max(1,lag//2)),('double',2*lag)):
            extras[method+'_hac_'+suffix]=statistic(values,r['theta'],bandwidth=length)
    return dict(cell_id=r['cell_id'],replication=r['replication'],statistics=extras)

def process_batch(args):
    out,path,c=args;start=time.monotonic();sha=digest(path)
    rows=[compute_record(r,c) for r in run.read(path)]
    run.dump(destination(out,path),dict(spec=SPEC,input_sha256=sha,seconds=time.monotonic()-start,rows=rows))
    return path.name,c['id'],len(rows)

def status(out,strict=False):
    cfg=run.configuration(out);paths=sorted((out/'draws').glob('*/*.json.gz'));done=0
    for path in paths:
        if check_saved(out,path):done+=len(run.read(destination(out,path))['rows'])
    expected=sum(cfg['counts'].values());result=dict(saved=done,expected=expected,complete=done==expected)
    run.dump(out/'bootstrap_status.json',result);print(result)
    if strict and not result['complete']:raise RuntimeError('Bootstrap incomplete; resume before final analysis')
    return result

def worker(out,workers=24,seconds=2700):
    from extend import complete
    complete(out)
    if workers<1 or workers>24:raise ValueError('Use 1 to 24 CPU workers')
    allocated=os.environ.get('SLURM_CPUS_PER_TASK')
    if allocated and workers>int(allocated):raise ValueError('Workers exceed allocated CPUs')
    lease=out/'BOOTSTRAP_RUNNING'
    try:lease.mkdir()
    except FileExistsError:raise RuntimeError('Bootstrap lease exists; check jobs before recovery')
    try:
        run.dump(lease/'owner.json',dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID')))
        import scipy,pandas
        env=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,
                 pandas=pandas.__version__,system=platform.system(),machine=platform.machine())
        execution=out/'bootstrap_execution.json'
        if execution.exists() and run.read(execution)!=env:raise RuntimeError('Bootstrap execution environment changed')
        if not execution.exists():run.dump(execution,env)
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGUSR1,stop)
        cfg=run.configuration(out);cells={c['id']:c for c in cfg['cells']}
        tasks=[(out,path,cells[int(path.parent.name.split('-')[1])]) for path in sorted((out/'draws').glob('*/*.json.gz')) if not check_saved(out,path)]
        todo=iter(tasks);start=time.monotonic()
        with ProcessPoolExecutor(max_workers=workers) as pool:
            pending=set()
            def dispatch():
                while len(pending)<workers and not STOP and not (out/'STOP').exists() and not (out.parent/'STOP').exists() and time.monotonic()-start<seconds-90:
                    task=next(todo,None)
                    if task is None:break
                    pending.add(pool.submit(process_batch,task))
            dispatch()
            while pending:
                finished,pending=wait(pending,return_when=FIRST_COMPLETED)
                for f in finished:
                    name,cell,n=f.result();print(f'Bootstrap cell={cell}, batch={name}, draws={n}',flush=True)
                dispatch()
        status(out)
    finally:
        (lease/'owner.json').unlink(missing_ok=True);lease.rmdir()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['worker','report','complete']);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=24);p.add_argument('--seconds',type=int,default=2700);a=p.parse_args()
    if a.command=='worker':worker(a.output,a.workers,a.seconds)
    else:status(a.output,a.command=='complete')
