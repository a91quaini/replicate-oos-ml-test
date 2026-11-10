#!/usr/bin/env python3
"""Immutable, resumable GPU experiment and timing-only allocation of replication counts."""
import argparse, fcntl, gzip, hashlib, json, math, os, platform, signal, sys, time, traceback, uuid
from pathlib import Path
import numpy as np
import pandas as pd
import design

HERE=Path(__file__).resolve().parent
STOP=False

def stop(*_):
    global STOP
    STOP=True

def read(path):
    p=Path(path); b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)

def dump(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    b=json.dumps(obj,allow_nan=False,indent=2).encode()
    tmp=p.with_name(p.name+f'.{uuid.uuid4().hex}.tmp')
    tmp.write_bytes(gzip.compress(b) if p.suffix=='.gz' else b);tmp.replace(p)

def sources():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.iterdir())
            if p.suffix in ('.py','.md','.sh','.sbatch','.txt')}

def configuration(out):
    c=read(out/'configuration.json')
    if c['sources']!=sources():raise RuntimeError('Use the frozen source/run.py in this output directory')
    return c

def prepare(out,profile,null_reps=512,power_reps=256,cells=None):
    if min(null_reps,power_reps)<1:raise ValueError('Positive counts required')
    cells=design.grid(profile=='pilot') if cells is None else cells
    cfg=dict(schema=1,profile=profile,seed=design.SEED,batch=design.BATCH,sources=sources(),cells=cells,
             counts={str(c['id']):(2*design.BATCH if profile=='pilot' else (null_reps if c['delta']==0 else power_reps)) for c in cells})
    out.mkdir(parents=True,exist_ok=True)
    if (out/'configuration.json').exists():
        if read(out/'configuration.json')!=cfg:raise RuntimeError('Existing configuration differs; use a NEW output')
        # Never overwrite a frozen source, even during an idempotent prepare.
        for name,digest in cfg['sources'].items():
            if hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError('Damaged frozen source: '+name)
    else:
        for name in cfg['sources']:
            dest=out/'source'/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes((HERE/name).read_bytes())
        dump(out/'configuration.json',cfg)
    pd.DataFrame(cells).to_csv(out/'design.csv',index=False)
    print(f'{profile}: {len(cells)} cells, {sum(cfg["counts"].values())} replications; {out}')
    return cfg

def tasks(cfg):
    cells=sorted(cfg['cells'],key=lambda c:(c['T'],c['delta']==0),reverse=True)
    offset=2000000 if cfg['profile']=='pilot' else 0
    return [(c,list(range(first,min(first+cfg['batch'],cfg['counts'][str(c['id'])]+1))),offset)
            for c in cells for first in range(1,cfg['counts'][str(c['id'])]+1,cfg['batch'])]

def environment(device):
    import torch,scipy
    return dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
                scipy=scipy.__version__,pandas=pd.__version__,system=platform.system(),machine=platform.machine(),
                device=torch.cuda.get_device_name(0) if device=='cuda' else 'cpu',cuda=torch.version.cuda,batch=design.BATCH)

def worker(out,seconds=6600,device='cuda'):
    if int(os.environ.get('SLURM_NTASKS','1'))!=1 or int(os.environ.get('SLURM_ARRAY_TASK_COUNT','1'))!=1:
        raise RuntimeError('This study permits exactly one worker; no multi-task arrays')
    lease=out/'WORKER_RUNNING'
    try:lease.mkdir()
    except FileExistsError:raise RuntimeError('Worker lease exists. Check Slurm before removing a stale WORKER_RUNNING directory.')
    try:
        dump(lease/'owner.json',dict(pid=os.getpid(),job=os.environ.get('SLURM_JOB_ID'),host=platform.node()))
        _worker(out,seconds,device)
    finally:
        (lease/'owner.json').unlink(missing_ok=True);lease.rmdir()

def _worker(out,seconds=6600,device='cuda'):
    import torch
    from experiment import simulate
    cfg=configuration(out)
    if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable: refusing CPU fallback')
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    env=environment(device)
    with (out/'environment.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (out/'execution.json').exists():
            if read(out/'execution.json')!=env:raise RuntimeError('Execution environment changed')
        else:dump(out/'execution.json',env)
    if cfg['profile']=='production':
        plan=read(out/'budget.json')
        if not plan['within_budget'] or plan['sources']!=cfg['sources'] or plan['pilot_execution']!=env:
            raise RuntimeError('Budget or pilot environment mismatch')
    signal.signal(signal.SIGUSR1,stop);signal.signal(signal.SIGTERM,stop)
    start=time.monotonic();last_batch=0.
    for c,reps,offset in tasks(cfg):
        if STOP or (out/'STOP').exists() or (out.parent/'STOP').exists() or time.monotonic()-start+max(60,1.5*last_batch)>seconds:break
        path=out/'draws'/f'cell-{c["id"]:03d}'/f'{reps[0]:05d}.json.gz'
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.with_suffix('.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if path.exists():
                old=read(path)
                if [r['replication'] for r in old]!=reps or any(r['failed'] for r in old):
                    raise RuntimeError('Saved failure/mismatch requires diagnosis: '+str(path))
                continue
            tic=time.monotonic()
            try:
                result=simulate(c,[r+offset for r in reps],device)
                if device=='cuda':torch.cuda.synchronize()
            except Exception:
                dump(path,[dict(cell_id=c['id'],replication=r,failed=True,error=traceback.format_exc()) for r in reps]);raise
            last_batch=time.monotonic()-tic
            for r,record in zip(reps,result):
                record.update(replication=r,seed_index=r+offset,batch_seconds=last_batch,batch_size=len(reps))
            dump(path,result)
            dump(path.with_suffix('.timing.json'),dict(cell_id=c['id'],seconds=time.monotonic()-tic,draws=len(reps)))
            print(f'cell={c["id"]} reps={reps[0]}:{reps[-1]} seconds={last_batch:.1f}',flush=True)
    print('Stopped cleanly; resume skips completed batches.',flush=True)

def report(out):
    cfg=configuration(out);seen=set();failures=0;timings=[]
    for path in sorted((out/'draws').glob('*/*.json.gz')):
        records=read(path)
        if records and not records[0]['failed']:
            timing=path.with_suffix('.timing.json')
            timings.append(read(timing) if timing.exists() else dict(cell_id=records[0]['cell_id'],seconds=records[0]['batch_seconds'],draws=len(records)))
        for r in records:
            key=(r['cell_id'],r['replication'])
            if key in seen or str(key[0]) not in cfg['counts'] or not 1<=key[1]<=cfg['counts'][str(key[0])]:
                raise RuntimeError('Unexpected/duplicate replication')
            seen.add(key);failures+=int(r['failed'])
    expected=sum(cfg['counts'].values())
    status=dict(saved=len(seen),expected=expected,failed=failures,complete=len(seen)==expected and failures==0,profile=cfg['profile'])
    dump(out/'status.json',status);pd.DataFrame(timings,columns=['cell_id','seconds','draws']).to_csv(out/'timings.csv',index=False)
    print(json.dumps(status,indent=2));return status

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('worker');a.add_argument('--output',type=Path,required=True);a.add_argument('--seconds',type=int,default=7500);a.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    for name in ('report','analyze'):
        a=sub.add_parser(name);a.add_argument('--output',type=Path,required=True)
    args=vars(p.parse_args());command=args.pop('command')
    if command=='worker':worker(args.pop('output'),**args)
    elif command=='report':report(args['output'])
    else:
        from extend import complete
        from resample import status
        complete(args['output']);status(args['output'],strict=True)
        from analysis import analyze
        analyze(args['output'])

if __name__=='__main__':main()
