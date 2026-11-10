"""Bounded parallel execution, atomic checkpoints, timing and reporting."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[name] = '1'
os.environ['MPLBACKEND'] = 'Agg'
import argparse
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import fcntl
import gzip
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import shutil
import signal
import time
import traceback
import numpy as np
import expanded_study as study

STOP = False


def request_stop(*_):
    global STOP
    STOP = True


def encode(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,np.generic): return value.item()
    raise TypeError(type(value).__name__)


def dump(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    payload = json.dumps(value,sort_keys=True,allow_nan=False,default=encode).encode()
    temporary.write_bytes(gzip.compress(payload) if path.suffix=='.gz' else payload)
    os.replace(temporary,path)


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_text())


def source_paths():
    paths = list(Path(__file__).parent.glob('*.py'))
    paths += [study.BASE/name for name in ('lasso_designs.py','calibration.py','inference.py','requirements.txt')]
    paths += [study.ROOT/'ideas/neural-networks/simulations'/name for name in ('core.py','network.py','selection.py')]
    return sorted(paths)


def configuration(args):
    cells = study.grid()
    if args.cells:
        wanted = set(map(int,args.cells.split(',')))
        cells = [c for c in cells if c['id'] in wanted]
        if len(cells)!=len(wanted): raise ValueError('Unknown cell ID')
    # Convert tuples before comparing the serialized configuration on resume.
    cells = json.loads(json.dumps(cells))
    counts = {str(c['id']):args.replications or (2 if args.profile=='pilot' else c['replications']) for c in cells}
    return dict(version=study.VERSION,seed=args.seed,bootstrap_draws=study.BOOTSTRAP_DRAWS,
        profile=args.profile,cells=cells,counts=counts,
        sources={str(p.relative_to(study.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths()},
        environment=dict(python=platform.python_version(),system=platform.system(),machine=platform.machine(),
            packages={p:importlib.metadata.version(p) for p in ('numpy','scipy','scikit-learn','pandas','matplotlib')}))


def chunks(output,config):
    for cell in config['cells']:
        count = config['counts'][str(cell['id'])]
        width = 1 if cell['schedule']=='daily' else 5
        for first in range(1,count+1,width):
            path = Path(output)/'draws'/f'cell-{cell["id"]:03d}'/f'{first:05d}.json.gz'
            yield cell,list(range(first,min(first+width,count+1))),path


def checked_records(path,cell,reps):
    rows = read(path) if path.exists() else []
    ids = [r['replication'] for r in rows]
    if len(ids)!=len(set(ids)) or not set(ids)<=set(reps):
        raise ValueError(f'Duplicate/unexpected replication in {path}')
    if any(r['cell_id']!=cell['id'] for r in rows):
        raise ValueError(f'Wrong cell in {path}')
    return rows


def run_chunk(output,config,cell,reps,path):
    # SIGUSR1 is sent to the parent only; workers finish their current draw and
    # check the shared stop file before starting another expensive fit.
    rows = checked_records(path,cell,reps)
    if any(r['failed'] for r in rows): raise RuntimeError(f'Saved failure: {path}')
    done = {r['replication'] for r in rows}
    for replication in reps:
        if replication in done: continue
        if (Path(output)/'STOP').exists(): break
        started = time.monotonic()
        try:
            row = study.simulate(cell,replication,config['seed'],config['bootstrap_draws'])
        except Exception:
            row = dict(cell_id=cell['id'],replication=replication,failed=True,error=traceback.format_exc())
        row['elapsed_seconds'] = time.monotonic()-started
        rows.append(row)
        dump(path,rows)
        if row['failed']: raise RuntimeError(f'Simulation failure saved to {path}: {row["error"]}')
    return len(rows)


def all_records(output,config):
    expected = set()
    for cell,reps,path in chunks(output,config):
        expected.add(path.resolve())
        for row in checked_records(path,cell,reps): yield row
    unexpected = set((Path(output)/'draws').glob('*/*.json.gz'))
    unexpected = {p.resolve() for p in unexpected}-expected
    if unexpected: raise ValueError(f'Unexpected checkpoint: {next(iter(unexpected))}')


def report(output,config):
    import pandas as pd
    output = Path(output)
    rows, paired, timings = [], [], []
    saved = failed = 0
    for record in all_records(output,config):
        saved += 1
        failed += int(record['failed'])
        if record['failed']: continue
        common = dict(cell_id=record['cell_id'],replication=record['replication'])
        timings.append(dict(**common,seconds=record['elapsed_seconds']))
        d = record['diagnostics']
        for method,s in record['statistics'].items():
            rows.append(dict(**common,method=method,valid=s['valid'],reject=s['reject'],covered=s['covered'],
                estimate=s['estimate'],variance=s['variance'],interval_length=s['interval_length'],
                error=s['estimate']-record['theta'],
                converged=d['all_fits_converged'],nested_converged=d['all_nested_fits_converged'],
                max_forecast_error=d['max_forecast_error'],max_representer_error=d['max_representer_error'],
                raw_drift=d['scaled_population_raw_drift'],corrected_drift=d['scaled_population_corrected_drift']))
        for other in ('raw','raw_bootstrap','orthogonal_bootstrap'):
            a,b = record['statistics']['orthogonal'],record['statistics'][other]
            if a['valid'] and b['valid']:
                paired.append(dict(**common,comparison='orthogonal minus '+other,
                                   difference=int(a['reject'])-int(b['reject'])))
    expected = sum(config['counts'].values())
    status = dict(saved=saved,expected=expected,failed=failed,complete=saved==expected and failed==0,
                  profile=config['profile'])
    dump(output/'status.json',status)
    if not rows: return status
    frame = pd.DataFrame(rows)
    summaries = []
    lookup = {c['id']:c for c in config['cells']}
    for (cell_id,method),group in frame.groupby(['cell_id','method'],sort=True):
        c = lookup[cell_id]
        valid = group[group.valid]
        count = len(valid)
        reject = float(valid.reject.mean()) if count else np.nan
        meta = study.description(c)
        summaries.append(dict(cell_id=cell_id,family=c['family'],n=c['n'],T=meta['T'],ratio=c['ratio'],
            complexity=str(c['complexity']),parameters_1=meta['parameters'][0],parameters_2=meta['parameters'][1],
            parameters_per_T_1=meta['parameters_per_training_observation'][0],
            parameters_per_T_2=meta['parameters_per_training_observation'][1],
            tuning=c['tuning_label'],schedule=c['schedule'],refits=c['refits'],delta=c['delta'],
            groups=';'.join(c['groups']),method=method,completed=len(group),valid=count,
            invalid=len(group)-count,valid_fraction=count/len(group),rejection=reject,
            rejection_mcse=np.sqrt(reject*(1-reject)/count) if count else np.nan,
            # This is an operational rate, not size conditional on valid variance.
            rejection_per_completed=float(valid.reject.sum()/len(group)),
            coverage=float(valid.covered.mean()) if count else np.nan,
            mean_interval_length=valid.interval_length.mean(),bias=group.error.mean(),
            rmse=np.sqrt(np.mean(group.error**2)),convergence_fraction=group.converged.mean(),
            nested_convergence_fraction=group.nested_converged.mean(),
            mean_max_forecast_error=group.max_forecast_error.mean(),
            mean_max_representer_error=group.max_representer_error.mean(),
            mean_scaled_raw_drift=group.raw_drift.mean(),mean_scaled_corrected_drift=group.corrected_drift.mean()))
    summary = pd.DataFrame(summaries)
    summary.to_csv(output/'summary.csv',index=False)
    summary[summary.delta==0].to_csv(output/'size.csv',index=False)
    summary[summary.delta!=0].to_csv(output/'local_power.csv',index=False)
    pd.DataFrame(timings).to_csv(output/'timings.csv',index=False)
    differences = []
    if paired:
        for (cell_id,comparison),g in pd.DataFrame(paired).groupby(['cell_id','comparison']):
            differences.append(dict(cell_id=cell_id,comparison=comparison,paired_valid=len(g),
                rejection_difference=g.difference.mean(),mcse=g.difference.std(ddof=1)/np.sqrt(len(g))))
    pd.DataFrame(differences).to_csv(output/'paired_test_differences.csv',index=False)
    plots(summary,output,config['profile'])
    return status


def plots(frame,output,profile):
    os.environ.setdefault('MPLCONFIGDIR',str(Path(output).resolve()/'.matplotlib'))
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    styles = [('raw','Raw HAC','o'),('orthogonal','Corrected HAC','s'),
              ('orthogonal_bootstrap','Corrected block bootstrap','^')]
    with PdfPages(output/'comparisons.pdf') as pdf:
        core = frame[frame.groups.str.contains('core|power_curve')]
        for (family,tuning),group in core.groupby(['family','tuning']):
            fig,axes = plt.subplots(2,3,figsize=(12,7),sharey=True,squeeze=False)
            for i,n in enumerate((500,1000)):
                for j,schedule in enumerate(study.SCHEDULES):
                    ax = axes[i,j]
                    for method,label,marker in styles:
                        g = group[(group.n==n)&(group.schedule==schedule)&(group.method==method)].sort_values('delta')
                        ax.errorbar(g.delta,g.rejection,yerr=1.96*g.rejection_mcse,label=label,marker=marker,capsize=2)
                    ax.axhline(.05,color='grey',linestyle=':')
                    ax.set(title=f'n={n}; {schedule}',xlabel='Local effect delta',ylim=(-.02,1.02))
                    if j==0: ax.set_ylabel('Rejection probability')
            axes[0,0].legend(fontsize=8)
            prefix = 'TIMING PILOT — NOT MONTE CARLO EVIDENCE. ' if profile=='pilot' else ''
            fig.suptitle(f'{prefix}{family}: {tuning}. Conditional on valid test; 95% MC error bars.',fontsize=11)
            fig.tight_layout()
            pdf.savefig(fig);plt.close(fig)
        for family in ('lasso','neural'):
            group = frame[(frame.family==family)&frame.groups.str.contains('complexity')&
                          (frame.method=='orthogonal')&(frame.delta==0)]
            tunings = list(group.tuning.unique())
            if not tunings:
                continue
            fig,axes = plt.subplots(1,len(tunings),figsize=(5*len(tunings),4),squeeze=False,sharey=True)
            for ax,tuning in zip(axes[0],tunings):
                for complexity,g in group[group.tuning==tuning].groupby('complexity'):
                    g = g.sort_values('ratio')
                    ax.errorbar(g.ratio,g.rejection,yerr=1.96*g.rejection_mcse,marker='o',capsize=2,label=complexity)
                ax.axhline(.05,color='grey',linestyle=':')
                ax.set(title=tuning,xlabel='Training/evaluation size ratio',ylim=(-.02,1.02))
                ax.legend(title='Dictionary / hidden widths',fontsize=8)
            axes[0,0].set_ylabel('Corrected HAC rejection probability')
            prefix = 'TIMING PILOT — NOT MONTE CARLO EVIDENCE. ' if profile=='pilot' else ''
            fig.suptitle(f'{prefix}{family}: complexity and training size, n=1000.',fontsize=11)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        for family in ('lasso','neural'):
            group=frame[(frame.family==family)&frame.groups.str.contains('complexity_power')]
            tunings=list(group.tuning.unique())
            if not tunings:
                continue
            fig,axes=plt.subplots(3,len(tunings),figsize=(4.2*len(tunings),9),squeeze=False,sharey=True)
            for i,ratio in enumerate((.5,1.,2.)):
                for j,tuning in enumerate(tunings):
                    ax=axes[i,j]
                    for method,label,marker in styles:
                        g=group[(group.ratio==ratio)&(group.tuning==tuning)&(group.method==method)].sort_values('delta')
                        ax.errorbar(g.delta,g.rejection,yerr=1.96*g.rejection_mcse,
                                    label=label,marker=marker,capsize=2)
                    ax.axhline(.05,color='grey',linestyle=':')
                    ax.set(title=f'{tuning}; T/n={ratio:g}',xlabel='Local effect delta',ylim=(-.02,1.02))
                    if j==0:ax.set_ylabel('Rejection probability')
            axes[0,0].legend(fontsize=8)
            prefix='TIMING PILOT — NOT MONTE CARLO EVIDENCE. ' if profile=='pilot' else ''
            complexity='p=256' if family=='lasso' else 'width=16'
            fig.suptitle(f'{prefix}{family}, {complexity}, n=1000, five refits.',fontsize=11)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)


def budget(pilot,production=None,limit=5000.):
    config = read(Path(pilot)/'configuration.json')
    scientific = configuration(argparse.Namespace(cells=None,replications=None,profile='production',seed=config['seed']))
    for key in ('version','sources','environment','bootstrap_draws'):
        if config[key]!=scientific[key]: raise ValueError(f'Pilot mismatch: {key}')
    times = {}
    for r in all_records(pilot,config):
        if r['failed']: raise ValueError('Pilot has failed draws')
        times.setdefault(r['cell_id'],[]).append(r['elapsed_seconds'])
    if set(times)!={c['id'] for c in study.grid()} or min(map(len,times.values()))<2:
        raise ValueError('Timing requires at least two completed replications of every cell')
    done = {}
    if production and (Path(production)/'configuration.json').exists():
        old = read(Path(production)/'configuration.json')
        if old!=scientific: raise ValueError('Production configuration differs from timing plan')
        for r in all_records(production,old):
            if r['failed']: raise ValueError('Production contains failures requiring diagnosis')
            done[r['cell_id']] = done.get(r['cell_id'],0)+1
    total = remaining = 0.
    for c in study.grid():
        seconds = max(times[c['id']])
        total += seconds*c['replications']
        remaining += seconds*(c['replications']-done.get(c['id'],0))
    # Genoa: one SBU per allocated core-hour. Timing is worker elapsed time,
    # then add a 50% contingency for I/O, load imbalance and pilot uncertainty.
    result = dict(measured_projection_sbu=total/3600,contingency_factor=1.5,
        projected_total_sbu_with_contingency=1.5*total/3600,
        projected_remaining_sbu_with_contingency=1.5*remaining/3600,
        projected_remaining_hours_on_192_cores=1.5*remaining/(3600*192),
        study_budget_sbu=limit,within_study_budget=1.5*total/3600<=limit,
        maximum_sbu_per_12_hour_submission=2304,
        warning='Estimate, not an allocation balance check. Check budget-overview before each submission.')
    dump(Path(pilot)/'budget.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def execute(args):
    output = Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    config = configuration(args)
    with (output/'.run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = output/'configuration.json'
        if path.exists():
            if read(path)!=config: raise ValueError('Source/configuration/environment changed; use a NEW directory')
        else:
            if list((output/'draws').glob('*/*.json.gz')): raise ValueError('Checkpoints without configuration')
            for name in config['sources']:
                target = output/'source'/name;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(study.ROOT/name,target)
            dump(path,config)
        if args.profile=='production' and not args.cells and not args.replications:
            if not args.timing: raise ValueError('Full production requires --timing PILOT_DIRECTORY')
            if not budget(args.timing,output,args.budget_sbu)['within_study_budget']:
                raise ValueError('Projected study exceeds budget. Review design/timing before spending more SBU.')
        (output/'STOP').unlink(missing_ok=True)
        todo = []
        for cell,reps,path in chunks(output,config):
            rows = checked_records(path,cell,reps)
            if any(r['failed'] for r in rows): raise ValueError(f'Saved failure in {path}; diagnose before resume')
            if len(rows)<len(reps): todo.append((cell,reps,path))
        # Expensive cells first reduce the final tail when most workers are idle.
        todo.sort(key=lambda t:-(t[0]['refits']*(6 if t[0]['tuning_label'] in ('cv','validation') else 1)))
        print(f'{len(todo)} remaining chunks; {args.workers} single-thread workers',flush=True)
        signal.signal(signal.SIGUSR1,request_stop)
        signal.signal(signal.SIGTERM,request_stop)
        iterator = iter(todo)
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            pending = set()
            for _ in range(min(2*args.workers,len(todo))):
                c,reps,p = next(iterator);pending.add(pool.submit(run_chunk,str(output),config,c,reps,p))
            finished = 0
            while pending:
                complete,pending = wait(pending,timeout=5,return_when=FIRST_COMPLETED)
                if STOP or (output/'STOP').exists():
                    request_stop()
                    (output/'STOP').touch()
                for future in complete:
                    try: future.result()
                    except Exception:
                        (output/'STOP').touch()
                        for queued in pending: queued.cancel()
                        raise
                    finished += 1
                    if finished==1 or finished%20==0 or finished==len(todo):
                        print(f'Finished {finished}/{len(todo)} chunks',flush=True)
                    if not STOP:
                        task = next(iterator,None)
                        if task:
                            c,reps,p = task;pending.add(pool.submit(run_chunk,str(output),config,c,reps,p))
        status = report(output,config)
        print(('COMPLETE' if status['complete'] else 'CHECKPOINTED; resume with same command')+': '+str(output),flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('plan','run','report','budget'))
    parser.add_argument('--output',default='results-comprehensive')
    parser.add_argument('--profile',choices=('pilot','production'),default='pilot')
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--seed',type=int,default=study.SEED)
    parser.add_argument('--cells',help='Comma-separated IDs; execution checks only')
    parser.add_argument('--replications',type=int,help='Override for execution checks only')
    parser.add_argument('--timing',help='Completed all-cell pilot directory')
    parser.add_argument('--budget-sbu',type=float,default=5000.)
    args = parser.parse_args()
    if args.workers<1 or (args.replications is not None and args.replications<1):
        parser.error('Workers and replications must be positive')
    if args.task=='plan':
        import pandas as pd
        out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
        rows=[dict(**c,**{k:v for k,v in study.description(c).items() if k not in c}) for c in study.grid()]
        dump(out/'design.json',rows)
        pd.DataFrame(rows).to_csv(out/'design.csv',index=False)
        print(f'{len(rows)} cells; {sum(c["replications"] for c in rows):,} production replications; {out}')
    elif args.task=='run': execute(args)
    elif args.task=='report': print(report(args.output,read(Path(args.output)/'configuration.json')))
    elif args.task=='budget':
        if not args.timing: parser.error('--timing is required')
        result=budget(args.timing,args.output,args.budget_sbu)
        if not result['within_study_budget']: raise SystemExit(2)


if __name__=='__main__': main()
