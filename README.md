# Replication code: out-of-sample tests for machine learning forecasts

This package reproduces the paper's current lasso and nonlinear probability
forecast experiments. It includes the saved production statistics, the Python
programs that generate the figures and tables, and the original scientific
simulation code. No institutional account or proprietary data is required.

## 1. Recreate the paper figures and tables

Use Python 3.11 or 3.12. From this directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py
```

This reads the supplied results and takes a few minutes on an ordinary computer;
it does not fit forecasts. It writes vector PDFs in `output/figures/`, LaTeX table
bodies in `output/lasso/tables/` and `output/neural/tables/`, and the underlying
plotted values, comparison statistics and audit manifests alongside them. The
LaTeX tables require `booktabs`. They are fragments, not standalone documents.
Use `--only lasso` or `--only neural` to rebuild one study, or `--verify-only` to
check the input data and scientific-source hashes without plotting.

| Study | Figures | Tables |
| --- | --- | --- |
| Lasso | `lasso_size`, `lasso_refitting`, `lasso_diagnostics`, `lasso_power`, `lasso_power_complexity` | `lasso_size`, `lasso_learning`, `lasso_size_tuning` |
| ReLU networks | `nn_size`, `nn_power`, `nn_diagnostics` | `nn_size`, `nn_sensitivity`, `nn_learning`, `nn_power_comparisons` |

The four displayed methods are Raw, Bootstrap Raw, Orthogonal and Bootstrap
Orthogonal. Size-adjusted power uses matched simulated null distributions; it
is a simulation diagnostic, not an additional implementable test. The plotting
programs preserve the paper's treatment of ties and paired Monte Carlo
uncertainty.

## 2. Rerun the simulations

The exact production grids, replication counts and seeds are in `configs/`:

- `lasso.json`: 111 configurations, 77,250 replications; bootstrap with 499
  resamples, already included in each replication.
- `neural.json`: 11 configurations, 7,168 replications; three null configurations
  with 1,024 draws and eight alternatives with 512 draws. The neural networks use
  five predictors, two ReLU layers of width eight, and validation selection.
  Bootstrap inference with 999 resamples is computed after training.

Start with a small smoke run. Fresh results go to a new directory under `runs/`;
the supplied `data/` are never overwritten. The simulation runner's `--help`
documents local execution, cell selection, replication limits and sharding.
Fresh execution targets Linux or macOS; on Windows, use a Linux environment.
The following local smoke run uses one lasso cell and two replications. It tests
execution and reporting; it cannot estimate rejection probabilities reliably.

```bash
python run_simulations.py prepare --study lasso --output runs/lasso-smoke \
  --cells 1 --max-replications 2
python run_simulations.py run --output runs/lasso-smoke --workers 2
python run_simulations.py analyze --output runs/lasso-smoke
```

For the full lasso grid, omit the cell and replication limits:

```bash
python run_simulations.py prepare --study lasso --output runs/lasso
python run_simulations.py run --output runs/lasso --workers 8
python run_simulations.py report --output runs/lasso
python run_simulations.py analyze --output runs/lasso
```

Set `--workers` to the CPU cores available to you. Each fit uses one numerical
thread. Full production is substantial; measure the smoke run on your hardware
before choosing the number of workers and wall time.

For neural fitting, use a separate environment matching its recorded software:

```bash
python3.11 -m venv .venv-neural
source .venv-neural/bin/activate
python -m pip install -r requirements-neural.txt
python -m pip install torch==2.7.1
```

With that environment active:

```bash
python run_simulations.py prepare --study neural --output runs/neural
python run_simulations.py run --output runs/neural --device cuda --workers 1
python run_simulations.py bootstrap --output runs/neural --workers 8
python run_simulations.py report --output runs/neural
python run_simulations.py analyze --output runs/neural
```

Use `--device cpu --workers 1` for CPU fitting, which is slower. For a neural
smoke run, add `--cells 0 --max-replications 2` to preparation and use a different
output directory. Neural training is batched in groups of 32 in full production;
the smoke run keeps the same epochs and tuning grid but uses fewer draws.
`analyze` writes fresh summary, power and diagnostic CSVs from the completed raw
checkpoints. It requires the full prepared run to finish; neural alternatives
also need their matching null cells. Smaller runs are marked as subsets.

Select a CPU or CUDA build of PyTorch appropriate for your machine. The published
neural simulations used Python 3.11.3, PyTorch 2.7.1 with CUDA 12.6 and an NVIDIA
H100. The lasso simulations used Python 3.11.3 and the package versions listed in
`requirements.txt`. Matching versions and hardware provides the closest
reproduction. Different linear-algebra libraries or CPU/GPU devices can change
floating-point optimization trajectories; exact agreement with supplied neural
draws is not promised on different hardware. Figure/table regeneration from
the supplied statistics does not require PyTorch or a GPU.

## 3. Run on a general cluster

Prepare the chosen study as above on storage shared with your compute nodes.
Activate its environment, then supply your own Slurm partition, account (if
required), resource limits and Python executable. For example, replace the
uppercase values below with those supplied by your cluster:

```bash
bash cluster.sh --output runs/lasso --partition CPU_PARTITION \
  --account YOUR_ACCOUNT --cpus 16 --memory 32G --time 04:00:00 --shards 4 \
  --python "$(command -v python)"
```

This submits four jobs, each with 16 CPU workers, covering disjoint groups of
cells. Resource values are examples, not universal requirements. For neural
fitting, a GPU example is:

```bash
bash cluster.sh --output runs/neural --partition GPU_PARTITION \
  --account YOUR_ACCOUNT --cpus 4 --memory 32G --time 04:00:00 --gpu \
  --python "$(command -v python)"
```

After all neural training finishes, submit its CPU bootstrap:

```bash
bash cluster.sh --output runs/neural --phase bootstrap \
  --partition CPU_PARTITION --account YOUR_ACCOUNT \
  --cpus 16 --memory 32G --time 02:00:00 \
  --python "$(command -v python)"
```

The wrapper uses standard Slurm options and does not load site-specific modules.
Omit `--account` if your site does not require it. Some sites use a different GPU
resource option; replace `--gpus=1` in `cluster.sh` as directed by your support
team. The wrapper has been syntax-checked; it cannot be submitted locally.
For other schedulers, run the same Python commands inside an allocated job. To
split work, use distinct `--shard-index 0 ... S-1` with the same `--shards S`.

Completed batches are retained when a run is resumed. To stop gently, create
`runs/STUDY/STOP`; workers finish their active work and stop taking new batches.
Remove that file before resuming with the same command. Check that all previous
workers have exited before changing the number of shards. A time-limit kill can
lose the current neural batch but not previously saved batches. Inspect your
scheduler's status and `run_simulations.py report` before resubmitting. The
scripts never submit follow-up jobs automatically.

## Files and scope

- `analysis/`: the paper's plotting and comparison calculations, adapted to
  read the compact public data rather than private checkpoint directories.
- `data/`: replication-level statistics, learning diagnostics and summaries.
  These are enough to regenerate all listed assets; full per-observation score
  sequences and trained network weights are not included.
- `engines/`: byte-preserved scientific modules used for the production studies.
  Some lasso modules import older network helpers; these are retained solely to
  preserve those imports. No superseded study is exposed by the public runner.
- `configs/`: the exact retained production cells and source hashes.
- `data_manifest.json`: integrity hashes for the compact data export.
- `run_simulations.py`, `cluster.sh`: portable orchestration for fresh runs.

In the saved tables, `cell_id` joins to the configuration, `replication` is the
within-cell simulation index, `method` identifies the statistic, and `valid`
and `reject` record its validity and nominal five-percent rejection. Replication
indices are intentionally paired across comparable cells. Pilots are excluded.
Some internal diagnostic fields and frozen engines retain population-oracle
quantities; oracle tests are not displayed in the paper figures or tables.

The probability study was expanded after examination of its initial production
results. This package retains the full final agreed grid and does not describe
that expansion as an independent prespecified confirmation. Superseded neural,
forest and kernel experiments are excluded.

The compact export was audited against the original raw checkpoints before
packaging. Its public checks verify data hashes, replication counts, rejection
rules and empirical-null calibrations; they cannot recheck omitted raw score
sequences. Fresh simulations save their own raw checkpoints and diagnostic
records. No external research dataset is needed for these simulation exercises.

## Package verification

The complete saved-data rebuild was run successfully. All seven generated LaTeX
table bodies matched the manuscript's current fragments byte for byte, and all
plotted numerical values matched within floating-point tolerance. A fresh lasso
replication, including both bootstrap tests and subsequent CSV analysis, passed.
Neural bootstrap and analysis were checked on an archived production score
record. Resume integrity, distinct shards and simulated Slurm submission/spool
execution were also checked. Fresh neural training was not exercised in the
packaging environment because PyTorch was unavailable; its scientific training
modules are unchanged from those used for the supplied production results.
