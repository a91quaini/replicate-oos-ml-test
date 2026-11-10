#!/usr/bin/env python3
"""Run the frozen experiments locally or in independent cluster shards."""
import os

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[name] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import contextmanager
import fcntl
import gzip
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import platform
import signal
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
STOP = False
_STUDY = None
_TORCH_READY = False


def read(path):
    path = Path(path)
    data = path.read_bytes()
    return json.loads(gzip.decompress(data) if path.suffix == ".gz" else data)


def dump(path, value):
    """Atomically publish a complete JSON checkpoint."""
    def encode(item):
        if hasattr(item, "tolist"):
            return item.tolist()
        raise TypeError(type(item).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, allow_nan=False, indent=2, default=encode).encode()
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_bytes(gzip.compress(data, mtime=0) if path.suffix == ".gz" else data)
    temp.replace(path)


def source_hashes(study):
    root = ROOT / "engines" / study
    files = sorted(root.rglob("*.py"))
    if not files:
        raise RuntimeError("Missing packaged engine: " + str(root))
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files}


def load_configuration(output):
    cfg = read(Path(output) / "configuration.json")
    metadata = cfg.get("replication")
    if not metadata:
        raise RuntimeError("Prepare a fresh output with this replication runner first")
    if metadata["engine_sources"] != source_hashes(metadata["study"]):
        raise RuntimeError("Engine code changed; restore it or prepare a new output directory")
    return cfg


@contextmanager
def lock_file(path, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise RuntimeError("Another process owns this checkpoint: " + str(path)) from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def prepare(study, output, cells=None, max_replications=None):
    cfg = read(ROOT / "configs" / (study + ".json"))
    declared = {str(Path(name).relative_to(Path("engines") / study)): digest
                for name, digest in cfg["engine_sources"].items()}
    if declared != source_hashes(study):
        raise RuntimeError("Packaged engine differs from its scientific manifest")
    if cells:
        wanted = set(map(int, cells.split(",")))
        cfg["cells"] = [c for c in cfg["cells"] if c["id"] in wanted]
        if {c["id"] for c in cfg["cells"]} != wanted:
            raise ValueError("Unknown cell ID")
    cfg["counts"] = {str(c["id"]): int(cfg["counts"][str(c["id"])])
                     for c in cfg["cells"]}
    if max_replications is not None:
        if max_replications < 1:
            raise ValueError("--max-replications must be positive")
        cfg["counts"] = {i: min(n, max_replications) for i, n in cfg["counts"].items()}
    cfg["replication"] = dict(study=study, engine_sources=source_hashes(study),
        scope="full" if cells is None and max_replications is None else "subset",
        description="Fresh rerun; original cell IDs and random seeds are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    with lock_file(output / "prepare.lock", blocking=True):
        path = output / "configuration.json"
        if path.exists():
            if read(path) != cfg:
                raise RuntimeError("Existing configuration differs; choose a NEW output directory")
        else:
            if any(output.iterdir()) and list(output.iterdir()) != [output / "prepare.lock"]:
                raise RuntimeError("Prepare requires an empty output directory")
            dump(path, cfg)
    print(f'{study}: {len(cfg["cells"])} cells, {sum(cfg["counts"].values())} replications; {output}')


def engine(study, training=True):
    global _STUDY
    if _STUDY is not None and _STUDY != study:
        raise RuntimeError("Run different studies in separate Python processes")
    if _STUDY is None:
        path = ROOT / "engines" / study
        if study == "lasso":
            path = path / "codes" / "simulations" / "comprehensive"
        sys.path.insert(0, str(path))
        _STUDY = study
    name = "expanded_study" if study == "lasso" else ("experiment" if training else "design")
    return importlib.import_module(name)


def setup_torch(device):
    global _TORCH_READY
    import torch
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu or install a CUDA-enabled PyTorch")
    if not _TORCH_READY:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        _TORCH_READY = True


def environment(output, study, phase, device):
    packages = ["numpy", "scipy", "pandas"]
    if study == "lasso":
        packages.append("scikit-learn")
    elif phase == "run":
        packages.append("torch")
    value = dict(python=platform.python_version(),
                 packages={p: importlib.metadata.version(p) for p in packages},
                 device=device if phase == "run" and study == "neural" else "cpu")
    path = output / (phase + "_environment.json")
    with lock_file(output / (phase + "_environment.lock"), blocking=True):
        if path.exists() and read(path) != value:
            raise RuntimeError("Execution environment changed; use the same environment to resume, or a NEW output")
        if not path.exists():
            dump(path, value)


def tasks(output, cfg, shard_index=0, shards=1):
    for index, cell in enumerate(sorted(cfg["cells"], key=lambda c: c["id"])):
        if index % shards != shard_index:
            continue
        count = cfg["counts"][str(cell["id"])]
        width = cfg.get("batch", 32) if cfg["replication"]["study"] == "neural" else (1 if cell["schedule"] == "daily" else 5)
        for first in range(1, count + 1, width):
            path = output / "draws" / f'cell-{cell["id"]:03d}' / f"{first:05d}.json.gz"
            yield cell, list(range(first, min(first + width, count + 1))), path


def checked_records(path, cell, reps):
    rows = read(path) if path.exists() else []
    ids = [r["replication"] for r in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= set(reps) or any(r["cell_id"] != cell["id"] for r in rows):
        raise RuntimeError("Unexpected/duplicate draw identity: " + str(path))
    if any(r["failed"] for r in rows):
        raise RuntimeError("Saved failure requires diagnosis: " + str(path))
    return rows


def run_batch(job):
    output, cfg, cell, reps, path, device, phase = job
    study = cfg["replication"]["study"]
    module = engine(study, training=phase == "run")
    lock = path.with_name(path.name + "." + phase + ".lock")
    with lock_file(lock):
        rows = checked_records(path, cell, reps)
        if phase == "bootstrap":
            if len(rows) != len(reps):
                raise RuntimeError("Finish this simulation batch before bootstrapping: " + str(path))
            resample = importlib.import_module("resample")
            if not resample.check_saved(output, path):
                resample.process_batch((output, path, cell))
            return cell["id"], reps[0], len(rows)
        if len(rows) == len(reps):
            return cell["id"], reps[0], len(rows)
        start = time.monotonic()
        try:
            if study == "lasso":
                done = {r["replication"] for r in rows}
                for rep in reps:
                    if rep in done:
                        continue
                    if (output / "STOP").exists():
                        break
                    tic = time.monotonic()
                    row = module.simulate(cell, rep, cfg["seed"], cfg["bootstrap_draws"])
                    row["elapsed_seconds"] = time.monotonic() - tic
                    rows.append(row)
                    dump(path, rows)
            else:
                if rows:
                    raise RuntimeError("Incomplete neural batch requires diagnosis: " + str(path))
                design = importlib.import_module("design")
                if cfg["seed"] != design.SEED:
                    raise RuntimeError("Configuration seed does not match the frozen neural engine")
                setup_torch(device)
                rows = module.simulate(cell, reps, device=device)
                if device == "cuda":
                    import torch
                    torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                if len(rows) != len(reps):
                    raise RuntimeError("Neural engine returned an unexpected number of draws")
                for rep, row in zip(reps, rows):
                    row.update(replication=rep, seed_index=rep, batch_seconds=elapsed, batch_size=len(reps))
                dump(path, rows)
                dump(path.with_suffix(".timing.json"), dict(cell_id=cell["id"], seconds=elapsed, draws=len(reps)))
        except Exception:
            dump(output / "errors" / path.parent.name / (path.name + ".json"),
                 dict(cell_id=cell["id"], replications=reps, error=traceback.format_exc()))
            raise
    return cell["id"], reps[0], len(rows)


def request_stop(*_):
    global STOP
    STOP = True


def run(output, workers=1, device="cpu", shards=1, shard_index=0, phase="run"):
    if workers < 1 or shards < 1 or not 0 <= shard_index < shards:
        raise ValueError("Positive workers/shards and 0 <= shard-index < shards are required")
    cfg = load_configuration(output)
    study = cfg["replication"]["study"]
    if phase == "bootstrap" and study != "neural":
        raise ValueError("Lasso computes its bootstrap tests during run")
    if phase == "run" and device == "cuda" and (study != "neural" or workers != 1):
        raise ValueError("Use one worker per allocated GPU; CUDA is available for the neural study")
    environment(output, study, phase, device)
    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1):
        signal.signal(signum, request_stop)
    jobs = ((output, cfg, c, reps, path, device, phase)
            for c, reps, path in tasks(output, cfg, shard_index, shards))
    with lock_file(output / f"{phase}-shard-{shard_index}-of-{shards}.lock"):
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            pending = set()
            def dispatch():
                while len(pending) < workers and not STOP and not (output / "STOP").exists():
                    job = next(jobs, None)
                    if job is None:
                        break
                    pending.add(pool.submit(run_batch, job))
            dispatch()
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    cell, first, count = future.result()
                    print(f"{phase}: cell={cell}, batch={first}, saved={count}", flush=True)
                dispatch()
    print("Stopped cleanly. Resume with the same command; completed draws are retained.")


def report(output):
    cfg = load_configuration(output)
    expected_paths, seen = set(), set()
    for cell, reps, path in tasks(output, cfg):
        expected_paths.add(path)
        for row in checked_records(path, cell, reps):
            key = (row["cell_id"], row["replication"])
            if key in seen:
                raise RuntimeError("Duplicate draw identity")
            seen.add(key)
    extra = set((output / "draws").glob("*/*.json.gz")) - expected_paths
    if extra:
        raise RuntimeError("Unexpected checkpoint: " + str(next(iter(extra))))
    expected = sum(cfg["counts"].values())
    status = dict(saved=len(seen), expected=expected, complete=len(seen) == expected,
                  study=cfg["replication"]["study"], scope=cfg["replication"]["scope"])
    if status["study"] == "neural":
        engine("neural", training=False)
        resample = importlib.import_module("resample")
        status["bootstrap_saved"] = sum(len(read(p)) for p in expected_paths
                                         if p.exists() and resample.check_saved(output, p))
        status["bootstrap_complete"] = status["bootstrap_saved"] == expected
    dump(output / "status.json", status)
    print(json.dumps(status, indent=2))
    return status


def analyze(output):
    status = report(output)
    if not status["complete"] or not status.get("bootstrap_complete", True):
        raise RuntimeError("Complete simulations and bootstrap before producing final summary tables")
    cfg = load_configuration(output)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".plot-cache"))
    engine(status["study"], training=False)
    original_run = importlib.import_module("run")
    if status["study"] == "lasso":
        original_run.plots = lambda *_: None
        original_run.report(output, cfg)
    else:
        original_run.configuration = load_configuration
        importlib.import_module("baseline").COUNTS = {}
        analysis = importlib.import_module("analysis")
        analysis.figures = lambda *_: None
        nulls = {(c["n"], c["T"], c["dgp"]) for c in cfg["cells"] if c["delta"] == 0}
        if any((c["n"], c["T"], c["dgp"]) not in nulls for c in cfg["cells"]):
            raise ValueError("Include the matching null cell to compute size-adjusted power")
        analysis.analyze(output)
    print("Summary CSV files saved to " + str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Freeze the study configuration in a fresh output directory")
    p.add_argument("--study", choices=("lasso", "neural"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cells", help="Optional comma-separated cell IDs")
    p.add_argument("--max-replications", type=int, help="Cap counts for a smoke test, not paper evidence")
    for command in ("run", "bootstrap"):
        p = sub.add_parser(command)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--workers", type=int, default=1)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        p.add_argument("--shards", type=int, default=1)
        p.add_argument("--shard-index", type=int, default=0)
    for command in ("report", "analyze"):
        sub.add_parser(command).add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    args["output"] = args["output"].expanduser().resolve()
    if command == "bootstrap":
        run(**args, phase="bootstrap")
    else:
        globals()[command](**args)


if __name__ == "__main__":
    main()
