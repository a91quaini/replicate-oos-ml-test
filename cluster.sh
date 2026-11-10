#!/usr/bin/env bash
# Generic Slurm submission. Prepare the output with run_simulations.py first.
set -euo pipefail

if [[ "${1:-}" == "__worker" ]]; then
    shift
    output=$1 phase=$2 cpus=$3 device=$4 shards=$5 python=$6 runner=$7
    workers=$cpus
    if [[ "$device" == cuda ]]; then workers=1; fi
    exec "$python" "$runner" "$phase" --output "$output" --workers "$workers" \
        --device "$device" --shards "$shards" --shard-index "${SLURM_ARRAY_TASK_ID:-0}"
fi

usage() {
    cat <<'EOF'
Usage: bash cluster.sh --output DIR --partition NAME --cpus N --memory SIZE \
       --time HH:MM:SS [--account NAME] [--shards N] [--gpu] \
       [--phase run|bootstrap] [--python /path/to/python]

Prepare DIR with run_simulations.py before submission. Supply your cluster's
partition, CPU count, memory (e.g. 16G), time limit, and account if required.
--shards submits disjoint groups of cells as a job array (default 1).
--gpu requests one GPU per shard and uses one neural training worker.
Run the neural bootstrap separately on CPUs after neural training completes.
Activate the desired environment before calling this script; its Python path
is carried into each job. The repository and output must be on shared storage.
EOF
}

output= partition= cpus= memory= walltime= account= shards=1 device=cpu phase=run
python="$(command -v python3)"
while (($#)); do
    case "$1" in
        --output) output=$2; shift 2;;
        --partition) partition=$2; shift 2;;
        --cpus) cpus=$2; shift 2;;
        --memory) memory=$2; shift 2;;
        --time) walltime=$2; shift 2;;
        --account) account=$2; shift 2;;
        --shards) shards=$2; shift 2;;
        --gpu) device=cuda; shift;;
        --phase) phase=$2; shift 2;;
        --python) python=$2; shift 2;;
        -h|--help) usage; exit 0;;
        *) echo "Unknown argument: $1" >&2; usage >&2; exit 2;;
    esac
done
if [[ -z "$output" || -z "$partition" || -z "$cpus" || -z "$memory" || -z "$walltime" ]]; then
    usage >&2; exit 2
fi
if [[ ! "$cpus" =~ ^[1-9][0-9]*$ || ! "$shards" =~ ^[1-9][0-9]*$ ]]; then
    echo "CPU and shard counts must be positive integers" >&2; exit 2
fi
if [[ "$phase" != run && "$phase" != bootstrap ]]; then
    echo "--phase must be run or bootstrap" >&2; exit 2
fi
if [[ "$phase" == bootstrap && "$device" == cuda ]]; then
    echo "Bootstrap uses CPUs; omit --gpu" >&2; exit 2
fi
output="$(cd "$output" && pwd)"
script="$(cd "$(dirname "$0")" && pwd)/cluster.sh"
runner="$(dirname "$script")/run_simulations.py"
if [[ ! -f "$output/configuration.json" ]]; then
    echo "Prepare this output with run_simulations.py first" >&2; exit 2
fi
mkdir -p "$output/logs"
args=(--parsable --job-name=oos-replication --nodes=1 --ntasks=1
      --cpus-per-task="$cpus" --mem="$memory" --time="$walltime"
      --partition="$partition" --array="0-$((shards-1))" --signal=B:USR1@120
      --output="$output/logs/%A_%a.out" --error="$output/logs/%A_%a.err")
if [[ -n "$account" ]]; then args+=(--account="$account"); fi
if [[ "$device" == cuda ]]; then args+=(--gpus=1); fi
sbatch "${args[@]}" "$script" __worker "$output" "$phase" "$cpus" "$device" "$shards" "$python" "$runner"
