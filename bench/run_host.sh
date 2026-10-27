#!/usr/bin/env bash
# Run one gpu_bench.py subcommand on one GPU of this host.
#
#   run_host.sh <physical GPU index> <out dir> "<description>" <idle|time|power> [gpu_bench.py args...]
#
# Environment: S (work directory for caches and tmp), PY (python), NSYS (nsys binary; `time` runs
# under it and records its version).
#   - refuses (exit 75) when another process is on that GPU; gpu_bench.py checks again before each chain;
#   - on zhang-x4, writes Start / End lines to /scratch/users/gpu-use-log/$USER.log;
#   - samples the GPU with nvidia-smi every 200 ms into <out dir>/smi-<subcommand>-<stamp>.csv;
#   - redirects the triton / XDG / tmp caches under $S, TRITON_CACHE_AUTOTUNING=1.
set -uo pipefail
gpu=$1 out=$2 desc=$3 sub=$4
shift 4
: "${S:?set S}" "${PY:?set PY}" "${NSYS:?set NSYS}"
here=$(cd "$(dirname "$0")" && pwd)

export TRITON_CACHE_DIR=$S/.triton XDG_CACHE_HOME=$S/.cache TMPDIR=$S/tmp
export TRITON_CACHE_AUTOTUNING=1 PYTHONDONTWRITEBYTECODE=1 NSYS
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu  # CUDA index = nvidia-smi / NVML index
mkdir -p "$TMPDIR" "$out/nsys"

uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep -F "$uuid" || true)
if [[ -n "$busy" ]]; then
  echo "GPU$gpu ($uuid) is busy: $busy" >&2
  exit 75
fi

stamp=$(date +%Y%m%d-%H%M%S)
nvidia-smi -i "$gpu" --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,noheader,nounits -lms 200 > "$out/smi-$sub-$stamp.csv" &
sampler=$!

log=""
[[ $(hostname) == en-ec-zhang-x4 ]] && log=/scratch/users/gpu-use-log/$USER.log
[[ -n $log ]] && echo "$(date '+%m/%d/%Y - %H:%M:%S') | $USER | GPU$gpu | Start | $desc" >> "$log"
if [[ $sub == time ]]; then
  # Only the cudaProfilerStart/Stop ranges (the 5 replays of each chain) are traced; each range
  # becomes its own report time-<stamp>.<k>.nsys-rep (capture-range-end=repeat).
  "$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node --capture-range=cudaProfilerApi \
    --capture-range-end=repeat --sample=none --cpuctxsw=none -o "$out/nsys/time-$stamp" \
    "$PY" "$here/gpu_bench.py" time --out "$out" "$@"
else
  "$PY" "$here/gpu_bench.py" "$sub" --out "$out" "$@"
fi
rc=$?
[[ -n $log ]] && echo "$(date '+%m/%d/%Y - %H:%M:%S') | $USER | GPU$gpu | End   | $desc (rc=$rc)" >> "$log"
kill "$sampler"
exit $rc
