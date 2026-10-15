#!/usr/bin/env bash
# Run one command on one zhang-x4 GPU, following the lab's GPU etiquette:
#   - refuse when another process is on that GPU;
#   - log Start / End lines in /scratch/users/gpu-use-log/yd383.log;
#   - sample SM / memory clocks with nvidia-smi every 200 ms while the command runs;
#   - redirect triton / XDG / tmp caches under the audit work directory.
#
# Usage: x4_gpu_run.sh <physical GPU index> <clock log path> "<log description>" <command...>
set -uo pipefail
gpu=$1 clock_log=$2 desc=$3
shift 3

A=/scratch/users/phd/yd383/v41-audit
LOG=/scratch/users/gpu-use-log/yd383.log
export TRITON_CACHE_DIR=$A/.triton XDG_CACHE_HOME=$A/.cache TMPDIR=$A/tmp
export TRITON_CACHE_AUTOTUNING=1 PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=$gpu

uuid=$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)
busy=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep -F "$uuid" || true)
if [[ -n "$busy" ]]; then
  echo "GPU$gpu ($uuid) is busy: $busy" >&2
  exit 75
fi

mkdir -p "$(dirname "$clock_log")"
nvidia-smi -i "$gpu" --query-gpu=timestamp,clocks.sm,clocks.mem,utilization.gpu,power.draw,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,noheader,nounits -lms 200 > "$clock_log" &
sampler=$!

echo "$(date '+%m/%d/%Y - %H:%M:%S') | yd383 | GPU$gpu | Start | $desc" >> "$LOG"
"$@"
rc=$?
echo "$(date '+%m/%d/%Y - %H:%M:%S') | yd383 | GPU$gpu | End   | $desc (rc=$rc)" >> "$LOG"
kill "$sampler"
exit $rc
