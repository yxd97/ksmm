#!/usr/bin/env bash
# The T8 (GPU bottleneck) supplementary measurements on zhang-x4, one GPU, in this order:
#   floor        empty-kernel graphs, one-CTA KSMM graphs, CTA-count sweep (t8_probe.py floor)
#   cfg-stage    KSMM Triton configs outside T4's autotune space on single stage shapes
#   cfg-chain    whole chains: default (T4 autotune), config 256/1/3 forced on every stage (best on single
#                stages at batch 256 / 4096), autotune over the extended space
#   ncu-scale    T4's ncu metrics on 2^14 / 2^16 scale1 chains, batch 1 (application replay, GC mode)
#   ncu-batch    T4's ncu metrics on bfly_4 at batch 64 and 1024 (fills T4's 1 / 16 / 256 / 4096)
#   ncu-limiter  ncu sections (pipes, warp stalls, scheduler) on a few kernels; kernel replay, L2 flushed
#   ncu-bw       T4's ncu metrics on the 19 KSMM points the byte estimate called bandwidth bound (G3-r1 B3)
#   ncu-best     T4's ncu metrics on bfly_4 batch 4096 with config 256/1/3 (application replay; G3-r1 S4)
#   flushcmp     G_cold with T4's zero_() flush vs a read-only flush: every published cuBLAS chain at batch 1
#                (v3-reference-data, v1, scale1) and KSMM controls (decision 2026-10-15 03:21 EDT)
#
# Usage: x4_run_t8.sh <physical GPU> [step ...]
# Triton cache: a copy of T4's cache ($A/.triton -> $A/.triton-t8), so the default autotune picks are T4's
# and T4's / T7's cache is not written to.
set -euo pipefail
gpu=$1
shift
steps=${*:-floor cfg-stage cfg-chain ncu-scale ncu-batch ncu-limiter}

A=/scratch/users/phd/yd383/v41-audit
PY=/scratch/users/phd/yd383/venv-cu126/bin/python
NCU=/usr/local/cuda-12.6/bin/ncu
HERE=$(cd "$(dirname "$0")" && pwd)
H=$HERE/a100_audit.py
P=$HERE/t8_probe.py
RUN=$HERE/x4_gpu_run.sh
D=$A/datasets
O=$A/runs/t8
mkdir -p "$O"
export TRITON_CACHE_DIR=$A/.triton-t8
[[ -d $TRITON_CACHE_DIR ]] || cp -a "$A/.triton" "$TRITON_CACHE_DIR"

ncu_metrics() {  # <name> <log text> <workload...>: T4's metric list, application replay
  local name=$1 text=$2
  shift 2
  "$RUN" "$gpu" "$O/clk-ncu-$name.csv" "v41-audit T8 ncu $text" \
    "$NCU" --replay-mode application --cache-control none --clock-control none \
    --nvtx --nvtx-include "prof/" --metrics "$("$PY" "$H" ncu-metrics)" -f -o "$O/ncu-$name" \
    "$@" > "$O/ncu-$name.log" 2>&1
  "$NCU" --import "$O/ncu-$name.ncu-rep" --csv --page raw --print-units base > "$O/ncu-raw-$name.csv"
}

ncu_sections() {  # <name> <log text> <launch count> <workload...>: limiter sections, kernel replay
  local name=$1 text=$2 count=$3
  shift 3
  "$RUN" "$gpu" "$O/clk-lim-$name.csv" "v41-audit T8 ncu limiter $text" \
    "$NCU" --replay-mode kernel --cache-control all --clock-control none \
    --nvtx --nvtx-include "prof/" -c "$count" \
    --section SpeedOfLight --section ComputeWorkloadAnalysis --section WarpStateStats \
    --section SchedulerStats --section Occupancy --section LaunchStats --section InstructionStats \
    --section MemoryWorkloadAnalysis -f -o "$O/lim-$name" \
    "$@" > "$O/lim-$name.log" 2>&1
  "$NCU" --import "$O/lim-$name.ncu-rep" --csv --page raw --print-units base > "$O/lim-raw-$name.csv"
}

for step in $steps; do
  case $step in
    floor)
      "$RUN" "$gpu" "$O/clk-floor.csv" "v41-audit T8 floor probe (empty kernels, 1-CTA KSMM, CTA sweep)" \
        "$PY" "$P" floor --out "$O/floor.jsonl"
      ;;
    cfg-stage)
      for spec in "64,2,2,32 1,256,4096" "2048,2,2,16 1" "16,4,4,64 1,4096" "256,8,9,1 1,4096"; do
        set -- $spec
        "$RUN" "$gpu" "$O/clk-cfg-stage-${1//,/x}.csv" "v41-audit T8 KSMM configs on stage $1 (batch $2)" \
          "$PY" "$P" configs --stage "$1" --batches "$2" --out "$O/cfg-stage.jsonl"
      done
      ;;
    cfg-chain)
      "$RUN" "$gpu" "$O/clk-cfg-chain-ref.csv" "v41-audit T8 KSMM extended autotune, bfly_4 b1/256/4096" \
        "$PY" "$P" configs --chains "$D/v3-reference-data/chains.csv" --only bfly_4 --batches 1,256,4096 \
        --configs 256/1/3 --extended-autotune --out "$O/cfg-chain.jsonl"
      "$RUN" "$gpu" "$O/clk-cfg-chain-ref2.csv" "v41-audit T8 KSMM extended autotune, debut_5 b1/4096" \
        "$PY" "$P" configs --chains "$D/v3-reference-data/chains.csv" --only debut_5 --batches 1,4096 \
        --configs 256/1/3 --extended-autotune --out "$O/cfg-chain.jsonl"
      "$RUN" "$gpu" "$O/clk-cfg-chain-scale.csv" "v41-audit T8 KSMM extended autotune, r2v_65536 b1" \
        "$PY" "$P" configs --chains "$D/scale1/chains.csv" --only r2v_65536 --batches 1 \
        --configs 256/1/3 --extended-autotune --out "$O/cfg-chain.jsonl"
      ;;
    ncu-scale)
      ncu_metrics scale-b1 "scale1 2^14/2^16 chains, ksmm+cublas, batch 1" \
        "$PY" "$P" profile --chains "$D/scale1/chains.csv" --dataset scale1 --impls ksmm,cublas --batch 1 \
        --only r2v_16384,r2v_65536,r2t_65536,r4_65536,debut_rect_65536
      ;;
    ncu-batch)
      for b in 64 1024; do
        ncu_metrics "ref-b$b" "bfly_4 ksmm+cublas batch $b" \
          "$PY" "$H" profile --chains "$D/v3-reference-data/chains.csv" --dataset v3-reference-data \
          --only bfly_4 --impls ksmm,cublas --batch "$b" --modes GC --reps 1 --range-name prof
      done
      ;;
    ncu-limiter)
      for b in 1 4096; do
        ncu_sections "bfly4-ksmm-b$b" "bfly_4 ksmm batch $b (3 kernels)" 3 \
          "$PY" "$P" profile --chains "$D/v3-reference-data/chains.csv" --dataset v3-reference-data \
          --only bfly_4 --impls ksmm --batch "$b"
      done
      ncu_sections "r2v65536-ksmm-b1" "r2v_65536 ksmm batch 1 (3 kernels)" 3 \
        "$PY" "$P" profile --chains "$D/scale1/chains.csv" --dataset scale1 --only r2v_65536 --impls ksmm --batch 1
      ncu_sections "bfly4-cublas-b4096" "bfly_4 cublas batch 4096" 1 \
        "$PY" "$P" profile --chains "$D/v3-reference-data/chains.csv" --dataset v3-reference-data \
        --only bfly_4 --impls cublas --batch 4096
      if [[ -n ${T8_BEST_4096:-} ]]; then
        ncu_sections "bfly4-ksmm-b4096-best" "bfly_4 ksmm batch 4096 config $T8_BEST_4096 (3 kernels)" 3 \
          "$PY" "$P" profile --chains "$D/v3-reference-data/chains.csv" --dataset v3-reference-data \
          --only bfly_4 --impls ksmm --batch 4096 --force-config "$T8_BEST_4096"
      fi
      ;;
    ncu-bw)
      # (batch, dataset, chains): the 19 KSMM points with byte-estimate bw_share >= 30 % in data/gpu-regimes.csv
      V1C="synthetic_radix8_4096,synthetic_depth5_4096,synthetic_rect_2048x4096,synthetic_rect_4096x2048"
      for spec in "256 v1 synthetic_radix8_4096" "512 v1 synthetic_radix8_4096,synthetic_depth5_4096" \
                  "1024 v1 $V1C" "2048 v1 $V1C,debut_vgg_256x2304" "4096 v1 $V1C,debut_vgg_256x2304" \
                  "2048 v3-reference-data debut_4" "4096 v3-reference-data debut_4"; do
        set -- $spec
        ncu_metrics "bw-$2-b$1" "KSMM 'bandwidth' points $2 batch $1 ($3)" \
          "$PY" "$H" profile --chains "$D/$2/chains.csv" --dataset "$2" --only "$3" --impls ksmm \
          --batch "$1" --modes GC --reps 1 --range-name prof
      done
      ;;
    ncu-best)
      ncu_metrics "best-b4096" "bfly_4 ksmm batch 4096 config 256/1/3" \
        "$PY" "$P" profile --chains "$D/v3-reference-data/chains.csv" --dataset v3-reference-data-cfg256 \
        --only bfly_4 --impls ksmm --batch 4096 --force-config 256/1/3
      ;;
    flushcmp)
      for ds in v3-reference-data v1 scale1; do
        "$RUN" "$gpu" "$O/clk-flushcmp-$ds.csv" "v41-audit T8 G_cold zero vs read-only L2 flush, $ds cublas (+ksmm controls)" \
          "$PY" "$P" flushcmp --chains "$D/$ds/chains.csv" --dataset "$ds" --impls cublas --out "$O/flushcmp.jsonl"
      done
      "$RUN" "$gpu" "$O/clk-flushcmp-ksmm.csv" "v41-audit T8 G_cold zero vs read-only L2 flush, ksmm controls" \
        "$PY" "$P" flushcmp --chains "$D/v3-reference-data/chains.csv" --dataset v3-reference-data \
        --only debut_1,debut_5,bfly_1,bfly_4 --impls ksmm --out "$O/flushcmp.jsonl"
      "$RUN" "$gpu" "$O/clk-flushcmp-ksmm2.csv" "v41-audit T8 G_cold zero vs read-only L2 flush, ksmm controls scale1" \
        "$PY" "$P" flushcmp --chains "$D/scale1/chains.csv" --dataset scale1 \
        --only r2v_8192,r2v_65536,r4_65536,debut_rect_65536 --impls ksmm --out "$O/flushcmp.jsonl"
      ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
