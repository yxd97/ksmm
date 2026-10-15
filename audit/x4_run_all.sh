#!/usr/bin/env bash
# The T4 measurement sequence on zhang-x4, one GPU, steps in this order:
#   checkpoint  E view of the 10 sheet chains, old harness call vs. this harness (3 rounds)
#   measure     E / G_cold / G_warm / h / numerics; ksmm and cublas at batch 1..4096, cusparse at batch 1
#   nsys        D: kernels per chain at batch 1 (E, GC, GW modes, 20 reps), parsed to CSV
#   ncu         per-kernel metrics for selected chains (GC mode, application replay, no ncu cache flush)
#
# Usage: x4_run_all.sh <physical GPU> <output dir> [step ...]
# The datasets' chains.csv files are copied from operations/datasets/<name>/ to $A/datasets/<name>/.
set -euo pipefail
gpu=$1 O=$2
shift 2
steps=${*:-checkpoint measure nsys ncu}

A=/scratch/users/phd/yd383/v41-audit
PY=/scratch/users/phd/yd383/venv-cu126/bin/python
NSYS=/usr/local/cuda-12.6/bin/nsys
NCU=/usr/local/cuda-12.6/bin/ncu
HERE=$(cd "$(dirname "$0")" && pwd)
H=$HERE/a100_audit.py
RUN=$HERE/x4_gpu_run.sh
D=$A/datasets
BATCHES=1,2,4,8,16,32,64,128,256,512,1024,2048,4096
mkdir -p "$O"

ncu_run() {  # <name> <dataset> <batch> <impls> <chains>
  local name=$1 ds=$2 batch=$3 impls=$4 chains=$5
  "$RUN" "$gpu" "$O/clk-ncu-$name.csv" "v41-audit T4 ncu $name" \
    "$NCU" --replay-mode application --cache-control none --clock-control none \
    --nvtx --nvtx-include "prof/" --metrics "$("$PY" "$H" ncu-metrics)" -f -o "$O/ncu-$name" \
    "$PY" "$H" profile --chains "$D/$ds/chains.csv" --dataset "$ds" --only "$chains" --impls "$impls" \
    --batch "$batch" --modes GC --reps 1 --range-name prof > "$O/ncu-$name.log" 2>&1
  "$NCU" --import "$O/ncu-$name.ncu-rep" --csv --page raw --print-units base > "$O/ncu-raw-$name.csv"
}

for step in $steps; do
  case $step in
    checkpoint)
      "$RUN" "$gpu" "$O/clk-checkpoint.csv" "v41-audit T4 checkpoint (E view, 10 sheet chains)" \
        "$PY" "$H" checkpoint --chains "$D/v3-reference-data/chains.csv" --dataset v3-reference-data \
        --sheet "$D/v3-reference-data/sheet-a100.csv" --rounds 3 --out "$O/checkpoint.jsonl"
      ;;
    measure)
      for ds in v3-reference-data v1; do
        "$RUN" "$gpu" "$O/clk-measure-$ds.csv" "v41-audit T4 measure $ds (3 impls, batch sweep)" \
          "$PY" "$H" measure --chains "$D/$ds/chains.csv" --dataset "$ds" --out "$O/measure-$ds.jsonl" \
          --batches-ksmm "$BATCHES" --batches-cublas "$BATCHES" --batches-cusparse 1
      done
      ;;
    nsys)
      for ds in v3-reference-data v1; do
        "$RUN" "$gpu" "$O/clk-nsys-$ds.csv" "v41-audit T4 nsys $ds (batch 1)" \
          "$NSYS" profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none \
          --force-overwrite=true -o "$O/nsys-$ds" \
          "$PY" "$H" profile --chains "$D/$ds/chains.csv" --dataset "$ds" --batch 1 --modes E,GC,GW --reps 20 \
          > "$O/nsys-$ds.log" 2>&1
        "$NSYS" export --type sqlite --force-overwrite=true -o "$O/nsys-$ds.sqlite" "$O/nsys-$ds.nsys-rep" > /dev/null
        "$PY" "$H" nsys-parse --sqlite "$O/nsys-$ds.sqlite" --out "$O/nsys-$ds.csv"
      done
      ;;
    ncu)
      ncu_run ref-b1 v3-reference-data 1 ksmm,cublas,cusparse debut_1,debut_2,debut_5,bfly_1,bfly_4
      ncu_run v1-b1 v1 1 ksmm,cublas,cusparse \
        debut_lenet_fc1_128x400,debut_vgg_512x4608,butterfly_512,butterfly_8192,synthetic_radix4_4096
      for b in 16 256 4096; do
        ncu_run "ref-b$b" v3-reference-data "$b" ksmm,cublas bfly_4,debut_5
      done
      "$PY" "$H" ncu-parse --csv "$O"/ncu-raw-*.csv --out "$O/ncu-kernels.csv"
      ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
