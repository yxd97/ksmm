"""Old harness call vs. this harness's ksmm call (batch 1), in one process on zhang-x4.

For each chain: host issue time h (a100_audit.host_issue_us, median of 5) and the E view's
p80 (do_bench, same parameters as the checkpoint) of both calls. Explains why the old call's
E is higher than ours. Appends one CSV row per chain to --out. Run with x4_gpu_run.sh, e.g.

  x4_gpu_run.sh 0 clk.csv "host_cmp" $PY audit/host_cmp.py --out host-cmp.csv --run 0
"""
import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import a100_audit as h  # noqa: E402

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--out", required=True)
p.add_argument("--run", type=int, required=True, help="run index written to the CSV")
p.add_argument("--chains", default="/scratch/users/phd/yd383/v41-audit/datasets/v3-reference-data/chains.csv")
p.add_argument("--legacy-root", default="/scratch/users/phd/yd383/do-bench-harness")
p.add_argument("--only", default="debut_1,debut_5,bfly_1,bfly_4")
args = p.parse_args()

torch, triton, ks_forward, ks_kernel = h.gpu_stack()
import triton.testing  # noqa: E402

sys.path.insert(0, args.legacy_root)
from a100bench.runner import _make_ksmm_call  # noqa: E402  (read-only use of the old harness)

chains = {c.name: c for c in h.read_chains(Path(args.chains))}
new = not os.path.exists(args.out)
with open(args.out, "a", encoding="utf-8", newline="") as stream:
    writer = csv.writer(stream, lineterminator="\n")
    if new:
        writer.writerow(["run", "chain", "stages", "h_legacy_us", "h_ours_us", "h_diff_us",
                         "e_p80_legacy_us", "e_p80_ours_us", "e_p80_diff_us", "loadavg1", "time_utc"])
    for name in args.only.split(","):
        c = chains[name]
        legacy, _ = _make_ksmm_call({"stages": [[a, b, cc, d, 1] for a, b, cc, d in c.stages], "batch": 1,
                                     "M": c.m_in}, torch, ks_forward)
        ours, _ = h.Built(torch, c).make_call(torch, ks_forward, "ksmm", h.make_input(torch, c, 1))
        hl = sorted(h.host_issue_us(torch, legacy) for _ in range(5))[2]
        ho = sorted(h.host_issue_us(torch, ours) for _ in range(5))[2]
        el = triton.testing.do_bench(legacy, warmup=h.WARMUP_MS, rep=h.REP_MS, quantiles=h.QUANTILES)[2] * 1000
        eo = triton.testing.do_bench(ours, warmup=h.WARMUP_MS, rep=h.REP_MS, quantiles=h.QUANTILES)[2] * 1000
        row = [args.run, name, len(c.stages), round(hl, 2), round(ho, 2), round(hl - ho, 2),
               round(el, 3), round(eo, 3), round(el - eo, 3), round(os.getloadavg()[0], 2), h.utc_now()]
        writer.writerow(row)
        stream.flush()
        print(*row, flush=True)
