"""Publish the T7 A100 runs of dataset scale1: a100-ksmm and a100-cublas at batch 1.

    python3 audit/publish_t7.py --in <copy of zhang-x4 $A/runs/t7> --ops <operations> --commit <ksmm commit>

Standard library only (runs on zhang-22). Inputs in --in, written by the T7 commands
on zhang-x4 (they are listed in the runs' description, RUN_CMDS below):
  measure-scale1.jsonl, nsys-scale1.csv, clk-measure-scale1.csv, clk-nsys-scale1.csv
Outputs:
  <ops>/datasets/scale1/runs/a100-<impl>/{run.json,results.csv}
  <ops>/tasks/<task>/data/scale1-a100-nsys.csv   (nsys per chain, impl and timing view)
Same timing views, numerics and columns as publish.py (T4), without ncu and cuSPARSE.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from publish import (COLUMNS, IMPL_CONFIG, NUMERICS, TIMING, TITLES, chain_order, clock_summary,
                     local_minute, r, read_jsonl, write_csv)

DATASET = "scale1"
RESULT_COLS = ("time_us", "e_p20", "e_p50", "e_p80", "g_warm_us", "d_us", "kernels", "gc_p50",
               "d_span_us", "d_eager_us", "h_us", "rel_l2", "note")
IMPLS = ("ksmm", "cublas")
# scale1 chains with the same shapes as chains T4 measured (different names, so other weights).
OVERLAP = {
    **{f"r2t_{2 ** e}": ("v3-reference-data", f"bfly_{e - 8}") for e in range(9, 13)},
    **{f"r2v_{2 ** e}": ("v1", f"butterfly_{2 ** e}") for e in range(9, 14)},
    "r4_4096": ("v1", "synthetic_radix4_4096"),
}
RUN_CMDS = r"""R=/scratch/users/phd/yd383/kroissant-project; A=/scratch/users/phd/yd383/v41-audit; X=zhang-x4.ece.cornell.edu
rsync -a $R/ksmm/ $X:$A/ksmm/                                     # ksmm 的 audit 分支
rsync -a $R/operations/datasets/scale1/chains.csv $X:$A/datasets/scale1/
# 在 zhang-x4 上，物理 GPU 1：
PY=/scratch/users/phd/yd383/venv-cu126/bin/python; NSYS=/usr/local/cuda-12.6/bin/nsys
H=$A/ksmm/audit/a100_audit.py; RUN=$A/ksmm/audit/x4_gpu_run.sh; D=$A/datasets; O=$A/runs/t7
$RUN 1 $O/clk-measure-scale1.csv "v41-audit T7 measure scale1 (ksmm, cublas; batch 1; 27 chains up to 2^16)" \
  $PY $H measure --chains $D/scale1/chains.csv --dataset scale1 --out $O/measure-scale1.jsonl --impls ksmm,cublas
$RUN 1 $O/clk-nsys-scale1.csv "v41-audit T7 nsys scale1 (ksmm, cublas; batch 1)" \
  $NSYS profile --trace=cuda,nvtx --cuda-graph-trace=node --sample=none --cpuctxsw=none --force-overwrite=true \
  -o $O/nsys-scale1 $PY $H profile --chains $D/scale1/chains.csv --dataset scale1 --impls ksmm,cublas \
  --batch 1 --modes E,GC,GW --reps 20 > $O/nsys-scale1.log 2>&1
$NSYS export --type sqlite --force-overwrite=true -o $O/nsys-scale1.sqlite $O/nsys-scale1.nsys-rep > /dev/null
$PY $H nsys-parse --sqlite $O/nsys-scale1.sqlite --out $O/nsys-scale1.csv
# 回到 zhang-22：
rsync -a --exclude '*.nsys-rep' --exclude '*.sqlite' $X:$A/runs/t7/ $R/scratch/runs/T7-a100/
python3 $R/ksmm/audit/publish_t7.py --in $R/scratch/runs/T7-a100 --ops $R/operations --commit <ksmm 提交>"""


def nsys_summary(path: Path) -> tuple[list[dict], dict]:
    """Aggregate the 20 reps per (mode, chain, impl), as publish.py does for T4."""
    groups = defaultdict(list)
    with open(path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            groups[(row["mode"], row["chain"], row["impl"])].append(row)
    rows, by_key = [], {}
    for (mode, chain, impl), reps in groups.items():
        d = [float(x["d_us"]) for x in reps]
        span = [float(x["span_us"]) for x in reps]
        per_kernel = list(zip(*[[float(v) for v in x["kernel_us"].split()] for x in reps]))
        kernels = sorted({int(x["kernels"]) for x in reps})
        rows.append({
            "mode": mode, "dataset": DATASET, "chain": chain, "impl": impl, "batch": reps[0]["batch"],
            "reps": len(reps), "kernels": " ".join(map(str, kernels)), "other_ops": reps[0]["other_ops"],
            "d_us_median": r(statistics.median(d)), "d_us_min": r(min(d)), "d_us_max": r(max(d)),
            "span_us_median": r(statistics.median(span)),
            "gaps_us_median": r(statistics.median(float(x["gaps_us"]) for x in reps)),
            "kernel_us_median": " ".join(f"{statistics.median(k):.3f}" for k in per_kernel),
            "kernel_names": reps[0]["kernel_names"], "grids": reps[0]["grids"],
            "blocks": reps[0]["blocks"], "api": reps[0]["api"],
        })
        by_key[(mode, chain, impl)] = (statistics.median(d), kernels, statistics.median(span))
    return rows, by_key


def overlap_note(ops: Path, impl: str, ours: dict) -> str:
    """G_cold and E p80 of scale1 chains against T4's runs of the same shapes."""
    t4 = {}
    for ds in {ds for ds, _ in OVERLAP.values()}:
        with open(ops / "datasets" / ds / "runs" / f"a100-{impl}" / "results.csv", encoding="utf-8") as fh:
            t4[ds] = {row["chain"]: row for row in csv.DictReader(fh)}
    gc, e80, pairs = [], [], []
    for name, (ds, old) in OVERLAP.items():
        a, b = ours.get(name), t4[ds].get(old)
        if not a or not b or not a.get("time_us") or not b.get("time_us"):
            continue
        gc.append((float(a["time_us"]) / float(b["time_us"]) - 1) * 100)
        e80.append(((float(a["e_p80"]) / float(b["e_p80"]) - 1) * 100, name, a, b))
        pairs.append(f"`{name}`↔{ds} `{old}`")
    if not pairs:
        return ""
    worst = max(e80, key=lambda x: abs(x[0]))
    return (f"- **与 T4 的同形状链核对**（{len(pairs)} 对：{'、'.join(pairs)}；链名不同，所以权重和输入不同）："
            f"G_cold 相差 {min(gc):+.1f}% 到 {max(gc):+.1f}%，E 的 p80 相差 {min(x[0] for x in e80):+.1f}% 到 "
            f"{max(x[0] for x in e80):+.1f}%（本运行 ÷ T4 − 1）。E 含露在外面的主机发射时间，波动比 G_cold 大："
            f"差得最多的 `{worst[1]}`，本运行 E 的 p20/p50/p80 是 {worst[2]['e_p20']}/{worst[2]['e_p50']}/{worst[2]['e_p80']} µs，"
            f"T4 是 {worst[3]['e_p20']}/{worst[3]['e_p50']}/{worst[3]['e_p80']} µs。\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--ops", required=True, help="operations repository root")
    p.add_argument("--task", default="2026-10-14-v41-baseline-audit")
    p.add_argument("--commit", required=True, help="ksmm commit to record in run.json")
    args = p.parse_args()
    inp, ops = Path(args.inp), Path(args.ops)

    nsys_rows, nsys_d = nsys_summary(inp / f"nsys-{DATASET}.csv")
    write_csv(ops / "tasks" / args.task / "data" / f"{DATASET}-a100-nsys.csv", nsys_rows)
    clk = clock_summary(inp / f"clk-measure-{DATASET}.csv")
    clk_nsys = clock_summary(inp / f"clk-nsys-{DATASET}.csv")

    rows = read_jsonl(inp / f"measure-{DATASET}.jsonl")
    prov_all = [x for x in rows if x["kind"] == "provenance"]
    prov = prov_all[-1]
    latest = {(x["chain"], x["impl"]): x for x in rows if x["kind"] == "point" and x["batch"] == 1}
    order = chain_order(ops, DATASET)
    for impl in IMPLS:
        out = []
        for chain in order:
            x = latest.get((chain, impl))
            if x is None:
                out.append({"chain": chain, "check": "未运行", **{k: "" for k in RESULT_COLS}})
                continue
            d, kernels, span = nsys_d.get(("GC", chain, impl), (None, [], None))
            de = nsys_d.get(("E", chain, impl), (None, [], None))[0]
            out.append({
                "chain": chain, "check": x["check"], "time_us": r(x.get("gc_mean")),
                "e_p20": r(x.get("e_p20")), "e_p50": r(x.get("e_p50")), "e_p80": r(x.get("e_p80")),
                "g_warm_us": r(x.get("gw_mean")), "d_us": r(d), "kernels": " ".join(map(str, kernels)),
                "gc_p50": r(x.get("gc_p50")), "d_span_us": r(span), "d_eager_us": r(de),
                "h_us": r(x.get("h_us")), "rel_l2": r(x.get("eager_rel_l2"), 6), "note": x.get("error", ""),
            })
        run_dir = ops / "datasets" / DATASET / "runs" / f"a100-{impl}"
        write_csv(run_dir / "results.csv", out)
        pts = list(latest.values())
        low = [x for x in pts if x["sm_mhz_end"] < 1410]
        low_note = (f"每个点测完时用 NVML 读一次 SM 时钟：{len(pts)} 个点里 {len(low)} 个低于 1410 MHz"
                    + (f"（{min(x['sm_mhz_end'] for x in low)}–{max(x['sm_mhz_end'] for x in low)} MHz："
                       + "、".join(f"{x['chain']} {x['impl']}" for x in low) + "）" if low else "")
                    + "。降频原因位 0x4 是软件功耗上限（SW power cap）")
        impl_text = IMPL_CONFIG[impl].replace("task data 的 a100-nsys.csv", "task data 的 scale1-a100-nsys.csv")
        cols = {k: v for k, v in COLUMNS.items() if not k.startswith("ncu_") and k != "csr_nnz"}
        run = {
            "format": "dataset-run/1",
            "title": TITLES[impl],
            "task": args.task,
            "time": local_minute(prov_all[0]["time_utc"]),
            "commits": {"ksmm": args.commit},
            "config": {
                "host": f"{prov['host']}（zhang-x4）",
                "gpu": f"{prov['gpu_name']}，物理 GPU {prov['gpu_index_nvml']}（CUDA_VISIBLE_DEVICES={prov['cuda_visible_devices']}），"
                       f"PCI {prov['gpu_pci_bus_id']}，{prov['gpu_uuid']}，VBIOS {prov['vbios']}，{prov['sm_count']} 个 SM",
                "driver": prov["driver"],
                "software": f"Python {prov['python']}（{prov['python_executable']}）；torch {prov['torch']}（CUDA {prov['torch_cuda']}）；"
                            f"triton {prov['triton']}；nvidia-cublas-cu12 {prov['nvidia_cublas_cu12']}；"
                            f"nvidia-cuda-runtime-cu12 {prov['nvidia_cuda_runtime_cu12']}；nsys 2024.5.1.113",
                "clocks": (f"不锁频（没有 root；applications clocks 是默认的 SM {prov['sm_clock_mhz_applications']} MHz、"
                           f"显存 {prov['mem_clock_mhz_applications']} MHz）。measure 期间 nvidia-smi 每 200 ms 采样，GPU 忙的 "
                           f"{clk.get('busy_samples')} 个样本：SM {clk.get('sm_mhz_min')}–{clk.get('sm_mhz_max')} MHz（中位数 "
                           f"{clk.get('sm_mhz_median')}，{100 * clk.get('share_at_1410', 0):.1f}% 的样本在 1410），显存 {clk.get('mem_mhz')} MHz，"
                           f"降频原因位 {clk.get('throttle_reasons')}，最高 {clk.get('temp_c_max')} °C、{clk.get('power_w_max')} W。"
                           "忙的样本里包括构建参考矩阵（fp32 稠密连乘）的时间，那段时间不计时。"
                           f"nsys 期间：SM {clk_nsys.get('sm_mhz_min')}–{clk_nsys.get('sm_mhz_max')} MHz。{low_note}"),
                "dtype": "fp16 输入、权重、输出；乘加在 fp32 里累加（KSMM 的 tl.dot、cuBLAS 的 CUBLAS_COMPUTE_32F）",
                "batch": 1,
                "layout": "特征维在前：输入 X 是 (M_in, batch)，输出 Y 是 (N_out, batch)；第 0 级最先作用在输入上",
                "timing": TIMING,
                "flush": f"{prov['flush_bytes']} 字节的 int32 缓冲 zero_()，一次约 {round(prov['flush_us_median'], 1)} µs",
                "numerics": NUMERICS,
                "impl": impl_text,
                "measured_at_commit": prov["ksmm_commit"],
                "measured_tree_dirty": prov["ksmm_dirty"],
                "loadavg_start": prov["loadavg_start"],
            },
            "description": (
                f"zhang-x4 上实测（任务 2026-10-14-v41-baseline-audit，步骤 T7，数据集 scale1）。实现：{impl_text}。\n\n"
                "- **`time_us` 是 G_cold**（CUDA graph 回放、窗口外清 L2、不含主机发射），是和 V3、V4.1 比对用的数。"
                "口径、harness 与 T4 的 `a100-*` 运行相同。\n"
                "- E 和结果表 sheet-a100 是同一个口径。E − G_cold 是主机发射开销里露在外面的部分，h_us 是主机发一次整条链的总时间。\n"
                "- 只测 batch = 1；没跑 ncu 和 cuSPARSE（T7 不要求）。nsys 每条链每种口径的明细在任务目录 `data/scale1-a100-nsys.csv`。\n"
                + overlap_note(ops, impl, {row["chain"]: row for row in out}) +
                f"\n复现（测量时 zhang-x4 上的 ksmm 在提交 `{prov['ksmm_commit'][:8]}`，工作树干净；`commits` 里的 `{args.commit[:8]}` "
                "比它多本发布脚本 `audit/publish_t7.py` 和 T4 审阅后的几个提交（publish.py 的说明文字、host_cmp.py，以及 "
                "a100_audit.py 里判断工作树是否干净的一行）；测量和计时的代码没变）：\n\n```bash\n" + RUN_CMDS + "\n```\n\n"
                "`x4_gpu_run.sh` 在占卡前检查有没有别人的进程，前后在 gpu-use-log 登记，运行期间用 nvidia-smi 每 200 ms 采样时钟。"
                "原始 nsys 报告留在 zhang-x4 的 `$A/runs/t7/`。"),
            "columns": cols,
        }
        (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print("published", len(order), "chains x", len(IMPLS), "impls,", len(nsys_rows), "nsys rows")


if __name__ == "__main__":
    main()
