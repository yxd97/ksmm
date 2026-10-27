"""Write one operations dataset run (run.json, results.csv) from a gpu_bench.py output directory.

    python3 bench/publish.py --in <copy of the run's --out dir> --chains <ops>/datasets/<ds>/chains.csv \
        --run-dir <ops>/datasets/<ds>/runs/<run> --task <task id> --title "<title>" [--nsys-bias "<text>"]

Reads time.jsonl, timing.jsonl and, when present, power.jsonl and idle.jsonl (power columns are
written only when power.jsonl exists, i.e. for B1). Standard library only; runs on zhang-22.
Rows follow the order of chains.csv; a chain without a time row gets check 未运行.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import gpu_bench as gb

TIME_COLS = ["knl_us", "knl_p20", "knl_p80", "device_us", "device_p20", "device_p80", "dkl_us", "kernels", "rel_l2"]
POWER_COLS = ["p_idle_w", "p_work_w", "p_useful_w", "e_run_uj", "power_n_runs", "busy_pct", "power_spread_pct",
              "sm_mhz", "temp_c"]
COLUMNS = {
    "knl_us": "KNL（µs）：一个 run 里各级 kernel 的 GPU 执行时间之和，100 个 run 的中位数（nsys 逐节点追踪）",
    "knl_p20": "KNL 的 20 百分位（µs）",
    "knl_p80": "KNL 的 80 百分位（µs）",
    "device_us": "device 时间（µs）：上一个 run 最后一个 kernel 结束到本 run 最后一个 kernel 结束，95 个 run 的中位数（每次回放的第 1 个 run 不算）",
    "device_p20": "device 时间的 20 百分位（µs）",
    "device_p80": "device 时间的 80 百分位（µs）",
    "dkl_us": "DKL（µs）：每个 run 的 device 时间 − KNL，95 个值的中位数",
    "kernels": "一个 run 的 kernel 数（nsys 实测，等于级数）",
    "rel_l2": "相对 L2 误差 ‖y − y_ref‖₂ / ‖y_ref‖₂，eager 输出和 graph 最后一个 run 的输出取较大者",
    "p_idle_w": "待机功率（W）：同一张卡不跑程序，第 30 s 到第 90 s 的能量计数器差 ÷ 时长",
    "p_work_w": "工作功率（W）：功率 graph 连续回放 120 s，第 30 s 到第 90 s 的能量计数器差 ÷ 时长",
    "p_useful_w": "有用功率（W）= p_work_w − p_idle_w",
    "e_run_uj": "每个 run 的有用能量（µJ）= p_useful_w × 窗口时长 ÷ 窗口里完成的 run 数",
    "power_n_runs": "功率 graph 里的 run 数：从 20 起，试跑 5 s 的 busy_pct < 95% 就翻倍",
    "busy_pct": "GPU 忙碌比例（%）= 窗口里的 run 数 × device_us ÷ 窗口时长（两个 CUDA event 之间的 GPU 时间）",
    "power_spread_pct": "稳态核对（%）：窗口里每 100 ms 的功率采样按 5 s 分成 12 段取平均，（最大 − 最小）÷ 平均",
    "sm_mhz": "功率窗口里 SM 时钟的中位数（MHz，NVML 每 100 ms 采样）",
    "temp_c": "功率窗口里的最高温度（°C）",
    "note": "失败原因、换卡、偏差",
}
IMPL_TEXT = {
    "triton": ("KSMM-Triton：每级调用一次 ks_triton_forward_impl(x, K, (A,B,C,D), layout='BSL')，K 的形状 (A·D, C, B)。"
               "录 graph 前先 eager 调用一次，Triton autotune 在这时选配置（TRITON_CACHE_AUTOTUNING=1 存盘），"
               "选中的配置记在 time.jsonl 的 config 字段"),
}
TIMING_TEXT = (f"每条链录一个 CUDA graph，里面 {gb.N_RUNS} 个首尾相接的 run，run 之间不清 L2。热身回放 {gb.WARMUP_REPLAYS} 次，"
               f"再在 nsys 下回放 {gb.TRACED_REPLAYS} 次（--trace=cuda,nvtx --cuda-graph-trace=node "
               "--capture-range=cudaProfilerApi --capture-range-end=repeat，每条链的 5 次回放是一个 cudaProfilerStart/Stop 区间，"
               "每次回放一个 NVTX 区间 <链名>/r<k>）。kernel 按启动它的 cudaGraphLaunch（correlationId 相同）归到回放，"
               "回放内按 graph 节点顺序（graphNodeId）每 `kernels` 个切成一个 run。KNL 100 个样本，device 95 个")
POWER_TEXT = (f"待机：只用 pynvml、不建 CUDA 上下文，测 {gb.POWER_S:.0f} s。工作：功率 graph 与计时是同一条链、同样的 run，"
              f"run 数从 {gb.N_RUNS} 起，不开追踪试跑 {gb.TRIAL_S:.0f} s，busy_pct < {gb.BUSY_MIN_PCT:.0f}% 就翻倍，最多 "
              f"{gb.POWER_N_RUNS_MAX}；选定后连续回放 {gb.POWER_S:.0f} s，主机最多领先 GPU {gb.LEAD} 次回放。"
              f"能量取第 {gb.WINDOW[0]:.0f} s、第 {gb.WINDOW[1]:.0f} s 两次读 nvmlDeviceGetTotalEnergyConsumption 的差，"
              "除以两次读数之间的实际时长；每 100 ms 采功率、SM 时钟、温度、降频原因。功率阶段不在 nsys 下运行")


def r(x, nd=3):
    return "" if x is None or x != x else round(x, nd)


def local_minute(utc_iso: str) -> str:
    t = datetime.fromisoformat(utc_iso).astimezone(ZoneInfo("America/New_York"))
    return t.strftime("%Y-%m-%dT%H:%M%z")[:-2] + ":" + t.strftime("%z")[-2:]


def rows_of(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")] if path.exists() else []


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--chains", required=True, help="operations chains.csv of the dataset")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--title", required=True)
    p.add_argument("--nsys-bias", default="未测", help="nsys tracing bias measured in the smoke step (text)")
    args = p.parse_args()
    inp, run_dir = Path(args.inp), Path(args.run_dir)

    time_rows = rows_of(inp / "time.jsonl")
    provs = [x for x in time_rows if x["kind"] == "provenance"]
    t_of = {x["chain"]: x for x in time_rows if x["kind"] == "point"}
    tm_of = {x["chain"]: x for x in rows_of(inp / "timing.jsonl") if x["kind"] == "timing"}
    power_rows = rows_of(inp / "power.jsonl")
    with_power = bool(power_rows)
    p_of = {x["chain"]: x for x in power_rows if x["kind"] == "point"}
    idle = [x for x in rows_of(inp / "idle.jsonl") if x["kind"] == "idle"]
    prov = provs[0]
    with open(args.chains, encoding="utf-8", newline="") as stream:
        order = [row["chain"] for row in csv.DictReader(stream)]

    out: list[dict] = []
    impls: set[str] = set()
    batches: set[int] = set()
    for chain in order:
        row = {"chain": chain, "check": "未运行", **{c: "" for c in TIME_COLS}}
        if with_power:
            row.update({c: "" for c in POWER_COLS})
        notes = []
        t = t_of.get(chain)
        if t is not None:
            impls.add(t["impl"])
            batches.add(t["batch"])
            row["check"] = t["check"]
            row["rel_l2"] = r(t.get("rel_l2"), 6)
            if t.get("error"):
                notes.append(t["error"])
            if t["gpu_uuid"] != prov["gpu_uuid"]:
                notes.append(f"换卡：{t['gpu_uuid']}")
            tm = tm_of.get(chain)
            if tm is None:
                notes.append("没有 nsys 计时")
            else:
                row.update({c: r(tm.get(c)) for c in TIME_COLS if c not in ("kernels", "rel_l2")})
                row["kernels"] = tm["kernels"] if tm["kernels"] is not None else ""
                notes += tm["problems"]
                if tm["knl_gt_device"]:
                    notes.append(f"{tm['knl_gt_device']}/{tm['device_n']} 个 run 的 KNL > device"
                                 f"（相邻 kernel 的 GPU 时间戳重叠 {tm['overlaps']} 处）")
        pw = p_of.get(chain)
        if with_power and pw is not None:
            if pw.get("error"):
                notes.append(f"功率：{pw['error']}")
            else:
                same = [x for x in idle if x["gpu_uuid"] == pw["gpu_uuid"]]
                p_idle = same[-1]["p_idle_w"] if same else None
                p_useful = pw["p_work_w"] - p_idle if p_idle is not None else None
                row.update(p_idle_w=r(p_idle), p_work_w=r(pw["p_work_w"]), p_useful_w=r(p_useful),
                           e_run_uj=r(p_useful * pw["window_s"] / pw["window_runs"] * 1e6, 6) if p_useful is not None else "",
                           power_n_runs=pw["power_n_runs"], busy_pct=r(pw["busy_pct"], 2),
                           power_spread_pct=r(pw["power_spread_pct"], 2), sm_mhz=r(pw["sm_mhz"], 1), temp_c=pw["temp_c"])
                if p_idle is None:
                    notes.append("这张卡没有待机功率")
                if pw["clock_event_reasons"] != "0x0":
                    notes.append(f"功率窗口的降频原因位 {pw['clock_event_reasons']}")
        row["note"] = "；".join(notes)
        out.append(row)

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "results.csv", "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)

    sessions = [{k: x.get(k) for k in ("time_utc", "argv", "ksmm_commit", "ksmm_dirty", "gpu_uuid", "loadavg_start",
                                       "gpu_processes_start")} for x in provs + [x for x in power_rows + rows_of(inp / "idle.jsonl")
                                                                                 if x["kind"] == "provenance"]]
    used_idle = sorted({(x["gpu_uuid"], round(x["p_idle_w"], 3), x["time_utc"]) for x in idle})
    impl = "、".join(sorted(impls))
    config = {
        "host": prov["host"],
        "gpu": (f"{prov['gpu_name']}，物理 GPU {prov['gpu_index_nvml']}（CUDA_VISIBLE_DEVICES={prov['cuda_visible_devices']}，"
                f"CUDA_DEVICE_ORDER=PCI_BUS_ID），PCI {prov['gpu_pci_bus_id']}，{prov['gpu_uuid']}，VBIOS {prov['vbios']}，"
                f"{prov['sm_count']} 个 SM"),
        "driver": prov["driver"],
        "software": (f"Python {prov['python']}（{prov['python_executable']}）；torch {prov['torch']}（CUDA {prov['torch_cuda']}，"
                     f"nvidia-cuda-runtime-cu12 {prov['cuda_runtime']}）；triton {prov['triton']}；{prov['nsys']}"),
        "clocks": (f"不锁频（没有 root）。applications clocks：SM {prov['sm_clock_mhz_applications']} MHz、显存 "
                   f"{prov['mem_clock_mhz_applications']} MHz；最高 SM {prov['sm_clock_mhz_max']} MHz"),
        "dtype": "fp16 输入、权重、输出",
        "batch": sorted(batches),
        "layout": "特征维在前（BSL）：输入 X 是 (M_in, batch)，输出 Y 是 (N_out, batch)；原始维度，不补零；第 0 级最先作用",
        "data": "权重 U(±√(3/C))、输入 U(±0.5)，种子是链名的 crc32（a100_audit.Built、make_input）",
        "n_runs": gb.N_RUNS,
        "timing": TIMING_TEXT,
        "nsys_bias": args.nsys_bias,
        "check": ("和 fp32 参考比相对 L2 误差；eager 输出和 graph 最后一个 run 的输出都 ≤ 1e-2 且有限才记「通过」。参考："
                  + ("各级稠密矩阵在 fp32 下连乘（a100_audit.Built）" if Path(args.chains).parent.name in gb.DENSE_REFERENCE
                     else "fp32 einsum 逐级算，不合成稠密矩阵")),
        "impl": "；".join(IMPL_TEXT.get(i, i) for i in sorted(impls)),
        "measured_at_commit": prov["ksmm_commit"],
        "measured_tree_dirty": prov["ksmm_dirty"],
        "sessions": sessions,
    }
    if with_power:
        config["power"] = POWER_TEXT
        config["idle"] = [{"gpu_uuid": u, "p_idle_w": w, "time_utc": tt} for u, w, tt in used_idle]
    n_check = {w: sum(1 for x in out if x["check"] == w) for w in ("通过", "失败", "未运行", "无法观测")}
    run = {
        "format": "dataset-run/1",
        "title": args.title,
        "task": args.task,
        "time": local_minute(prov["time_utc"]),
        "commits": {"ksmm": prov["ksmm_commit"]},
        "config": config,
        "description": (f"{prov['host']} 上 {impl} 的 GPU 计时" + ("与功率" if with_power else "")
                        + f"（ksmm `bench/gpu_bench.py`，提交 `{prov['ksmm_commit'][:8]}`）。"
                        + f"{len(out)} 条链：" + "，".join(f"{w} {n}" for w, n in n_check.items() if n) + "。"
                        + (f" KNL、device 的中位数见各列；DKL 中位数的中位数 "
                           f"{r(statistics.median(x['dkl_us'] for x in tm_of.values() if 'dkl_us' in x))} µs。"
                           if any("dkl_us" in x for x in tm_of.values()) else "")),
        "columns": {c: COLUMNS[c] for c in TIME_COLS + (POWER_COLS if with_power else []) + ["note"]},
    }
    (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{run_dir}: {len(out)} chains, " + ", ".join(f"{w} {n}" for w, n in n_check.items()))


if __name__ == "__main__":
    main()
