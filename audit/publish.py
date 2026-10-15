"""Turn the zhang-x4 outputs of x4_run_all.sh into operations dataset runs and task data files.

Standard library only (runs on zhang-22). Inputs, all in --in (a copy of the x4 output dir):
  checkpoint.jsonl, measure-<dataset>.jsonl, nsys-<dataset>.csv, ncu-kernels.csv, clk-*.csv,
  host-cmp.csv (from host_cmp.py)
Also reads the V4.1 and V3 runs of each dataset (runs/v41 `padded`, runs/v3-eb02 `shape_run`) to
state where their zero-padded shapes differ from the original dimensions A100 runs.
Outputs:
  <ops>/datasets/<dataset>/runs/a100-<impl>/{run.json,results.csv}   (batch 1)
  <ops>/tasks/<task>/data/a100-{checkpoint,batch-sweep,nsys,ncu,clocks,host-cmp}.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

DATASETS = ("v3-reference-data", "v1")
IMPLS = ("ksmm", "cublas", "cusparse")
TITLES = {
    "ksmm": "A100 KSMM-Triton 逐级（本任务实测）",
    "cublas": "A100 cuBLAS 稠密（整条链合成一个矩阵，本任务实测）",
    "cusparse": "A100 cuSPARSE CSR（整条链合成一个矩阵，本任务实测）",
}

TIMING = {
    "time_us": "G_cold 的均值（µs）",
    "E": "eager：triton.testing.do_bench(call, warmup=25, rep=100)（与旧 harness 相同），对全部样本算 p20/p50/p80（公式与 triton 3.6 的 _quantile 相同）。do_bench 每次调用前把 256 MB 缓冲清零来清 L2，清零在计时窗口外；但主机在清零执行期间（约 150 µs）已经开始逐级发射，所以 E 含主机发射时间里没被清零盖住的部分",
    "G_cold": "整条链录成一个 CUDA graph；每次回放前把同一个 256 MB 缓冲清零（窗口外），窗口里只有 graph.replay()。清零执行时主机已经提交了回放，所以窗口里没有主机发射时间。预热 25 ms，测约 100 ms（样本数见 task data 的 gc_n）。窗口里含 graph 在设备端的启动延迟：nsys 量到清零 kernel 结束到链的第一个 kernel 开始是 1.95–3.94 µs（eager 约 1.0 µs；G1 第 2 轮 partC 审阅第 4 节）。所以小链的 G_cold 里有 2–4 µs 不是 kernel 时间（对比 time_us 和 d_span_us），分析时把它算作 GPU 的固定开销",
    "G_warm": "同一个 graph；每次回放前先跑一个不访存的空转 kernel（torch.cuda._sleep(100000)，约 70 µs）让 GPU 保持忙，L2 里还是上一次回放的数据",
    "D": "nsys 2024.5.1（--trace=cuda,nvtx --cuda-graph-trace=node）：在 G_cold 的条件下回放 20 次，每次把这次回放启动的 kernel 的设备时间相加，取中位数。kernel 按启动它的 API 调用（cudaGraphLaunch、cuLaunchKernelEx 等，correlationId 相同）是否落在这次回放的 NVTX 区间里归属",
    "E_minus_G": "E − G_cold 近似主机发射时间里露在外面的部分：同样清 L2，差别在 kernel 由主机逐个发射还是由 graph 一次提交。两点限定：(1) G_cold 窗口含 graph 在设备端的启动延迟（约 2–4 µs，见 G_cold），eager 的第一个 kernel 在清零后约 1 µs 就开始；(2) 单次调用、主机发射被清零盖住的情况（cuBLAS、cuSPARSE、短链 KSMM），E − G_cold 接近 0，可以为负。E 还随进程变，见 E",
    "h": "主机发一次整条链的时间：GPU 被约 70 ms 的空转 kernel 占着，主机连发 20 次、不等 GPU，取平均（µs）",
    "event_resolution": "这块 A100 上 CUDA event 计时以 1.024 µs 为步长：所有分位数都是 1.024 µs 的整数倍（结果表里 sheet-a100 的 10 个数也是）。所以 G_cold、G_warm 报均值，分位数另报",
}
NUMERICS = ("和 fp32 稠密参考比：参考 = 各级稠密矩阵在 fp32 下连乘得到的 N_out × M_in 矩阵乘 fp32 输入（构建时另用 4 维 einsum 逐级算一遍，"
            "两者相对误差 < 1e-4 才继续）。指标是相对 L2 误差 ‖y − y_ref‖₂ / ‖y_ref‖₂；eager 输出和 graph 回放输出都 ≤ 1e-2 且都是有限值，"
            "才记「通过」。权重取 U(±√(3/C))（每级保持方差），输入取 U(±0.5)，种子是链名的 crc32")
IMPL_CONFIG = {
    "ksmm": ("每级调用一次 ks_triton_forward_impl(x, K, (A,B,C,D), layout='BSL')，K 的形状 (A·D, C, B)，与旧 harness 是同一份 kernel 源码。"
             "Triton autotune 在 6 个配置里选（块 16×16×32，warps 4/8，stages 3/4/5），用 TRITON_CACHE_AUTOTUNING=1 存盘，nsys、ncu 进程用同一个配置。"
             "和旧 harness 的唯一差别：输入直接是 (M_in, batch)，链首尾不做 transpose().contiguous()（batch = 1 时这两步不启动 kernel，只有主机开销）"),
    "cublas": ("整条链在 fp32 下合成一个 N_out × M_in 矩阵，转 fp16，torch.mm(W, X)。实际调用 cublasGemmEx（CUBLAS_OP_N，A/B/C 都是 CUDA_R_16F，"
               "CUBLAS_COMPUTE_32F，CUBLAS_GEMM_DEFAULT_TENSOR_OP；CUBLAS_LOGINFO_DBG 日志为证）。batch = 1 时 cuBLAS 对小矩阵选 gemvx kernel，"
               "对大矩阵选 ampere_fp16_s16816gemm（kernel 名见 task data 的 a100-nsys.csv）"),
    "cusparse": ("同一个 fp16 合成矩阵转 CSR（行偏移、列号都是 int32），batch = 1 用 torch.mv，实际调用 cusparseSpMV（CSR，CUDA_R_16F，"
                 "computeType CUDA_R_32F，SPMV_CSR_ALG1；CUSPARSE_LOG_LEVEL=5 日志为证），每次 3 个 kernel：csrmv_v3_partition_kernel、"
                 "vector_scalar_multiply_kernel、csrmv_v3_kernel。这些链合成后的矩阵基本是稠密的（见 csr_nnz 列），所以这是在稠密矩阵上做 CSR SpMV"),
}
COLUMNS = {
    "time_us": "G_cold（µs）：整条链录成一个 CUDA graph 回放，回放前在计时窗口外清 L2，取均值",
    "e_p20": "E 的 20 百分位（µs）：eager do_bench，每次清 L2",
    "e_p50": "E 的中位数（µs）",
    "e_p80": "E 的 80 百分位（µs）；结果表 sheet-a100 取的就是它",
    "g_warm_us": "G_warm（µs）：graph 回放，L2 热，取均值",
    "d_us": "D（µs）：nsys 在 G_cold 条件下看到的各 kernel 设备时间之和，20 次的中位数",
    "kernels": "整条链调用一次的 kernel 数（nsys）",
    "gc_p50": "G_cold 的中位数（µs），以 1.024 µs 为步长",
    "d_span_us": "nsys，G_cold 条件：第一个 kernel 开始到最后一个 kernel 结束（µs），中位数；减去 d_us 就是 kernel 之间的空隙",
    "d_eager_us": "nsys，E 条件（eager 发射、清 L2）：各 kernel 设备时间之和（µs），中位数",
    "h_us": "主机发一次整条链的时间（µs），不等 GPU",
    "rel_l2": "eager 输出对 fp32 稠密参考的相对 L2 误差；≤ 1e-2 记通过（graph 输出另核，见 task data）",
    "nnz_ratio_padded": "V4.1 补零后链的 nnz ÷ 原始链的 nnz（按运行 v41 的 padded 列算），即链的规模之比；1 表示没补零。A100 算原始链，V4.1 算补零后更大的链。KSMM 逐级算，有效乘加数就是原始 nnz，所以对 KSMM 这就是 V4.1 与 A100 的有效乘加之比；cuBLAS、cuSPARSE 算合成后的原始 N_out × M_in 矩阵，乘加数见 macs 列，不能用这个倍数比工作量。见 config.shape",
    "macs": "batch 1 调用一次的有效乘加数（不算 kernel 内部按块补齐的部分）：ksmm 是原始链的 nnz（Σ A·B·C·D）；cublas 是合成矩阵的 N_out·M_in；cusparse 是 csr_nnz",
    "csr_nnz": "合成矩阵转 CSR 后的非零数",
    "ncu_dram_pct": "ncu：DRAM 吞吐占峰值（%，dram__throughput.avg.pct_of_peak_sustained_elapsed），各 kernel 按时长加权；空着表示这条链没跑 ncu",
    "ncu_l2_hit_pct": "ncu：L2 命中率（%）= 各 kernel 的 lts__t_sectors_lookup_hit 之和 ÷（hit + miss 之和）",
    "ncu_sm_pct": "ncu：SM 吞吐占峰值（%，sm__throughput.avg.pct_of_peak_sustained_elapsed），按时长加权",
    "ncu_occ_pct": "ncu：实际占用率（%，sm__warps_active.avg.pct_of_peak_sustained_active），按时长加权",
    "ncu_tensor_pct": "ncu：张量核 HMMA 管线忙的周期占比（%，sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active），按时长加权",
    "ncu_waves": "ncu：各 kernel waves（launch__waves_per_multiprocessor）的最大值",
    "note": "出错时的报错",
}
API_LOG_CMD = (
    "A=/scratch/users/phd/yd383/v41-audit; K=<ksmm audit 分支的检出>; O=$A/runs/t4-fix; PY=/scratch/users/phd/yd383/venv-cu126/bin/python; "
    "for b in 1 4; do CUBLAS_LOGINFO_DBG=1 CUBLAS_LOGDEST_DBG=$O/cublas-api-b$b.log CUSPARSE_LOG_LEVEL=5 "
    "CUSPARSE_LOG_FILE=$O/cusparse-api-b$b.log $K/audit/x4_gpu_run.sh 0 $O/clk-apilog-b$b.csv apilog $PY $K/audit/a100_audit.py profile "
    "--chains $A/datasets/v3-reference-data/chains.csv --dataset v3-reference-data --only debut_1,bfly_4 --impls cublas,cusparse "
    "--batch $b --modes E --reps 1; done; "
    "grep -A20 'cublasGemmEx(' $O/cublas-api-b1.log | grep -E '^i!  (transa|transb|m|n|k|Atype|Btype|Ctype|computeType|algo):' | sort | uniq -c; "
    "grep -o '\\[Api\\]\\[cusparseSp[A-Za-z_]*\\]' $O/cusparse-api-b1.log | sort | uniq -c"
)
API_LOG = {
    "cublas": ("cuBLAS API 日志（CUBLAS_LOGINFO_DBG=1，2026-10-15 01:58 EDT，ksmm 38a7fd2，debut_1 和 bfly_4，batch 1 和 4）："
               "torch.mm 每次调用一次 cublasGemmEx，每个 batch 12 次，参数全部是 transa = transb = CUBLAS_OP_N、m = batch、"
               "n = N_out、k = M_in（debut_1 是 16，bfly_4 是 4096）、Atype = Btype = Ctype = CUDA_R_16F、"
               "computeType = CUBLAS_COMPUTE_32F、algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP。日志头写的是「cuBLAS (v12.3)」，"
               "但进程实际载入的是 venv 里的 libcublas.so.12，cublasGetProperty 读出 12.6.4（G1 第 2 轮 partC 审阅第 7 节），"
               "与 pip 包 nvidia-cublas-cu12 12.6.4.1 一致。日志在 zhang-x4 的 /scratch/users/phd/yd383/v41-audit/runs/t4-fix/"),
    "cusparse": ("cuSPARSE API 日志（CUSPARSE_LOG_LEVEL=5，2026-10-15 01:58 EDT，ksmm 38a7fd2，debut_1 和 bfly_4）：batch 1 时 torch.mv "
                 "每次调用 cusparseSpMV_bufferSize 和 cusparseSpMV 各一次（各 12 次），参数是 format = CSR、offsetType = columnType = 32I、"
                 "valuesType = CUDA_R_16F、computeType = CUDA_R_32F、alg = SPMV_CSR_ALG1，trace 里的 kernel 是 csrmv_v3_partition_kernel、"
                 "vector_scalar_multiply_kernel、csrmv_v3_kernel；batch 4 时 torch.mm 走 cusparseSpMM（alg = CSR_ALG2，kernel "
                 "csrmm_alg2_partition_kernel、matrix_scalar_multiply_kernel、csrmm_alg2_kernel）。cusparseGetProperty 读出 12.5.4"
                 "（G1 第 2 轮 partC 审阅第 7 节），与 pip 包 nvidia-cusparse-cu12 12.5.4.2 一致。日志在 zhang-x4 的 "
                 "/scratch/users/phd/yd383/v41-audit/runs/t4-fix/"),
}
HOST_CMP_CMD = ("A=/scratch/users/phd/yd383/v41-audit; K=<ksmm audit 分支的检出>; O=$A/runs/t4-fix; "
                "PY=/scratch/users/phd/yd383/venv-cu126/bin/python; for r in 0 1 2; do $K/audit/x4_gpu_run.sh 0 "
                "$O/clk-host-cmp-$r.csv host_cmp $PY $K/audit/host_cmp.py --out $O/host-cmp.csv --run $r; done")
NCU_CONFIG = ("ncu 2024.3.2（/usr/local/cuda-12.6/bin/ncu）：--replay-mode application --cache-control none --clock-control none "
              "--nvtx --nvtx-include prof/；每一遍重跑整个进程，harness 在 graph 回放前清 L2，所以每个 kernel 看到的缓存状态和 G_cold 相同；"
              "不锁频。只测部分链：v3-reference-data 的 debut_1、debut_2、debut_5、bfly_1、bfly_4，v1 的 debut_lenet_fc1_128x400、"
              "debut_vgg_512x4608、butterfly_512、butterfly_8192、synthetic_radix4_4096（三种实现，batch 1）；另测 bfly_4、debut_5 在 batch 16、256、4096"
              "（ksmm、cublas）")


def describe(impl: str, ds: str, commit: str, measured_at: str, ck_note: str, shape_note: str, e_spread: str) -> str:
    extra = {
        "ksmm": f"- **检查点**（只对 v3-reference-data）：{ck_note}\n",
        "cublas": "",
        "cusparse": "- 只测 batch = 1。\n",
    }[impl]
    return (
        f"zhang-x4 上实测（任务 2026-10-14-v41-baseline-audit，步骤 T4）。实现：{IMPL_CONFIG[impl]}。\n\n"
        "- **`time_us` 是 G_cold**（CUDA graph 回放、窗口外清 L2、不含主机发射），是和 V3、V4.1 比对用的数。\n"
        "- E 和结果表 sheet-a100 是同一个口径。E − G_cold 近似露在外面的主机发射时间（限定见 `config.timing.E_minus_G`），"
        f"h_us 是主机发一次整条链的总时间。E 受主机速度影响，跨进程会差到{e_spread}，看主机开销用 h_us（见 `config.timing.E`）。\n"
        f"- **形状**：{shape_note}\n"
        "- 口径、时钟、软件版本见 `config`；batch 扫描、nsys 汇总、ncu 明细、检查点、时钟采样在任务目录的 "
        "`data/a100-batch-sweep.csv`、`a100-nsys.csv`、`a100-ncu.csv`、`a100-checkpoint.csv`、`a100-clocks.csv`。\n"
        + extra +
        f"\n复现（测量时 ksmm 在提交 `{measured_at[:8]}`；`commits` 里的 `{commit[:8]}` 在它之后只改了 nsys-parse 写 kernel 名的方式、"
        "判断工作区是否干净时也算未跟踪文件，加了发布脚本和 host_cmp.py，测量代码没变）：\n\n"
        "```bash\n"
        "R=/scratch/users/phd/yd383/kroissant-project; A=/scratch/users/phd/yd383/v41-audit; X=zhang-x4.ece.cornell.edu\n"
        "rsync -a $R/ksmm/ $X:$A/ksmm/                                   # ksmm 的 audit 分支\n"
        "for ds in v3-reference-data v1; do rsync -a $R/operations/datasets/$ds/chains.csv $X:$A/datasets/$ds/; done\n"
        "rsync -a $R/operations/datasets/v3-reference-data/runs/sheet-a100/results.csv $X:$A/datasets/v3-reference-data/sheet-a100.csv\n"
        "ssh $X $A/ksmm/audit/x4_run_all.sh 0 $A/runs/t4 checkpoint measure nsys ncu   # 物理 GPU 0\n"
        "rsync -a --exclude '*.nsys-rep' --exclude '*.sqlite' --exclude '*.ncu-rep' $X:$A/runs/t4/ $R/scratch/a100-t4/\n"
        "python3 $R/ksmm/audit/publish.py --in $R/scratch/a100-t4 --ops $R/operations --commit <ksmm 提交>\n"
        "```\n\n"
        + (f"host_cmp（检查点说明里的主机时间差，输出拷成 `$R/scratch/a100-t4/host-cmp.csv` 再发布）：`{HOST_CMP_CMD}`。\n\n"
           if impl == "ksmm" else "")
        + (f"API 日志的生成命令：`{API_LOG_CMD}`。\n\n" if impl != "ksmm" else "")
        + 
        "`x4_gpu_run.sh` 在占卡前检查有没有别人的进程，前后在 gpu-use-log 登记，运行期间用 nvidia-smi 每 200 ms 采样时钟。"
        "原始 nsys、ncu 报告留在 zhang-x4 的 `$A/runs/t4/`。"
    )



def r(x, nd=3):
    return "" if x is None or x != x else round(x, nd)


def local_minute(utc_iso: str) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    t = datetime.fromisoformat(utc_iso).astimezone(ZoneInfo("America/New_York"))
    return t.strftime("%Y-%m-%dT%H:%M%z")[:-2] + ":" + t.strftime("%z")[-2:]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def chain_order(ops: Path, dataset: str) -> list[str]:
    with open(ops / "datasets" / dataset / "chains.csv", encoding="utf-8", newline="") as stream:
        return [row["chain"] for row in csv.DictReader(stream)]


def chain_stages(ops: Path, dataset: str) -> dict:
    out = {}
    with open(ops / "datasets" / dataset / "chains.csv", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stages, k = [], 0
            while row.get(f"A{k}"):
                stages.append(tuple(int(row[f"{x}{k}"]) for x in "ABCD"))
                k += 1
            out[row["chain"]] = stages
    return out


def run_shapes(ops: Path, dataset: str, run: str, column: str) -> dict:
    """Per-chain stage shapes from another run's column, e.g. '256,2,2,1;128,2,2,2'."""
    with open(ops / "datasets" / dataset / "runs" / run / "results.csv", encoding="utf-8", newline="") as stream:
        return {row["chain"]: [tuple(int(v) for v in s.split(",")) for s in row[column].split(";")]
                for row in csv.DictReader(stream) if row.get(column)}


def nnz(stages) -> int:
    return sum(a * b * c * d for a, b, c, d in stages)


def m_in(stages) -> int:
    a, _b, c, d = stages[0]
    return a * c * d


def n_out(stages) -> int:
    a, b, _c, d = stages[-1]
    return a * b * d


def pct(x: float) -> str:
    """+7.1% / −12.7% / 0.0% (Unicode minus, no signed zero)."""
    t = f"{x:+.1f}%"
    return "0.0%" if t in ("+0.0%", "-0.0%") else t.replace("-", "−")


def fmt_ratio(x: float):
    return int(x) if x == int(x) else round(x, 4)


def shape_info(ops: Path, dataset: str):
    """Chain sizes of A100 (original) vs. V4.1 / V3 (zero-padded), and the impl-independent part of config.shape."""
    orig = chain_stages(ops, dataset)
    v41 = run_shapes(ops, dataset, "v41", "padded")
    v3 = run_shapes(ops, dataset, "v3-eb02", "shape_run")
    with open(ops / "datasets" / dataset / "runs" / "v3-eb02" / "results.csv", encoding="utf-8", newline="") as stream:
        v3_rows = {row["chain"]: row for row in csv.DictReader(stream)}
    ratio = {c: nnz(v41[c]) / nnz(st) for c, st in orig.items()}
    changed = [c for c in orig if v41[c] != orig[c]]
    items = []
    for c in changed:
        o, pz = orig[c], v41[c]
        item = f"{c} {fmt_ratio(ratio[c])} 倍（nnz {nnz(o)}→{nnz(pz)}，M_in {m_in(o)}→{m_in(pz)}"
        if n_out(o) != n_out(pz):
            item += f"，N_out {n_out(o)}→{n_out(pz)}"
        items.append(item + "）")
    text = ("A100 的三种实现都按 chains.csv 的原始维度算，不补零。V3（运行 v3-eb02 的 shape_run 列）和 V4.1（运行 v41 的 padded 列）"
            f"跑的是补零后的形状。本数据集 {len(orig)} 条链里有 {len(changed)} 条补零后形状变了")
    if changed:
        text += (f"，V4.1 算的链比 A100 的大，链的规模（nnz）是原始的 {fmt_ratio(min(ratio[c] for c in changed))}–"
                 f"{fmt_ratio(max(ratio[c] for c in changed))} 倍：" + "；".join(items) + "。")
        v3diff = [c for c in changed if c in v3 and v3[c] != v41[c]]
        if v3diff:
            b1 = all(any(b == 1 for _a, b, _c, _d in orig[c]) for c in v3diff)
            text += (f"其中 {len(v3diff)} 条" + ("有 B = 1 的级，" if b1 else "") + "V3 的补法和 V4.1 不同，V3 的倍数是："
                     + "；".join(f"{c} {fmt_ratio(nnz(v3[c]) / nnz(orig[c]))} 倍" for c in v3diff) + "。")
            skipped = [c for c in v3diff if not v3_rows[c].get("cycles") and v3_rows[c]["note"].startswith("V3 求解器判为 SKIP")]
            if skipped:
                text += (f"V3 在其中 {len(skipped)} 条（{'、'.join(skipped)}）上求解器判为 SKIP，运行 v3-eb02 里 check 是"
                         f"「{v3_rows[skipped[0]]['check']}」、没有周期。")
    else:
        text += "。"
    text += "其余链三方形状相同。列 nnz_ratio_padded 按 V4.1 算。"
    return ratio, {c: nnz(v41[c]) for c in orig}, changed, text


def shape_impl(impl: str, changed: list, padded_nnz: dict, macs: dict) -> tuple:
    """Impl-specific end of config.shape and the description bullet: what the nnz ratio means for this impl."""
    if impl == "ksmm":
        tail = ("KSMM 逐级算原始链，有效乘加数就是原始 nnz（macs 列），所以这些链上 V4.1 比 A100 多做了补零的部分，"
                "倍数就是 nnz_ratio_padded")
        short = "对 KSMM，这就是 V4.1 与 A100 的有效乘加之比"
    else:
        what = "稠密矩阵，有效乘加数是 N_out·M_in" if impl == "cublas" else "矩阵转成的 CSR，有效乘加数是 csr_nnz"
        tail = (f"{'cuBLAS' if impl == 'cublas' else 'cuSPARSE'} 算的是合成后的原始 N_out × M_in {what}（macs 列），"
                "由合成矩阵的大小决定，不能用 nnz_ratio_padded 比工作量")
        if changed:
            k = [macs[c] / padded_nnz[c] for c in changed]
            tail += f"：上面这些链上，它是 V4.1 补零后 nnz 的 {min(k):.1f}–{max(k):.1f} 倍"
        short = f"{'cuBLAS' if impl == 'cublas' else 'cuSPARSE'} 的乘加数由合成矩阵决定（macs 列），不能用这个倍数比工作量"
    return tail, short


def clock_summary(path: Path) -> dict:
    """Samples of nvidia-smi taken every 200 ms while one x4_gpu_run.sh command ran."""
    busy = []
    for row in csv.reader(open(path, encoding="utf-8")):
        if len(row) < 7:
            continue
        _ts, sm, mem, util, power, temp, reasons = (x.strip() for x in row[:7])
        if int(util) > 0:
            busy.append((int(sm), int(mem), float(power), int(temp), reasons))
    if not busy:
        return {"log": path.name, "busy_samples": 0}
    sms = [b[0] for b in busy]
    return {
        "log": path.name,
        "busy_samples": len(busy),
        "sm_mhz_min": min(sms),
        "sm_mhz_median": statistics.median(sms),
        "sm_mhz_max": max(sms),
        "share_at_1410": round(sum(1 for s in sms if s == 1410) / len(sms), 4),
        "mem_mhz": " ".join(str(m) for m in sorted({b[1] for b in busy})),
        "power_w_max": max(b[2] for b in busy),
        "temp_c_max": max(b[3] for b in busy),
        "throttle_reasons": " ".join(sorted({b[4] for b in busy})),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--ops", required=True, help="operations repository root")
    p.add_argument("--task", default="2026-10-14-v41-baseline-audit")
    p.add_argument("--commit", required=True, help="ksmm commit to record in run.json")
    args = p.parse_args()
    inp, ops = Path(args.inp), Path(args.ops)
    data = ops / "tasks" / args.task / "data"

    # ---- checkpoint
    ck = [row for row in read_jsonl(inp / "checkpoint.jsonl") if row["kind"] == "checkpoint"]
    by_chain = defaultdict(list)
    for row in ck:
        by_chain[row["chain"]].append(row)
    ck_rows = []
    for chain in chain_order(ops, "v3-reference-data"):
        rows = sorted(by_chain[chain], key=lambda x: x["round"])
        sheet = rows[0]["sheet_p80"]
        leg = statistics.median(x["legacy_p80"] for x in rows)
        ours = statistics.median(x["ours_p80"] for x in rows)
        ck_rows.append({
            "chain": chain, "sheet_p80_us": sheet,
            **{f"legacy_p80_r{x['round']}": r(x["legacy_p80"], 2) for x in rows},
            "legacy_p80_median": r(leg, 2), "legacy_dev_pct": r((leg / sheet - 1) * 100, 1),
            **{f"ours_p80_r{x['round']}": r(x["ours_p80"], 2) for x in rows},
            "ours_p80_median": r(ours, 2), "ours_dev_pct": r((ours / sheet - 1) * 100, 1),
            "legacy_p50_median": r(statistics.median(x["legacy_p50"] for x in rows), 2),
            "sm_mhz_end": " ".join(str(x["sm_mhz_end"]) for x in rows),
            "ksmm_configs": rows[0]["ksmm_configs"],
        })
    write_csv(data / "a100-checkpoint.csv", ck_rows)

    # ---- nsys: aggregate the 20 reps per (mode, dataset, chain, impl)
    nsys = defaultdict(list)
    for ds in DATASETS:
        with open(inp / f"nsys-{ds}.csv", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                nsys[(row["mode"], row["dataset"], row["chain"], row["impl"])].append(row)
    nsys_rows = []
    nsys_d = {}
    for (mode, ds, chain, impl), reps in nsys.items():
        d = [float(x["d_us"]) for x in reps]
        span = [float(x["span_us"]) for x in reps]
        per_kernel = list(zip(*[[float(v) for v in x["kernel_us"].split()] for x in reps]))
        kernels = {int(x["kernels"]) for x in reps}
        nsys_rows.append({
            "mode": mode, "dataset": ds, "chain": chain, "impl": impl, "batch": reps[0]["batch"], "reps": len(reps),
            "kernels": " ".join(str(k) for k in sorted(kernels)),
            "other_ops": reps[0]["other_ops"],
            "d_us_median": r(statistics.median(d)), "d_us_min": r(min(d)), "d_us_max": r(max(d)),
            "span_us_median": r(statistics.median(span)),
            "gaps_us_median": r(statistics.median(float(x["gaps_us"]) for x in reps)),
            "kernel_us_median": " ".join(f"{statistics.median(k):.3f}" for k in per_kernel),
            "kernel_names": reps[0]["kernel_names"],
            "grids": reps[0]["grids"], "blocks": reps[0]["blocks"], "api": reps[0]["api"],
        })
        nsys_d[(mode, ds, chain, impl)] = (statistics.median(d), sorted(kernels), statistics.median(span))
    write_csv(data / "a100-nsys.csv", nsys_rows)

    # ---- ncu: per kernel, and a per-chain duration-weighted summary
    with open(inp / "ncu-kernels.csv", encoding="utf-8", newline="") as stream:
        ncu = list(csv.DictReader(stream))
    write_csv(data / "a100-ncu.csv", ncu)
    ncu_sum = {}
    groups = defaultdict(list)
    for k in ncu:
        groups[(k["dataset"], k["chain"], k["impl"], int(k["batch"]))].append(k)
    for key, ks in groups.items():
        w = [float(k["duration_us"]) for k in ks]

        def wmean(col):
            vals = [(float(k[col]), wi) for k, wi in zip(ks, w) if k[col] != ""]
            return sum(v * wi for v, wi in vals) / sum(wi for _v, wi in vals)

        hits = sum(int(k["l2_hit_sectors"]) for k in ks)
        miss = sum(int(k["l2_miss_sectors"]) for k in ks)
        ncu_sum[key] = {
            "ncu_dram_pct": r(wmean("dram_pct"), 2), "ncu_l2_hit_pct": r(100 * hits / (hits + miss), 1),
            "ncu_sm_pct": r(wmean("sm_pct"), 2), "ncu_occ_pct": r(wmean("occ_achieved_pct"), 1),
            "ncu_tensor_pct": r(wmean("tensor_pct"), 2), "ncu_waves": r(max(float(k["waves"]) for k in ks), 2),
        }

    # ---- clocks
    clock_rows = [clock_summary(f) for f in sorted(inp.glob("clk-*.csv"))]
    write_csv(data / "a100-clocks.csv", clock_rows)
    clocks = {c["log"]: c for c in clock_rows}
    worst = max(ck_rows, key=lambda x: abs(x["legacy_dev_pct"]))
    # ---- checkpoint note and the E caveat: E of host-bound chains moves with host speed
    meas_ref = {x["chain"]: x for x in read_jsonl(inp / "measure-v3-reference-data.jsonl")
                if x["kind"] == "point" and x["impl"] == "ksmm" and x["batch"] == 1}
    leg_dev = [x["legacy_dev_pct"] for x in ck_rows]
    ours_dev = [x["ours_dev_pct"] for x in ck_rows]
    meas_dev = [(meas_ref[x["chain"]]["e_p80"] / x["sheet_p80_us"] - 1) * 100 for x in ck_rows]
    with open(inp / "host-cmp.csv", encoding="utf-8", newline="") as stream:
        hc = list(csv.DictReader(stream))
    write_csv(data / "a100-host-cmp.csv", hc)
    hd = [float(x["h_diff_us"]) for x in hc]
    ed = [float(x["e_p80_diff_us"]) for x in hc]
    share = [float(x["h_diff_us"]) / float(x["e_p80_diff_us"]) * 100 for x in hc if float(x["e_p80_diff_us"]) > 0]
    same_e = sorted({x["chain"] for x in hc if float(x["e_p80_diff_us"]) == 0})
    ck_note = (f"用旧 harness 自己的调用、同样的 do_bench（3 轮取 p80 的中位数），10 条对表是 {pct(min(leg_dev))} 到 "
               f"{pct(max(leg_dev))}，都在 ±10% 以内；审阅在另一个进程复跑 1 轮是 −8.1% 到 +7.1%（G1 第 2 轮 partC 审阅第 3 节）。"
               f"本 harness 的调用对表的偏差随进程变：检查点那个进程 {pct(min(ours_dev))} 到 {pct(max(ours_dev))}，"
               f"measure 进程（本运行的 e_p80）{pct(min(meas_dev))} 到 {pct(max(meas_dev))}，审阅复跑的检查点 −20.6% 到 0%；"
               "变的原因是主机速度，见 `config.timing.E`。旧调用的 E 比本调用高：host_cmp.py 在 3 个进程里测得，旧调用发一次整条链的"
               f"主机时间多 {min(hd):.1f}–{max(hd):.1f} µs（它在链首尾各多一次 transpose().contiguous()），同一进程里两种调用的 E p80 差 "
               f"{min(ed):.1f}–{max(ed):.1f} µs（{'、'.join(same_e) or '没有链'} 两者相同）；E p80 差大于 0 的链上，主机时间差只解释了 E 差的 "
               f"{min(share):.0f}–{max(share):.0f}%，其余原因没有查清。数据见任务 data/a100-host-cmp.csv、data/a100-checkpoint.csv。")
    d5_hc = [x for x in hc if x["chain"] == "debut_5"]
    d5_e = [float(x["e_p80_ours_us"]) for x in d5_hc]
    d5_h = [float(x["h_ours_us"]) for x in d5_hc]
    h_all = d5_h + [meas_ref["debut_5"]["h_us"], 175.8]
    e_all = d5_e + [meas_ref["debut_5"]["e_p80"], 52.22,
                    next(x["ours_p80_median"] for x in ck_rows if x["chain"] == "debut_5")]
    e_spread = f"约 {(max(e_all) - min(e_all)) / max(e_all) * 100:.0f}%（{max(e_all):.2f} 对 {min(e_all):.2f} µs）"
    e_note = ("。h_us 大于清零时间（约 150 µs）的链，E 受主机发射限制，跟着主机速度一比一变，不同进程之间实测相差可到"
              f"{e_spread}："
              "本 harness 同一个 ksmm 调用的 debut_5 E p80，检查点进程 "
              f"{next(x['ours_p80_median'] for x in ck_rows if x['chain'] == 'debut_5')} µs，measure 进程 "
              f"{round(meas_ref['debut_5']['e_p80'], 2)} µs，host_cmp 的 3 个进程 {min(d5_e):.2f}–{max(d5_e):.2f} µs，审阅复测 52.22 µs"
              "（G1 第 2 轮 partC 审阅第 2 节）。跨进程比较 E 或 E − G_cold 时要带上这个误差。看主机开销用 h_us：debut_5 的 h_us "
              f"在这几次之间只差 {(max(h_all) - min(h_all)) / max(h_all) * 100:.1f}%（measure {round(meas_ref['debut_5']['h_us'], 1)} µs，"
              f"host_cmp {min(d5_h):.1f}–{max(d5_h):.1f} µs，审阅复测 175.8 µs）。G_cold、G_warm 是设备时间，审阅复测 21 个点偏差在 "
              "−3.9% 到 +1.7% 以内")
    timing = dict(TIMING)
    timing["E"] = TIMING["E"] + e_note

    # ---- measurements
    sweep_rows = []
    provenance = {}
    for ds in DATASETS:
        rows = read_jsonl(inp / f"measure-{ds}.jsonl")
        provenance[ds] = [x for x in rows if x["kind"] == "provenance"]
        latest = {}
        for x in rows:
            if x["kind"] == "point":
                latest[(x["chain"], x["impl"], x["batch"])] = x
        order = chain_order(ops, ds)
        ratio, padded_nnz, changed, shape_common = shape_info(ops, ds)
        for chain in order:
            for impl in IMPLS:
                for (c, i, b), x in sorted(latest.items(), key=lambda kv: kv[0][2]):
                    if c != chain or i != impl:
                        continue
                    sweep_rows.append({
                        "dataset": ds, "chain": chain, "impl": impl, "batch": b, "stages": x["stages"],
                        "m_in": x["m_in"], "n_out": x["n_out"], "nnz": x["nnz"], "status": x["status"],
                        "check": x["check"], "rel_l2": r(x.get("eager_rel_l2"), 6),
                        "graph_rel_l2": r(x.get("graph_rel_l2"), 6),
                        **{f"{v}_{q}": r(x.get(f"{v}_{q}")) for v in ("e", "gc", "gw")
                           for q in ("p20", "p50", "p80", "mean", "n")},
                        "h_us": r(x.get("h_us")), "csr_nnz": x.get("csr_nnz", ""),
                        "sm_mhz_end": x.get("sm_mhz_end"), "ksmm_configs": x.get("ksmm_configs", ""),
                        "error": x.get("error", ""),
                    })
        for impl in IMPLS:
            out = []
            macs = {}
            for chain in order:
                x = latest.get((chain, impl, 1))
                if x is None:
                    out.append({"chain": chain, "check": "未运行"})
                    continue
                d, kernels, span = nsys_d.get(("GC", ds, chain, impl), (None, [], None))
                de = nsys_d.get(("E", ds, chain, impl), (None, [], None))[0]
                row = {
                    "chain": chain, "check": x["check"], "time_us": r(x.get("gc_mean")),
                    "e_p20": r(x.get("e_p20")), "e_p50": r(x.get("e_p50")), "e_p80": r(x.get("e_p80")),
                    "g_warm_us": r(x.get("gw_mean")), "d_us": r(d), "kernels": " ".join(map(str, kernels)),
                    "gc_p50": r(x.get("gc_p50")), "d_span_us": r(span), "d_eager_us": r(de),
                    "h_us": r(x.get("h_us")), "rel_l2": r(x.get("eager_rel_l2"), 6),
                }
                if impl == "cusparse":
                    row["csr_nnz"] = x.get("csr_nnz", "")
                row.update(ncu_sum.get((ds, chain, impl, 1), {k: "" for k in (
                    "ncu_dram_pct", "ncu_l2_hit_pct", "ncu_sm_pct", "ncu_occ_pct", "ncu_tensor_pct", "ncu_waves")}))
                row["nnz_ratio_padded"] = fmt_ratio(ratio[chain])
                macs[chain] = {"ksmm": x["nnz"], "cublas": x["m_in"] * x["n_out"]}.get(impl, x.get("csr_nnz"))
                row["macs"] = macs[chain]
                row["note"] = x.get("error", "")
                out.append(row)
            run_dir = ops / "datasets" / ds / "runs" / f"a100-{impl}"
            write_csv(run_dir / "results.csv", out)
            tail, impl_short = shape_impl(impl, changed, padded_nnz, macs)
            shape_text = shape_common + tail
            shape_short = (f"A100 跑原始维度，V3、V4.1 跑补零后的形状；本数据集有 {len(changed)} 条链 V3、V4.1 算的链比 A100 的大"
                           f"（`nnz_ratio_padded` > 1，链的规模之比）。{impl_short}。详见 `config.shape`。")
            cols = dict(COLUMNS)
            if impl != "cusparse":
                cols.pop("csr_nnz")
            prov = provenance[ds][-1]
            clk = clocks.get(f"clk-measure-{ds}.csv", {})
            pts = list(latest.values())
            low = [x for x in pts if x["sm_mhz_end"] < 1410]
            low_note = (f"每个点测完时用 NVML 读一次 SM 时钟：{len(pts)} 个点里 {len(low)} 个低于 1410 MHz"
                        + (f"（{min(x['sm_mhz_end'] for x in low)}–{max(x['sm_mhz_end'] for x in low)} MHz，batch 都 ≥ "
                           f"{min(x['batch'] for x in low)}，其中 {sum(1 for x in low if x['impl'] == 'cublas')} 个是 cublas）"
                           if low else "")
                        + f"；batch = 1 的点 {sum(1 for x in low if x['batch'] == 1)} 个低于 1410。"
                        "降频原因位 0x4 是软件功耗上限（SW power cap）")
            run = {
                "format": "dataset-run/1",
                "title": TITLES[impl],
                "task": args.task,
                "time": local_minute(provenance[ds][0]["time_utc"]),
                "commits": {"ksmm": args.commit},
                "config": {
                    "host": f"{prov['host']}（zhang-x4）",
                    "gpu": f"{prov['gpu_name']}，物理 GPU {prov['gpu_index_nvml']}（CUDA_VISIBLE_DEVICES={prov['cuda_visible_devices']}），"
                           f"PCI {prov['gpu_pci_bus_id']}，{prov['gpu_uuid']}，VBIOS {prov['vbios']}，{prov['sm_count']} 个 SM",
                    "driver": prov["driver"],
                    "software": f"Python {prov['python']}（{prov['python_executable']}）；torch {prov['torch']}（CUDA {prov['torch_cuda']}）；"
                                f"triton {prov['triton']}；nvidia-cublas-cu12 {prov['nvidia_cublas_cu12']}；"
                                f"nvidia-cusparse-cu12 {prov['nvidia_cusparse_cu12']}；nvidia-cuda-runtime-cu12 "
                                f"{prov['nvidia_cuda_runtime_cu12']}；nsys 2024.5.1.113；ncu 2024.3.2.0",
                    "clocks": (f"不锁频（没有 root；applications clocks 是默认的 SM {prov['sm_clock_mhz_applications']} MHz、"
                               f"显存 {prov['mem_clock_mhz_applications']} MHz）。测量期间 nvidia-smi 每 200 ms 采样，GPU 忙的 "
                               f"{clk.get('busy_samples')} 个样本：SM {clk.get('sm_mhz_min')}–{clk.get('sm_mhz_max')} MHz（中位数 "
                               f"{clk.get('sm_mhz_median')}，{100 * clk.get('share_at_1410', 0):.1f}% 的样本在 1410），显存 {clk.get('mem_mhz')} MHz，"
                               f"降频原因位 {clk.get('throttle_reasons')}，最高 {clk.get('temp_c_max')} °C、{clk.get('power_w_max')} W。{low_note}"),
                    "dtype": "fp16 输入、权重、输出；乘加在 fp32 里累加（KSMM 的 tl.dot、cuBLAS 的 CUBLAS_COMPUTE_32F、cuSPARSE 的 CUDA_R_32F）",
                    "batch": 1,
                    "layout": "特征维在前：输入 X 是 (M_in, batch)，输出 Y 是 (N_out, batch)；第 0 级最先作用在输入上",
                    "shape": shape_text,
                    "timing": timing,
                    "flush": f"{prov['flush_bytes']} 字节的 int32 缓冲 zero_()，一次约 {round(prov['flush_us_median'], 1)} µs",
                    "numerics": NUMERICS,
                    "impl": IMPL_CONFIG[impl],
                    **({"api_log": API_LOG[impl]} if impl in API_LOG else {}),
                    "ncu": NCU_CONFIG,
                    "measured_at_commit": prov["ksmm_commit"],
                    "measured_tree_dirty": prov["ksmm_dirty"],
                    "loadavg_start": prov["loadavg_start"],
                },
                "description": describe(impl, ds, args.commit, prov["ksmm_commit"], ck_note, shape_short, e_spread),
                "columns": cols,
            }
            (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    write_csv(data / "a100-batch-sweep.csv", sweep_rows)

    print("published", len(sweep_rows), "sweep rows,", len(nsys_rows), "nsys rows,", len(ncu), "ncu kernels")


if __name__ == "__main__":
    main()
