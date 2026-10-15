"""A100 measurement harness for operations task 2026-10-14-v41-baseline-audit (step T4).

Reads an operations dataset `chains.csv` (stage 0 acts first on the input; each stage is
(A, B, C, D) with out[a][b][d] += W[a][b][c][d] * in[a][c][d]) and times three GPU
implementations of the whole chain y = S_{K-1} ... S_1 S_0 x:

  ksmm      KSMM-Triton `ks_triton_forward_impl`, BSL layout, one kernel per stage
            (the kernel behind the old do-bench-harness numbers).
  cublas    the chain composed into one dense N x M fp16 matrix, then one torch.mm
            (cuBLAS GEMM; a GEMV-shaped GEMM at batch 1).
  cusparse  the same composed matrix converted to CSR (int32 indices), then torch.mv
            (batch 1, cuSPARSE SpMV) or torch.mm (batch > 1, cuSPARSE SpMM).

Input and output are feature-major (BSL): X is (M_in, batch), Y is (N_out, batch).

Timing views (all in microseconds):

  E        triton.testing.do_bench(call, warmup=25, rep=100, quantiles=[.2,.5,.8]):
           eager launches; do_bench zeroes a 256 MB buffer before every call, outside the
           event window, but the host keeps launching while that zeroing runs.
  G_cold   the whole chain captured into one CUDA graph; before every replay the same
           256 MB buffer is zeroed (outside the event window); the window holds only
           graph.replay().
  G_warm   same graph; before every replay a spin kernel (torch.cuda._sleep, no memory
           traffic) keeps the GPU busy while the host enqueues the replay, so L2 keeps the
           previous replay's data.
  h        host time to issue one eager call while the GPU is kept busy (no blocking).

D (sum of kernel durations) comes from Nsight Systems: run `profile` under nsys, then
`nsys-parse` on the exported sqlite file.

ncu metrics come from `profile` under Nsight Compute, then `ncu-parse`.

Subcommands: measure, checkpoint, profile, nsys-parse, ncu-parse. See `--help` of each.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import socket
import sqlite3
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # ksmm repository root
sys.path.insert(0, str(ROOT / "src"))

QUANTILES = [0.2, 0.5, 0.8]
WARMUP_MS, REP_MS = 25, 100  # same as the old harness and triton's defaults
FLUSH_BYTES = 256 * 1024 * 1024  # same size as triton do_bench's L2-clearing buffer
SLEEP_CYCLES = 100_000  # about 70 us at 1410 MHz
TOL_REL_L2 = 1e-2  # pass threshold on ||y - y_ref||_2 / ||y_ref||_2
# Largest fp32 composed matrix built by multiplying the stage matrices (65536 x 65536,
# the T4 and T7-phase-1 sizes). Larger chains (T7 phase 2, up to 2^20 points) get the
# fp32 reference stage by stage and the cuBLAS fp16 matrix column block by column block.
DENSE32_MAX_BYTES = 16 << 30
BLOCK_BYTES = 1 << 30  # fp32 identity block per column block of the fp16 composition
IMPLS = ("ksmm", "cublas", "cusparse")


@dataclass
class Chain:
    name: str
    source: str
    stages: list  # list of (A, B, C, D), stage 0 first

    @property
    def m_in(self) -> int:
        a, _b, c, d = self.stages[0]
        return a * c * d

    @property
    def n_out(self) -> int:
        a, b, _c, d = self.stages[-1]
        return a * b * d

    @property
    def nnz(self) -> int:
        return sum(a * b * c * d for a, b, c, d in self.stages)


def read_chains(path: Path) -> list[Chain]:
    chains = []
    with open(path, encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            stages, k = [], 0
            while row.get(f"A{k}"):
                stages.append(tuple(int(row[f"{x}{k}"]) for x in "ABCD"))
                k += 1
            for k, ((a, b, _c, d), (a2, _b2, c2, d2)) in enumerate(zip(stages, stages[1:])):
                if a * b * d != a2 * c2 * d2:
                    raise ValueError(f"{row['chain']}: chain condition fails between stage {k} and {k + 1}")
            chains.append(Chain(row["chain"], row["source"], stages))
    return chains


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def quantiles(times_us: list[float]) -> list[float]:
    """Linear-interpolation quantiles, the same formula as triton 3.6 testing._quantile."""
    a = sorted(times_us)
    out = []
    for q in QUANTILES:
        point = q * (len(a) - 1)
        lo, hi = math.floor(point), math.ceil(point)
        out.append((1 - (point - lo)) * a[lo] + (point - lo) * a[hi])
    return out


def summary(times_us: list[float]) -> dict:
    """p20 / p50 / p80, mean and sample count.

    CUDA event times on this A100 come in steps of 1.024 us, so the quantiles are
    quantized; the mean over hundreds of samples averages the step out.
    """
    p20, p50, p80 = quantiles(times_us)
    return {"p20": p20, "p50": p50, "p80": p80, "mean": sum(times_us) / len(times_us), "n": len(times_us)}


# --------------------------------------------------------------------------- GPU side


def gpu_stack():
    import torch
    import triton
    import triton.testing

    from ksmm_triton.ksmm_triton_tc import ks_fused_kernel, ks_triton_forward_impl

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return torch, triton, ks_triton_forward_impl, ks_fused_kernel


def chain_seed(name: str) -> int:
    return zlib.crc32(name.encode("utf-8")) & 0x7FFFFFFF


class Built:
    """Weights of one chain on the GPU, plus the fp32 composed matrix."""

    def __init__(self, torch, chain: Chain):
        self.chain = chain
        gen = torch.Generator(device="cuda").manual_seed(chain_seed(chain.name))
        self.w4 = []  # fp16 (A, B, C, D)
        for a, b, c, d in chain.stages:
            bound = math.sqrt(3.0 / c)  # variance preserving: values stay in fp16's normal range
            w = (torch.rand((a, b, c, d), generator=gen, device="cuda") * 2 - 1) * bound
            self.w4.append(w.half())
        # KSMM layout: K[a*D + d, c, b] = W[a, b, c, d]
        self.k_bmm = [w.permute(0, 3, 2, 1).reshape(a * d, c, b).contiguous()
                      for w, (a, b, c, d) in zip(self.w4, chain.stages)]
        small = chain.n_out * chain.m_in * 4 <= DENSE32_MAX_BYTES
        self.dense32 = self._compose(torch) if small else None
        if small:
            self._self_check(torch)

    def _stage_dense(self, torch, k: int):
        a, b, c, d = self.chain.stages[k]
        ia, ib, ic, id_ = torch.meshgrid(*(torch.arange(n, device="cuda") for n in (a, b, c, d)), indexing="ij")
        rows = ((ia * b + ib) * d + id_).reshape(-1)
        cols = ((ia * c + ic) * d + id_).reshape(-1)
        s = torch.zeros((a * b * d, a * c * d), dtype=torch.float32, device="cuda")
        s[rows, cols] = self.w4[k].float().reshape(-1)
        return s

    def _compose(self, torch):
        total = self._stage_dense(torch, 0)
        for k in range(1, len(self.chain.stages)):
            total = self._stage_dense(torch, k) @ total
        return total  # (N_out, M_in) fp32

    def reference(self, torch, x32):
        """fp32 reference output: composed matrix times x, or stage by stage for large chains."""
        return self.dense32 @ x32 if self.dense32 is not None else self.stagewise32(torch, x32)

    def composed16(self, torch):
        """fp16 N_out x M_in matrix of the whole chain.

        Small chains: the fp32 composed matrix rounded. Large chains: column blocks of the
        identity pushed through the stages in fp32 (stagewise32), each block rounded to fp16;
        refuses when the fp16 matrix alone does not fit in free device memory.
        """
        if self.dense32 is not None:
            return self.dense32.half()
        n_out, m_in = self.chain.n_out, self.chain.m_in
        free, _total = torch.cuda.mem_get_info()
        need = n_out * m_in * 2
        if need + 4 * BLOCK_BYTES > free:
            raise RuntimeError(f"{self.chain.name}: composed fp16 matrix {need / 2**30:.1f} GiB "
                               f"does not fit ({free / 2**30:.1f} GiB free)")
        w16 = torch.empty((n_out, m_in), dtype=torch.float16, device="cuda")
        step = max(1, BLOCK_BYTES // (4 * m_in))
        for j0 in range(0, m_in, step):
            j1 = min(m_in, j0 + step)
            eye = torch.zeros((m_in, j1 - j0), dtype=torch.float32, device="cuda")
            eye[torch.arange(j0, j1, device="cuda"), torch.arange(j1 - j0, device="cuda")] = 1.0
            w16[:, j0:j1] = self.stagewise32(torch, eye).half()
            del eye
        return w16

    def stagewise32(self, torch, x32):
        """fp32 chain applied stage by stage with einsum on the 4-D definition."""
        y = x32
        n = x32.shape[1]
        for w, (a, b, c, d) in zip(self.w4, self.chain.stages):
            y = torch.einsum("abcd,acdn->abdn", w.float(), y.reshape(a, c, d, n)).reshape(a * b * d, n)
        return y

    def _self_check(self, torch):
        gen = torch.Generator(device="cuda").manual_seed(1)
        v = torch.rand((self.chain.m_in, 4), generator=gen, device="cuda") - 0.5
        ref = self.stagewise32(torch, v)
        got = self.dense32 @ v
        err = (torch.linalg.vector_norm(got - ref) / torch.linalg.vector_norm(ref)).item()
        if not err < 1e-4:
            raise RuntimeError(f"{self.chain.name}: composed fp32 matrix disagrees with the stagewise einsum (rel {err:.3e})")

    def make_call(self, torch, ks_forward, impl: str, x):
        """Return (call, extra) where call() -> (N_out, batch) or (N_out,) output."""
        if impl == "ksmm":
            plan = list(zip(self.chain.stages, self.k_bmm))

            def call():
                y = x
                for pattern, k in plan:
                    y = ks_forward(y, k, pattern, layout="BSL")
                return y

            return call, {}
        w16 = self.composed16(torch)
        if impl == "cublas":
            return (lambda: torch.mm(w16, x)), {}
        if impl == "cusparse":
            csr = w16.to_sparse_csr()
            csr = torch.sparse_csr_tensor(csr.crow_indices().int(), csr.col_indices().int(), csr.values(),
                                          size=csr.shape)
            extra = {"csr_nnz": int(csr.values().numel())}
            del w16
            if x.shape[1] == 1:
                xv = x.view(-1)
                return (lambda: torch.mv(csr, xv)), extra
            return (lambda: torch.mm(csr, x)), extra
        raise ValueError(impl)


def rel_errors(torch, out, ref32):
    out32 = out.float().reshape(ref32.shape)
    finite = bool(torch.isfinite(out32).all().item())
    rel_l2 = (torch.linalg.vector_norm(out32 - ref32) / torch.linalg.vector_norm(ref32)).item()
    max_rel = ((out32 - ref32).abs().max() / ref32.abs().max()).item()
    return finite, rel_l2, max_rel


def capture_graph(torch, call):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = call()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def bench_graph(torch, graph, pre, warmup_ms=WARMUP_MS, rep_ms=REP_MS):
    """Event-timed graph replays; `pre` runs before each replay, outside the window."""
    est_s = torch.cuda.Event(enable_timing=True)
    est_e = torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        pre()
        graph.replay()
    torch.cuda.synchronize()
    est_s.record()
    for _ in range(5):
        pre()
        graph.replay()
    est_e.record()
    torch.cuda.synchronize()
    per_iter_ms = max(est_s.elapsed_time(est_e) / 5, 1e-3)
    n_warm = max(1, int(warmup_ms / per_iter_ms))
    n_rep = max(10, int(rep_ms / per_iter_ms))
    for _ in range(n_warm):
        pre()
        graph.replay()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_rep)]
    for i in range(n_rep):
        pre()
        starts[i].record()
        graph.replay()
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]


def host_issue_us(torch, call, reps=20):
    """Host time to issue one eager call while a spin kernel keeps the GPU busy.

    NaN when the spin kernel had already finished at the end of the loop, i.e. the
    loop may have waited on the GPU.
    """
    call()
    torch.cuda.synchronize()
    torch.cuda._sleep(100_000_000)  # about 70 ms at 1410 MHz
    spin_done = torch.cuda.Event()
    spin_done.record()
    t0 = time.perf_counter()
    for _ in range(reps):
        call()
    t1 = time.perf_counter()
    still_spinning = not spin_done.query()
    torch.cuda.synchronize()
    return (t1 - t0) / reps * 1e6 if still_spinning else float("nan")


def flush_time_us(torch, flush):
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(20):
        s.record()
        flush.zero_()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)
    return quantiles(times)[1]


class Clock:
    """NVML reads of the visible GPU (instantaneous SM / memory clock)."""

    def __init__(self, uuid: str):
        import pynvml

        self.nvml = pynvml
        pynvml.nvmlInit()
        self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)

    def sm_mem(self):
        n = self.nvml
        return (n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_SM),
                n.nvmlDeviceGetClockInfo(self.handle, n.NVML_CLOCK_MEM))

    def info(self) -> dict:
        n = self.nvml
        h = self.handle

        def safe(fn, *a):
            try:
                v = fn(*a)
                return v.decode() if isinstance(v, bytes) else v
            except n.NVMLError as exc:  # not supported on this board / driver
                return f"unavailable: {exc}"

        return {
            "driver": safe(n.nvmlSystemGetDriverVersion),
            "gpu_name": safe(n.nvmlDeviceGetName, h),
            "gpu_uuid": safe(n.nvmlDeviceGetUUID, h),
            "gpu_pci_bus_id": safe(lambda: n.nvmlDeviceGetPciInfo(h).busId),
            "gpu_index_nvml": safe(n.nvmlDeviceGetIndex, h),
            "vbios": safe(n.nvmlDeviceGetVbiosVersion, h),
            "sm_clock_mhz_now": safe(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_SM),
            "mem_clock_mhz_now": safe(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_MEM),
            "sm_clock_mhz_max": safe(n.nvmlDeviceGetMaxClockInfo, h, n.NVML_CLOCK_SM),
            "sm_clock_mhz_applications": safe(n.nvmlDeviceGetApplicationsClock, h, n.NVML_CLOCK_SM),
            "mem_clock_mhz_applications": safe(n.nvmlDeviceGetApplicationsClock, h, n.NVML_CLOCK_MEM),
            "sm_clock_mhz_default_applications": safe(n.nvmlDeviceGetDefaultApplicationsClock, h, n.NVML_CLOCK_SM),
            "gpu_locked_clocks": "not set (no root); applications clocks at default",
            "persistence_mode": safe(n.nvmlDeviceGetPersistenceMode, h),
        }


def git_state() -> dict:
    def run(*cmd):
        return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True).stdout.strip()

    return {"ksmm_commit": run("git", "rev-parse", "HEAD"),
            "ksmm_dirty": bool(run("git", "status", "--porcelain"))}  # untracked files count too


def provenance(torch, triton, clock: Clock | None, flush_us: float) -> dict:
    from importlib import metadata

    def ver(pkg):
        try:
            return metadata.version(pkg)
        except metadata.PackageNotFoundError:
            return None

    props = torch.cuda.get_device_properties(0)
    prov = {
        "time_utc": utc_now(),
        "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_device_name": props.name,
        "torch_device_uuid": str(props.uuid),
        "sm_count": props.multi_processor_count,
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "triton": triton.__version__,
        "nvidia_cublas_cu12": ver("nvidia-cublas-cu12"),
        "nvidia_cusparse_cu12": ver("nvidia-cusparse-cu12"),
        "nvidia_cuda_runtime_cu12": ver("nvidia-cuda-runtime-cu12"),
        "triton_cache_autotuning": os.environ.get("TRITON_CACHE_AUTOTUNING"),
        "loadavg_start": os.getloadavg(),
        "flush_bytes": FLUSH_BYTES,
        "flush_us_median": flush_us,
        "sleep_cycles_warm": SLEEP_CYCLES,
        **git_state(),
    }
    if clock is not None:
        prov.update(clock.info())
    return prov


def ksmm_configs(ks_kernel, chain: Chain, batch: int) -> str:
    """The autotuned config of each stage (cache keyed by a, b, c, d, B, dtypes): BBxBCxBBATCH/warps/stages."""
    out = []
    for a, b, c, d in chain.stages:
        hits = [cfg for key, cfg in ks_kernel.cache.items() if tuple(key[:5]) == (a, b, c, d, batch)]
        if hits:
            kw = hits[0].kwargs
            out.append(f"{kw['BLOCK_SIZE_B']}x{kw['BLOCK_SIZE_C']}x{kw['BLOCK_SIZE_BATCH']}"
                       f"/w{hits[0].num_warps}/s{hits[0].num_stages}")
        else:
            out.append("none")
    return " ".join(out)


def write_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def done_keys(path: Path) -> set:
    keys = set()
    if path.exists():
        for line in open(path, encoding="utf-8"):
            row = json.loads(line)
            if row.get("kind") == "point":
                keys.add((row["dataset"], row["chain"], row["impl"], row["batch"]))
    return keys


def parse_batches(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x]


def select(chains: list[Chain], names: str | None) -> list[Chain]:
    if not names:
        return chains
    wanted = names.split(",")
    by_name = {c.name: c for c in chains}
    return [by_name[n] for n in wanted]


def make_input(torch, chain: Chain, batch: int):
    gen = torch.Generator(device="cuda").manual_seed(chain_seed(chain.name) ^ batch)
    return ((torch.rand((chain.m_in, batch), generator=gen, device="cuda") - 0.5)).half()


def cmd_measure(args) -> None:
    torch, triton, ks_forward, ks_kernel = gpu_stack()
    import triton.testing

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clock = Clock("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    write_jsonl(out_path, {"kind": "provenance", "dataset": args.dataset, "argv": sys.argv,
                           **provenance(torch, triton, clock, flush_time_us(torch, flush))})
    skip = done_keys(out_path)
    impls = args.impls.split(",")
    batch_sets = {impl: parse_batches(getattr(args, f"batches_{impl}")) for impl in IMPLS}
    for chain in select(read_chains(Path(args.chains)), args.only):
        todo = [(impl, b) for impl in impls for b in batch_sets[impl]
                if (args.dataset, chain.name, impl, b) not in skip]
        if not todo:
            continue
        built = Built(torch, chain)
        for impl, batch in todo:
            started = time.perf_counter()
            row = {"kind": "point", "dataset": args.dataset, "chain": chain.name, "impl": impl, "batch": batch,
                   "stages": len(chain.stages), "m_in": chain.m_in, "n_out": chain.n_out, "nnz": chain.nnz,
                   "time_utc": utc_now()}
            graph = call = out = gout = None
            try:
                x = make_input(torch, chain, batch)
                ref = built.reference(torch, x.float())
                call, extra = built.make_call(torch, ks_forward, impl, x)
                row.update(extra)
                out = call()
                torch.cuda.synchronize()
                finite, rel_l2, max_rel = rel_errors(torch, out, ref)
                row.update(eager_finite=finite, eager_rel_l2=rel_l2, eager_max_rel=max_rel)
                e = summary([t * 1000 for t in triton.testing.do_bench(
                    call, warmup=WARMUP_MS, rep=REP_MS, return_mode="all")])
                row.update({f"e_{k}": v for k, v in e.items()})
                graph, gout = capture_graph(torch, call)
                gfinite, grel_l2, gmax_rel = rel_errors(torch, gout, ref)
                row.update(graph_finite=gfinite, graph_rel_l2=grel_l2, graph_max_rel=gmax_rel)
                gc = summary(bench_graph(torch, graph, flush.zero_))
                gw = summary(bench_graph(torch, graph, lambda: torch.cuda._sleep(SLEEP_CYCLES)))
                row.update({f"gc_{k}": v for k, v in gc.items()})
                row.update({f"gw_{k}": v for k, v in gw.items()})
                row["h_us"] = host_issue_us(torch, call)
                if impl == "ksmm":
                    row["ksmm_configs"] = ksmm_configs(ks_kernel, chain, batch)
                ok = finite and gfinite and rel_l2 <= TOL_REL_L2 and grel_l2 <= TOL_REL_L2
                row["check"] = "通过" if ok else "失败"
                row["status"] = "ok"
            except Exception as exc:  # record and continue with the next point
                row.update(status=f"error:{type(exc).__name__}", error=str(exc)[:2000], check="失败")
            finally:
                del graph, call, out, gout
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            row["sm_mhz_end"], row["mem_mhz_end"] = clock.sm_mem()
            row["elapsed_s"] = round(time.perf_counter() - started, 3)
            write_jsonl(out_path, row)
            print(f"{chain.name:34s} {impl:8s} b={batch:<5d} {row.get('check')} "
                  f"E80={row.get('e_p80', float('nan')):9.2f} Gc={row.get('gc_mean', float('nan')):9.2f} "
                  f"Gw={row.get('gw_mean', float('nan')):9.2f} h={row.get('h_us', float('nan')):8.2f} "
                  f"rel={row.get('eager_rel_l2', float('nan')):.2e} {row['status']}", flush=True)
        del built
        torch.cuda.empty_cache()


def cmd_checkpoint(args) -> None:
    """E view of the ten sheet chains: the old harness's own call vs. this harness's ksmm call."""
    torch, triton, ks_forward, ks_kernel = gpu_stack()
    import triton.testing

    sys.path.insert(0, args.legacy_root)
    from a100bench.runner import _make_ksmm_call  # read-only use of the old harness

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clock = Clock("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    write_jsonl(out_path, {"kind": "provenance", "dataset": args.dataset, "argv": sys.argv,
                           **provenance(torch, triton, clock, flush_time_us(torch, flush))})
    sheet = {r["chain"]: float(r["time_us"]) for r in csv.DictReader(open(args.sheet, encoding="utf-8"))}
    chains = read_chains(Path(args.chains))
    for rnd in range(args.rounds):
        for chain in chains:
            point = {"stages": [[a, b, c, d, 1] for a, b, c, d in chain.stages], "batch": 1, "M": chain.m_in}
            torch.manual_seed(chain_seed(chain.name))
            legacy_call, legacy_meta = _make_ksmm_call(point, torch, ks_forward)
            legacy_call()
            torch.cuda.synchronize()
            legacy = triton.testing.do_bench(legacy_call, warmup=WARMUP_MS, rep=REP_MS, quantiles=QUANTILES)
            built = Built(torch, chain)
            x = make_input(torch, chain, 1)
            call, _ = built.make_call(torch, ks_forward, "ksmm", x)
            call()
            torch.cuda.synchronize()
            ours = triton.testing.do_bench(call, warmup=WARMUP_MS, rep=REP_MS, quantiles=QUANTILES)
            sm, mem = clock.sm_mem()
            row = {"kind": "checkpoint", "round": rnd, "chain": chain.name, "sheet_p80": sheet[chain.name],
                   "legacy_p20": legacy[0] * 1000, "legacy_p50": legacy[1] * 1000, "legacy_p80": legacy[2] * 1000,
                   "ours_p20": ours[0] * 1000, "ours_p50": ours[1] * 1000, "ours_p80": ours[2] * 1000,
                   "legacy_layout": legacy_meta["layout"], "ksmm_configs": ksmm_configs(ks_kernel, chain, 1),
                   "sm_mhz_end": sm, "mem_mhz_end": mem, "loadavg": os.getloadavg(), "time_utc": utc_now()}
            write_jsonl(out_path, row)
            print(f"r{rnd} {chain.name:8s} sheet={row['sheet_p80']:7.2f} legacy={row['legacy_p80']:7.2f} "
                  f"ours={row['ours_p80']:7.2f} dev={(row['legacy_p80'] / row['sheet_p80'] - 1) * 100:+6.1f}% "
                  f"sm={sm}", flush=True)
            del legacy_call, built, call
            torch.cuda.empty_cache()


def cmd_profile(args) -> None:
    """Workload for nsys / ncu: NVTX ranges around eager calls and graph replays."""
    torch, triton, ks_forward, _ks_kernel = gpu_stack()
    nvtx = torch.cuda.nvtx
    flush = torch.empty(FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    modes = args.modes.split(",")
    for chain in select(read_chains(Path(args.chains)), args.only):
        built = Built(torch, chain)
        for impl in args.impls.split(","):
            x = make_input(torch, chain, args.batch)
            call, _ = built.make_call(torch, ks_forward, impl, x)
            call()
            torch.cuda.synchronize()
            graph, _gout = capture_graph(torch, call)
            tag = f"{args.dataset}|{chain.name}|{impl}|{args.batch}"
            for mode in modes:
                for rep in range(args.reps):
                    if mode == "GW":
                        graph.replay()
                        torch.cuda._sleep(SLEEP_CYCLES)
                    else:
                        flush.zero_()
                    nvtx.range_push(f"{mode}|{tag}|{rep}")
                    if args.range_name:  # inner fixed name for ncu --nvtx-include
                        nvtx.range_push(args.range_name)
                    if mode == "E":
                        call()
                    else:
                        graph.replay()
                    if args.range_name:
                        nvtx.range_pop()
                    nvtx.range_pop()
                torch.cuda.synchronize()
            del graph, call
            torch.cuda.empty_cache()
        del built
        torch.cuda.empty_cache()


def run_length(names: list[str]) -> str:
    """'a a a b' -> 'a*3 b' (kernel names in launch order)."""
    out: list[list] = []
    for name in names:
        if out and out[-1][0] == name:
            out[-1][1] += 1
        else:
            out.append([name, 1])
    return " ".join(f"{n}*{k}" if k > 1 else n for n, k in out)


def cmd_nsys_parse(args) -> None:
    """Sum GPU activity per NVTX range of an exported nsys sqlite file.

    A kernel (or memset / memcpy) belongs to a range when the CUDA API call that launched it
    (same correlationId; for graph nodes the cudaGraphLaunch call) starts inside the range on
    the same thread.
    """
    db = sqlite3.connect(args.sqlite)
    cur = db.cursor()
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges = []
    for start, end, text, text_id, tid in cur.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND eventType = 59"):
        label = text if text is not None else strings.get(text_id)
        if label and label.count("|") >= 4:
            ranges.append((start, end, label, tid))
    api = list(cur.execute("SELECT start, correlationId, globalTid, nameId FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    api.sort()
    gpu = {}
    if "CUPTI_ACTIVITY_KIND_KERNEL" in tables:
        for start, end, corr, name_id, gx, gy, gz, bx, by, bz in cur.execute(
                "SELECT start, end, correlationId, demangledName, gridX, gridY, gridZ, blockX, blockY, blockZ "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL"):
            gpu.setdefault(corr, []).append((start, end, short_name(strings.get(name_id, "?")), f"{gx}x{gy}x{gz}",
                                             f"{bx}x{by}x{bz}"))
    for table, kind in (("CUPTI_ACTIVITY_KIND_MEMSET", "memset"), ("CUPTI_ACTIVITY_KIND_MEMCPY", "memcpy")):
        if table in tables:
            for start, end, corr in cur.execute(f"SELECT start, end, correlationId FROM {table}"):
                gpu.setdefault(corr, []).append((start, end, kind, "", ""))
    import bisect

    starts = [a[0] for a in api]
    out_rows = []
    for r_start, r_end, label, tid in ranges:
        lo, hi = bisect.bisect_left(starts, r_start), bisect.bisect_right(starts, r_end)
        acts = []
        api_names = set()
        for a_start, corr, a_tid, name_id in api[lo:hi]:
            if a_tid == tid and corr in gpu:
                acts.extend(gpu[corr])
                api_names.add(strings.get(name_id, "?"))
        acts.sort()
        mode, dataset, chain, impl, batch, rep = label.split("|")
        dur = [(e - s) / 1000.0 for s, e, *_ in acts]
        out_rows.append({
            "mode": mode, "dataset": dataset, "chain": chain, "impl": impl, "batch": int(batch), "rep": int(rep),
            "kernels": sum(1 for a in acts if a[2] not in ("memset", "memcpy")),
            "other_ops": sum(1 for a in acts if a[2] in ("memset", "memcpy")),
            "d_us": round(sum(dur), 3),
            "span_us": round((max(e for _s, e, *_ in acts) - min(s for s, *_ in acts)) / 1000.0, 3) if acts else None,
            "gaps_us": round((max(e for _s, e, *_ in acts) - min(s for s, *_ in acts)) / 1000.0 - sum(dur), 3)
            if acts else None,
            "kernel_us": " ".join(f"{d:.3f}" for d in dur),
            "kernel_names": run_length([a[2] for a in acts]),
            "grids": " ".join(a[3] for a in acts),
            "blocks": " ".join(sorted({a[4] for a in acts if a[4]})),
            "api": " ; ".join(sorted(api_names)),
        })
    with open(args.out, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(out_rows[0]))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"{len(out_rows)} ranges -> {args.out}")


NCU_METRICS = {  # output column -> ncu metric (import with --print-units base)
    "duration_us": "gpu__time_duration.sum",
    "sm_clock_ghz": "sm__cycles_elapsed.avg.per_second",
    "dram_pct": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_read_bytes": "dram__bytes_read.sum",
    "dram_write_bytes": "dram__bytes_write.sum",
    "l2_hit_sectors": "lts__t_sectors_lookup_hit.sum",
    "l2_miss_sectors": "lts__t_sectors_lookup_miss.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "occ_achieved_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theoretical_pct": "sm__maximum_warps_per_active_cycle_pct",
    "grid": "launch__grid_size",
    "block": "launch__block_size",
    "waves": "launch__waves_per_multiprocessor",
    "regs": "launch__registers_per_thread",
    "tensor_pct": "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active",
    "hmma_inst": "sm__inst_executed_pipe_tensor_op_hmma.sum",
}


def short_name(name: str) -> str:
    """Demangled kernel name without template arguments, parameters and return type."""
    out, depth = [], 0
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out).split("(")[0].split()[-1]


def cmd_ncu_parse(args) -> None:
    """Per-kernel rows from `ncu --import X --csv --page raw --print-units base` files."""
    import re

    tag_re = re.compile(r"(E|GC|GW)\|([^|\s]+)\|([^|\s]+)\|([^|\s]+)\|(\d+)\|(\d+)")
    rows = []
    for path in args.csv:
        with open(path, encoding="utf-8", newline="") as stream:
            table = list(csv.reader(stream))
        head, units, body = table[0], table[1], table[2:]
        col = {n: i for i, n in enumerate(head)}
        nvtx_col = next(i for i, n in enumerate(head) if n.startswith("thread Domain:Push/Pop_Range"))
        for metric in NCU_METRICS.values():
            unit = units[col[metric]]
            if metric.startswith("gpu__time_duration") and unit != "ns":
                raise ValueError(f"{path}: {metric} unit {unit!r}; import with --print-units base")
        order: dict = {}
        for r in body:
            m = tag_re.search(r[nvtx_col])
            if not m:
                raise ValueError(f"{path}: kernel {r[col['ID']]} has no harness NVTX tag: {r[nvtx_col][:200]}")
            mode, dataset, chain, impl, batch, _rep = m.groups()
            key = (mode, dataset, chain, impl, int(batch))
            order[key] = order.get(key, -1) + 1
            row = {"mode": mode, "dataset": dataset, "chain": chain, "impl": impl, "batch": int(batch),
                   "idx": order[key], "kernel": short_name(r[col["Kernel Name"]])}
            for out_col, metric in NCU_METRICS.items():
                v = float(r[col[metric]].replace(",", ""))
                if out_col == "duration_us":
                    v /= 1000.0
                elif out_col == "sm_clock_ghz":
                    v /= 1e9
                row[out_col] = int(v) if out_col in ("grid", "block", "regs", "dram_read_bytes", "dram_write_bytes",
                                                      "hmma_inst", "l2_hit_sectors", "l2_miss_sectors") else round(v, 4)
            # Hit rate from the two counters, so it stays within [0, 100] even when ncu's own
            # lts__t_sector_hit_rate.pct mixes passes (seen above 100 % on tiny kernels).
            lookups = row["l2_hit_sectors"] + row["l2_miss_sectors"]
            row["l2_hit_pct"] = round(100.0 * row["l2_hit_sectors"] / lookups, 2) if lookups else None
            rows.append(row)
    with open(args.out, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} kernels -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("measure", help="E, G_cold, G_warm, h and numerics per chain x impl x batch")
    m.add_argument("--chains", required=True)
    m.add_argument("--dataset", required=True)
    m.add_argument("--out", required=True, help="JSONL; existing points are skipped (resume)")
    m.add_argument("--impls", default=",".join(IMPLS))
    m.add_argument("--only", help="comma-separated chain names")
    for impl in IMPLS:
        m.add_argument(f"--batches-{impl}", default="1")
    m.set_defaults(fn=cmd_measure)

    c = sub.add_parser("checkpoint", help="E view of the sheet chains with the old harness's call")
    c.add_argument("--chains", required=True)
    c.add_argument("--dataset", required=True)
    c.add_argument("--sheet", required=True, help="runs/sheet-a100/results.csv")
    c.add_argument("--legacy-root", default="/scratch/users/phd/yd383/do-bench-harness")
    c.add_argument("--rounds", type=int, default=3)
    c.add_argument("--out", required=True)
    c.set_defaults(fn=cmd_checkpoint)

    r = sub.add_parser("profile", help="workload for nsys / ncu")
    r.add_argument("--chains", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--impls", default=",".join(IMPLS))
    r.add_argument("--only")
    r.add_argument("--batch", type=int, default=1)
    r.add_argument("--modes", default="E,GC,GW", help="E eager+flush, GC graph+flush, GW graph after a warm replay")
    r.add_argument("--reps", type=int, default=20)
    r.add_argument("--range-name", help="inner NVTX range with this fixed name (for ncu --nvtx-include)")
    r.set_defaults(fn=cmd_profile)

    n = sub.add_parser("nsys-parse", help="per-NVTX-range GPU time from an nsys sqlite export")
    n.add_argument("--sqlite", required=True)
    n.add_argument("--out", required=True)
    n.set_defaults(fn=cmd_nsys_parse)

    q = sub.add_parser("ncu-parse", help="per-kernel metrics from ncu raw CSV exports")
    q.add_argument("--csv", nargs="+", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(fn=cmd_ncu_parse)

    sub.add_parser("ncu-metrics", help="print the ncu --metrics list").set_defaults(
        fn=lambda _args: print(",".join(NCU_METRICS.values())))

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
