"""GPU probes for operations task 2026-10-14-v41-baseline-audit, step T8 (GPU bottleneck analysis).

No new compute kernel: every kernel here is either the KSMM-Triton kernel `ks_fused_kernel`
(same source as T4), cuBLAS through torch.mm, or PyTorch's own spin kernel
`torch.cuda._sleep(0)` used as an empty kernel. Timing reuses `a100_audit.bench_graph`
(CUDA graph replay, event timing, L2 flushed outside the window for G_cold).

Subcommands
  floor     fixed costs: graphs of K empty kernels (K = 1..64), graphs of K one-CTA KSMM
            kernels, and graphs of 1 and 8 KSMM kernels of shape (a, 2, 2, 1) for a range
            of CTA counts a. Per-kernel cost = (G(8) - G(1)) / 7; intercept = graph launch.
  configs   Triton configurations of ks_fused_kernel outside T4's 6-config autotune space.
            Each config is forced through a one-config autotuner around the same JIT
            function (`ks_fused_kernel.fn`); only BLOCK_SIZE_BATCH, num_warps, num_stages
            change. Outputs are compared with the default (T4-autotuned) output.
  flushcmp  G_cold with T4's L2 flush (zero_() of a 256 MB buffer, leaving dirty lines in L2) against a
            read-only flush (sum() over another 256 MB buffer, leaving clean lines); random operands.
  profile   ncu / nsys workload in the format of `a100_audit.py profile` (same NVTX tags),
            but with random weights and no fp32 composition, so 2^16-point chains and
            their dense matrices can be profiled quickly. Optional --force-config.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import a100_audit as base  # noqa: E402  (same directory)

KEY = ["a", "b", "c", "d", "B"]


# --------------------------------------------------------------------------- helpers


def parse_config(text: str):
    """'BBATCH/warps/stages', e.g. '64/4/3' -> triton.Config with BLOCK_SIZE_B = C = 16."""
    import triton

    bb, w, s = (int(x) for x in text.split("/"))
    return triton.Config({"BLOCK_SIZE_B": 16, "BLOCK_SIZE_C": 16, "BLOCK_SIZE_BATCH": bb},
                         num_warps=w, num_stages=s)


def config_text(cfg) -> str:
    kw = cfg.kwargs
    return f"{kw['BLOCK_SIZE_B']}x{kw['BLOCK_SIZE_C']}x{kw['BLOCK_SIZE_BATCH']}/w{cfg.num_warps}/s{cfg.num_stages}"


class KernelSwap:
    """Replace ksmm_triton_tc.ks_fused_kernel (looked up at call time by ks_triton_forward_impl)."""

    def __init__(self):
        import ksmm_triton.ksmm_triton_tc as tc

        self.tc = tc
        self.default = tc.ks_fused_kernel

    def use(self, configs):
        import triton

        if configs is None:
            self.tc.ks_fused_kernel = self.default
        else:
            self.tc.ks_fused_kernel = triton.autotune(configs=list(configs), key=KEY)(self.default.fn)
        return self.tc.ks_fused_kernel


def random_ksmm(torch, stages, batch, seed=0):
    """Random fp16 KSMM weights in (A*D, C, B) layout and a (M_in, batch) input; no reference."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    ks = []
    for a, b, c, d in stages:
        bound = math.sqrt(3.0 / c)
        ks.append(((torch.rand((a * d, c, b), generator=gen, device="cuda") * 2 - 1) * bound).half())
    a, _b, c, d = stages[0]
    x = (torch.rand((a * c * d, batch), generator=gen, device="cuda") - 0.5).half()
    return ks, x


def ksmm_call(ks_forward, stages, ks, x):
    plan = list(zip(stages, ks))

    def call():
        y = x
        for pattern, k in plan:
            y = ks_forward(y, k, pattern, layout="BSL")
        return y

    return call


def timed(torch, graph, flush, n_min=0):
    gc = base.summary(base.bench_graph(torch, graph, flush.zero_))
    gw = base.summary(base.bench_graph(torch, graph, lambda: torch.cuda._sleep(base.SLEEP_CYCLES)))
    return {"gc_mean": gc["mean"], "gc_p50": gc["p50"], "gc_n": gc["n"],
            "gw_mean": gw["mean"], "gw_p50": gw["p50"], "gw_n": gw["n"]}


def setup(args, kind):
    torch, triton, ks_forward, _ = base.gpu_stack()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    clock = base.Clock("GPU-" + str(torch.cuda.get_device_properties(0).uuid))
    flush = torch.empty(base.FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    base.write_jsonl(out, {"kind": "provenance", "probe": kind, "argv": sys.argv,
                           **base.provenance(torch, triton, clock, base.flush_time_us(torch, flush))})
    return torch, triton, ks_forward, clock, flush, out


# --------------------------------------------------------------------------- floor


def cmd_floor(args) -> None:
    torch, _triton, ks_forward, clock, flush, out = setup(args, "floor")

    def record(row):
        row["sm_mhz_end"], row["mem_mhz_end"] = clock.sm_mem()
        base.write_jsonl(out, row)
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)

    # 1. graphs of K empty kernels (PyTorch's spin kernel with 0 cycles: 1 block, 1 thread)
    for k in (1, 2, 4, 8, 16, 32, 64):
        graph, _ = base.capture_graph(torch, lambda k=k: [torch.cuda._sleep(0) for _ in range(k)])
        record({"kind": "empty", "kernels": k, **timed(torch, graph, flush)})
        del graph

    # 2. graphs of K one-CTA KSMM kernels, stage (1, 2, 2, 1) repeated, batch 1
    for k in (1, 2, 4, 8, 16, 32):
        stages = [(1, 2, 2, 1)] * k
        ks, x = random_ksmm(torch, stages, 1)
        graph, _ = base.capture_graph(torch, ksmm_call(ks_forward, stages, ks, x))
        record({"kind": "ksmm_1cta", "kernels": k, "pattern": "1,2,2,1", "batch": 1, **timed(torch, graph, flush)})
        del graph, ks, x

    # 3. CTA-count sweep: (a, b, b, 1) repeated R times, R = 1 and 8, batch 1
    for radix in (2, 4):
        for a in (1, 16, 108, 216, 432, 864, 1728, 2048, 4096, 8192, 16384, 32768, 65536):
            for reps in (1, 8):
                stages = [(a, radix, radix, 1)] * reps
                ks, x = random_ksmm(torch, stages, 1)
                graph, _ = base.capture_graph(torch, ksmm_call(ks_forward, stages, ks, x))
                record({"kind": "ksmm_ctas", "radix": radix, "ctas": a, "kernels": reps,
                        "pattern": f"{a},{radix},{radix},1", "batch": 1, **timed(torch, graph, flush)})
                del graph, ks, x
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- configs


def cfg_space(batch: int) -> list[str]:
    bbs = (16, 32) if batch <= 32 else (16, 32, 64, 128, 256)
    return [f"{bb}/{w}/{s}" for bb in bbs for w in (1, 2, 4, 8) for s in (1, 3)]


def cmd_configs(args) -> None:
    torch, _triton, ks_forward, clock, flush, out = setup(args, "configs")
    swap = KernelSwap()
    if args.chains:
        chains = base.select(base.read_chains(Path(args.chains)), args.only)
        jobs = [(c.name, c.stages) for c in chains]
    else:  # a single stage shape repeated
        a, b, c, d = (int(v) for v in args.stage.split(","))
        reps = args.repeat if b == c else 1
        jobs = [(f"stage_{a}x{b}x{c}x{d}x{reps}", [(a, b, c, d)] * reps)]
    for batch in base.parse_batches(args.batches):
        space = cfg_space(batch)
        forced = args.configs.split(",") if args.configs else space
        for name, stages in jobs:
            ks, x = random_ksmm(torch, stages, batch, seed=base.chain_seed(name))
            # reference: the T4 autotuner (same TRITON_CACHE_DIR contents as T4, so the same picks)
            swap.use(None)
            ref = ksmm_call(ks_forward, stages, ks, x)()
            torch.cuda.synchronize()
            ref32 = ref.float()
            variants = [("default", None)] + [(t, [parse_config(t)]) for t in forced if t != "none"]
            if args.extended_autotune:
                variants.append(("autotune-ext", [parse_config(t) for t in space]))
            for label, cfgs in variants:
                row = {"kind": "config", "name": name, "stages": len(stages), "batch": batch,
                       "pattern0": ",".join(map(str, stages[0])), "variant": label}
                started = time.perf_counter()
                graph = None
                try:
                    kern = swap.use(cfgs)
                    call = ksmm_call(ks_forward, stages, ks, x)
                    y = call()
                    torch.cuda.synchronize()
                    row["compile_s"] = round(time.perf_counter() - started, 3)
                    diff = (torch.linalg.vector_norm(y.float() - ref32) / torch.linalg.vector_norm(ref32)).item()
                    row["rel_diff_vs_default"] = diff
                    row["check"] = "通过" if diff <= 1e-3 and bool(torch.isfinite(y).all().item()) else "失败"
                    if label in ("default", "autotune-ext"):
                        row["picked"] = " ".join(
                            config_text(cfg) for key, cfg in kern.cache.items()
                            if tuple(key[:5]) in {(a, b, c, d, batch) for a, b, c, d in stages})
                    graph, _ = base.capture_graph(torch, call)
                    row.update(timed(torch, graph, flush))
                    row["status"] = "ok"
                except Exception as exc:  # e.g. resources exceeded for a config
                    row.update(status=f"error:{type(exc).__name__}", error=str(exc)[:500], check="失败")
                finally:
                    del graph
                    torch.cuda.synchronize()
                row["elapsed_s"] = round(time.perf_counter() - started, 3)
                row["sm_mhz_end"], row["mem_mhz_end"] = clock.sm_mem()
                base.write_jsonl(out, row)
                print(f"{name:28s} b={batch:<5d} {label:14s} {row['status']:10s} "
                      f"Gc={row.get('gc_mean', float('nan')):9.2f} Gw={row.get('gw_mean', float('nan')):9.2f} "
                      f"diff={row.get('rel_diff_vs_default', float('nan')):.1e} {row.get('picked', '')[:60]}",
                      flush=True)
            swap.use(None)
            del ks, x, ref, ref32
            torch.cuda.empty_cache()


# --------------------------------------------------------------------------- flushcmp


def random_call(torch, ks_forward, chain, impl: str, batch: int):
    """Random operands (timing does not depend on values) and the call of one implementation."""
    if impl == "ksmm":
        ks, x = random_ksmm(torch, chain.stages, batch, seed=base.chain_seed(chain.name))
        return ksmm_call(ks_forward, chain.stages, ks, x)
    if impl == "cublas":  # random dense N_out x M_in matrix: GEMV / GEMM time does not depend on values
        gen = torch.Generator(device="cuda").manual_seed(base.chain_seed(chain.name))
        w = (torch.rand((chain.n_out, chain.m_in), generator=gen, device="cuda", dtype=torch.float16) - 0.5)
        x = (torch.rand((chain.m_in, batch), generator=gen, device="cuda", dtype=torch.float16) - 0.5)
        return lambda: torch.mm(w, x)
    raise ValueError(impl)


def cmd_flushcmp(args) -> None:
    """G_cold with T4's flush (zero_() a 256 MB buffer: L2 left with dirty lines) against a read-only flush
    (sum() over another 256 MB buffer: L2 left with clean lines). Same graph, alternating, `--rounds` each."""
    torch, _triton, ks_forward, clock, flush, out = setup(args, "flushcmp")
    clean_buf = torch.ones(base.FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    every = {"zero": flush.zero_, "clean": lambda: clean_buf.sum()}
    pres = {m: every[m] for m in args.order.split(",")}  # measured in this order in every round
    for chain in base.select(base.read_chains(Path(args.chains)), args.only):
        for impl in args.impls.split(","):
            row = {"kind": "flushcmp", "tag": args.tag, "order": args.order, "dataset": args.dataset,
                   "chain": chain.name, "impl": impl,
                   "batch": args.batch, "m_in": chain.m_in, "n_out": chain.n_out, "stages": len(chain.stages)}
            graph = call = None
            try:
                call = random_call(torch, ks_forward, chain, impl, args.batch)
                call()
                torch.cuda.synchronize()
                graph, _ = base.capture_graph(torch, call)
                samples = {m: [] for m in pres}
                for _ in range(args.rounds):
                    for mode, pre in pres.items():
                        samples[mode].append(base.summary(base.bench_graph(torch, graph, pre))["mean"])
                for mode, v in samples.items():
                    row[f"{mode}_us"] = sum(v) / len(v)
                    row[f"{mode}_rounds"] = v
                row["status"] = "ok"
            except Exception as exc:
                row.update(status=f"error:{type(exc).__name__}", error=str(exc)[:500])
            finally:
                del graph, call
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            row["sm_mhz_end"], row["mem_mhz_end"] = clock.sm_mem()
            base.write_jsonl(out, row)
            print(f"{chain.name:28s} {impl:7s} zero={row.get('zero_us', float('nan')):9.2f} "
                  f"clean={row.get('clean_us', float('nan')):9.2f} {row['status']}", flush=True)


# --------------------------------------------------------------------------- profile


def cmd_profile(args) -> None:
    """Same NVTX tags as a100_audit profile ('MODE|dataset|chain|impl|batch|rep', inner 'prof')."""
    torch, _triton, ks_forward, _ = base.gpu_stack()
    nvtx = torch.cuda.nvtx
    swap = KernelSwap()
    if args.force_config:
        swap.use([parse_config(args.force_config)])
    flush = torch.empty(base.FLUSH_BYTES // 4, dtype=torch.int32, device="cuda")
    for chain in base.select(base.read_chains(Path(args.chains)), args.only):
        for impl in args.impls.split(","):
            call = random_call(torch, ks_forward, chain, impl, args.batch)
            call()
            torch.cuda.synchronize()
            graph, _ = base.capture_graph(torch, call)
            tag = f"{args.dataset}|{chain.name}|{impl}|{args.batch}"
            for rep in range(args.reps):
                flush.zero_()
                nvtx.range_push(f"GC|{tag}|{rep}")
                nvtx.range_push(args.range_name)
                graph.replay()
                nvtx.range_pop()
                nvtx.range_pop()
            torch.cuda.synchronize()
            del graph, call
            torch.cuda.empty_cache()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("floor", help="fixed costs: empty kernels, one-CTA KSMM, CTA-count sweep")
    f.add_argument("--out", required=True)
    f.set_defaults(fn=cmd_floor)

    c = sub.add_parser("configs", help="KSMM Triton configs outside T4's autotune space")
    c.add_argument("--out", required=True)
    c.add_argument("--chains", help="chains.csv; with --only")
    c.add_argument("--only")
    c.add_argument("--stage", help="A,B,C,D: one stage shape instead of chains")
    c.add_argument("--repeat", type=int, default=8, help="repeat a square stage this many times")
    c.add_argument("--batches", default="1")
    c.add_argument("--configs", help="comma-separated BBATCH/warps/stages to force one by one ('none': only "
                                     "default and autotune-ext); default: the whole space of the batch")
    c.add_argument("--extended-autotune", action="store_true",
                   help="also let the autotuner pick per stage from the whole space of the batch")
    c.set_defaults(fn=cmd_configs)

    fc = sub.add_parser("flushcmp", help="G_cold with T4's zero_() flush vs a read-only flush")
    fc.add_argument("--out", required=True)
    fc.add_argument("--chains", required=True)
    fc.add_argument("--dataset", required=True)
    fc.add_argument("--only")
    fc.add_argument("--impls", default="cublas")
    fc.add_argument("--batch", type=int, default=1)
    fc.add_argument("--rounds", type=int, default=2)
    fc.add_argument("--order", default="zero,clean", help="flush kinds in measuring order within a round")
    fc.add_argument("--tag", default="", help="free text stored with every row (e.g. a repeat)")
    fc.set_defaults(fn=cmd_flushcmp)

    r = sub.add_parser("profile", help="ncu / nsys workload without fp32 composition")
    r.add_argument("--chains", required=True)
    r.add_argument("--dataset", required=True)
    r.add_argument("--impls", default="ksmm")
    r.add_argument("--only")
    r.add_argument("--batch", type=int, default=1)
    r.add_argument("--reps", type=int, default=1)
    r.add_argument("--range-name", default="prof")
    r.add_argument("--force-config", help="BBATCH/warps/stages for every ksmm stage")
    r.set_defaults(fn=cmd_profile)

    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
