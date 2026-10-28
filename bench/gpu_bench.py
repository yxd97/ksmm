"""GPU time and power of KSMM chains (operations task 2026-10-27-gpu-bench, step T3).

Workload: one CUDA graph per chain holding n_runs back-to-back runs of the chain (no L2 flush
between runs). A run applies the whole chain once to the input X (M_in, batch) and gives
Y (N_out, batch), fp16, feature-major (BSL), stage 0 first, original dimensions. Weights and
input come from a100_audit.Built / make_input (seed: crc32 of the chain name).

Subcommands (run them through run_host.sh, which picks the GPU):

  idle    idle power of the GPU in CUDA_VISIBLE_DEVICES; NVML only, no CUDA context.
  time    per chain: eager run + check, capture the n_runs graph + check, WARMUP_S seconds of
          continuous replays (stable clocks), then R back-to-back replays with a CUDA event between replays, all submitted while a
          wait kernel holds the GPU; device_us = event gap / n_runs. No tracing.
  power   per chain: pick power_n_runs (busy_pct >= 95 %), replay 120 s, energy from 30 s to 90 s.

Outputs in --out: time.jsonl, power.jsonl, idle.jsonl (one row per chain or measurement, plus
a provenance row per process), power-samples/*.csv. Each command skips chains already in its
JSONL (resume). See README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, NamedTuple

ROOT = Path(__file__).resolve().parents[1]  # ksmm repository root
sys.path.insert(0, str(ROOT / "audit"))
import a100_audit as audit  # noqa: E402  (also puts ksmm/src on sys.path)

N_RUNS = 20  # runs per graph for timing
WARMUP_S = 0.5  # continuous replays before timing, so the GPU reaches its steady clock (calibrated, README)
REPLAYS = 100  # timed replays R; fewer for long replays:
LONG_REPLAY_MS = 10.0  # a replay longer than this gets R = about TIMED_TOTAL_MS of replays,
TIMED_TOTAL_MS = 1000.0
MIN_REPLAYS = 10  # but at least this many
WAIT_CYCLES = 100_000_000  # first length of the wait kernel (torch.cuda._sleep), doubled when too short
WAIT_TRIES = 6
DENSE_REFERENCE = {"b1"}  # datasets checked against the fp32 composed matrix; others: einsum stage by stage
POWER_S = 120.0  # replay (or idle) time per power measurement
WINDOW = (30.0, 90.0)  # energy window, seconds after the start
SPREAD_BIN_S = 5.0  # power_spread_pct: bin means over the window
SAMPLE_S = 0.1  # NVML sampling period
TRIAL_S = 5.0  # trial replay time per power_n_runs candidate
BUSY_MIN_PCT = 95.0
POWER_N_RUNS_MAX = N_RUNS * 2 ** 10
LEAD = 8  # the host stays at most this many replays ahead of the GPU
EXIT_BUSY = 75  # another process is on the GPU


# --------------------------------------------------------------------------- implementations


class Impl(NamedTuple):
    run: Callable[[], object]  # one run: the whole chain on x once -> (N_out, batch) fp16 tensor
    config: Callable[[], str]  # kernel configuration used; called after the first (eager) run


def make_triton(torch, built, x) -> Impl:
    """KSMM-Triton: ks_triton_forward_impl per stage, K of shape (A*D, C, B), BSL layout."""
    from ksmm_triton.ksmm_triton_tc import ks_fused_kernel, ks_triton_forward_impl

    plan = list(zip(built.chain.stages, built.k_bmm))

    def run():
        y = x
        for pattern, k in plan:
            y = ks_triton_forward_impl(y, k, pattern, layout="BSL")
        return y

    return Impl(run, lambda: audit.ksmm_configs(ks_fused_kernel, built.chain, x.shape[1]))


def make_cuda(torch, built, x) -> Impl:
    """KSMM-CUDA: ksmm's bs_last fp16 kernels, batch 25088 hyperparameters only (ksmm_cuda/build.py)."""
    import ksmm_cuda

    return Impl(*ksmm_cuda.make(torch, built, x))


# name -> make(torch, built, x) -> Impl. built: a100_audit.Built (built.chain; built.w4: fp16
# (A, B, C, D) weights per stage; built.k_bmm: the same as (A*D, C, B)); x: (M_in, batch) fp16.
IMPLS: dict[str, Callable] = {"triton": make_triton, "cuda": make_cuda}


# --------------------------------------------------------------------------- helpers


def read_rows(path: Path, kind: str) -> list[dict]:
    if not path.exists():
        return []
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    return [r for r in rows if r.get("kind") == kind]


def latest(path: Path, kind: str) -> dict:
    """chain -> last row of this kind."""
    return {r["chain"]: r for r in read_rows(path, kind)}


def gpu_processes() -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name", "--format=csv,noheader"],
                         capture_output=True, text=True, check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.splitlines() if line.strip()]


def exit_if_busy(uuid: str) -> None:
    """Stop with exit code 75 when another process is on this GPU; finished rows stay."""
    others = [p for p in gpu_processes() if p[0] == uuid and int(p[1]) != os.getpid()]
    if others:
        print(f"{uuid} is busy: {others}; stopping (exit {EXIT_BUSY})", flush=True)
        sys.exit(EXIT_BUSY)


def provenance(clock, torch=None, triton=None) -> dict:
    from importlib import metadata

    prov = {
        "kind": "provenance", "time_utc": audit.utc_now(), "argv": sys.argv, "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(), "python_executable": sys.executable,
        "loadavg_start": os.getloadavg(), "gpu_processes_start": gpu_processes(),
        **audit.git_state(), **clock.info(),
    }
    if torch is not None and triton is not None:
        props = torch.cuda.get_device_properties(0)
        prov.update(torch=torch.__version__, torch_cuda=torch.version.cuda, triton=triton.__version__,
                    cuda_runtime=metadata.version("nvidia-cuda-runtime-cu12"),
                    torch_device_name=props.name, torch_device_uuid=f"GPU-{props.uuid}",
                    sm_count=props.multi_processor_count,
                    triton_cache_autotuning=os.environ.get("TRITON_CACHE_AUTOTUNING"))
    return prov


def setup(args):
    """Torch stack, chains to do, the impl factory and the NVML clock of device 0."""
    torch, triton, _ks_forward, _ks_kernel = audit.gpu_stack()
    dataset = Path(args.chains).parent.name
    if dataset not in DENSE_REFERENCE:
        audit.DENSE32_MAX_BYTES = 0  # Built then skips the composed matrix; reference() goes stage by stage
    chains = audit.select(audit.read_chains(Path(args.chains)), args.only)
    uuid = f"GPU-{torch.cuda.get_device_properties(0).uuid}"
    return torch, triton, dataset, chains, IMPLS[args.impl], uuid, audit.Clock(uuid)


def run_set(run: Callable[[], object], n: int):
    """n back-to-back runs; returns the output of the last one."""
    y = None
    for _ in range(n):
        y = run()
    return y


def base_row(dataset: str, chain, args, uuid: str) -> dict:
    return {"kind": "point", "dataset": dataset, "chain": chain.name, "impl": args.impl, "batch": args.batch,
            "stages": len(chain.stages), "m_in": chain.m_in, "n_out": chain.n_out, "nnz": chain.nnz,
            "gpu_uuid": uuid, "time_utc": audit.utc_now()}


# --------------------------------------------------------------------------- time


def warm_up(torch, graph, seconds: float) -> float:
    """Replay continuously for `seconds` as in the power phase; returns the mean GPU ms per replay."""
    feeder = Feeder(torch, graph)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        feeder.step()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / feeder.n


def timed_replays(torch, graph, replays: int) -> dict:
    """`replays` back-to-back replays with a CUDA event before the first and after each one.

    A wait kernel (torch.cuda._sleep) holds the GPU while the host submits all replays and
    events, so the host cannot slow the GPU down. wait_ok: the first event had not completed when
    the last submission returned, i.e. the wait kernel was still running; otherwise the wait is
    doubled and the measurement repeated (at most WAIT_TRIES times).
    """
    cycles, tries, waited, host_ms, ev = WAIT_CYCLES, 0, False, 0.0, []
    while not waited and tries < WAIT_TRIES:
        if tries:
            cycles *= 2
        tries += 1
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(replays + 1)]
        torch.cuda._sleep(cycles)
        ev[0].record()
        t0 = time.perf_counter()
        for e in ev[1:]:
            graph.replay()
            e.record()
        host_ms = (time.perf_counter() - t0) * 1000.0
        waited = not ev[0].query()
        torch.cuda.synchronize()
    return {"replay_ms": [a.elapsed_time(b) for a, b in zip(ev, ev[1:])], "wait_ok": waited,
            "wait_cycles": cycles, "wait_tries": tries, "host_submit_ms": host_ms}


def cmd_time(args) -> None:
    torch, triton, dataset, chains, make, uuid, clock = setup(args)
    out = Path(args.out)
    path = out / "time.jsonl"
    prov = provenance(clock, torch, triton)
    if args.impl == "cuda":
        import ksmm_cuda

        prov["cuda_build"] = ksmm_cuda.build_info()
    audit.write_jsonl(path, prov)
    done = audit.done_keys(path)
    for chain in chains:
        if (dataset, chain.name, args.impl, args.batch) in done:
            continue
        exit_if_busy(uuid)
        started = time.perf_counter()
        row = {**base_row(dataset, chain, args, uuid), "n_runs": args.n_runs}
        try:
            built = audit.Built(torch, chain)
            x = audit.make_input(torch, chain, args.batch)
            ref = built.reference(torch, x.float())
            impl = make(torch, built, x)
            y = impl.run()  # eager; Triton autotunes here, before the capture
            torch.cuda.synchronize()
            efin, erel, _ = audit.rel_errors(torch, y, ref)
            row["config"] = impl.config()
            run = impl.run
            graph, gy = audit.capture_graph(torch, lambda r=run: run_set(r, args.n_runs))
            gfin, grel, _ = audit.rel_errors(torch, gy, ref)
            y = gy = ref = None  # free before the replays
            ok = efin and gfin and erel <= audit.TOL_REL_L2 and grel <= audit.TOL_REL_L2
            row.update(eager_finite=efin, eager_rel_l2=erel, graph_finite=gfin, graph_rel_l2=grel,
                       rel_l2=max(erel, grel), check="通过" if ok else "失败")
            per_ms = warm_up(torch, graph, WARMUP_S)
            replays = REPLAYS if per_ms <= LONG_REPLAY_MS else max(MIN_REPLAYS, round(TIMED_TOTAL_MS / per_ms))
            res = timed_replays(torch, graph, replays)
            dev = [ms * 1000.0 / args.n_runs for ms in res["replay_ms"]]
            p20, p50, p80 = audit.quantiles(dev)
            row.update(device_us=p50, device_p20=p20, device_p80=p80, replays=replays, warmup_s=WARMUP_S,
                       warmup_replay_ms=per_ms, **res)
        except Exception as exc:  # record and go on with the next chain
            row.update(check="失败", error=f"{type(exc).__name__}: {str(exc)[:2000]}")
        finally:
            built = x = ref = impl = run = graph = None  # free the GPU memory of this chain
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        row["sm_mhz_end"], row["mem_mhz_end"] = clock.sm_mem()
        row["elapsed_s"] = round(time.perf_counter() - started, 3)
        audit.write_jsonl(path, row)
        print(f"time {chain.name:36s} {row['check']} rel={row.get('rel_l2', float('nan')):.2e} "
              f"device={row.get('device_us', float('nan')):.3f} us (p20 {row.get('device_p20', float('nan')):.3f}, "
              f"p80 {row.get('device_p80', float('nan')):.3f}) R={row.get('replays')} wait_ok={row.get('wait_ok')} "
              f"cfg={row.get('config', '')} {row.get('error', '')}", flush=True)


# --------------------------------------------------------------------------- power


class Sampler(threading.Thread):
    """Every SAMPLE_S: (t since t0, power W, SM MHz, temperature C, clock event reasons)."""

    def __init__(self, clock, t0: float):
        super().__init__(daemon=True)
        self.clock, self.t0, self.rows, self.halt = clock, t0, [], threading.Event()

    def run(self) -> None:
        n, h = self.clock.nvml, self.clock.handle
        k = 0
        while not self.halt.is_set():
            self.rows.append((round(time.perf_counter() - self.t0, 3), n.nvmlDeviceGetPowerUsage(h) / 1000.0,
                              n.nvmlDeviceGetClockInfo(h, n.NVML_CLOCK_SM),
                              n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU),
                              n.nvmlDeviceGetCurrentClocksEventReasons(h)))
            k += 1
            self.halt.wait(max(0.0, self.t0 + k * SAMPLE_S - time.perf_counter()))


def energy_window(clock, step: Callable[[], None], at_mark: Callable[[int], None]) -> dict:
    """Call step() for POWER_S seconds; read the energy counter at WINDOW[0] and WINDOW[1].

    at_mark(i) runs right before the i-th energy read (i = 0, 1).
    """
    n, h = clock.nvml, clock.handle
    t0 = time.perf_counter()
    sampler = Sampler(clock, t0)
    sampler.start()
    marks: list[tuple[float, int]] = []
    while (t := time.perf_counter() - t0) < POWER_S:
        if len(marks) < 2 and t >= WINDOW[len(marks)]:
            at_mark(len(marks))
            marks.append((time.perf_counter() - t0, n.nvmlDeviceGetTotalEnergyConsumption(h)))
        step()
    sampler.halt.set()
    sampler.join()
    (ta, ea), (tb, eb) = marks
    win = [s for s in sampler.rows if ta <= s[0] < tb]
    nbins = round((WINDOW[1] - WINDOW[0]) / SPREAD_BIN_S)
    bins = [statistics.mean(s[1] for s in win if ta + i * SPREAD_BIN_S <= s[0] < ta + (i + 1) * SPREAD_BIN_S)
            for i in range(nbins)]
    reasons = 0
    for s in win:
        reasons |= s[4]
    return {"p_w": (eb - ea) / 1000.0 / (tb - ta), "energy_j": (eb - ea) / 1000.0, "window_s": tb - ta,
            "power_spread_pct": (max(bins) - min(bins)) / statistics.mean(bins) * 100.0,
            "sm_mhz": statistics.median(s[2] for s in win), "temp_c": max(s[3] for s in win),
            "clock_event_reasons": hex(reasons), "samples": sampler.rows}


def write_samples(path: Path, samples: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as stream:
        w = csv.writer(stream, lineterminator="\n")
        w.writerow(["t_s", "power_w", "sm_mhz", "temp_c", "clock_event_reasons"])
        w.writerows([t, p, sm, tc, hex(r)] for t, p, sm, tc, r in samples)


def cmd_idle(args) -> None:
    """Idle power: NVML only (no torch import, no CUDA context)."""
    import pynvml

    pynvml.nvmlInit()
    index = int(os.environ["CUDA_VISIBLE_DEVICES"])  # the physical index; NVML ignores CUDA_VISIBLE_DEVICES
    uuid = pynvml.nvmlDeviceGetUUID(pynvml.nvmlDeviceGetHandleByIndex(index))
    clock = audit.Clock(uuid)
    out = Path(args.out)
    path = out / "idle.jsonl"
    audit.write_jsonl(path, provenance(clock))
    exit_if_busy(uuid)
    res = energy_window(clock, lambda: time.sleep(0.01), lambda _i: None)
    exit_if_busy(uuid)  # somebody started during the window: no row
    name = f"idle-{audit.utc_now().replace(':', '')}.csv"
    write_samples(out / "power-samples" / name, res.pop("samples"))
    row = {"kind": "idle", "gpu_uuid": uuid, "time_utc": audit.utc_now(), "p_idle_w": res.pop("p_w"),
           "samples_file": f"power-samples/{name}", **res}
    audit.write_jsonl(path, row)
    print(f"idle {uuid} p_idle={row['p_idle_w']:.2f} W spread={row['power_spread_pct']:.2f}% "
          f"sm={row['sm_mhz']} temp={row['temp_c']}", flush=True)


class Feeder:
    """Submits graph replays, never more than LEAD ahead of the GPU."""

    def __init__(self, torch, graph):
        self.graph, self.n = graph, 0
        self.done = [torch.cuda.Event() for _ in range(LEAD)]

    def step(self) -> None:
        slot = self.done[self.n % LEAD]
        if self.n >= LEAD:
            slot.synchronize()  # replay n - LEAD has finished
        self.graph.replay()
        slot.record()
        self.n += 1


def busy_pct(replays: int, n_runs: int, device_us: float, window_ms: float) -> float:
    return replays * n_runs * device_us / (window_ms * 1000.0) * 100.0


def trial(torch, feeder: Feeder, n_runs: int, device_us: float) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    first, t0 = feeder.n, time.perf_counter()
    while time.perf_counter() - t0 < TRIAL_S:
        feeder.step()
    end.record()
    torch.cuda.synchronize()
    return busy_pct(feeder.n - first, n_runs, device_us, start.elapsed_time(end))


def cmd_power(args) -> None:
    torch, triton, dataset, chains, make, uuid, clock = setup(args)
    out = Path(args.out)
    path = out / "power.jsonl"
    audit.write_jsonl(path, provenance(clock, torch, triton))
    done = audit.done_keys(path)
    timing = latest(out / "time.jsonl", "point")
    for chain in chains:
        if (dataset, chain.name, args.impl, args.batch) in done:
            continue
        exit_if_busy(uuid)
        started = time.perf_counter()
        row = base_row(dataset, chain, args, uuid)
        try:
            device_us = timing[chain.name]["device_us"]  # KeyError: time this chain first
            built = audit.Built(torch, chain)
            x = audit.make_input(torch, chain, args.batch)
            impl = make(torch, built, x)
            run = impl.run
            run()
            torch.cuda.synchronize()
            n_runs, trials = N_RUNS, []
            while True:
                graph, _ = audit.capture_graph(torch, lambda r=run, n=n_runs: run_set(r, n))
                feeder = Feeder(torch, graph)
                trials.append([n_runs, round(trial(torch, feeder, n_runs, device_us), 2)])
                if trials[-1][1] >= BUSY_MIN_PCT or n_runs >= POWER_N_RUNS_MAX:
                    break
                del graph, feeder
                torch.cuda.empty_cache()
                n_runs *= 2
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
            at = [0, 0]

            def mark(i: int, fd: Feeder = feeder) -> None:
                ev[i].record()
                at[i] = fd.n

            res = energy_window(clock, feeder.step, mark)
            del mark  # it holds the feeder and so the graph
            torch.cuda.synchronize()
            replays = at[1] - at[0]
            window_ms = ev[0].elapsed_time(ev[1])
            write_samples(out / "power-samples" / f"{chain.name}.csv", res.pop("samples"))
            row.update(power_n_runs=n_runs, trials=trials, device_us=device_us, p_work_w=res.pop("p_w"),
                       window_replays=replays, window_runs=replays * n_runs, window_gpu_s=window_ms / 1000.0,
                       busy_pct=busy_pct(replays, n_runs, device_us, window_ms),
                       samples_file=f"power-samples/{chain.name}.csv", **res)
        except Exception as exc:  # record and go on with the next chain
            row.update(error=f"{type(exc).__name__}: {str(exc)[:2000]}")
        finally:
            built = x = impl = run = graph = feeder = None  # free the GPU memory of this chain
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        row["elapsed_s"] = round(time.perf_counter() - started, 3)
        audit.write_jsonl(path, row)
        print(f"power {chain.name:36s} n_runs={row.get('power_n_runs')} p_work={row.get('p_work_w', float('nan')):.2f} W "
              f"busy={row.get('busy_pct', float('nan')):.1f}% spread={row.get('power_spread_pct', float('nan')):.2f}% "
              f"sm={row.get('sm_mhz')} trials={row.get('trials')} {row.get('error', '')}", flush=True)


# --------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("time", cmd_time), ("power", cmd_power)):
        s = sub.add_parser(name)
        s.add_argument("--chains", required=True, help="<dataset>/chains.csv; the dataset is the directory name")
        s.add_argument("--only", help="comma-separated chain names")
        s.add_argument("--impl", choices=sorted(IMPLS), default="triton")
        s.add_argument("--batch", type=int, required=True)
        s.add_argument("--out", required=True, help="output directory of this run")
        if name == "time":
            s.add_argument("--n-runs", type=int, default=N_RUNS)
        s.set_defaults(fn=fn)
    s = sub.add_parser("idle")
    s.add_argument("--out", required=True, help="output directory of this run")
    s.set_defaults(fn=cmd_idle)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
