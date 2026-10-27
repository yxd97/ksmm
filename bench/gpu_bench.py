"""GPU time and power of KSMM chains (operations task 2026-10-27-gpu-bench, step T3).

Workload: one CUDA graph per chain holding n_runs back-to-back runs of the chain (no L2 flush
between runs). A run applies the whole chain once to the input X (M_in, batch) and gives
Y (N_out, batch), fp16, feature-major (BSL), stage 0 first, original dimensions. Weights and
input come from a100_audit.Built / make_input (seed: crc32 of the chain name).

Subcommands (run them through run_host.sh, which picks the GPU and wraps `time` in nsys):

  idle    idle power of the GPU in CUDA_VISIBLE_DEVICES; NVML only, no CUDA context.
  time    per chain: eager run + check, capture the n_runs graph + check, 3 warm-up replays,
          then 5 replays inside one cudaProfilerStart/Stop range (one NVTX range per replay).
  parse   export every nsys report of the output directory to sqlite and compute KNL, device
          and DKL time per chain from the GPU timestamps of the kernels.
  power   per chain: pick power_n_runs (busy_pct >= 95 %), replay 120 s, energy from 30 s to 90 s.

Outputs in --out: time.jsonl, timing.jsonl, power.jsonl, idle.jsonl (one row per chain or
measurement, plus a provenance row per process), nsys/*.nsys-rep|sqlite, power-samples/*.csv.
Each command skips chains already in its JSONL (resume). See README.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import socket
import sqlite3
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
WARMUP_REPLAYS = 3
TRACED_REPLAYS = 5
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


# name -> make(torch, built, x) -> Impl. built: a100_audit.Built (built.chain; built.w4: fp16
# (A, B, C, D) weights per stage; built.k_bmm: the same as (A*D, C, B)); x: (M_in, batch) fp16.
IMPLS: dict[str, Callable] = {"triton": make_triton}


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


def tool_version(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True).stdout.strip() or None
    except OSError:
        return None


def provenance(clock, torch=None, triton=None) -> dict:
    from importlib import metadata

    prov = {
        "kind": "provenance", "time_utc": audit.utc_now(), "argv": sys.argv, "host": socket.gethostname(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "python": platform.python_version(), "python_executable": sys.executable,
        "nsys": tool_version([os.environ["NSYS"], "--version"]) if os.environ.get("NSYS") else None,
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


def cmd_time(args) -> None:
    torch, triton, dataset, chains, make, uuid, clock = setup(args)
    out = Path(args.out)
    path = out / "time.jsonl"
    audit.write_jsonl(path, provenance(clock, torch, triton))
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
            for _ in range(WARMUP_REPLAYS):
                graph.replay()
            torch.cuda.synchronize()
            torch.cuda.profiler.start()
            for k in range(TRACED_REPLAYS):
                torch.cuda.nvtx.range_push(f"{chain.name}/r{k}")
                graph.replay()
                torch.cuda.nvtx.range_pop()
            torch.cuda.synchronize()
            torch.cuda.profiler.stop()
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
              f"cfg={row.get('config', '')} {row.get('error', '')}", flush=True)


# --------------------------------------------------------------------------- parse


def parse_report(sqlite_path: Path, time_rows: dict) -> list[dict]:
    """KNL, device and DKL per chain from one exported nsys report.

    A kernel belongs to replay r<k> of a chain when the CUDA API call that launched it (same
    correlationId: the cudaGraphLaunch of that replay) starts inside the NVTX range `<chain>/r<k>`
    on the same thread. The NVTX range is host time and ends before the kernels run, so it only
    selects the cudaGraphLaunch call, never kernels by GPU time. A replay's kernels are put in
    graph node order (graphNodeId, assigned in capture order = launch order; GPU start times of
    neighbours can overlap by tens of ns) and cut into n_runs runs of `stages` kernels.
    """
    db = sqlite3.connect(sqlite_path)
    cur = db.cursor()
    strings = dict(cur.execute("SELECT id, value FROM StringIds"))
    tables = {r[0] for r in cur.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    ranges: dict[str, list] = {}
    for start, end, text, text_id, tid in cur.execute(
            "SELECT start, end, text, textId, globalTid FROM NVTX_EVENTS WHERE end IS NOT NULL AND eventType = 59"):
        label = text if text is not None else strings.get(text_id, "")
        if "/r" in label:
            chain, k = label.rsplit("/r", 1)
            ranges.setdefault(chain, []).append((int(k), start, end, tid))
    api = sorted(cur.execute("SELECT start, correlationId, globalTid FROM CUPTI_ACTIVITY_KIND_RUNTIME"))
    kernels: dict[int, list] = {}
    for node, start, end, corr, gx, gy, gz in cur.execute(
            "SELECT graphNodeId, start, end, correlationId, gridX, gridY, gridZ FROM CUPTI_ACTIVITY_KIND_KERNEL"):
        kernels.setdefault(corr, []).append((node, start, end, f"{gx}x{gy}x{gz}"))
    others: dict[int, int] = {}
    for table in ("CUPTI_ACTIVITY_KIND_MEMSET", "CUPTI_ACTIVITY_KIND_MEMCPY"):
        if table in tables:
            for (corr,) in cur.execute(f"SELECT correlationId FROM {table}"):
                others[corr] = others.get(corr, 0) + 1
    import bisect

    starts = [a[0] for a in api]
    rows = []
    for chain, reps in ranges.items():
        t = time_rows.get(chain)
        if t is None or "n_runs" not in t:
            continue  # no finished time row: the chain gets measured again
        stages, n = t["stages"], t["n_runs"]
        knl, device, dkl, problems, grids = [], [], [], [], None
        other_ops = knl_gt_device = start_inversions = overlaps = 0
        for k, r_start, r_end, tid in sorted(reps):
            lo, hi = bisect.bisect_left(starts, r_start), bisect.bisect_right(starts, r_end)
            acts = []
            for _s, corr, a_tid in api[lo:hi]:
                if a_tid == tid:
                    acts.extend(kernels.get(corr, []))
                    other_ops += others.get(corr, 0)
            acts.sort()  # graph node order
            start_inversions += sum(b[1] < a[1] for a, b in zip(acts, acts[1:]))
            overlaps += sum(b[1] < a[2] for a, b in zip(acts, acts[1:]))
            if len(acts) != n * stages:
                problems.append(f"r{k}: {len(acts)} kernels, expected {n}x{stages}")
                continue
            runs = [acts[i * stages:(i + 1) * stages] for i in range(n)]
            for j, run in enumerate(runs):
                g = " ".join(a[3] for a in run)
                grids = grids or g
                if g != grids:
                    problems.append(f"r{k} run {j}: grids {g} differ from {grids}")
                knl_us = sum(e - s for _n, s, e, _g in run) / 1000.0
                knl.append(knl_us)
                if j:  # the first run of a replay follows the graph launch: KNL only
                    dev_us = (run[-1][2] - runs[j - 1][-1][2]) / 1000.0
                    device.append(dev_us)
                    dkl.append(dev_us - knl_us)
                    knl_gt_device += knl_us > dev_us
        row = {"kind": "timing", "chain": chain, "impl": t["impl"], "batch": t["batch"],
               "report": sqlite_path.with_suffix(".nsys-rep").name, "replays": len(reps),
               "kernels": stages if knl else None, "grids": grids, "other_ops": other_ops,
               "knl_n": len(knl), "device_n": len(device), "knl_gt_device": knl_gt_device,
               "start_inversions": start_inversions, "overlaps": overlaps}
        if knl and device:
            (k20, k50, k80), (d20, d50, d80) = audit.quantiles(knl), audit.quantiles(device)
            row.update(knl_us=k50, knl_p20=k20, knl_p80=k80, device_us=d50, device_p20=d20, device_p80=d80,
                       dkl_us=statistics.median(dkl))
        if len(reps) != TRACED_REPLAYS:
            problems.append(f"{len(reps)} NVTX replay ranges, expected {TRACED_REPLAYS}")
        if other_ops:
            problems.append(f"{other_ops} memset/memcpy in the replays")
        row["problems"] = problems
        rows.append(row)
    db.close()
    return rows


def cmd_parse(args) -> None:
    out = Path(args.out)
    path = out / "timing.jsonl"
    time_rows = latest(out / "time.jsonl", "point")
    parsed = {r["report"] for r in read_rows(path, "timing")}
    def order(rep: Path):  # time-<stamp>.<k>.nsys-rep, k = 1, 2, ... per cudaProfilerStart/Stop range
        stamp, k = rep.name.split(".")[:2]
        return stamp, int(k)

    for rep in sorted((out / "nsys").glob("*.nsys-rep"), key=order):
        if rep.name in parsed:
            continue
        sql = rep.with_suffix(".sqlite")
        if not sql.exists():
            subprocess.run([os.environ["NSYS"], "export", "--type", "sqlite", "--force-overwrite=true",
                            "-o", str(sql), str(rep)], check=True, stdout=subprocess.DEVNULL)
        for row in parse_report(sql, time_rows):
            audit.write_jsonl(path, row)
            print(f"parse {row['chain']:36s} knl={row.get('knl_us', float('nan')):9.3f} "
                  f"device={row.get('device_us', float('nan')):9.3f} dkl={row.get('dkl_us', float('nan')):8.3f} "
                  f"kernels={row['kernels']} n={row['knl_n']}/{row['device_n']} {' ; '.join(row['problems'])}",
                  flush=True)


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
    timing = latest(out / "timing.jsonl", "timing")
    for chain in chains:
        if (dataset, chain.name, args.impl, args.batch) in done:
            continue
        exit_if_busy(uuid)
        started = time.perf_counter()
        row = base_row(dataset, chain, args, uuid)
        try:
            device_us = timing[chain.name]["device_us"]  # KeyError: time and parse this chain first
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
    for name, fn in (("idle", cmd_idle), ("parse", cmd_parse)):
        s = sub.add_parser(name)
        s.add_argument("--out", required=True, help="output directory of this run")
        s.set_defaults(fn=fn)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
