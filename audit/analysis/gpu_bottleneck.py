#!/usr/bin/env python3
"""GPU bottleneck analysis for operations task 2026-10-14-v41-baseline-audit, step T8.

Reads (all paths under --ops, the operations repository, unless noted):
  datasets/{v3-reference-data,v1}/chains.csv, runs/a100-{ksmm,cublas,cusparse}/results.csv, runs/v41/results.csv
  datasets/scale1/chains.csv, runs/a100-{ksmm,cublas}/results.csv, runs/v41/results.csv   (used when published)
  tasks/<task>/data/a100-batch-sweep.csv, a100-nsys.csv, a100-ncu.csv, v41-v3ref-stages.csv, v41-v1-stages.csv
  --t8 (raw T8 outputs copied from zhang-x4 runs/t8): floor.jsonl, cfg-stage.jsonl, cfg-chain.jsonl,
       ncu-raw-*.csv (T4's metric list), lim-raw-*.csv (ncu sections)
Writes tasks/<task>/data/gpu-*.csv and gpu-*.vl.json; prints the numbers the report quotes.

Run: python3 audit/analysis/gpu_bottleneck.py --ops <operations> --t8 <copy of runs/t8>
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import a100_audit  # noqa: E402  (ncu-parse, the same parser T4 used)

TASK = "2026-10-14-v41-baseline-audit"
PEAK_BW = 1935e9   # B/s, A100 80GB PCIe HBM2e (NVIDIA A100 datasheet); same constant as T5's model.py
PEAK_TC = 312e12   # FLOP/s, fp16 tensor core, dense (datasheet)
L2_BYTES = 40 * 2**20
# Regime thresholds (see gpu-bottleneck.md, "判定规则")
TC_SHARE = 0.30    # useful tensor FLOP/s >= 30 % of peak -> compute (tensor core) bound
BW_SHARE = 0.30    # estimated DRAM bytes / time >= 30 % of peak -> bandwidth bound
FIX_SHARE = 0.50   # the implementation's own floor >= 50 % of G_cold -> fixed-overhead / latency bound
DATASETS = ("v3-reference-data", "v1", "scale1")
IMPLS = ("ksmm", "cublas", "cusparse")
R_FIX, R_SM, R_BW, R_TC, R_MID = "固定开销/延迟", "SM（CTA 吞吐）", "带宽", "算力（张量核）", "中间"


# --------------------------------------------------------------------------- io


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", encoding="utf-8", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})
    size = path.stat().st_size
    assert size < 2_000_000, f"{path} is {size} bytes"
    print(f"wrote {path.name}: {len(rows)} rows, {size} bytes")


def fmt(v):
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:.4g}" if abs(v) < 1e-3 and v != 0 else f"{v:.4f}".rstrip("0").rstrip(".")
    return "" if v is None else v


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def num(v, default=float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def stages_of(row: dict) -> list[tuple[int, int, int, int]]:
    out, k = [], 0
    while row.get(f"A{k}"):
        out.append(tuple(int(row[f"{x}{k}"]) for x in "ABCD"))
        k += 1
    return out


def fit(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares line y = a + b x -> (a, b)."""
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b


# --------------------------------------------------------------------------- inputs


class Inputs:
    def __init__(self, ops: Path, t8: Path):
        self.ops, self.t8 = ops, t8
        self.data = ops / "tasks" / TASK / "data"
        self.chains: dict = {}       # (ds, chain) -> stages
        self.results: dict = {}      # (ds, impl) -> {chain: row}
        self.v41: dict = {}          # (ds, chain) -> v41 results row
        for ds in DATASETS:
            d = ops / "datasets" / ds
            if not (d / "chains.csv").exists():
                continue
            for r in read_csv(d / "chains.csv"):
                self.chains[(ds, r["chain"])] = stages_of(r)
            for impl in IMPLS:
                f = d / "runs" / f"a100-{impl}" / "results.csv"
                if f.exists():
                    self.results[(ds, impl)] = {r["chain"]: r for r in read_csv(f)}
            f = d / "runs" / "v41" / "results.csv"
            if f.exists():
                for r in read_csv(f):
                    self.v41[(ds, r["chain"])] = r
        self.sweep = read_csv(self.data / "a100-batch-sweep.csv")
        self.nsys = read_csv(self.data / "a100-nsys.csv")
        self.v41_stages = {}
        for ds, name in (("v3-reference-data", "v41-v3ref-stages.csv"), ("v1", "v41-v1-stages.csv")):
            for r in read_csv(self.data / name):
                self.v41_stages.setdefault((ds, r["chain"]), []).append(r)
        self.scale1_published = ("scale1", "ksmm") in self.results
        self.ncu = self._ncu()

    def _ncu(self) -> list[dict]:
        """T4's per-kernel ncu rows plus T8's (parsed with T4's parser from the raw exports)."""
        rows = [dict(r, source="T4 data/a100-ncu.csv") for r in read_csv(self.data / "a100-ncu.csv")]
        raw = sorted(self.t8.glob("ncu-raw-*.csv"))
        if raw:
            out = self.t8 / "ncu-kernels-t8.csv"
            a100_audit.cmd_ncu_parse(Namespace(csv=[str(p) for p in raw], out=str(out)))
            rows += [dict(r, source="T8 " + out.name) for r in read_csv(out)]
        return rows


# --------------------------------------------------------------------------- fixed costs (floor probe)


def floor_constants(inp: Inputs) -> tuple[dict, list[dict]]:
    recs = jsonl(inp.t8 / "floor.jsonl")
    rows = [r for r in recs if r["kind"] != "provenance"]
    c: dict = {"flush_us": next(r["flush_us_median"] for r in recs if r["kind"] == "provenance")}
    for kind, tag in (("empty", "empty"), ("ksmm_1cta", "1cta")):
        pts = sorted((r["kernels"], r["gc_mean"], r["gw_mean"]) for r in rows if r["kind"] == kind)
        a, b = fit([p[0] for p in pts], [p[1] for p in pts])
        aw, bw = fit([p[0] for p in pts], [p[2] for p in pts])
        c[f"{tag}_launch_us"], c[f"{tag}_per_kernel_us"] = a, b
        c[f"{tag}_launch_warm_us"], c[f"{tag}_per_kernel_warm_us"] = aw, bw
    c["graph_launch_us"] = c["empty_launch_us"]          # G_cold window before / around the first kernel
    c["kernel_floor_us"] = c["empty_per_kernel_us"]      # one empty kernel inside a graph (incl. its gap)
    c["ksmm_1cta_us"] = c["1cta_per_kernel_us"]          # one one-CTA KSMM kernel inside a graph
    # one-CTA KSMM kernel minus an empty kernel: parameters (imc_miss), barrier, L2 / DRAM access, MMA (not split);
    # the cold - warm part of it is what DRAM costs over L2
    c["ksmm_onecta_extra_us"] = c["ksmm_1cta_us"] - c["kernel_floor_us"]
    c["ksmm_onecta_extra_warm_us"] = c["1cta_per_kernel_warm_us"] - c["empty_per_kernel_warm_us"]
    curve = []
    by = defaultdict(dict)
    for r in rows:
        if r["kind"] == "ksmm_ctas":
            by[(r["radix"], r["ctas"])][r["kernels"]] = r
    for (radix, ctas), d in sorted(by.items()):
        per = (d[8]["gc_mean"] - d[1]["gc_mean"]) / 7
        perw = (d[8]["gw_mean"] - d[1]["gw_mean"]) / 7
        curve.append({"radix": radix, "ctas": ctas, "per_kernel_us": per, "per_kernel_warm_us": perw,
                      "g1_us": d[1]["gc_mean"], "g8_us": d[8]["gc_mean"],
                      "intercept_us": d[1]["gc_mean"] - per, "fixed_share": c["ksmm_1cta_us"] / per,
                      "ns_per_cta": (per - c["ksmm_1cta_us"]) * 1000 / ctas if ctas > 1 else float("nan")})
    big = [p for p in curve if p["ctas"] >= 8192]
    for radix in (2, 4):
        pts = [p for p in big if p["radix"] == radix]
        a, b = fit([p["ctas"] for p in pts], [p["per_kernel_us"] for p in pts])
        c[f"r{radix}_cta_rate_per_us"] = 1 / b   # CTAs per microsecond at large CTA counts
        c[f"r{radix}_cta_intercept_us"] = a
    # CTA count at which the CTA term equals the one-CTA cost (fixed share 50 %), radix 2, interpolated
    pts = sorted((p["ctas"], p["per_kernel_us"]) for p in curve if p["radix"] == 2)
    target = 2 * c["ksmm_1cta_us"]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if y0 <= target < y1:
            c["r2_ctas_at_half_fixed"] = x0 * (x1 / x0) ** ((target - y0) / (y1 - y0))
    for r in rows:
        if r["kind"] in ("empty", "ksmm_1cta"):
            curve.append({"radix": "", "ctas": "", "kind": r["kind"], "kernels": r["kernels"],
                          "g1_us": r["gc_mean"], "g1_warm_us": r["gw_mean"]})
    return c, curve


# --------------------------------------------------------------------------- ncu summaries


def ncu_summary(inp: Inputs) -> dict:
    g = defaultdict(list)
    for r in inp.ncu:
        g[(r["dataset"], r["chain"], r["impl"], int(r["batch"]))].append(r)
    out = {}
    for key, ks in g.items():
        dur = [num(k["duration_us"]) for k in ks]
        tot = sum(dur)

        def w(col):
            return sum(num(k[col]) * d for k, d in zip(ks, dur)) / tot

        hit = sum(num(k["l2_hit_sectors"]) for k in ks)
        miss = sum(num(k["l2_miss_sectors"]) for k in ks)
        out[key] = {"ncu_kernels": len(ks), "ncu_dram_pct": w("dram_pct"), "ncu_sm_pct": w("sm_pct"),
                    "ncu_tensor_pct": w("tensor_pct"), "ncu_occ_pct": w("occ_achieved_pct"),
                    "ncu_l2_hit_pct": 100 * hit / (hit + miss) if hit + miss else float("nan"),
                    "ncu_waves_max": max(num(k["waves"]) for k in ks),
                    "ncu_dram_bytes": sum(num(k["dram_read_bytes"]) + num(k["dram_write_bytes"]) for k in ks),
                    "ncu_kernel_us_median": statistics.median(dur), "ncu_source": ks[0]["source"]}
    return out


# --------------------------------------------------------------------------- regimes


def chain_work(stages, impl: str, batch: int, csr_nnz: float) -> dict:
    m_in = stages[0][0] * stages[0][2] * stages[0][3]
    n_out = stages[-1][0] * stages[-1][1] * stages[-1][3]
    nnz = sum(a * b * c * d for a, b, c, d in stages)
    if impl == "ksmm":   # each stage reads its weights, input and writes its output (no inter-stage reuse)
        nbytes = sum(2 * (a * b * c * d + batch * (a * c * d + a * b * d)) for a, b, c, d in stages)
        macs = nnz * batch
        act = max(2 * batch * (a * c * d + a * b * d) for a, b, c, d in stages)
    elif impl == "cublas":
        nbytes = 2 * (m_in * n_out + batch * (m_in + n_out))
        macs = m_in * n_out * batch
        act = 2 * batch * (m_in + n_out)
    else:  # cusparse CSR: fp16 values + int32 column indices + int32 row pointers
        nbytes = csr_nnz * 6 + (n_out + 1) * 4 + 2 * batch * (m_in + n_out)
        macs = csr_nnz * batch
        act = 2 * batch * (m_in + n_out)
    return {"m_in": m_in, "n_out": n_out, "nnz": nnz, "bytes_est": nbytes, "macs": macs,
            "stage_act_bytes_max": act}


def classify(impl, tc_share, bw_share, fix_share, n: dict) -> tuple[str, str]:
    """(regime, basis). KSMM's byte estimate (no inter-stage reuse) is an upper bound on DRAM bytes, so a KSMM
    point is called bandwidth bound only when ncu measured DRAM >= 50 %; otherwise it stays CTA throughput."""
    if impl == "ksmm" and tc_share < TC_SHARE and bw_share >= BW_SHARE:
        if not n:
            return R_SM, "规则；带宽估算 ≥ 30% 但没有 ncu，未核实"
        if n["ncu_dram_pct"] >= 50:
            return R_BW, "ncu DRAM ≥ 50%"
        return R_SM, "带宽估算 ≥ 30%，ncu DRAM < 50%，按 ncu 改判"
    return _rule(impl, tc_share, bw_share, fix_share), "规则"


def _rule(impl, tc_share, bw_share, fix_share) -> str:
    if tc_share >= TC_SHARE:
        return R_TC
    if bw_share >= BW_SHARE:
        return R_BW
    if fix_share >= FIX_SHARE:
        return R_FIX
    return R_SM if impl == "ksmm" else R_MID


def ncu_check(regime: str, n: dict) -> str:
    if not n:
        return ""
    dram, sm, tc = n["ncu_dram_pct"], n["ncu_sm_pct"], n["ncu_tensor_pct"]
    # the fixed / CTA-throughput boundary comes from the timing decomposition (floor probe); ncu checks that neither
    # is memory bound and that SM load is low in the first and higher in the second (the two ranges overlap on purpose)
    if regime == R_MID:
        return "无核对条件"
    ok = {R_FIX: dram < 10 and sm < 50, R_SM: sm >= 30 and sm >= dram, R_BW: dram >= 50,
          R_TC: tc >= 50}[regime]
    return "一致" if ok else "不一致"


def regimes(inp: Inputs, c: dict, ncu: dict) -> list[dict]:
    pts = []
    for r in inp.sweep:
        pts.append((r["dataset"], r["chain"], r["impl"], int(r["batch"]), {
            "g_cold": num(r["gc_mean"]), "g_warm": num(r["gw_mean"]), "e_p50": num(r["e_p50"]),
            "e_p80": num(r["e_p80"]), "h_us": num(r["h_us"]), "csr_nnz": num(r["csr_nnz"], 0.0),
            "ksmm_configs": r["ksmm_configs"]}))
    if inp.scale1_published:
        for impl in ("ksmm", "cublas"):
            for name, r in inp.results.get(("scale1", impl), {}).items():
                pts.append(("scale1", name, impl, 1, {
                    "g_cold": num(r["time_us"]), "g_warm": num(r.get("g_warm_us")), "e_p50": num(r.get("e_p50")),
                    "e_p80": num(r.get("e_p80")), "h_us": num(r.get("h_us")), "csr_nnz": 0.0, "ksmm_configs": ""}))
    floors = {impl: min(p[4]["g_cold"] for p in pts if p[2] == impl) for impl in IMPLS}
    nsys_k = {(r["dataset"], r["chain"], r["impl"]): int(r["kernels"])
              for r in inp.nsys if r["mode"] == "GC" and r["batch"] == "1"}
    # eager (E mode) under nsys: share of the span from first kernel start to last kernel end with no kernel running
    e_idle = {(r["dataset"], r["chain"], r["impl"]): num(r["gaps_us_median"]) / num(r["span_us_median"])
              for r in inp.nsys if r["mode"] == "E" and r["batch"] == "1" and num(r["span_us_median"]) > 0}
    out = []
    for ds, name, impl, batch, m in pts:
        stages = inp.chains[(ds, name)]
        w = chain_work(stages, impl, batch, m["csr_nnz"])
        k = len(stages) if impl == "ksmm" else nsys_k.get((ds, name, impl), 1 if impl == "cublas" else 3)
        g = m["g_cold"]
        fixed = c["graph_launch_us"] + k * c["ksmm_1cta_us"] if impl == "ksmm" else floors[impl]
        tc_share = 2 * w["macs"] / (g * 1e-6) / PEAK_TC
        bw_share = w["bytes_est"] / (g * 1e-6) / PEAK_BW
        fix_share = min(1.0, fixed / g)
        n = ncu.get((ds, name, impl, batch), {})
        reg, basis = classify(impl, tc_share, bw_share, fix_share, n)
        ctas = (sum(a * d for a, _b, _c, d in stages) / len(stages)) * math.ceil(batch / 32) if impl == "ksmm" else ""
        out.append({
            "dataset": ds, "chain": name, "impl": impl, "batch": batch, "stages": len(stages), "kernels": k,
            "n_in": w["m_in"], "nnz": w["nnz"], "macs": w["macs"], "bytes_est": w["bytes_est"],
            "stage_act_bytes_max": w["stage_act_bytes_max"], "ctas_per_kernel_mean": ctas,
            "g_cold_us": g, "g_warm_us": m["g_warm"], "e_p50_us": m["e_p50"], "e_p80_us": m["e_p80"],
            "h_us": m["h_us"], "h_over_gcold": m["h_us"] / g, "e_over_gcold": m["e_p50"] / g,
            "eager_regime": "发射受限" if m["h_us"] > g else "不受发射限制",
            "nsys_e_idle_share": e_idle.get((ds, name, impl), "") if batch == 1 else "",
            "per_kernel_us": (g - c["graph_launch_us"]) / k if impl == "ksmm" else "",
            "floor_us": fixed, "fixed_share": fix_share, "bw_share": bw_share, "tc_share": tc_share,
            "device_regime": reg, "regime_basis": basis, **{key: n.get(key, "") for key in (
                "ncu_dram_pct", "ncu_sm_pct", "ncu_tensor_pct", "ncu_occ_pct", "ncu_l2_hit_pct", "ncu_waves_max",
                "ncu_source")},
            "ncu_agrees": ncu_check(reg, n), "ksmm_configs": m["ksmm_configs"]})
    return out


# --------------------------------------------------------------------------- per-stage comparison with V4.1


def stage_compare(inp: Inputs, c: dict) -> tuple[list[dict], list[dict]]:
    gc_rows = {(r["dataset"], r["chain"]): r for r in inp.nsys
               if r["mode"] == "GC" and r["impl"] == "ksmm" and r["batch"] == "1"}
    stage_rows, chain_rows = [], []
    for (ds, name), ns in sorted(gc_rows.items()):
        res = inp.results[(ds, "ksmm")][name]
        stages = inp.chains[(ds, name)]
        v41 = inp.v41_stages.get((ds, name))
        kus = [float(x) for x in ns["kernel_us_median"].split()]
        if not v41 or len(kus) != len(stages) or len(v41) != len(stages):
            continue
        g = float(res["time_us"])
        L = c["graph_launch_us"]
        scale = (g - L) / sum(kus)
        tot = defaultdict(float)
        for k, ((a, b, cc, d), knsys, v) in enumerate(zip(stages, kus, v41)):
            t = knsys * scale
            cyc = float(v["stage_cycles"])
            parts = {"compute": num(v["t_bfly"]) + num(v["t_add"]), "memory": num(v["t_dma"]),
                     "movement": num(v["t_trans"]), "control": num(v["t_gti"]) + num(v["t_nop"]),
                     "idle": num(v["t_idle"])}
            s = sum(parts.values())
            row = {"dataset": ds, "chain": name, "stage": k, "A": a, "B": b, "C": cc, "D": d, "ctas": a * d,
                   "v41_ns": cyc, **{f"v41_{p}_ns": cyc * x / s for p, x in parts.items()},
                   "gpu_nsys_ns": knsys * 1000, "gpu_ns": t * 1000,
                   "gpu_floor_ns": c["kernel_floor_us"] * 1000,
                   "gpu_onecta_extra_ns": c["ksmm_onecta_extra_us"] * 1000,
                   "gpu_cta_ns": (t - c["ksmm_1cta_us"]) * 1000,
                   "gpu_bw_peak_ns": 2 * (a * b * cc * d + a * cc * d + a * b * d) / PEAK_BW * 1e9,
                   "gpu_tc_peak_ns": 2 * a * b * cc * d / PEAK_TC * 1e9,
                   "ratio_gpu_v41": t * 1000 / cyc}
            stage_rows.append(row)
            for key in ("v41_ns", "gpu_ns", "gpu_floor_ns", "gpu_onecta_extra_ns", "gpu_cta_ns", "gpu_bw_peak_ns",
                        "gpu_tc_peak_ns", "v41_compute_ns", "v41_memory_ns", "v41_movement_ns", "v41_control_ns",
                        "v41_idle_ns"):
                tot[key] += row[key]
        v41_total = float(inp.v41[(ds, name)]["cycles"])
        chain_rows.append({
            "dataset": ds, "chain": name, "stages": len(stages), "nnz_ratio_padded": res.get("nnz_ratio_padded", ""),
            "g_cold_us": g, "graph_launch_us": L, "kernel_floor_us": tot["gpu_floor_ns"] / 1000,
            "onecta_extra_us": tot["gpu_onecta_extra_ns"] / 1000, "cta_us": tot["gpu_cta_ns"] / 1000,
            "bw_peak_us": tot["gpu_bw_peak_ns"] / 1000, "tc_peak_us": tot["gpu_tc_peak_ns"] / 1000,
            "v41_us": v41_total / 1000, "v41_stage_sum_us": tot["v41_ns"] / 1000,
            **{f"v41_{p}_us": tot[f"v41_{p}_ns"] / 1000 for p in ("compute", "memory", "movement", "control", "idle")},
            "ratio": g * 1000 / v41_total,
            "share_launch": L / g, "share_floor": tot["gpu_floor_ns"] / 1000 / g,
            "share_onecta_extra": tot["gpu_onecta_extra_ns"] / 1000 / g, "share_cta": tot["gpu_cta_ns"] / 1000 / g,
            "share_fixed_total": (L + tot["gpu_floor_ns"] / 1000 + tot["gpu_onecta_extra_ns"] / 1000) / g,
            "per_stage_gpu_us": (g - L) / len(stages), "per_stage_v41_us": v41_total / 1000 / len(stages)})
    return stage_rows, chain_rows


# --------------------------------------------------------------------------- launch removed


def launch_removed(inp: Inputs, c: dict, fb: list[dict]) -> list[dict]:
    """A100 / V4.1 under each view, KSMM batch 1, plus E - G_cold split into an eager device-side part and the
    host issue time that shows in E.

    E's window starts after a ~150 us L2 flush during which the host keeps issuing. Chains whose whole issue time
    h_us is shorter than the flush have nothing of the host left in E: their E - G_cold is the device-side
    difference between eager and graph launches (eager kernels run longer and are ~1 us apart; the graph adds
    its own start). A line in the number of stages fitted on those chains estimates that device part for the
    longer chains (an extrapolation); the rest of E - G_cold is host issue time showing in E.
    """
    rows = []
    for ds in DATASETS:
        ks = inp.results.get((ds, "ksmm"), {})
        cb = inp.results.get((ds, "cublas"), {})
        for name, r in ks.items():
            if (ds, name) not in inp.v41:
                continue
            k = len(inp.chains[(ds, name)])
            v = float(inp.v41[(ds, name)]["cycles"]) / 1000
            g, gw = num(r["time_us"]), num(r.get("g_warm_us"))
            cg = num(cb.get(name, {}).get("time_us"))
            cgw = num(cb.get(name, {}).get("g_warm_us"))
            rows.append({"dataset": ds, "chain": name, "stages": k, "v41_us": v,
                         "ksmm_e_p50_us": num(r.get("e_p50")), "ksmm_e_p80_us": num(r.get("e_p80")),
                         "ksmm_h_us": num(r.get("h_us")), "ksmm_g_cold_us": g, "ksmm_g_warm_us": gw,
                         "ksmm_no_graph_launch_us": g - c["graph_launch_us"],
                         "ksmm_floor_no_graph_us": k * c["ksmm_1cta_us"],
                         "cublas_g_cold_us": cg, "cublas_g_warm_us": cgw})
    corr_cublas = {(r["dataset"], r["chain"]): r["g_cold_corrected_us"] for r in fb if r["impl"] == "cublas"}
    short = [(x["stages"], x["ksmm_e_p50_us"] - x["ksmm_g_cold_us"]) for x in rows if x["ksmm_h_us"] < c["flush_us"]]
    a, b = fit([k for k, _ in short], [d for _, d in short])
    c["eager_dev_intercept_us"], c["eager_dev_per_stage_us"], c["eager_dev_fit_chains"] = a, b, len(short)
    c["eager_dev_fit_stages"] = f"{min(k for k, _ in short)}-{max(k for k, _ in short)}"
    for x in rows:
        g, e, v = x["ksmm_g_cold_us"], x["ksmm_e_p50_us"], x["v41_us"]
        host_hidden = x["ksmm_h_us"] < c["flush_us"]
        dev = e - g if host_hidden else a + b * x["stages"]
        best = min(z for z in (g, x["cublas_g_cold_us"]) if not math.isnan(z))
        corr = corr_cublas.get((x["dataset"], x["chain"]), float("nan"))
        best_corr = min(g, corr) if not math.isnan(corr) else best
        x.update({
            "host_hidden_by_flush": "是" if host_hidden else "否",
            "eager_device_extra_us": dev, "host_in_e_us": (e - g) - dev,
            "best_gpu_us": best, "best_gpu_impl": "ksmm" if best == g else "cublas",
            "ratio_e": e / v, "ratio_e_p80": x["ksmm_e_p80_us"] / v, "ratio_g_cold": g / v,
            "ratio_g_warm": x["ksmm_g_warm_us"] / v, "ratio_no_graph_launch": x["ksmm_no_graph_launch_us"] / v,
            "ratio_floor_no_graph": x["ksmm_floor_no_graph_us"] / v,
            "ratio_cublas": x["cublas_g_cold_us"] / v, "ratio_best": best / v,
            "cublas_g_cold_corrected_us": corr, "ratio_cublas_corrected": corr / v,
            "best_gpu_corrected_us": best_corr, "best_gpu_corrected_impl": "ksmm" if best_corr == g else "cublas",
            "ratio_best_corrected": best_corr / v,
            "host_share_of_e": ((e - g) - dev) / e, "eager_device_share_of_e": dev / e,
            "graph_launch_share_of_g": c["graph_launch_us"] / g})
    return rows


# --------------------------------------------------------------------------- L2 flush bias


def flush_bias(inp: Inputs) -> list[dict]:
    """G_cold re-measured with T4's flush (zero_(): dirty lines left in L2) and a read-only flush (clean lines);
    random operands, batch 1 (t8_probe.py flushcmp). Published G_cold from the a100-* runs.

    Each flush kind is measured in several rounds; the first rounds on a freshly built multi-GB matrix are slow
    whatever the flush (flushcmp-big.jsonl measures clean first), so each kind takes its fastest round.
    flushcmp-big.jsonl (3 rounds, clean first) replaces flushcmp.jsonl (2 rounds, zero first) where both exist.
    bias = zero / clean - 1 is what the dirty lines add; the corrected G_cold is published / (1 + bias).
    """
    recs = {}
    for name in ("flushcmp.jsonl", "flushcmp-big.jsonl"):
        f = inp.t8 / name
        if f.exists():
            for r in jsonl(f):
                if r.get("kind") == "flushcmp":
                    recs[(r["dataset"], r["chain"], r["impl"], r["batch"])] = dict(r, file=name)
    out = []
    for (ds, chain, impl, batch), r in recs.items():
        pub = num(inp.results.get((ds, impl), {}).get(chain, {}).get("time_us"))
        v41 = num(inp.v41.get((ds, chain), {}).get("cycles")) / 1000
        zr, cr = r.get("zero_rounds", []), r.get("clean_rounds", [])
        z, cl = (min(zr), min(cr)) if zr and cr else (float("nan"), float("nan"))
        bias = z / cl - 1
        corr = pub / (1 + bias)
        out.append({"dataset": ds, "chain": chain, "impl": impl, "batch": batch, "m_in": r["m_in"], "n_out": r["n_out"],
                    "matrix_mb": 2 * r["m_in"] * r["n_out"] / 1e6 if impl == "cublas" else "",
                    "status": r["status"], "file": r["file"], "order": r.get("order", "zero,clean"),
                    "zero_rounds_us": " ".join(f"{x:.2f}" for x in zr), "clean_rounds_us": " ".join(f"{x:.2f}" for x in cr),
                    "g_cold_published_us": pub, "g_cold_zero_us": z, "g_cold_clean_us": cl,
                    "bias_pct": 100 * bias, "published_vs_zero_pct": 100 * (pub / z - 1),
                    "g_cold_corrected_us": corr, "v41_us": v41,
                    "ratio_published": pub / v41, "ratio_corrected": corr / v41, "ratio_clean": cl / v41,
                    "sm_mhz_end": r.get("sm_mhz_end", "")})
    out.sort(key=lambda x: (x["impl"], DATASETS.index(x["dataset"]), x["m_in"] * x["n_out"], x["chain"]))
    return out


# --------------------------------------------------------------------------- batch crossovers


def crossovers(reg: list[dict]) -> list[dict]:
    by = defaultdict(dict)
    for r in reg:
        by[(r["dataset"], r["chain"], r["impl"])][r["batch"]] = r
    out = []
    for (ds, name, impl), pts in sorted(by.items()):
        if impl != "ksmm" or len(pts) < 2:
            continue
        cb = by.get((ds, name, "cublas"), {})
        bs = sorted(pts)

        def first(pred):
            return next((b for b in bs if pred(b)), "")

        out.append({
            "dataset": ds, "chain": name, "stages": pts[1]["stages"], "n_in": pts[1]["n_in"],
            "launch_bound_until": max((b for b in bs if pts[b]["eager_regime"] == "发射受限"), default=""),
            "first_batch_not_fixed": first(lambda b: pts[b]["device_regime"] != R_FIX),
            "first_batch_cublas_faster": first(lambda b: b in cb and cb[b]["g_cold_us"] < pts[b]["g_cold_us"]),
            "batches_ksmm_faster": " ".join(str(b) for b in bs if b in cb and pts[b]["g_cold_us"] < cb[b]["g_cold_us"]),
            "ksmm_regimes": " ".join(f"{b}:{pts[b]['device_regime']}" for b in bs),
            "cublas_regimes": " ".join(f"{b}:{cb[b]['device_regime']}" for b in bs if b in cb)})
    return out


# --------------------------------------------------------------------------- KSMM configs


def configs(inp: Inputs, reg: list[dict]) -> list[dict]:
    rows = []
    cublas = {(r["dataset"], r["chain"], r["batch"]): r["g_cold_us"] for r in reg if r["impl"] == "cublas"}
    for fname in ("cfg-stage.jsonl", "cfg-chain.jsonl"):
        f = inp.t8 / fname
        if not f.exists():
            continue
        recs = [r for r in jsonl(f) if r.get("kind") == "config"]
        default = {(r["name"], r["batch"]): r for r in recs if r["variant"] == "default"}
        for r in recs:
            d = default.get((r["name"], r["batch"]))
            ds = "v3-reference-data" if r["name"] in ("bfly_4", "debut_5") else ("scale1" if fname == "cfg-chain.jsonl" else "")
            rows.append({"file": fname, "name": r["name"], "stages": r["stages"], "pattern0": r["pattern0"],
                         "batch": r["batch"], "variant": r["variant"], "status": r["status"],
                         "check": r.get("check", ""), "rel_diff_vs_default": r.get("rel_diff_vs_default", ""),
                         "g_cold_us": r.get("gc_mean", ""), "g_warm_us": r.get("gw_mean", ""),
                         "speedup_vs_default": d["gc_mean"] / r["gc_mean"] if d and r.get("gc_mean") else "",
                         "cublas_g_cold_us": cublas.get((ds, r["name"], r["batch"]), ""),
                         "picked": r.get("picked", ""), "sm_mhz_end": r.get("sm_mhz_end", "")})
    return rows


# --------------------------------------------------------------------------- ncu limiter sections

LIM_METRICS = {
    "duration_us": "gpu__time_duration.sum",
    "sm_pct": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram_pct": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex_pct": "l1tex__throughput.avg.pct_of_peak_sustained_active",
    "l2_pct": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "issue_active_pct": "sm__inst_issued.avg.pct_of_peak_sustained_active",
    "ipc_issued": "sm__inst_issued.avg.per_cycle_active",
    "occ_achieved_pct": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "occ_theoretical_pct": "sm__maximum_warps_per_active_cycle_pct",
    "blocks_limit_regs": "launch__occupancy_limit_registers",
    "blocks_limit_smem": "launch__occupancy_limit_shared_mem",
    "blocks_limit_warps": "launch__occupancy_limit_warps",
    "blocks_limit_blocks": "launch__occupancy_limit_blocks",
    "grid": "launch__grid_size", "block": "launch__block_size", "regs": "launch__registers_per_thread",
    "smem_per_block": "launch__shared_mem_per_block_static",
    "smem_dyn_per_block": "launch__shared_mem_per_block_dynamic",
    "waves": "launch__waves_per_multiprocessor",
    "inst_executed": "smsp__inst_executed.sum",
    "warp_cycles_per_inst": "smsp__average_warp_latency_per_inst_issued.ratio",
}
STALL_PREFIX = "smsp__average_warps_issue_stalled_"
STALL_SUFFIX = "_per_issue_active.ratio"
PIPE_PREFIX = "sm__inst_executed_pipe_"
PIPE_SUFFIX = ".avg.pct_of_peak_sustained_active"


def limiter(inp: Inputs) -> list[dict]:
    out = []
    for path in sorted(inp.t8.glob("lim-raw-*.csv")):
        table = list(csv.reader(open(path, encoding="utf-8", newline="")))
        head, body = table[0], table[2:]
        col = {n: i for i, n in enumerate(head)}
        nvtx_col = next(i for i, n in enumerate(head) if n.startswith("thread Domain:Push/Pop_Range"))
        for idx, r in enumerate(body):
            def get(metric):
                i = col.get(metric)
                return num(r[i].replace(",", "")) if i is not None else float("nan")

            row = {"run": path.stem.replace("lim-raw-", ""), "idx": idx,
                   "kernel": a100_audit.short_name(r[col["Kernel Name"]]),
                   "tag": r[nvtx_col].split("prof")[0].strip(" :|")[-80:]}
            for k, m in LIM_METRICS.items():
                row[k] = get(m)
            row["duration_us"] /= 1000.0
            stalls = {n[len(STALL_PREFIX):-len(STALL_SUFFIX)]: get(n) for n in head
                      if n.startswith(STALL_PREFIX) and n.endswith(STALL_SUFFIX) and "not_issued" not in n}
            tot = sum(v for v in stalls.values() if not math.isnan(v))
            top = sorted(((v, k) for k, v in stalls.items() if not math.isnan(v)), reverse=True)[:4]
            row["stall_cycles_per_issue"] = tot
            row["stall_top"] = " ".join(f"{k}:{v:.2f}" for v, k in top)
            pipes = {n[len(PIPE_PREFIX):-len(PIPE_SUFFIX)]: get(n) for n in head
                     if n.startswith(PIPE_PREFIX) and n.endswith(PIPE_SUFFIX)}
            ptop = sorted(((v, k) for k, v in pipes.items() if not math.isnan(v)), reverse=True)[:4]
            row["pipe_top"] = " ".join(f"{k}:{v:.1f}" for v, k in ptop)
            out.append(row)
    return out


# --------------------------------------------------------------------------- clocks during the T8 runs


def clocks(inp: Inputs) -> list[dict]:
    """nvidia-smi samples (every 200 ms) written by x4_gpu_run.sh: SM / memory clock while the GPU was busy."""
    out = []
    for path in sorted(inp.t8.glob("clk-*.csv")):
        rows = [[x.strip() for x in line.split(",")] for line in open(path, encoding="utf-8") if line.strip()]
        busy = [r for r in rows if len(r) >= 7 and num(r[3], 0) > 0]
        if not busy:
            out.append({"log": path.name, "busy_samples": 0})
            continue
        sm = [num(r[1]) for r in busy]
        out.append({"log": path.name, "busy_samples": len(busy), "sm_mhz_min": min(sm),
                    "sm_mhz_median": statistics.median(sm), "sm_mhz_max": max(sm),
                    "share_at_1410": sum(1 for x in sm if x == 1410) / len(sm),
                    "mem_mhz": " ".join(sorted({r[2] for r in busy})),
                    "power_w_max": max(num(r[4]) for r in busy), "temp_c_max": max(num(r[5]) for r in busy),
                    "throttle_reasons": " ".join(sorted({r[6] for r in busy}))})
    return out


# --------------------------------------------------------------------------- charts


def vl(path: Path, spec: dict) -> None:
    spec = {"$schema": "https://vega.github.io/schema/vega-lite/v5.json", **spec}
    path.write_text(json.dumps(spec, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {path.name}")


REG_SCALE = {"domain": [R_FIX, R_SM, R_BW, R_TC, R_MID],
             "range": ["#9ca3af", "#f59e0b", "#2563eb", "#dc2626", "#a78bfa"]}


def charts(out: Path, scale1: bool) -> None:
    num_parse = {k: "number" for k in ("batch", "g_cold_us", "h_us", "e_p50_us", "stages", "n_in", "nnz",
                                       "fixed_share", "per_kernel_us")}
    # 1. the map: chain x batch, colour = device regime, cross = launch bound (eager)
    vl(out / "gpu-map.vl.json", {
        "description": "瓶颈地图（GPU 一侧，G_cold 口径）：每格是一条链在一个 batch 下的判定。颜色 = 去掉主机发射后 GPU 被什么卡住；"
                       "叉号 = 主机发一次整条链的时间 h_us 大于 G_cold（连续 eager 调用时发射受限）；白圈 = 有 ncu 的格子（其余格子是按规则推断的）。行按 nnz 排序。数据：gpu-regimes.csv",
        "data": {"url": "gpu-regimes.csv", "format": {"type": "csv", "parse": num_parse}},
        "transform": [{"filter": "datum.dataset != 'scale1' && datum.impl != 'cusparse'"},
                      {"calculate": "(datum.dataset == 'v1' ? 'v1/' : 'ref/') + datum.chain + ' (' + datum.stages + ' 级)'",
                       "as": "who"}],
        "facet": {"column": {"field": "impl", "type": "nominal", "title": None, "sort": ["ksmm", "cublas"]}},
        "spec": {"width": 260, "height": {"step": 11}, "layer": [
            {"mark": {"type": "rect"},
             "encoding": {"x": {"field": "batch", "type": "ordinal", "title": "batch"},
                          "y": {"field": "who", "type": "nominal", "title": None, "axis": {"labelLimit": 300},
                                "sort": {"field": "nnz", "op": "max", "order": "ascending"}},
                          "color": {"field": "device_regime", "type": "nominal", "title": "GPU 被什么卡住",
                                    "scale": REG_SCALE},
                          "tooltip": [{"field": "who"}, {"field": "impl"}, {"field": "batch"},
                                      {"field": "device_regime"}, {"field": "eager_regime"},
                                      {"field": "g_cold_us", "format": ".1f"}, {"field": "h_us", "format": ".1f"},
                                      {"field": "fixed_share", "format": ".2f"}, {"field": "bw_share", "type": "quantitative",
                                                                                 "format": ".2f"},
                                      {"field": "tc_share", "type": "quantitative", "format": ".2f"}]}},
            {"transform": [{"filter": "datum.ncu_source != null && datum.ncu_source != ''"}],
             "mark": {"type": "point", "shape": "circle", "size": 70, "color": "white", "filled": False, "strokeWidth": 1.5},
             "encoding": {"x": {"field": "batch", "type": "ordinal"},
                          "y": {"field": "who", "type": "nominal", "sort": {"field": "nnz", "op": "max",
                                                                          "order": "ascending"}}}},
            {"transform": [{"filter": "datum.eager_regime == '发射受限'"}],
             "mark": {"type": "point", "shape": "cross", "size": 30, "color": "black", "filled": True},
             "encoding": {"x": {"field": "batch", "type": "ordinal"},
                          "y": {"field": "who", "type": "nominal", "sort": {"field": "nnz", "op": "max",
                                                                          "order": "ascending"}}}}]},
        "resolve": {"scale": {"y": "shared"}}})
    # 2. batch axis: G_cold and host issue time vs batch for three chains
    vl(out / "gpu-batch.vl.json", {
        "description": "batch 轴：G_cold 随 batch（实线，点形 = 判定），以及主机发一次整条链的时间 h（KSMM，虚线）。h 高于 G_cold 的区间在 eager 下发射受限。数据：gpu-regimes.csv",
        "data": {"url": "gpu-regimes.csv", "format": {"type": "csv", "parse": num_parse}},
        "transform": [{"filter": {"field": "chain", "oneOf": ["debut_1", "debut_5", "bfly_4"]}},
                      {"filter": "datum.dataset == 'v3-reference-data' && datum.impl != 'cusparse'"}],
        "facet": {"column": {"field": "chain", "type": "nominal", "title": None, "sort": ["debut_1", "debut_5", "bfly_4"]}},
        "spec": {"width": 220, "height": 220, "layer": [
            {"mark": {"type": "line", "point": False, "color": "#9ca3af"},
             "encoding": {"x": {"field": "batch", "type": "quantitative", "scale": {"type": "log", "base": 2}},
                          "y": {"field": "g_cold_us", "type": "quantitative", "scale": {"type": "log"},
                                "title": "µs"},
                          "detail": {"field": "impl"}}},
            {"mark": {"type": "point", "filled": True, "size": 50},
             "encoding": {"x": {"field": "batch", "type": "quantitative"}, "y": {"field": "g_cold_us", "type": "quantitative"},
                          "color": {"field": "device_regime", "type": "nominal", "title": "判定", "scale": REG_SCALE},
                          "shape": {"field": "impl", "type": "nominal", "title": "实现"},
                          "tooltip": [{"field": "impl"}, {"field": "batch"}, {"field": "g_cold_us", "format": ".1f"},
                                      {"field": "device_regime"}]}},
            {"transform": [{"filter": "datum.impl == 'ksmm'"}],
             "mark": {"type": "line", "strokeDash": [4, 3], "color": "black"},
             "encoding": {"x": {"field": "batch", "type": "quantitative"}, "y": {"field": "h_us", "type": "quantitative"}}}]}})
    # 3. stages axis: v1 synthetic_depth* (all 4096 points) and butterflies, batch 1
    vl(out / "gpu-stages.vl.json", {
        "description": "级数轴（batch 1，KSMM）：v1 的 synthetic_depth5…11_4096（N 都是 4096）。G_cold、E p50 和主机发一次整条链的时间 h 随级数的变化。数据：gpu-regimes.csv",
        "data": {"url": "gpu-regimes.csv", "format": {"type": "csv", "parse": num_parse}},
        "transform": [{"filter": "datum.impl == 'ksmm' && datum.batch == 1 && datum.dataset == 'v1' && indexof(datum.chain, 'synthetic_depth') == 0"},
                      {"fold": ["g_cold_us", "e_p50_us", "h_us"], "as": ["col", "us"]},
                      {"calculate": "{'g_cold_us': 'G_cold', 'e_p50_us': 'E p50', 'h_us': '主机发射 h'}[datum.col]",
                       "as": "口径"}],
        "width": 360, "height": 240,
        "mark": {"type": "line", "point": True},
        "encoding": {"x": {"field": "stages", "type": "quantitative", "title": "级数（= kernel 数）",
                           "scale": {"zero": False}, "axis": {"tickMinStep": 1, "format": "d"}},
                     "y": {"field": "us", "type": "quantitative", "title": "µs"},
                     "color": {"field": "口径", "type": "nominal", "title": None,
                               "scale": {"domain": ["主机发射 h", "E p50", "G_cold"],
                                         "range": ["#6b7280", "#f59e0b", "#2563eb"]}},
                     "tooltip": [{"field": "chain"}, {"field": "口径"}, {"field": "us", "format": ".1f"}]}})
    # 3b. scale axis on measured chains (batch 1)
    fam = ("test(/^(butterfly_|r2v_)/, datum.chain) ? 'radix-2，v1 的级序' : "
           "test(/^(bfly_|r2t_)/, datum.chain) ? 'radix-2，表的级序' : "
           "test(/^(r4_|synthetic_radix4)/, datum.chain) ? 'radix-4' : "
           "test(/^debut_rect_/, datum.chain) ? 'DeBut 矩形（scale1）' : null")
    vl(out / "gpu-scale.vl.json", {
        "description": "规模轴（batch 1）：G_cold 随输入维度 N。点形 = 判定；黑色虚线 = KSMM 的主机发射时间 h（高于 G_cold 时 eager 发射受限）。"
                       + ("含 scale1（T7）到 2^16。" if scale1 else "scale1 未发布，只有 v1、v3-reference-data 到 8192。")
                       + "数据：gpu-regimes.csv",
        "data": {"url": "gpu-regimes.csv", "format": {"type": "csv", "parse": num_parse}},
        "transform": [{"filter": "datum.batch == 1 && datum.impl != 'cusparse'"},
                      {"calculate": fam, "as": "family"}, {"filter": "datum.family != null"}],
        "facet": {"field": "family", "type": "nominal", "title": None}, "columns": 2,
        "spec": {"width": 230, "height": 180, "layer": [
            {"mark": {"type": "line", "color": "#9ca3af"},
             "encoding": {"x": {"field": "n_in", "type": "quantitative", "scale": {"type": "log", "base": 2}, "title": "N"},
                          "y": {"field": "g_cold_us", "type": "quantitative", "scale": {"type": "log"}, "title": "µs"},
                          "detail": [{"field": "dataset"}, {"field": "impl"}]}},
            {"mark": {"type": "point", "filled": True, "size": 50},
             "encoding": {"x": {"field": "n_in", "type": "quantitative"}, "y": {"field": "g_cold_us", "type": "quantitative"},
                          "color": {"field": "device_regime", "type": "nominal", "title": "判定", "scale": REG_SCALE},
                          "shape": {"field": "impl", "type": "nominal", "title": "实现"},
                          "tooltip": [{"field": "chain"}, {"field": "impl"}, {"field": "g_cold_us", "format": ".1f"},
                                      {"field": "h_us", "format": ".1f"}, {"field": "device_regime"}]}},
            {"transform": [{"filter": "datum.impl == 'ksmm'"}],
             "mark": {"type": "line", "strokeDash": [4, 3], "color": "black"},
             "encoding": {"x": {"field": "n_in", "type": "quantitative"}, "y": {"field": "h_us", "type": "quantitative"},
                          "detail": {"field": "dataset"}}}]},
        "resolve": {"scale": {"x": "independent"}}})
    # 4. scale axis: per-kernel time vs CTA count (floor probe)
    vl(out / "gpu-cta.vl.json", {
        "description": "规模轴：一个 KSMM kernel（batch 1）在 graph 里的时间随 CTA 数（= 每级的 A·D，N/2 或 N/4）的变化。"
                       "点 = T8 探针 (a, r, r, 1) 级重复 8 次与 1 次之差 ÷ 7；虚线 = 只有一个 CTA 的 KSMM kernel；"
                       "点线 = 空 kernel。数据：gpu-floor.csv",
        "data": {"url": "gpu-floor.csv", "format": {"type": "csv", "parse": {"ctas": "number", "per_kernel_us": "number"}}},
        "width": 380, "height": 260,
        "layer": [
            {"transform": [{"filter": "isValid(datum.ctas) && datum.ctas > 0"}],
             "mark": {"type": "line", "point": True},
             "encoding": {"x": {"field": "ctas", "type": "quantitative", "scale": {"type": "log", "base": 2},
                                "title": "每个 kernel 的 CTA 数"},
                          "y": {"field": "per_kernel_us", "type": "quantitative", "scale": {"type": "log"},
                                "title": "每个 kernel 的时间（µs，G_cold）"},
                          "color": {"field": "radix", "type": "nominal", "title": "radix"},
                          "tooltip": [{"field": "radix"}, {"field": "ctas"}, {"field": "per_kernel_us", "format": ".2f"}]}},
            {"transform": [{"filter": "datum.kind == 'const'"}],
             "mark": {"type": "rule", "strokeDash": [4, 3]},
             "encoding": {"y": {"field": "per_kernel_us", "type": "quantitative"},
                          "color": {"value": "black"}, "tooltip": [{"field": "label"}]}}]})
    # 5. per-chain split of G_cold (KSMM, batch 1) with V4.1 as a tick
    vl(out / "gpu-split.vl.json", {
        "description": "KSMM 一条链的 G_cold（batch 1）拆成四段：graph 启动、每个 kernel 的空 kernel 下限、单 CTA kernel 比空 kernel 多出的时间、其余（CTA 数带来的部分）。"
                       "黑色竖线 = V4.1 的时间。数据：gpu-chain-split.csv",
        "data": {"url": "gpu-chain-split.csv", "format": {"type": "csv", "parse": {
            "graph_launch_us": "number", "kernel_floor_us": "number", "onecta_extra_us": "number", "cta_us": "number",
            "v41_us": "number", "g_cold_us": "number"}}},
        "transform": [{"calculate": "(datum.dataset == 'v1' ? 'v1/' : 'ref/') + datum.chain", "as": "who"}],
        "height": {"step": 12}, "width": 420,
        "layer": [
            {"transform": [{"fold": ["graph_launch_us", "kernel_floor_us", "onecta_extra_us", "cta_us"], "as": ["段", "us"]},
                           {"calculate": "{'graph_launch_us': 'graph 启动', 'kernel_floor_us': '空 kernel 下限 × 级数', "
                                         "'onecta_extra_us': '单 CTA kernel 比空 kernel 多出的时间 × 级数', 'cta_us': 'CTA 数带来的部分'}[datum['段']]",
                            "as": "segment"}],
             "mark": "bar",
             "encoding": {"y": {"field": "who", "type": "nominal", "title": None,
                                "sort": {"field": "g_cold_us", "op": "max", "order": "ascending"}},
                          "x": {"field": "us", "type": "quantitative", "title": "µs（G_cold）", "stack": "zero"},
                          "color": {"field": "segment", "type": "nominal", "title": None,
                                    "sort": ["graph 启动", "空 kernel 下限 × 级数", "单 CTA kernel 比空 kernel 多出的时间 × 级数", "CTA 数带来的部分"]},
                          "tooltip": [{"field": "who"}, {"field": "segment"}, {"field": "us", "format": ".2f"}]}},
            {"mark": {"type": "tick", "color": "black", "thickness": 2},
             "encoding": {"y": {"field": "who", "type": "nominal", "sort": {"field": "g_cold_us", "op": "max", "order": "ascending"}},
                          "x": {"field": "v41_us", "type": "quantitative"},
                          "tooltip": [{"field": "who"}, {"field": "v41_us", "format": ".3f"}]}}]})
    # 6. KSMM configs at batch 4096 / 256 / 1
    vl(out / "gpu-config.vl.json", {
        "description": "KSMM 的 Triton 配置（单级形状，重复 8 次）：G_cold 随 BLOCK_SIZE_BATCH（横轴）和 warps（颜色）变化，num_stages 取 3。"
                       "虚线 = T4 的 autotune 选的配置（default）。数据：gpu-config.csv",
        "data": {"url": "gpu-config.csv", "format": {"type": "csv", "parse": {"g_cold_us": "number", "batch": "number"}}},
        "transform": [{"filter": "datum.file == 'cfg-stage.jsonl' && datum.status == 'ok'"},
                      {"calculate": "datum.name + ' · batch ' + datum.batch", "as": "case"},
                      {"calculate": "datum.variant == 'default' ? null : toNumber(split(datum.variant, '/')[0])", "as": "bbatch"},
                      {"calculate": "datum.variant == 'default' ? null : 'w' + split(datum.variant, '/')[1]", "as": "warps"},
                      {"calculate": "datum.variant == 'default' ? null : split(datum.variant, '/')[2]", "as": "stages_cfg"}],
        "facet": {"field": "case", "type": "nominal", "title": None}, "columns": 4,
        "spec": {"width": 150, "height": 130, "layer": [
            {"transform": [{"filter": "datum.variant != 'default' && datum.stages_cfg == '3'"}],
             "mark": {"type": "line", "point": True},
             "encoding": {"x": {"field": "bbatch", "type": "ordinal", "title": "BLOCK_SIZE_BATCH"},
                          "y": {"field": "g_cold_us", "type": "quantitative", "title": "µs", "scale": {"zero": False}},
                          "color": {"field": "warps", "type": "nominal", "title": "warps"},
                          "tooltip": [{"field": "variant"}, {"field": "g_cold_us", "format": ".1f"}]}},
            {"transform": [{"filter": "datum.variant == 'default'"}],
             "mark": {"type": "rule", "strokeDash": [4, 3], "color": "black"},
             "encoding": {"y": {"field": "g_cold_us", "type": "quantitative"}}}]},
        "resolve": {"scale": {"y": "independent"}}})


# --------------------------------------------------------------------------- report numbers


def report(c, reg, chain_split, lr, xo, cfg, lim, inp) -> None:
    print("\n== floor constants (us)")
    for k, v in c.items():
        print(f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")
    print("\n== regime counts (dataset, impl, regime)")
    cnt = defaultdict(int)
    for r in reg:
        cnt[(r["dataset"], r["impl"], r["device_regime"])] += 1
    for k, v in sorted(cnt.items()):
        print(f"  {k}: {v}")
    print("\n== regime basis (ksmm points with byte estimate >= 30 %)")
    for r in reg:
        if r["regime_basis"] != "规则":
            print(f"  {r['dataset'][:4]} {r['chain']:26s} b={r['batch']:<5} est={r['bw_share']:.3f} "
                  f"dram={r['ncu_dram_pct']} sm={r['ncu_sm_pct']} -> {r['device_regime']} ({r['regime_basis']})")
    print("\n== ncu agreement")
    agree = defaultdict(int)
    for r in reg:
        if r["ncu_agrees"]:
            agree[r["ncu_agrees"]] += 1
            if r["ncu_agrees"] == "不一致":
                print("  disagree:", r["dataset"], r["chain"], r["impl"], r["batch"], r["device_regime"],
                      r["ncu_dram_pct"], r["ncu_sm_pct"], r["ncu_tensor_pct"])
    print(" ", dict(agree))
    print("\n== launch bound (eager), batch 1")
    for impl in IMPLS:
        b1 = [r for r in reg if r["impl"] == impl and r["batch"] == 1]
        lb = [r for r in b1 if r["eager_regime"] == "发射受限"]
        hg = [r["h_over_gcold"] for r in b1]
        print(f"  {impl}: {len(lb)}/{len(b1)} launch bound; h/G_cold {min(hg):.2f}-{max(hg):.2f}")
    print("\n== eager idle share under nsys (E mode, batch 1, ksmm) by stages")
    by_k = defaultdict(list)
    for r in reg:
        if r["impl"] == "ksmm" and r["batch"] == 1 and r["nsys_e_idle_share"] != "":
            by_k[r["stages"]].append(r["nsys_e_idle_share"])
    for k in sorted(by_k):
        print(f"  K={k:2d}: idle {min(by_k[k]):.2f}-{max(by_k[k]):.2f} (n={len(by_k[k])})")
    print("\n== stages axis: v1 synthetic_depth*_4096, ksmm batch 1 (slope per stage, us)")
    dep = sorted((r["stages"], r) for r in reg if r["impl"] == "ksmm" and r["batch"] == 1
                 and r["dataset"] == "v1" and r["chain"].startswith("synthetic_depth"))
    for col in ("g_cold_us", "e_p50_us", "h_us"):
        a, b = fit([k for k, _ in dep], [r[col] for _, r in dep])
        print(f"  {col}: {a:.2f} + {b:.2f} * K  ({', '.join(f'{r[col]:.1f}' for _, r in dep)})")
    print("\n== host issue time per kernel (ksmm, batch 1): h / K")
    hk = [r["h_us"] / r["stages"] for r in reg if r["impl"] == "ksmm" and r["batch"] == 1]
    print(f"  {min(hk):.1f}-{max(hk):.1f} us, median {statistics.median(hk):.1f}")
    print("\n== chain split (ksmm, batch 1)")
    for r in chain_split:
        print(f"  {r['dataset'][:4]} {r['chain']:30s} K={r['stages']:2d} G={r['g_cold_us']:7.2f} launch={r['share_launch']:.2f} "
              f"floor={r['share_floor']:.2f} onecta={r['share_onecta_extra']:.2f} cta={r['share_cta']:.2f} "
              f"fixed_total={r['share_fixed_total']:.2f} ratio={r['ratio']:6.1f} "
              f"stageGPU={r['per_stage_gpu_us']:.2f} stageV41={r['per_stage_v41_us']:.3f}")
    print("\n== launch removed")
    for r in lr:
        if r["dataset"] == "v3-reference-data" or r["chain"] in ("butterfly_8192", "r2v_65536", "r2t_65536"):
            print(f"  {r['dataset'][:4]} {r['chain']:22s} E={r['ksmm_e_p50_us']:7.1f} Gc={r['ksmm_g_cold_us']:7.1f} "
                  f"Gw={r['ksmm_g_warm_us']:7.1f} floor0={r['ksmm_floor_no_graph_us']:6.1f} cub={r['cublas_g_cold_us']:8.1f} "
                  f"V41={r['v41_us']:.3f} | dev={r['eager_device_extra_us']:6.1f} host={r['host_in_e_us']:6.1f} "
                  f"hidden={r['host_hidden_by_flush']} | xE={r['ratio_e']:.1f} xE80={r['ratio_e_p80']:.1f} "
                  f"xG={r['ratio_g_cold']:.1f} xGw={r['ratio_g_warm']:.1f} xG-L={r['ratio_no_graph_launch']:.1f} "
                  f"xfloor0={r['ratio_floor_no_graph']:.1f} xcub={r['ratio_cublas']:.1f} xbest={r['ratio_best']:.1f}")
    print("\n== crossovers (ksmm)")
    for r in xo:
        if r["dataset"] == "v3-reference-data":
            print(f"  {r['chain']:10s} launch-bound<= {r['launch_bound_until']} first-not-fixed={r['first_batch_not_fixed']} "
                  f"cublas-faster-from={r['first_batch_cublas_faster']} ksmm-faster-at=[{r['batches_ksmm_faster']}]")
    print("\n== configs")
    for r in cfg:
        if r["variant"] in ("default", "autotune-ext", "256/1/3") or (r["file"] == "cfg-stage.jsonl" and r["speedup_vs_default"] and r["speedup_vs_default"] > 1.05):
            print(f"  {r['file'][:9]} {r['name']:24s} b={r['batch']:<5} {r['variant']:13s} {r['status']:6s} {r['check']} "
                  f"Gc={num(r['g_cold_us']):9.2f} x{num(r['speedup_vs_default']):.2f} cublas={r['cublas_g_cold_us']} {r['picked'][:70]}")
    print("\n== limiter")
    for r in lim:
        print(f"  {r['run']:22s} {r['idx']} {r['kernel'][:28]:28s} {r['duration_us']:8.2f}us sm={r['sm_pct']:.1f} dram={r['dram_pct']:.1f} "
              f"issue={r['issue_active_pct']:.1f} occ={r['occ_achieved_pct']:.1f}/{r['occ_theoretical_pct']:.1f} "
              f"pipes[{r['pipe_top']}] stalls[{r['stall_top']}]")
    best = ncu_summary(inp).get(("v3-reference-data-cfg256", "bfly_4", "ksmm", 4096))
    if best:
        print("\n== bfly_4 ksmm batch 4096, config 256/1/3, application replay (T4 metrics):",
              {k: (round(v, 2) if isinstance(v, float) else v) for k, v in best.items()})
    print("\nscale1 published:", inp.scale1_published)


def report_flush(fb: list[dict], lr: list[dict]) -> None:
    print("\n== L2 flush bias (G_cold zero_() vs read-only flush, batch 1, fastest round of each)")
    for impl in ("cublas", "ksmm"):
        rows = [r for r in fb if r["impl"] == impl and r["status"] == "ok"]
        if not rows:
            continue
        b = [r["bias_pct"] for r in rows]
        print(f"  {impl}: n={len(rows)} bias {min(b):.1f}..{max(b):.1f}% median {statistics.median(b):.1f}%")
        for r in rows:
            mb = "" if r["matrix_mb"] == "" else f"{r['matrix_mb']:.1f}"
            print(f"    {r['dataset'][:4]} {r['chain']:28s} {mb:>8} MB pub={r['g_cold_published_us']:9.2f} "
                  f"zero={r['g_cold_zero_us']:9.2f} clean={r['g_cold_clean_us']:9.2f} bias={r['bias_pct']:6.1f}% "
                  f"pub/zero={r['published_vs_zero_pct']:6.1f}% x_pub={r['ratio_published']:.2f} "
                  f"x_corr={r['ratio_corrected']:.2f} [{r['zero_rounds_us']} | {r['clean_rounds_us']}] {r['file']}")
    errs = [r for r in fb if r["status"] != "ok"]
    print("  errors:", [(r["dataset"], r["chain"], r["impl"], r["status"]) for r in errs])
    for ds in DATASETS:
        rows = [r for r in lr if r["dataset"] == ds and not math.isnan(r["ratio_cublas_corrected"])]
        if rows:
            print(f"  {ds}: cublas/V41 median {statistics.median(r['ratio_cublas'] for r in rows):.2f} -> "
                  f"{statistics.median(r['ratio_cublas_corrected'] for r in rows):.2f}; best/V41 median "
                  f"{statistics.median(r['ratio_best'] for r in rows):.2f} -> "
                  f"{statistics.median(r['ratio_best_corrected'] for r in rows):.2f}; best impl changes: "
                  f"{[r['chain'] for r in rows if r['best_gpu_impl'] != r['best_gpu_corrected_impl']]}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ops", required=True, type=Path)
    p.add_argument("--t8", required=True, type=Path)
    args = p.parse_args()
    inp = Inputs(args.ops, args.t8)
    out = inp.data
    c, curve = floor_constants(inp)
    curve += [{"kind": "const", "label": "一个 CTA 的 KSMM kernel（G_cold）", "per_kernel_us": c["ksmm_1cta_us"]},
              {"kind": "const", "label": "空 kernel（G_cold）", "per_kernel_us": c["kernel_floor_us"]}]
    ncu = ncu_summary(inp)
    reg = regimes(inp, c, ncu)
    stage_rows, chain_split = stage_compare(inp, c)
    fb = flush_bias(inp)
    lr = launch_removed(inp, c, fb)
    xo = crossovers(reg)
    cfg = configs(inp, reg)
    lim = limiter(inp)
    write_csv(out / "gpu-floor.csv", curve)
    write_csv(out / "gpu-regimes.csv", reg)
    write_csv(out / "gpu-stage-compare.csv", stage_rows)
    write_csv(out / "gpu-chain-split.csv", chain_split)
    write_csv(out / "gpu-launch-removed.csv", lr)
    write_csv(out / "gpu-crossover.csv", xo)
    if cfg:
        write_csv(out / "gpu-config.csv", cfg)
    if lim:
        write_csv(out / "gpu-limiter.csv", lim)
    write_csv(out / "gpu-t8-clocks.csv", clocks(inp))
    if fb:
        write_csv(out / "gpu-flush-bias.csv", fb)
    charts(out, inp.scale1_published)
    report(c, reg, chain_split, lr, xo, cfg, lim, inp)
    report_flush(fb, lr)


if __name__ == "__main__":
    main()
