"""ksmm-cuda implementation for gpu_bench.py (--impl cuda): the extension built by build.py.

The extension is loaded from $KSMM_CUDA_BUILD, default $S/ksmm-cuda-build (build.py --out).
"""
from __future__ import annotations

import importlib.machinery
import importlib.util
import os
from pathlib import Path

NAME = "ksmm_cuda_bsl_half"
_ext = None


def extension():
    """The built extension module (import torch first)."""
    global _ext
    if _ext is None:
        build = Path(os.environ.get("KSMM_CUDA_BUILD") or Path(os.environ["S"]) / "ksmm-cuda-build")
        so = build / "build" / f"{NAME}.so"
        loader = importlib.machinery.ExtensionFileLoader(NAME, str(so))
        spec = importlib.util.spec_from_file_location(NAME, so, loader=loader)
        if spec is None:
            raise ImportError(f"cannot load {so}")
        _ext = importlib.util.module_from_spec(spec)
        loader.exec_module(_ext)
    return _ext


def make(torch, built, x):
    """(run, config) of KSMM-CUDA: one ksmm bs_last fp16 kernel per stage, weights as (A*D*C, B), batch 25088 only."""
    ext = extension()
    stages = built.chain.stages
    missing = [s for s in stages if not ext.has(*s)]
    if missing:
        raise ValueError(f"no batch-{ext.batch} bs_last half hyperparameters for {missing}")
    if x.shape[1] != ext.batch:
        raise ValueError(f"ksmm-cuda is built for batch {ext.batch}, not {x.shape[1]}")
    plan = [(k.view(a * d * c, b), (a, b, c, d)) for k, (a, b, c, d) in zip(built.k_bmm, stages)]

    def run():
        y = x
        for w, pattern in plan:
            y = ext.forward(y, w, *pattern)
        return y

    return run, lambda: " ".join(ext.config(*s) for s in stages)
