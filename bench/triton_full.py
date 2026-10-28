"""KSMM-Triton with the author's full autotune space (operations task 2026-10-27-gpu-bench, step T4b).

src/ksmm_triton/ksmm_triton_tc.py:104-107 narrows BLOCK_SIZE_B/C to [16] and BLOCK_SIZE_BATCH to
[32] (6 configs); the full lists [16, 32, 64, 128] are commented out there. This module rebuilds
the config list with the full lists and the num_warps / num_stages rule of :109-137 (copied
verbatim), wraps the same JIT kernel (ks_fused_kernel.fn) in a new triton.autotune with the same
key, and runs ks_triton_forward_impl with that autotuner in place of ks_fused_kernel. src/ is
not modified.
"""
from __future__ import annotations

import types

import triton
from ksmm_triton import ksmm_triton_tc as tc

BLOCK_SIZES = [16, 32, 64, 128]
BATCH_SIZES = [16, 32, 64, 128]


def full_configs() -> list:
    """The config loop of ksmm_triton_tc.py:109-137 with the full block and batch lists."""
    configs = []
    for BB in BLOCK_SIZES:
        for BC in BLOCK_SIZES:
            for BBATCH in BATCH_SIZES:
                total_work = BB * BC * BBATCH
                if total_work <= 1024:
                    warp_options = [1, 2]
                elif total_work <= 4096:
                    warp_options = [2, 4]
                else:
                    warp_options = [4, 8]
                for num_warps in warp_options:
                    if total_work <= 2048:
                        stage_options = [2, 3, 4]
                    elif total_work <= 8192:
                        stage_options = [3, 4, 5]
                    else:
                        stage_options = [4, 5]
                    for num_stages in stage_options:
                        configs.append(triton.Config(
                            {'BLOCK_SIZE_B': BB, 'BLOCK_SIZE_C': BC, 'BLOCK_SIZE_BATCH': BBATCH},
                            num_warps=num_warps, num_stages=num_stages))
    return configs


ks_fused_kernel_full = triton.autotune(configs=full_configs(), key=['a', 'b', 'c', 'd', 'B'])(
    tc.ks_fused_kernel.fn)  # pyright: ignore[reportFunctionMemberAccess]  (an Autotuner; .fn is the JITFunction)

# ks_triton_forward_impl looks ks_fused_kernel up in its module globals; this copy sees the
# full-space autotuner under that name and is otherwise the same function.
_impl = tc.ks_triton_forward_impl
ks_triton_forward_full = types.FunctionType(
    _impl.__code__, {**_impl.__globals__, 'ks_fused_kernel': ks_fused_kernel_full},
    _impl.__name__, _impl.__defaults__, _impl.__closure__)
