"""Host issue time of the old harness call vs. this harness's ksmm call (batch 1), on zhang-x4.

Explains why the checkpoint's E of the old call is higher than ours. Run with x4_gpu_run.sh.
"""
import sys, torch
sys.path.insert(0, "/scratch/users/phd/yd383/v41-audit/ksmm/audit")
sys.path.insert(0, "/scratch/users/phd/yd383/do-bench-harness")
import a100_audit as h
from a100bench.runner import _make_ksmm_call
torch_, triton, ks_forward, _k = h.gpu_stack()
chains = {c.name: c for c in h.read_chains(h.Path("/scratch/users/phd/yd383/v41-audit/datasets/v3-reference-data/chains.csv"))}
for name in ["debut_1", "debut_5", "bfly_1", "bfly_4"]:
    c = chains[name]
    legacy, _ = _make_ksmm_call({"stages": [[a, b, cc, d, 1] for a, b, cc, d in c.stages], "batch": 1, "M": c.m_in}, torch, ks_forward)
    built = h.Built(torch, c)
    ours, _ = built.make_call(torch, ks_forward, "ksmm", h.make_input(torch, c, 1))
    hl = sorted(h.host_issue_us(torch, legacy) for _ in range(5))[2]
    ho = sorted(h.host_issue_us(torch, ours) for _ in range(5))[2]
    x = torch.empty((1, c.m_in), dtype=torch.float16, device="cuda")
    import time
    t0 = time.perf_counter()
    for _ in range(1000):
        y = x.transpose(0, 1).contiguous()
    t1 = time.perf_counter()
    print(f"{name:8s} stages={len(c.stages):2d} h_legacy={hl:7.2f} h_ours={ho:7.2f} diff={hl-ho:6.2f} us; one transpose().contiguous() = {(t1-t0)/1000*1e6:.2f} us")
