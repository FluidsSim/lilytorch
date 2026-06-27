"""Peak-GPU-memory experiment for **Kernel B** — the load-bearing claim.

Kernel-mode's headline benefit is *memory*: the BDIM weights ``mu0, mu1`` and the
three unit-normal components are computed in **thread registers** and never
materialised as global tensors.  The PyTorch "python path" cannot do that — it
must allocate full-grid intermediates (``mu0_{u,v,w}``, ``mu1_{u,v,w}``,
``n{x,y,z}``, the three ∂-derivatives, …), which is exactly the transient that
the 3-D memory work (5.749 GiB python-path peak in ``_recompute_mu_normals``)
set out to remove.

This bench measures the **peak GPU memory above the steady-state baseline** for
three implementations of the *same* computation:

  * ``python``  — full-grid torch intermediates (what kernel mode replaces);
  * ``native``  — the hand-written CUDA ``bdim_coeff_3d`` (register-resident);
  * ``warp``    — the single-source ``@wp.kernel`` port.

Two measurements per path:
  * **torch peak Δ** via ``torch.cuda.max_memory_allocated`` (catches every
    allocation the torch caching allocator makes — i.e. the python path, and any
    spill the Warp kernel might make through a *torch* tensor);
  * **driver Δ** via ``torch.cuda.mem_get_info`` (driver-level free memory,
    which *also* sees Warp's separate allocator) — to confirm Warp does not
    quietly spill mu/normals into a Warp-allocated global.

The point we want to land: ``native ≈ warp ≈ 0`` extra (registers), while
``python`` balloons with the grid — and all three agree numerically (printed
parity), so it is genuinely the same kernel, just different memory profiles.

Run:  python -m lilytorch.warp_poc.bench_bdim_memory --grids 96 128 160
"""
from __future__ import annotations

import argparse
import torch

import lilytorch.src.kernels  # noqa: F401
from lilytorch.src.kernels.ops import bdim_coeff_3d as native_3d
from lilytorch.warp_poc.warp_bdim import bdim_coeff_3d_warp

RHO, DT = 1000.0, 1e-3
MIB = 1024.0 ** 2


# ─── manufactured problem ────────────────────────────────────────────────────

def build(N, dev, dtype=torch.float64):
    h = 1.0 / N
    eps = 2 * h
    torch.manual_seed(0)
    g = torch.Generator(device=dev).manual_seed(0)
    xs = (torch.arange(N, device=dev) - N / 2.0) * h
    X, Y, Z = torch.meshgrid(xs, xs, xs, indexing="ij")
    R = 0.3

    def sph(o):
        return (torch.sqrt((X - o) ** 2 + Y ** 2 + Z ** 2) - R).to(dtype)

    def rnd():
        return torch.randn(N, N, N, dtype=dtype, device=dev, generator=g)

    prob = dict(
        h=h, eps=eps, N=N,
        sdf_u=sph(0.5 * h), sdf_v=sph(0.0), sdf_w=sph(0.0),
        body_u=rnd(), body_v=rnd(), body_w=rnd(),
        up=rnd(), vp=rnd(), wp=rnd(),
    )
    del X, Y, Z
    return prob


def fresh_outputs(P, dev, dtype=torch.float64):
    N = P["N"]
    u0, v0, w0 = P["up"].clone(), P["vp"].clone(), P["wp"].clone()
    base = DT / RHO
    ch = torch.full((N - 1, N - 2, N - 2), base, dtype=dtype, device=dev)
    cv = torch.full((N - 2, N - 1, N - 2), base, dtype=dtype, device=dev)
    cw = torch.full((N - 2, N - 2, N - 1), base, dtype=dtype, device=dev)
    return u0, v0, w0, ch, cv, cw


# ─── the three implementations (identical math) ──────────────────────────────

def run_native(P, outs):
    u0, v0, w0, ch, cv, cw = outs
    N = P["N"]
    native_3d(P["up"], P["vp"], P["wp"],
              P["sdf_u"], P["sdf_v"], P["sdf_w"],
              P["body_u"], P["body_v"], P["body_w"],
              u0, v0, w0, ch, cv, cw,
              P["eps"], RHO, DT, P["h"], 0, 0, 0, N, N, N, 1)


def run_warp(P, outs):
    u0, v0, w0, ch, cv, cw = outs
    N = P["N"]
    bdim_coeff_3d_warp(P["up"], P["vp"], P["wp"],
                       P["sdf_u"], P["sdf_v"], P["sdf_w"],
                       P["body_u"], P["body_v"], P["body_w"],
                       u0, v0, w0, ch, cv, cw,
                       P["eps"], RHO, DT, P["h"], 0, 0, 0, N, N, N, 1)


def _mu0_mu1(phi, eps):
    deps = phi / eps
    s = torch.sin(torch.pi * deps)
    c = torch.cos(torch.pi * deps)
    mu0_band = 0.5 * (1.0 + deps + s / torch.pi)
    mu1_band = eps * (0.25 - 0.25 * deps * deps
                      - (s * deps + (1.0 + c) / torch.pi) / (2.0 * torch.pi))
    inside = phi <= -eps
    outside = phi >= eps
    mu0 = torch.where(inside, torch.zeros_like(phi),
                      torch.where(outside, torch.ones_like(phi), mu0_band))
    mu1 = torch.where(inside | outside, torch.zeros_like(phi), mu1_band)
    return mu0, mu1


def _clamp_central(a, ax, inv_2h):
    """Central difference with replicate-clamped edges (matches the .cu)."""
    up = torch.cat([a.narrow(ax, 1, a.shape[ax] - 1),
                    a.narrow(ax, a.shape[ax] - 1, 1)], dim=ax)
    dn = torch.cat([a.narrow(ax, 0, 1),
                    a.narrow(ax, 0, a.shape[ax] - 1)], dim=ax)
    return (up - dn) * inv_2h


def _zero_boundary(d, ax):
    """Zero the two boundary planes along ax (the .cu sets ∂=0 at i∈{0,N-1})."""
    n = d.shape[ax]
    d.narrow(ax, 0, 1).zero_()
    d.narrow(ax, n - 1, 1).zero_()
    return d


def _python_axis(phi, phi_prime, body, eps, inv_2h, mu0_proj):
    """Full-grid torch path for one face axis → (u0_axis, coeff_full)."""
    mu0, mu1 = _mu0_mu1(phi, eps)
    nx = _clamp_central(phi, 0, inv_2h)
    ny = _clamp_central(phi, 1, inv_2h)
    nz = _clamp_central(phi, 2, inv_2h)
    nn = torch.sqrt(nx * nx + ny * ny + nz * nz)
    scale = torch.where(nn > 0, 1.0 / nn, torch.zeros_like(nn))
    nx = nx * scale; ny = ny * scale; nz = nz * scale
    pmb = phi_prime - body
    ddx = _zero_boundary(_clamp_central(pmb, 0, inv_2h), 0)
    ddy = _zero_boundary(_clamp_central(pmb, 1, inv_2h), 1)
    ddz = _zero_boundary(_clamp_central(pmb, 2, inv_2h), 2)
    nd = nx * ddx + ny * ddy + nz * ddz
    u0 = mu0 * pmb + body + mu1 * nd
    coeff = (DT * mu0 if mu0_proj else DT) / RHO
    if not mu0_proj:
        coeff = torch.full_like(phi, DT / RHO)
    return u0, coeff


def run_python(P, outs):
    u0, v0, w0, ch, cv, cw = outs
    N = P["N"]
    inv_2h = 0.5 / P["h"]
    eps = P["eps"]
    uu, cu = _python_axis(P["sdf_u"], P["up"], P["body_u"], eps, inv_2h, 1)
    vv, cvf = _python_axis(P["sdf_v"], P["vp"], P["body_v"], eps, inv_2h, 1)
    ww, cwf = _python_axis(P["sdf_w"], P["wp"], P["body_w"], eps, inv_2h, 1)
    u0.copy_(uu); v0.copy_(vv); w0.copy_(ww)
    # compact face grids: padded i∈[1,N-1], j∈[1,N-2], k∈[1,N-2] → store (i-1,…)
    ch.copy_(cu[1:N, 1:N - 1, 1:N - 1])
    cv.copy_(cvf[1:N - 1, 1:N, 1:N - 1])
    cw.copy_(cwf[1:N - 1, 1:N - 1, 1:N])


# ─── measurement harness ─────────────────────────────────────────────────────

def peak_delta(fn, P, dev):
    """Peak torch-allocated memory ABOVE the post-setup baseline, in MiB,
    plus driver-level free-memory drop during the call (catches Warp's
    separate allocator).  Warms up once so allocators are primed."""
    outs = fresh_outputs(P, dev)
    fn(P, outs)                      # warmup (compile / prime caches)
    torch.cuda.synchronize()
    del outs

    outs = fresh_outputs(P, dev)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base_alloc = torch.cuda.memory_allocated()
    free0, _ = torch.cuda.mem_get_info()
    torch.cuda.reset_peak_memory_stats()

    fn(P, outs)
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() - base_alloc
    free1, _ = torch.cuda.mem_get_info()
    driver = free0 - free1           # >0 means GPU memory got consumed
    res = outs
    return peak / MIB, driver / MIB, res


def parity(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[96, 128, 160])
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA required."); return
    dev = "cuda:0"

    print(f"\nKernel B peak-memory experiment (float64, {dev})")
    print("'torch Δ' = max_memory_allocated above baseline; "
          "'drv Δ' = driver free-memory drop (sees Warp's allocator).\n")
    hdr = (f"{'grid':>6} {'out MiB':>9} | "
           f"{'python torchΔ':>13} {'drvΔ':>7} | "
           f"{'native torchΔ':>13} {'drvΔ':>7} | "
           f"{'warp torchΔ':>12} {'drvΔ':>7} | {'parity':>9}")
    print(hdr); print("-" * len(hdr))

    for N in args.grids:
        P = build(N, dev)
        out_bytes = (3 * N ** 3 + (N - 1) * (N - 2) * (N - 2)
                     + (N - 2) * (N - 1) * (N - 2)
                     + (N - 2) * (N - 2) * (N - 1)) * 8 / MIB

        pp, pd, rp = peak_delta(run_python, P, dev)
        npk, nd, rn = peak_delta(run_native, P, dev)
        wpk, wd, rw = peak_delta(run_warp, P, dev)

        par = max(parity(rn, rp), parity(rn, rw))
        print(f"{N:>6} {out_bytes:>9.1f} | "
              f"{pp:>13.1f} {pd:>7.1f} | "
              f"{npk:>13.1f} {nd:>7.1f} | "
              f"{wpk:>12.1f} {wd:>7.1f} | {par:>9.1e}")
        del P, rp, rn, rw
        torch.cuda.empty_cache()

    print("\nReading: 'native'/'warp' torchΔ ≈ 0 → mu/normals stay in registers "
          "(no global buffers).\n'python' torchΔ scales with the grid → the "
          "full-grid mu/normal intermediates kernel mode removes.")


if __name__ == "__main__":
    main()
