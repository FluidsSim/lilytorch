"""Speed + peak-memory benchmarks for the 2-D Warp port round (P1–P4).

For each ported kernel: wall-clock ratio vs the native CUDA op under CUDA-graph
capture (the production-relevant timing, HANDOFF lesson 3), plus — for the
memory-critical kernels (Kernel B, cvof) — peak GPU memory ABOVE the
pre-allocated outputs, measured both by torch (`max_memory_allocated`) and the
driver (`mem_get_info`, which also sees Warp's separate allocator, lesson 12),
against a PyTorch "python path" reference that materialises the full-grid
intermediates the fused kernel keeps in registers.

    python -m lilytorch.warp_poc.bench_2d --grids 256 512 1024
"""
from __future__ import annotations

import argparse, math, time
import torch, warp as wp

try:
    import lilytorch.src.kernels  # noqa: F401
    from lilytorch.src.kernels.ops import (
        streaming_sdf_stag_2d_multi as nat_kA,
        bdim_coeff_2d as nat_kB,
        rbgs_sweep_2d as nat_rbgs,
        mg_residual_2d as nat_resid,
        poisson_solve_multigrid_2d as nat_mg,
    )
    _NATIVE = True
except Exception as e:
    print(f"[warn] native unavailable: {e}")
    _NATIVE = False

from lilytorch.warp_poc.scene_2d import make_synthetic_scene_2d
from lilytorch.warp_poc.warp_kernels_2d import WarpStreamingSDF2D
from lilytorch.warp_poc.warp_bdim_2d import bdim_coeff_2d_warp
from lilytorch.warp_poc.warp_poisson_2d import WarpRBGS2D
from lilytorch.warp_poc.warp_cvof import cvof_sweep_warp
from lilytorch.warp_poc.warp_multigrid_2d import mg_residual_2d_warp, WarpVCycle2D

RHO, DT = 1000.0, 1e-3


def time_ms(fn, warmup=5, reps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps * 1000.0


def _ratio(t, t_nat):
    return f"{t/t_nat:.2f}×" if not math.isnan(t_nat) and t_nat > 0 else "n/a"


def _ms(v):
    return f"{v:.3f}ms" if not math.isnan(v) else "  n/a"


# ── peak-memory helper ───────────────────────────────────────────────────────

def peak_above_baseline(fn, dev="cuda:0"):
    """(torch_peak_MiB, driver_used_MiB) the op transiently allocates beyond
    whatever is live at entry.  Run AFTER a warmup so JIT/graph pools exist."""
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    free0, _ = torch.cuda.mem_get_info()
    fn()
    torch.cuda.synchronize()
    torch_peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    free1, _ = torch.cuda.mem_get_info()
    driver = max(0.0, (free0 - free1) / 2**20)
    return torch_peak, driver


# ═══ Kernel A 2-D ════════════════════════════════════════════════════════════

def bench_kernelA(grids, B, device):
    print(f"\n  Kernel A 2-D (streaming_sdf_stag_2d_multi), float32, B={B}")
    for Ng in grids:
        sc = make_synthetic_scene_2d(Ng, Ng, B, device=device)
        Ngx, Ngy = sc["Ngx"], sc["Ngy"]
        di0, dj0, dAi, dAj = sc["dirty_bounds"]

        def call_nat():
            sc["sdf_cc"].fill_(1e4); sc["sdf_u"].fill_(1e4); sc["sdf_v"].fill_(1e4)
            sc["body_u"].zero_(); sc["body_v"].zero_()
            nat_kA(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
                   sc["kin"], sc["aabb_lo"], sc["aabb_dim"], sc["gx"], sc["gy"],
                   float(sc["h"]), int(sc["max_vol"]),
                   sc["sdf_cc"].view(Ngx, Ngy), sc["sdf_u"].view(Ngx, Ngy),
                   sc["sdf_v"].view(Ngx, Ngy), sc["body_u"].view(Ngx, Ngy),
                   sc["body_v"].view(Ngx, Ngy), sc["key_cc"], sc["key_u"], sc["key_v"],
                   0, di0, dj0, dAi, dAj, sc["num_u"], sc["num_v"], sc["den_u"],
                   sc["den_v"], 0.0)
        t_nat = time_ms(call_nat) if _NATIVE else float("nan")

        w = WarpStreamingSDF2D(Ngx, Ngy, device=device)
        w.setup(sc["F_flat"], sc["F_offsets"], sc["body_shapes"], sc["body_meta"],
                sc["gx"], sc["gy"], float(sc["h"]), int(sc["max_vol"]))
        w.update_kinematics(sc["kin"], sc["aabb_lo"], sc["aabb_dim"])
        N = Ngx * Ngy
        out = [wp.full(N, 1e4, dtype=wp.float32, device=device) for _ in range(3)]
        bvel = [wp.zeros(N, dtype=wp.float32, device=device) for _ in range(2)]
        args = (out[0], out[1], out[2], bvel[0], bvel[1])
        w.run_fanned_eager(*args)
        with wp.ScopedCapture(device=device) as cap:
            w.run_fanned_eager(*args)
        g = cap.graph
        t_g = time_ms(lambda: wp.capture_launch(g))
        print(f"   {Ng:>5}²  native={_ms(t_nat)}  warp-graph={_ms(t_g)} ({_ratio(t_g, t_nat)})")
        del sc, out, bvel; torch.cuda.empty_cache()


# ═══ Kernel B 2-D (speed + memory vs python path) ════════════════════════════

def _kB_fields(N, dev):
    g = torch.Generator(device=dev).manual_seed(0)
    h = 1.0 / N
    xs = (torch.arange(N, device=dev) - N / 2.0) * h
    X, Y = torch.meshgrid(xs, xs, indexing="ij")
    disc = lambda o: (torch.sqrt((X - o) ** 2 + Y ** 2) - 0.3).double()
    rnd = lambda: torch.randn(N, N, dtype=torch.float64, device=dev, generator=g)
    return dict(h=h, eps=2 * h, N=N, su=disc(0.5 * h), sv=disc(0.0),
                bu=rnd(), bv=rnd(), up=rnd(), vp=rnd())


def _kB_py(P, u0, v0, ch, cv):
    """python path: materialise mu0/mu1/normals as FULL-GRID tensors (what the
    fused kernel keeps in registers) for one face, to expose the memory cost."""
    eps, h = P["eps"], P["h"]
    for sdf, body, pp, out, c in ((P["su"], P["bu"], P["up"], u0, ch),
                                  (P["sv"], P["bv"], P["vp"], v0, cv)):
        deps = (sdf / eps).clamp(-1, 1)
        mu0 = 0.5 * (1 + deps + torch.sin(math.pi * deps) / math.pi)
        mu1 = eps * (0.25 - 0.25 * deps**2
                     - (torch.sin(math.pi*deps)*deps + (1+torch.cos(math.pi*deps))/math.pi)/(2*math.pi))
        gx = torch.gradient(sdf, dim=0)[0] / 1.0
        gy = torch.gradient(sdf, dim=1)[0] / 1.0
        nn = torch.sqrt(gx*gx + gy*gy).clamp_min(1e-30)
        nx, ny = gx/nn, gy/nn
        d = pp - body
        ddx = torch.gradient(d, dim=0)[0]; ddy = torch.gradient(d, dim=1)[0]
        nd = nx*ddx + ny*ddy
        out.copy_(mu0*(pp-body) + body + mu1*nd)
        c.copy_(DT * mu0 / RHO)


def bench_kernelB(grids, device):
    print(f"\n  Kernel B 2-D (bdim_coeff_2d), float64")
    for N in grids:
        P = _kB_fields(N, device)
        base = DT / RHO
        mk = lambda: (P["up"].clone(), P["vp"].clone(),
                      torch.full((N, N), base, dtype=torch.float64, device=device),
                      torch.full((N, N), base, dtype=torch.float64, device=device))

        outs = mk()
        call_nat = lambda: nat_kB(P["up"], P["vp"], P["su"], P["sv"], P["bu"], P["bv"],
                                  *outs, P["eps"], RHO, DT, P["h"], 0, 0, N, N, 1)
        t_nat = time_ms(call_nat) if _NATIVE else float("nan")

        call_w = lambda: bdim_coeff_2d_warp(P["up"], P["vp"], P["su"], P["sv"],
                                            P["bu"], P["bv"], *outs, P["eps"], RHO, DT,
                                            P["h"], 0, 0, N, N, 1)
        call_w()
        with wp.ScopedCapture(device=device) as cap:
            call_w()
        g = cap.graph
        t_g = time_ms(lambda: wp.capture_launch(g))

        # memory: warp vs native vs python path (above the preallocated outs)
        m_nat = peak_above_baseline(call_nat) if _NATIVE else (float('nan'),)*2
        m_w   = peak_above_baseline(lambda: wp.capture_launch(g))
        outs_py = mk()
        m_py = peak_above_baseline(lambda: _kB_py(P, *outs_py))
        print(f"   {N:>5}²  native={_ms(t_nat)}  warp-graph={_ms(t_g)} ({_ratio(t_g, t_nat)})")
        print(f"          peak MiB above outputs — native(torch/drv)="
              f"{m_nat[0]:.0f}/{m_nat[1]:.0f}  warp={m_w[0]:.0f}/{m_w[1]:.0f}  "
              f"python={m_py[0]:.0f}/{m_py[1]:.0f}")
        del P, outs, outs_py; torch.cuda.empty_cache()


# ═══ smoother 2-D (N sweeps) ═════════════════════════════════════════════════

def bench_smoother(grids, device):
    print(f"\n  RBGS 2-D smoother, float32  (native=tiled SHARED-MEM multi-sweep FUSION;")
    print(f"  warp=thread-per-cell global, color-compacted flat launch — no `wp.tile` in 1.14)")
    print(f"  Reported at nsmoothing=2 (the real V-cycle nu) AND 10 (the fusion-max microbench).")
    for N in grids:
        torch.manual_seed(1)
        coeffs = [(0.5 + torch.rand(N, N, dtype=torch.float32, device=device)) for _ in range(4)]
        f = torch.randn(N, N, dtype=torch.float32, device=device)
        p0 = torch.zeros(N + 2, N + 2, dtype=torch.float32, device=device)
        p0[1:-1, 1:-1] = torch.randn(N, N, device=device)
        cells = []
        for ns in (2, 10):
            pn = p0.clone()
            t_nat = time_ms(lambda: nat_rbgs(pn, f, *coeffs, 1e-30, ns)) if _NATIVE else float("nan")
            # native fusion advantage (vs native run unfused, 1 sweep × ns) — the
            # part Warp 1.14 structurally can't match (no shared-mem tile API).
            t_nat_unfused = (time_ms(lambda: [nat_rbgs(pn, f, *coeffs, 1e-30, 1) for _ in range(ns)])
                             if _NATIVE else float("nan"))
            s = WarpRBGS2D(N, N, device=device)
            s.setup(p0.clone(), f, coeffs)
            s.capture_sweeps(ns)
            t_g = time_ms(s.run_graph)
            cells.append((ns, t_nat, t_nat_unfused, t_g))
        for ns, t_nat, t_nat_unf, t_g in cells:
            fus = f"{t_nat_unf/t_nat:.1f}×fusion" if not math.isnan(t_nat) else ""
            print(f"   {N:>5}² ns={ns:>2}  native-fused={_ms(t_nat)}  native-unfused={_ms(t_nat_unf)} "
                  f"({fus})  warp={_ms(t_g)} (vs-fused {_ratio(t_g, t_nat)}, "
                  f"vs-unfused {_ratio(t_g, t_nat_unf)})")
        torch.cuda.empty_cache()


# ═══ cvof_sweep (speed + memory vs python path) ══════════════════════════════

def _cvof_py(a, u, cfl, fd, out):
    """python path: the ~8 full-grid temporaries cvof_sweep fuses."""
    def shift(t, n):
        idx = (torch.arange(t.shape[fd], device=t.device) - n).clamp(0, t.shape[fd]-1)
        return t.index_select(fd, idx)
    am1, am2, ap1 = shift(a, 1), shift(a, 2), shift(a, -1)
    C = u * cfl
    def vl(db, df):
        den = torch.where(db+df == 0, torch.ones_like(db), db+df)
        s = 2*db*df/den
        return torch.where(db*df > 0, s, torch.zeros_like(s))
    fpos = am1 + 0.5*(1-C)*vl(am1-am2, a-am1)
    fneg = a - 0.5*(1+C)*vl(a-am1, ap1-a)
    F = u * torch.where(C >= 0, fpos, fneg)
    sl = [slice(None)]*a.dim()
    sli = sl.copy(); sli[fd] = slice(1, -1)
    slL = sl.copy(); slL[fd] = slice(1, -1)
    slR = sl.copy(); slR[fd] = slice(2, None)
    out[tuple(sli)] = a[tuple(sli)] + cfl*((F[tuple(slL)]-F[tuple(slR)])
                                           + a[tuple(sli)]*(u[tuple(slR)]-u[tuple(slL)]))


def _cvof_native_3d_throughput(device):
    """FAIR efficiency reference: native cvof's per-cell throughput when its
    block map is well-configured (3-D).  The native 2-D op idles ~97% of each
    32-wide block (transverse size 1), so 'Warp vs native-2D' is NOT like-for-
    like; Warp vs this number is."""
    if not _NATIVE:
        return float("nan")
    N = 101
    a = torch.rand(N, N, N, dtype=torch.float64, device=device)
    u = torch.randn(N, N, N, dtype=torch.float64, device=device)
    o = a.clone()
    t = time_ms(lambda: torch.ops.lilytorch_kernels.cvof_sweep(a, u, 0.3, 0, o))
    return N**3 / 1e6 / t  # Mcells/ms


def bench_cvof(grids, device):
    print(f"\n  cvof_sweep 2-D (two-phase W&Y), float64, face_dim=0")
    nat3d = _cvof_native_3d_throughput(device)
    print(f"  FAIR reference — native cvof at proper occupancy (3-D): {nat3d:.1f} Mcells/ms")
    for N in grids:
        torch.manual_seed(3)
        a = torch.rand(N, N, dtype=torch.float64, device=device)
        u = torch.randn(N, N, dtype=torch.float64, device=device)
        cfl, fd = 0.3, 0
        out = a.clone()
        call_nat = lambda: torch.ops.lilytorch_kernels.cvof_sweep(a, u, cfl, fd, out)
        t_nat = time_ms(call_nat) if _NATIVE else float("nan")
        call_w = lambda: cvof_sweep_warp(a, u, cfl, fd, out)
        call_w()
        with wp.ScopedCapture(device=device) as cap:
            call_w()
        g = cap.graph
        t_g = time_ms(lambda: wp.capture_launch(g))
        m_nat = peak_above_baseline(call_nat) if _NATIVE else (float('nan'),)*2
        m_w = peak_above_baseline(lambda: wp.capture_launch(g))
        out_py = a.clone()
        m_py = peak_above_baseline(lambda: _cvof_py(a, u, cfl, fd, out_py))
        tput_w = N*N / 1e6 / t_g
        tput_n = N*N / 1e6 / t_nat if not math.isnan(t_nat) else float('nan')
        print(f"   {N:>5}²  warp={_ms(t_g)} ({tput_w:.0f} Mcells/ms ≈ native-3D {nat3d:.0f})  "
              f"native-2D={_ms(t_nat)} ({tput_n:.1f} Mcells/ms, ~{nat3d/max(tput_n,1e-9):.0f}× under-occupied)")
        print(f"          peak MiB above output — native={m_nat[0]:.0f}/{m_nat[1]:.0f}  "
              f"warp={m_w[0]:.0f}/{m_w[1]:.0f}  python={m_py[0]:.0f}/{m_py[1]:.0f}")
        torch.cuda.empty_cache()


# ═══ V-cycle vs native multigrid driver ══════════════════════════════════════

def bench_vcycle(grids, device):
    print(f"\n  Poisson V-cycle: WarpVCycle2D vs native multigrid driver, float64")
    for N in grids:
        torch.manual_seed(0)
        f = torch.randn(N, N, dtype=torch.float64, device=device); f -= f.mean()
        vc = WarpVCycle2D(N, device=device)
        vc.set_rhs(f)
        vc.capture()
        t_g = time_ms(vc.run_graph, warmup=3, reps=20)
        # native multigrid solve (full solve, not one cycle) for a rough scale ref
        if _NATIVE:
            base = DT / RHO
            ch = torch.ones(N+1, N, dtype=torch.float64, device=device); ch[0]=0; ch[N]=0
            cv = torch.ones(N, N+1, dtype=torch.float64, device=device); cv[:,0]=0; cv[:,N]=0
            coeffs = [ch[1:].contiguous(), ch[:-1].contiguous(),
                      cv[:,1:].contiguous(), cv[:,:-1].contiguous()]
            p = torch.zeros(N+2, N+2, dtype=torch.float64, device=device)
            try:
                t_nat = time_ms(lambda: nat_mg(p.clone(), f, *coeffs, 1e-10, "rbgs"),
                                warmup=2, reps=10)
            except Exception:
                t_nat = float("nan")
        else:
            t_nat = float("nan")
        print(f"   {N:>5}²  warp-1cycle-graph={_ms(t_g)}   "
              f"native-full-solve={_ms(t_nat)} (full multigrid solve, not 1 cycle)")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    print(f"\n{'═'*74}\n  2-D Warp port — speed (ratio vs native, CUDA-graph) + peak memory")
    print(f"  device={a.device}   (<1.00× = Warp faster)\n{'═'*74}")
    bench_kernelA(a.grids, a.B, a.device)
    bench_kernelB(a.grids, a.device)
    bench_smoother(a.grids, a.device)
    bench_cvof(a.grids, a.device)
    bench_vcycle(a.grids, a.device)
    print(f"{'═'*74}\n")
