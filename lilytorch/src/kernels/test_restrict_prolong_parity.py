"""Parity tests for multigrid transfer CUDA kernels.

Compares the new native kernels (lilytorch_kernels::restrict_residual_*,
restrict_face_*, prolongate_add_*) against the reference PyTorch chain
in lilytorch.src.poisson_mult (_restrict_residual_*, _restrict_face_*,
_prolongate_*).
"""
import sys
import torch
sys.path.insert(0, '/data/andreaferrario/lilytorch')

from lilytorch.src.kernels import _C  # noqa: F401  load .so
from lilytorch.src.kernels import ops as K
from lilytorch.src.poisson_mult import (
    _restrict_residual_2d, _restrict_residual_3d,
    _restrict_face_2d, _restrict_face_3d,
    _prolongate_2d, _prolongate_3d,
)

device = torch.device("cuda:0")
torch.manual_seed(0)

PASS = []
FAIL = []

def check(name, ref, got, atol_f32=1e-5, atol_f64=1e-12):
    atol = atol_f64 if ref.dtype == torch.float64 else atol_f32
    diff = (ref - got).abs().max().item()
    ok = diff <= atol
    (PASS if ok else FAIL).append((name, diff, atol))
    flag = "OK " if ok else "FAIL"
    print(f"  [{flag}] {name:<48s} max|diff|={diff:.3e}  tol={atol:.1e}  dtype={ref.dtype}")


for dtype in (torch.float32, torch.float64):
    print(f"\n=== dtype = {dtype} ===")

    # ----- residual restriction 2-D ----------------------------------
    for Nx, Ny in [(8, 8), (16, 32), (15, 17)]:
        r = torch.randn(Nx, Ny, dtype=dtype, device=device)
        rc_ref = _restrict_residual_2d(r).contiguous()
        rc_got = torch.empty_like(rc_ref)
        K.restrict_residual_2d(r, rc_got)
        check(f"restrict_residual_2d {Nx}x{Ny}", rc_ref, rc_got)

    # ----- residual restriction 3-D ----------------------------------
    for Nx, Ny, Nz in [(8, 8, 8), (16, 8, 12), (7, 9, 11)]:
        r = torch.randn(Nx, Ny, Nz, dtype=dtype, device=device)
        rc_ref = _restrict_residual_3d(r).contiguous()
        rc_got = torch.empty_like(rc_ref)
        K.restrict_residual_3d(r, rc_got)
        check(f"restrict_residual_3d {Nx}x{Ny}x{Nz}", rc_ref, rc_got)

    # ----- face restriction 2-D --------------------------------------
    for Nx, Ny in [(8, 8), (16, 24)]:
        ch = torch.randn(Nx + 1, Ny, dtype=dtype, device=device)
        cv = torch.randn(Nx, Ny + 1, dtype=dtype, device=device)
        ch_ref, cv_ref = _restrict_face_2d(ch, cv)
        ch_ref = ch_ref.contiguous(); cv_ref = cv_ref.contiguous()
        ch_got = torch.empty_like(ch_ref)
        cv_got = torch.empty_like(cv_ref)
        K.restrict_face_2d(ch, ch_got, face_dim=0)
        K.restrict_face_2d(cv, cv_got, face_dim=1)
        check(f"restrict_face_2d ch {Nx}x{Ny}", ch_ref, ch_got)
        check(f"restrict_face_2d cv {Nx}x{Ny}", cv_ref, cv_got)

    # ----- face restriction 3-D --------------------------------------
    for Nx, Ny, Nz in [(8, 8, 8), (12, 8, 16)]:
        ch = torch.randn(Nx + 1, Ny, Nz, dtype=dtype, device=device)
        cv = torch.randn(Nx, Ny + 1, Nz, dtype=dtype, device=device)
        cw = torch.randn(Nx, Ny, Nz + 1, dtype=dtype, device=device)
        ch_ref, cv_ref, cw_ref = _restrict_face_3d(ch, cv, cw)
        ch_ref = ch_ref.contiguous(); cv_ref = cv_ref.contiguous(); cw_ref = cw_ref.contiguous()
        ch_got = torch.empty_like(ch_ref)
        cv_got = torch.empty_like(cv_ref)
        cw_got = torch.empty_like(cw_ref)
        K.restrict_face_3d(ch, ch_got, face_dim=0)
        K.restrict_face_3d(cv, cv_got, face_dim=1)
        K.restrict_face_3d(cw, cw_got, face_dim=2)
        check(f"restrict_face_3d ch {Nx}x{Ny}x{Nz}", ch_ref, ch_got)
        check(f"restrict_face_3d cv {Nx}x{Ny}x{Nz}", cv_ref, cv_got)
        check(f"restrict_face_3d cw {Nx}x{Ny}x{Nz}", cw_ref, cw_got)

    # ----- prolongation + add 2-D ------------------------------------
    # F.interpolate uses float32 even for double; relax f32 tol slightly.
    p2d_atol_f32 = 5e-5
    p2d_atol_f64 = 1e-12
    for Nx_c, Ny_c, Nx_f, Ny_f in [(4, 4, 8, 8), (8, 6, 16, 12)]:
        ec = torch.randn(Nx_c + 2, Ny_c + 2, dtype=dtype, device=device)
        # Ref: F.interpolate on interior, then add to p[interior]
        p_ref = torch.randn(Nx_f + 2, Ny_f + 2, dtype=dtype, device=device)
        p_got = p_ref.clone()
        err = _prolongate_2d(ec, (Nx_f, Ny_f))
        p_ref[1:-1, 1:-1] = p_ref[1:-1, 1:-1] + err
        K.prolongate_add_2d(ec, p_got)
        atol = p2d_atol_f64 if dtype == torch.float64 else p2d_atol_f32
        diff = (p_ref - p_got).abs().max().item()
        ok = diff <= atol
        (PASS if ok else FAIL).append((f"prolongate_add_2d", diff, atol))
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] prolongate_add_2d {Nx_c}x{Ny_c}->{Nx_f}x{Ny_f:<6}  max|diff|={diff:.3e}  tol={atol:.1e}")

    # ----- prolongation + add 3-D ------------------------------------
    for Nx_c, Ny_c, Nz_c, Nx_f, Ny_f, Nz_f in [(4, 4, 4, 8, 8, 8), (6, 4, 8, 12, 8, 16)]:
        ec = torch.randn(Nx_c + 2, Ny_c + 2, Nz_c + 2, dtype=dtype, device=device)
        p_ref = torch.randn(Nx_f + 2, Ny_f + 2, Nz_f + 2, dtype=dtype, device=device)
        p_got = p_ref.clone()
        err = _prolongate_3d(ec, (Nx_f, Ny_f, Nz_f))
        p_ref[1:-1, 1:-1, 1:-1] = p_ref[1:-1, 1:-1, 1:-1] + err
        K.prolongate_add_3d(ec, p_got)
        atol = p2d_atol_f64 if dtype == torch.float64 else p2d_atol_f32
        diff = (p_ref - p_got).abs().max().item()
        ok = diff <= atol
        (PASS if ok else FAIL).append((f"prolongate_add_3d", diff, atol))
        flag = "OK " if ok else "FAIL"
        print(f"  [{flag}] prolongate_add_3d {Nx_c}x{Ny_c}x{Nz_c}->{Nx_f}x{Ny_f}x{Nz_f:<6}  max|diff|={diff:.3e}  tol={atol:.1e}")


print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for n, d, t in FAIL:
        print(f"  FAILED: {n} diff={d:.3e} tol={t:.1e}")
    sys.exit(1)
