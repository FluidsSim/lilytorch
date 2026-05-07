"""Integration parity test for the fused 2-D `apply_bcs_2d` dispatch in
`adv_diff.AdvectionDiffusion.set_BCs`.

Builds a small 2-D `AdvectionDiffusion` instance with a representative
mix of BCs (Dirichlet on west/east for u, Neumann everywhere else),
runs `set_BCs` once via the fused CUDA path (when available) and once
via the legacy slice-copy fallback, and asserts the (u, v) tensors are
bit-identical.

The test always exercises the legacy slice-copy fallback (CPU keeps
that branch active even with the descriptor pack present), and the
fused path when CUDA is available.

Run with::

    python -m lilytorch.src.kernels.test_set_bcs_2d_fused
"""
from __future__ import annotations

import torch

import lilytorch.src.kernels  # noqa: F401  -- registers the namespace
from lilytorch.src.adv_diff import AdvDiffSolver


def _make_solver(device, dtype):
    Nx, Ny = 24, 18
    h = 0.05
    x = torch.arange(Nx, dtype=dtype, device=device) * h
    y = torch.arange(Ny, dtype=dtype, device=device) * h
    return AdvDiffSolver(
        device=device, dt=0.01, x=x, y=y, nu=1e-3,
        BC_type_u=("D", "D", "N", "N"),
        BC_values_u=(1.0, 0.0, 0.0, 0.0),
        BC_type_v=("N", "N", "D", "D"),
        BC_values_v=(0.0, 0.0, 0.0, -2.5),
        method="quick",
    )


def _legacy_set_bcs(adv, u, v):
    """Reproduce the precomputed Python slice-copy path (always-correct
    reference, never dispatches to the fused op)."""
    for comp, dst, src in adv._bc_neumann_ops:
        (u, v)[comp][dst] = (u, v)[comp][src]
    for comp, dst, val in adv._bc_dirichlet_ops:
        (u, v)[comp][dst] = val


def _check(device, dtype, label):
    adv = _make_solver(device, dtype)

    # Same RNG, on CPU then ship to device, so CPU and CUDA inputs match
    # bit-for-bit.
    torch.manual_seed(0)
    Nx_u, Ny_u = adv.nx, adv.ny
    Nx_v, Ny_v = adv.nx, adv.ny
    u_init = torch.randn(Nx_u, Ny_u, dtype=dtype).to(device)
    v_init = torch.randn(Nx_v, Ny_v, dtype=dtype).to(device)

    # 1. Reference: legacy Python loop
    u_ref = u_init.clone()
    v_ref = v_init.clone()
    _legacy_set_bcs(adv, u_ref, v_ref)

    # 2. Public path: set_BCs (will dispatch to apply_bcs_2d on CUDA)
    u_kern = u_init.clone()
    v_kern = v_init.clone()
    adv.set_BCs(u_kern, v_kern)

    du = (u_kern - u_ref).abs().max().item()
    dv = (v_kern - v_ref).abs().max().item()
    print(f"  [{label}]  max |Δu|={du:.3e}  max |Δv|={dv:.3e}")
    assert du == 0.0, f"{label}: u mismatch ({du:.3e})"
    assert dv == 0.0, f"{label}: v mismatch ({dv:.3e})"

    # Sanity: the configured Dirichlet face must have been overwritten.
    assert torch.all(u_kern[0,  :] == 1.0), f"{label}: u west Dirichlet not applied"
    assert torch.all(u_kern[-1, :] == 0.0), f"{label}: u east Dirichlet not applied"
    assert torch.all(v_kern[:, -1] == -2.5), f"{label}: v north Dirichlet not applied"


def main():
    print("== CPU fallback (legacy slice-copy path) ==")
    _check("cpu", torch.float64, "cpu f64")
    _check("cpu", torch.float32, "cpu f32")

    if torch.cuda.is_available():
        print("== CUDA fused apply_bcs_2d path ==")
        _check("cuda", torch.float64, "cuda f64")
        _check("cuda", torch.float32, "cuda f32")
    else:
        print("CUDA not available, skipping fused-path test.")

    print("test_set_bcs_2d_fused: PASSED")


if __name__ == "__main__":
    main()
