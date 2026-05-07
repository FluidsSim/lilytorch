"""Self-contained correctness test for ``apply_bcs_2d``.

2-D analogue of the 3-D ``apply_bcs_3d`` op. Builds a small (u, v)
field, fires a representative descriptor mix (4 Neumann faces × 2
components + a pair of Dirichlet writes) through the kernel, and
compares against a pure-Python reference that does the same writes
slice-by-slice.

The reference path is the un-fused PyTorch slice-copy that the kernel
replaces in the solver, so a successful match also confirms the BC
descriptor encoding is the one the solver expects to use.

Runs on CPU. When CUDA is available, the same descriptor set is run on
CUDA and cross-checked against the CPU output.

Run with::

    python -m lilytorch.src.kernels.test_apply_bcs_2d_self
"""
from __future__ import annotations

import torch

import lilytorch.src.kernels  # noqa: F401  -- registers the namespace
from lilytorch.src.kernels import apply_bcs_2d


def _reference_apply_bcs_2d(u, v, neu_desc, dir_desc, dir_val):
    """Slice-copy reference matching the kernel's behaviour."""
    bases = [u, v]
    N_neu = neu_desc.shape[0]
    for op in range(N_neu):
        comp, axis, side = [int(x) for x in neu_desc[op].tolist()]
        sz = bases[comp].shape[axis]
        if side == 0:
            dst, src = 0, 1
        else:
            dst, src = sz - 1, sz - 2
        if axis == 0:
            bases[comp][dst, :] = bases[comp][src, :]
        else:
            bases[comp][:, dst] = bases[comp][:, src]
    N_dir = dir_desc.shape[0]
    for d in range(N_dir):
        comp, axis, offset = [int(x) for x in dir_desc[d].tolist()]
        sz = bases[comp].shape[axis]
        dst = offset if offset >= 0 else (sz + offset)
        value = float(dir_val[d])
        if axis == 0:
            bases[comp][dst, :] = value
        else:
            bases[comp][:, dst] = value


def _make_scene(*, dtype, device):
    # Always generate on CPU so that both the CPU and CUDA tests start from
    # bit-identical tensors (CPU and CUDA RNGs produce different values even
    # with the same seed, breaking the cross-device parity check).
    torch.manual_seed(0)
    Nx_u, Ny_u = 32, 24
    Nx_v, Ny_v = 30, 26
    u = torch.randn(Nx_u, Ny_u, dtype=dtype).to(device)
    v = torch.randn(Nx_v, Ny_v, dtype=dtype).to(device)
    shapes = torch.tensor([
        [Nx_u, Ny_u],
        [Nx_v, Ny_v],
    ], dtype=torch.int64, device=device)
    return u, v, shapes


def _make_descriptors(device):
    # Neumann: copy ghost row from neighbour on every external face,
    # for both u and v. (comp, axis, side)
    neu = []
    for comp in (0, 1):
        for axis in (0, 1):
            for side in (0, 1):
                neu.append([comp, axis, side])
    neu_desc = torch.tensor(neu, dtype=torch.int32, device=device)

    # Dirichlet: u(x=0,:) = 1.0  ;  v(:, y=-1) = -2.5 (offset -1).
    dir_desc = torch.tensor([
        [0, 0, 0],
        [1, 1, -1],
    ], dtype=torch.int32, device=device)
    dir_val = torch.tensor([1.0, -2.5], dtype=torch.float64, device=device)
    return neu_desc, dir_desc, dir_val


def _max_line_dim(shapes_t):
    return int(shapes_t.max().item())


def _run_and_check(*, dtype, device, label):
    u, v, shapes = _make_scene(dtype=dtype, device=device)
    neu_desc, dir_desc, dir_val_f64 = _make_descriptors(device)
    dir_val = dir_val_f64.to(dtype)

    u_kernel = u.clone()
    v_kernel = v.clone()
    apply_bcs_2d(
        u_kernel, v_kernel, shapes,
        neu_desc, dir_desc, dir_val,
        _max_line_dim(shapes),
    )

    u_ref = u.clone()
    v_ref = v.clone()
    _reference_apply_bcs_2d(u_ref, v_ref, neu_desc, dir_desc, dir_val)

    du = (u_kernel - u_ref).abs().max().item()
    dv = (v_kernel - v_ref).abs().max().item()
    print(f"  [{label}]  max |Δu|={du:.3e}  max |Δv|={dv:.3e}")
    assert du == 0.0, f"{label}: u mismatch ({du:.3e})"
    assert dv == 0.0, f"{label}: v mismatch ({dv:.3e})"

    # Spot-check: dirichlet rows must equal the requested value.
    assert torch.all(u_kernel[0, :] == dir_val[0]), \
        f"{label}: u Dirichlet row not applied"
    assert torch.all(v_kernel[:, -1] == dir_val[1]), \
        f"{label}: v Dirichlet column not applied"

    # Spot-check: Neumann ghost cells must equal their interior neighbour.
    assert torch.all(u_kernel[-1, :] == u_kernel[-2, :]), \
        f"{label}: u Neumann hi-x ghost not copied"
    assert torch.all(v_kernel[0, :]  == v_kernel[1, :]), \
        f"{label}: v Neumann lo-x ghost not copied"

    return u_kernel, v_kernel


def main():
    print("== CPU apply_bcs_2d vs reference ==")
    u_cpu, v_cpu = _run_and_check(dtype=torch.float64, device="cpu", label="cpu f64")
    _run_and_check(dtype=torch.float32, device="cpu", label="cpu f32")

    if torch.cuda.is_available():
        print("== CUDA apply_bcs_2d vs reference ==")
        u_cuda, v_cuda = _run_and_check(dtype=torch.float64,
                                         device="cuda", label="cuda f64")
        _run_and_check(dtype=torch.float32, device="cuda", label="cuda f32")

        # Cross-device parity check (deterministic seed).
        du = (u_cpu - u_cuda.cpu()).abs().max().item()
        dv = (v_cpu - v_cuda.cpu()).abs().max().item()
        print(f"  CPU vs CUDA: max |Δu|={du:.3e}  max |Δv|={dv:.3e}")
        assert du == 0.0, f"CPU/CUDA disagree on u (Δ={du:.3e})"
        assert dv == 0.0, f"CPU/CUDA disagree on v (Δ={dv:.3e})"
    else:
        print("CUDA not available, skipping CUDA cross-check.")

    print("test_apply_bcs_2d_self: PASSED")


if __name__ == "__main__":
    main()
