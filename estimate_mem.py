"""Memory estimation pipeline for the 3-D BDIM fluid solver.

Replaces the static 25-grid hand-wave with a component-wise model that
takes into account:

* grid size (``nx``, ``ny``, ``nz``) and dtype (``fp32`` / ``fp64``)
* Poisson solver mode (``"neumann"`` or ``"free"``) — free-space mode
  uses a zero-padded complex Green's-function FFT that is ~3 GB extra
  on the reference grid.
* body count (``nbodies``) and which ``CompositeBody`` flavour is in
  use (``"analytical"`` = legacy ``CompositeBodyAnalytical`` with per-
  child caches, or ``"multianimat"`` = sparse AABB storage).
* whether ``compile_forces=True`` (which requires a dense
  ``(B, Nx, Ny, Nz)`` SDF stack unless the sparse path is taken).
* ``force_delta_order`` (order 2 allocates an extra |∇SDF| tensor).
* ``sparse_sdf`` / ``aabb_fill_fraction`` for the ``_sdf_sparse`` path.

Run ``python estimate_mem.py --help`` for CLI options.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Iterable, List, Literal, Optional

# ``torch`` is imported lazily so the script still works on a machine
# without PyTorch (e.g. for a pure memory estimate on a laptop).
try:  # pragma: no cover - environment probe
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    _HAS_TORCH = False


# ---------------------------------------------------------------------------
# dtype helpers (works with or without torch)
# ---------------------------------------------------------------------------

_DTYPE_NAME_TO_BYTES = {
    "float64": 8, "fp64": 8, "double": 8,
    "float32": 4, "fp32": 4, "float": 4,
    "float16": 2, "fp16": 2, "half": 2,
    "bfloat16": 2, "bf16": 2,
}


def _dtype_bytes(dtype) -> int:
    """Return itemsize in bytes for a torch.dtype *or* string alias."""
    if dtype is None:
        return 8
    if isinstance(dtype, str):
        key = dtype.lower().replace("torch.", "")
        if key not in _DTYPE_NAME_TO_BYTES:
            raise ValueError(f"Unknown dtype string: {dtype!r}")
        return _DTYPE_NAME_TO_BYTES[key]
    if _HAS_TORCH and isinstance(dtype, torch.dtype):
        return torch.empty((), dtype=dtype).element_size()
    raise TypeError(f"Unsupported dtype argument: {type(dtype).__name__}")


def _dtype_label(dtype) -> str:
    if isinstance(dtype, str):
        return dtype
    if _HAS_TORCH and isinstance(dtype, torch.dtype):
        return str(dtype).replace("torch.", "")
    return str(dtype)


# ---------------------------------------------------------------------------
# Component model
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class Component:
    """One memory-consuming tensor (or family of tensors).

    ``count`` counts how many grid-sized tensors of ``elem_bytes`` each
    we hold.  ``per_body`` is True when the count scales with ``nbodies``.
    """
    name: str
    count: float                   # number of N-sized tensors (can be fractional)
    elem_bytes: int                # bytes per cell (4 for fp32, 8 for fp64, 16 for cplx128)
    cells: int                     # number of cells in one tensor (full grid or aabb)
    per_body: bool = False
    note: str = ""

    @property
    def total_bytes(self) -> float:
        return self.count * self.elem_bytes * self.cells

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 1024 ** 2


def _fmt_mb(mb: float) -> str:
    if mb >= 1024:
        return f"{mb / 1024:8.2f} GB"
    return f"{mb:8.2f} MB"


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _grid_components(
    N: int,
    cell_bytes_fluid: int,
    cell_bytes_geom: int,
    bc_type: str,
) -> List[Component]:
    """Grid-scaling (O(N)) components: fluid state, MAC grids, Poisson.

    Estimates correspond to what lilytorch's ``FluidSolver`` + ``poisson_fft``
    actually allocate on the reference 3-D grid.  Values come from reading
    ``lilytorch/src/solver.py`` and ``lilytorch/src/poisson_fft.py``.
    """
    c: List[Component] = []

    # --- fluid state (two time-levels for u/v/w + pressure + divergence + phi)
    # u0, u1, v0, v1, w0, w1, p, div, phi  (+ a handful of rhs/work buffers
    # that live across step boundaries).  Fluid state stays in the chosen
    # solver dtype (typically fp64 for stiff incompressible solvers).
    c.append(Component(
        name="fluid state (u0,u1,v0,v1,w0,w1,p,div,phi)",
        count=9, elem_bytes=cell_bytes_fluid, cells=N,
        note="solver.py step buffers (two-stage time integration)",
    ))
    c.append(Component(
        name="fluid rhs / work buffers",
        count=4, elem_bytes=cell_bytes_fluid, cells=N,
        note="BDIM residuals, rhs for Poisson, μ₁-smoothed fields",
    ))

    # --- MAC staggered coordinate grids (X, Y, Z, Xu, Yu, Xv, Yv, Xw, Yw, Zw, ...)
    # Shared via _StaggeredGrids so allocated once per (x,y,z); count ~10.
    c.append(Component(
        name="MAC staggered coord grids (X,Y,Z,Xu..Zw)",
        count=10, elem_bytes=cell_bytes_geom, cells=N,
        note="_StaggeredGrids in body.py; shared across bodies",
    ))

    # --- BDIM smoothed-mu / normal fields (cell-centre and 3 faces)
    c.append(Component(
        name="BDIM μ₀/μ₁ fields + normals",
        count=6, elem_bytes=cell_bytes_geom, cells=N,
        note="mu0, mu1 on staggered faces + 3 normal components",
    ))

    # --- Poisson solver
    if bc_type == "free":
        # Zero-padded work buffer U of shape (2nx, 2ny, 2nz) — 8× N in the
        # real dtype (fluid), plus complex Green's-function FFT of the same
        # padded shape in complex-equivalent (2 × cell_bytes_fluid).
        c.append(Component(
            name="Poisson free-space zero-padded U",
            count=8, elem_bytes=cell_bytes_fluid, cells=N,
            note="shape (2nx,2ny,2nz) — poisson_fft.py:103",
        ))
        c.append(Component(
            name="Poisson Green's-function FFT (complex)",
            count=8, elem_bytes=2 * cell_bytes_fluid, cells=N,
            note="shape (2nx,2ny,2nz) complex — poisson_fft.py:290",
        ))
    else:  # neumann / dirichlet via DCT
        c.append(Component(
            name="Poisson Neumann eigenvalues / twiddles",
            count=2, elem_bytes=cell_bytes_fluid, cells=N,
            note="eig (real) + complex twiddle ≈ 1 real + 1 complex grid",
        ))
        c.append(Component(
            name="Poisson Neumann complex work (Z, V)",
            count=2, elem_bytes=2 * cell_bytes_fluid, cells=N,
            note="per-axis DCT: X.to(cdtype) * W_mod, fft result",
        ))

    return c


def _body_components(
    N: int,
    cell_bytes_geom: int,
    B: int,
    composite_kind: str,
    compile_forces: bool,
    force_delta_order: int,
    sparse_sdf: bool,
    aabb_fill_fraction: float,
    legacy_analytical_cache: bool = False,
    batched_forces_max_dense_bytes: int = 1 * 1024**3,
) -> List[Component]:
    """Body-scaling components (O(B · something))."""

    c: List[Component] = []
    aabb_cells = max(1, int(N * aabb_fill_fraction))

    if composite_kind == "analytical":
        # CompositeBodyAnalytical normally caches 7 full-grid tensors per
        # CHILD (sdf_val, sdf_u/v/w, body_u/v/w) between iterations.
        # Step 2 of the implementation plan releases those tensors as
        # soon as they have been merged into the composite's streaming
        # union, bringing the per-body cost to ~0.  If you want to
        # estimate the pre-fix (legacy) cost, pass
        # ``legacy_analytical_cache=True``.
        if legacy_analytical_cache:
            c.append(Component(
                name="per-child analytical SDF/vel caches (legacy)",
                count=7 * B, elem_bytes=cell_bytes_geom, cells=N,
                per_body=True,
                note="body.sdf_val, sdf_u/v/w, body_u/v/w kept on each "
                     "child BEFORE step-2 streaming release",
            ))
        else:
            c.append(Component(
                name="per-child analytical SDF/vel caches",
                count=0, elem_bytes=cell_bytes_geom, cells=N,
                per_body=True,
                note="released per iteration by step-2 streaming union",
            ))
    elif composite_kind == "multianimat":
        if sparse_sdf:
            # _sdf_sparse keeps only per-body AABB sub-blocks.  Each body
            # stores one SDF sub-grid of size ~aabb_fill_fraction × N.
            c.append(Component(
                name="_sdf_sparse AABB sub-blocks",
                count=B, elem_bytes=cell_bytes_geom, cells=aabb_cells,
                per_body=True,
                note=f"one per body, ~{aabb_fill_fraction*100:.1f}% of grid",
            ))
        else:
            # Legacy dense path: sdf_vals of shape (B, *gs)
            c.append(Component(
                name="composite.sdf_vals (legacy dense)",
                count=B, elem_bytes=cell_bytes_geom, cells=N,
                per_body=True,
                note="dense (B, Nx, Ny, Nz) — body.py:2169",
            ))
    else:
        raise ValueError(
            f"composite_kind must be 'analytical' or 'multianimat', "
            f"got {composite_kind!r}"
        )

    # --- batched-forces dense stacks (peak, not steady-state)
    if compile_forces:
        # Step 3 fix: when the sparse path is available AND the dense
        # stack would exceed ``batched_forces_max_dense_bytes`` (1 GiB
        # default), the solver auto-falls back to the per-body sparse
        # loop, so no dense stack is allocated.
        dense_stack_bytes = B * N * cell_bytes_geom
        sparse_available = sparse_sdf and composite_kind == "multianimat"
        fallback = sparse_available and dense_stack_bytes > batched_forces_max_dense_bytes
        if sparse_available and not fallback:
            c.append(Component(
                name="batched-forces sdf_all (sparse avoided)",
                count=0, elem_bytes=cell_bytes_geom, cells=N,
                per_body=True,
                note="dense stack skipped — sparse path active "
                     "(under threshold)",
            ))
        elif fallback:
            c.append(Component(
                name="batched-forces sdf_all (step-3 sparse fallback)",
                count=0, elem_bytes=cell_bytes_geom, cells=N,
                per_body=True,
                note=f"dense stack would be "
                     f"{dense_stack_bytes / 1024**3:.2f} GB > "
                     f"{batched_forces_max_dense_bytes / 1024**3:.2f} GB "
                     "threshold — auto-falls back to per-body sparse loop",
            ))
        else:
            c.append(Component(
                name="batched-forces sdf_all stack",
                count=B, elem_bytes=cell_bytes_geom, cells=N,
                per_body=True,
                note="(B, Nx, Ny, Nz) — solver.py:1408",
            ))
        if force_delta_order == 2 and not sparse_available:
            c.append(Component(
                name="batched-forces |∇SDF| stack",
                count=B, elem_bytes=cell_bytes_geom, cells=N,
                per_body=True,
                note="(B, Nx, Ny, Nz) Towers 2nd-order — solver.py:1438",
            ))

    return c


def _transient_factor(bc_type: str) -> float:
    """Multiplier that accounts for transient peaks (autograd tape,
    Poisson forward+inverse FFTs, ``_bdim`` compiled-graph replay)."""
    if bc_type == "free":
        return 1.6  # FFT buffers are already counted explicitly; less slack
    return 2.3      # DCT path allocates more transient complex tensors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def estimate_memory(
    nx: int = 1024 + 2,
    ny: int = 128 + 2,
    nz: int = 128 + 2,
    nbodies: int = 1,
    *,
    dtype=None,
    sdf_dtype=None,
    bc_type: Literal["neumann", "free"] = "neumann",
    compile_forces: bool = True,
    force_delta_order: int = 1,
    composite_kind: Literal["analytical", "multianimat"] = "multianimat",
    sparse_sdf: bool = True,
    aabb_fill_fraction: float = 0.05,
    legacy_analytical_cache: bool = False,
    batched_forces_max_dense_bytes: int = 1 * 1024**3,
    gpu_budget_gb: Optional[float] = None,
    verbose: bool = True,
):
    """Estimate peak VRAM for a given 3-D BDIM solver configuration.

    Parameters
    ----------
    nx, ny, nz : int
        Grid dimensions INCLUDING the +2 ghost layers that
        ``FluidSolver`` appends.  Defaults match the reference 3-D
        zebrafish simulation.
    nbodies : int
        Number of immersed bodies ``B``.
    dtype : torch.dtype or str, default float64
        Dtype of the *fluid* state (u, v, w, p, …).
    sdf_dtype : torch.dtype or str, optional
        Dtype of *geometry* fields (SDF, MAC grids, body velocities).
        Defaults to ``dtype``.  Setting this to ``float32`` while
        ``dtype=float64`` halves the body-scaling cost (step 5).
    bc_type : {"neumann", "free"}
        Poisson boundary condition.  ``"free"`` triggers the zero-padded
        complex-FFT Green's-function solver (substantially larger).
    compile_forces : bool
        Whether ``FluidSolver._compile_forces`` is True.  The batched
        CUDA-graph path reconstructs a ``(B, Nx, Ny, Nz)`` dense stack
        unless the ``multianimat`` + ``sparse_sdf`` route is used.
    force_delta_order : {1, 2}
        Towers delta order; 2 allocates an additional |∇SDF| stack.
    composite_kind : {"analytical", "multianimat"}
        Which ``CompositeBody`` class is in use.
    sparse_sdf : bool
        Use the ``_sdf_sparse`` AABB storage (``multianimat`` only).
    aabb_fill_fraction : float
        Fraction of the full grid covered by one body's AABB.  Typical
        values: 0.02–0.2 depending on body size vs. domain.
    gpu_budget_gb : float, optional
        If given, print a verdict ("fits on a Xg GPU").
    verbose : bool
        If True (default) print the breakdown table and scaling curve.

    Returns
    -------
    dict
        ``{"steady_mb", "peak_mb", "components", "cells", …}``.
    """
    # ---- dtype resolution -------------------------------------------------
    if dtype is None:
        dtype = "float64"
    cell_bytes_fluid = _dtype_bytes(dtype)
    cell_bytes_geom = _dtype_bytes(sdf_dtype) if sdf_dtype is not None else cell_bytes_fluid

    N = int(nx) * int(ny) * int(nz)

    comps: List[Component] = []
    comps.extend(_grid_components(N, cell_bytes_fluid, cell_bytes_geom, bc_type))
    comps.extend(_body_components(
        N, cell_bytes_geom, int(nbodies),
        composite_kind=composite_kind,
        compile_forces=compile_forces,
        force_delta_order=force_delta_order,
        sparse_sdf=sparse_sdf,
        aabb_fill_fraction=aabb_fill_fraction,
        legacy_analytical_cache=legacy_analytical_cache,
        batched_forces_max_dense_bytes=batched_forces_max_dense_bytes,
    ))

    grid_bytes_fluid = cell_bytes_fluid * N
    grid_bytes_geom = cell_bytes_geom * N

    steady_bytes = sum(c.total_bytes for c in comps)
    peak_bytes = steady_bytes * _transient_factor(bc_type)

    steady_mb = steady_bytes / 1024 ** 2
    peak_mb = peak_bytes / 1024 ** 2

    if verbose:
        _print_breakdown(
            comps, nx, ny, nz, nbodies,
            dtype=dtype, sdf_dtype=sdf_dtype if sdf_dtype is not None else dtype,
            bc_type=bc_type, compile_forces=compile_forces,
            force_delta_order=force_delta_order,
            composite_kind=composite_kind,
            sparse_sdf=sparse_sdf,
            aabb_fill_fraction=aabb_fill_fraction,
            grid_bytes_fluid=grid_bytes_fluid,
            grid_bytes_geom=grid_bytes_geom,
            steady_mb=steady_mb, peak_mb=peak_mb,
            gpu_budget_gb=gpu_budget_gb,
        )

    return {
        "steady_mb": steady_mb,
        "peak_mb": peak_mb,
        "steady_gb": steady_mb / 1024,
        "peak_gb": peak_mb / 1024,
        "cells": N,
        "grid_bytes_fluid": grid_bytes_fluid,
        "grid_bytes_geom": grid_bytes_geom,
        "components": comps,
    }


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------

def _print_breakdown(
    comps: List[Component], nx, ny, nz, nbodies, *,
    dtype, sdf_dtype, bc_type, compile_forces, force_delta_order,
    composite_kind, sparse_sdf, aabb_fill_fraction,
    grid_bytes_fluid, grid_bytes_geom,
    steady_mb, peak_mb, gpu_budget_gb,
):
    bar = "=" * 78
    sep = "-" * 78
    print(bar)
    print("lilytorch 3-D solver memory estimate")
    print(bar)
    print(f"  grid                : {nx} × {ny} × {nz}  =  {nx*ny*nz:,} cells")
    print(f"  fluid dtype         : {_dtype_label(dtype)}  "
          f"({grid_bytes_fluid / 1024**2:.2f} MB / full grid)")
    print(f"  geom/SDF dtype      : {_dtype_label(sdf_dtype)}  "
          f"({grid_bytes_geom / 1024**2:.2f} MB / full grid)")
    print(f"  nbodies             : {nbodies}")
    print(f"  composite_kind      : {composite_kind}")
    print(f"  sparse_sdf          : {sparse_sdf}   "
          f"(aabb fill ≈ {aabb_fill_fraction*100:.1f}% of grid)")
    print(f"  compile_forces      : {compile_forces}")
    print(f"  force_delta_order   : {force_delta_order}")
    print(f"  Poisson bc_type     : {bc_type}")
    print(sep)

    # --- grouped table -----------------------------------------------------
    header = (f"  {'component':44s}  {'count':>6s}  "
              f"{'bytes/elem':>10s}  {'MB':>10s}  B?")
    print(header)
    print(sep)

    def _print_group(title, predicate):
        group = [c for c in comps if predicate(c)]
        if not group:
            return 0.0
        print(f"  [{title}]")
        total = 0.0
        for c in group:
            flag = "Y" if c.per_body else "-"
            count_str = f"{c.count:g}"
            mb = c.total_mb
            total += mb
            name = c.name
            if c.note:
                name = f"{name}"
            print(f"  {name[:44]:44s}  {count_str:>6s}  "
                  f"{c.elem_bytes:>10d}  {_fmt_mb(mb)}  {flag}")
            if c.note:
                print(f"     ↳ {c.note}")
        print(f"  {'  → subtotal':44s}  {'':>6s}  {'':>10s}  {_fmt_mb(total)}")
        print()
        return total

    grid_total = _print_group("grid-scaling  (O(N), independent of nbodies)",
                              lambda c: not c.per_body)
    body_total = _print_group("body-scaling  (O(B · …))",
                              lambda c: c.per_body)

    transient = (steady_mb * _transient_factor("free" if any("free-space" in c.name for c in comps)
                                                else "neumann")
                  - steady_mb)
    print(f"  [transient / peak multiplier]")
    print(f"  {'autograd + compiled-graph replay slack':44s}  "
          f"{'':>6s}  {'':>10s}  {_fmt_mb(transient)}  -")
    print()

    print(sep)
    print(f"  STEADY-STATE total  : {_fmt_mb(steady_mb)}")
    print(f"  PEAK VRAM estimate  : {_fmt_mb(peak_mb)}")
    print(sep)

    # --- dominant term & suggestions --------------------------------------
    if body_total > grid_total:
        dom = "body-scaling"
        hint = (
            "Body-scaling dominates.  Suggestions:\n"
            "    * if composite_kind='analytical', upgrade to "
            "'multianimat' or rely on step-2 streaming release;\n"
            "    * enable sparse_sdf=True and tune aabb_fill_fraction;\n"
            "    * set sdf_dtype=float32 (step 5) — halves this term;\n"
            "    * consider narrow-band SDF storage (step 6)."
        )
    else:
        dom = "grid-scaling"
        hint = (
            "Grid-scaling dominates.  Suggestions:\n"
            "    * if bc_type='free', the Green's-function FFT alone is "
            "~3 GB at the reference grid — switch to 'neumann' if the\n"
            "      physics allows;\n"
            "    * reduce nx·ny·nz;\n"
            "    * set dtype=float32 for the fluid state "
            "(experimental — validate numerics)."
        )
    print(f"  Dominant term       : {dom}")
    print(f"  Hint: {hint}")

    if gpu_budget_gb is not None:
        ok = peak_mb <= gpu_budget_gb * 1024
        verdict = "FITS" if ok else "OOM RISK"
        print()
        print(f"  vs. {gpu_budget_gb:.0f} GB GPU  : {verdict} "
              f"({peak_mb / 1024:.2f} / {gpu_budget_gb:.2f} GB)")
    print(bar)


def print_scaling_table(
    nx=1024 + 2, ny=128 + 2, nz=128 + 2, *,
    body_counts: Iterable[int] = (1, 5, 10, 20, 50),
    dtype="float64",
    sdf_dtype=None,
    bc_type: str = "neumann",
    compile_forces: bool = True,
    force_delta_order: int = 1,
    composite_kind: str = "multianimat",
    sparse_sdf: bool = True,
    aabb_fill_fraction: float = 0.05,
    legacy_analytical_cache: bool = False,
    batched_forces_max_dense_bytes: int = 1 * 1024**3,
    gpu_budgets_gb: Iterable[float] = (24, 40, 80),
    title_suffix: str = "",
):
    """Tabulate peak VRAM vs. nbodies for a configuration.

    Used both as a standalone sanity check at module import and as a
    regression monitor — any future refactor that silently regresses
    memory will show up here.
    """
    bar = "=" * 78
    print(bar)
    hdr_label = (f"PEAK VRAM vs nbodies  ({composite_kind}, "
                 f"sparse_sdf={sparse_sdf}, compile_forces={compile_forces}, "
                 f"bc={bc_type}{title_suffix})")
    print(hdr_label)
    print(bar)
    hdr = f"  {'B':>4s}  {'steady (GB)':>12s}  {'peak (GB)':>12s}"
    for g in gpu_budgets_gb:
        hdr += f"  {int(g):>3d}GB"
    print(hdr)
    for B in body_counts:
        r = estimate_memory(
            nx, ny, nz, B,
            dtype=dtype, sdf_dtype=sdf_dtype, bc_type=bc_type,
            compile_forces=compile_forces, force_delta_order=force_delta_order,
            composite_kind=composite_kind, sparse_sdf=sparse_sdf,
            aabb_fill_fraction=aabb_fill_fraction,
            legacy_analytical_cache=legacy_analytical_cache,
            batched_forces_max_dense_bytes=batched_forces_max_dense_bytes,
            verbose=False,
        )
        row = (f"  {B:>4d}  {r['steady_gb']:>12.2f}  {r['peak_gb']:>12.2f}")
        for g in gpu_budgets_gb:
            row += f"   {'OK' if r['peak_gb'] <= g else 'OOM':>5s}"
        print(row)
    print(bar)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--nx", type=int, default=1024 + 2, help="cells in x (incl. ghosts)")
    p.add_argument("--ny", type=int, default=128 + 2, help="cells in y (incl. ghosts)")
    p.add_argument("--nz", type=int, default=128 + 2, help="cells in z (incl. ghosts)")
    p.add_argument("-B", "--nbodies", type=int, default=1, help="number of bodies")
    p.add_argument("--dtype", default="float64", choices=list(_DTYPE_NAME_TO_BYTES),
                   help="fluid-state dtype")
    p.add_argument("--sdf-dtype", default=None, choices=list(_DTYPE_NAME_TO_BYTES),
                   help="geometry-field dtype (default: same as --dtype)")
    p.add_argument("--bc", default="neumann", choices=("neumann", "free"),
                   help="Poisson boundary condition")
    p.add_argument("--no-compile-forces", action="store_true",
                   help="disable compile_forces (avoids batched sdf_all stack)")
    p.add_argument("--force-delta-order", type=int, choices=(1, 2), default=1)
    p.add_argument("--composite", default="multianimat",
                   choices=("analytical", "multianimat"),
                   help="CompositeBody flavour")
    p.add_argument("--no-sparse-sdf", action="store_true",
                   help="use the legacy dense (B, *gs) SDF storage")
    p.add_argument("--aabb-fill", type=float, default=0.05,
                   help="fraction of the grid covered by one body's AABB")
    p.add_argument("--legacy-analytical-cache", action="store_true",
                   help="estimate pre-step-2 (legacy) analytical caches")
    p.add_argument("--batched-forces-max-dense-gb", type=float, default=1.0,
                   help="step-3 sparse-fallback threshold in GB (default 1)")
    p.add_argument("--gpu-gb", type=float, default=None,
                   help="print OOM verdict against this GPU budget (GB)")
    p.add_argument("--scaling", action="store_true",
                   help="also print peak-VRAM-vs-B scaling table")
    p.add_argument(
        "--body-counts", type=int, nargs="+", default=[1, 5, 10, 20, 50],
        help="body counts used in the scaling table",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    dense_thresh_bytes = int(args.batched_forces_max_dense_gb * 1024**3)
    estimate_memory(
        args.nx, args.ny, args.nz, args.nbodies,
        dtype=args.dtype,
        sdf_dtype=args.sdf_dtype,
        bc_type=args.bc,
        compile_forces=not args.no_compile_forces,
        force_delta_order=args.force_delta_order,
        composite_kind=args.composite,
        sparse_sdf=not args.no_sparse_sdf,
        aabb_fill_fraction=args.aabb_fill,
        legacy_analytical_cache=args.legacy_analytical_cache,
        batched_forces_max_dense_bytes=dense_thresh_bytes,
        gpu_budget_gb=args.gpu_gb,
    )
    if args.scaling:
        print()
        print_scaling_table(
            args.nx, args.ny, args.nz,
            body_counts=args.body_counts,
            dtype=args.dtype,
            sdf_dtype=args.sdf_dtype,
            bc_type=args.bc,
            compile_forces=not args.no_compile_forces,
            force_delta_order=args.force_delta_order,
            composite_kind=args.composite,
            sparse_sdf=not args.no_sparse_sdf,
            aabb_fill_fraction=args.aabb_fill,
            legacy_analytical_cache=args.legacy_analytical_cache,
            batched_forces_max_dense_bytes=dense_thresh_bytes,
        )
    return 0


# ---------------------------------------------------------------------------
# Backward-compatible entry point + validation sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())

    # Backward-compatible default: reference config, verbose breakdown.
    estimate_memory()

    # ---- parametric regression monitor (step 7 of the plan) --------------
    # Tabulate peak VRAM for B = 1, 5, 10, 20, 50 under each composite
    # flavour so future refactors that silently regress memory are
    # visible at a glance.
    print()
    print_scaling_table(composite_kind="multianimat", sparse_sdf=True,
                        title_suffix=", post-step-3 sparse path")
    print()
    print_scaling_table(composite_kind="multianimat", sparse_sdf=False,
                        title_suffix=", legacy dense (B, *gs) stack")
    print()
    print_scaling_table(composite_kind="analytical",
                        title_suffix=", post-step-2 streaming release")
    print()
    print_scaling_table(composite_kind="analytical",
                        legacy_analytical_cache=True,
                        title_suffix=", pre-step-2 legacy caches")
