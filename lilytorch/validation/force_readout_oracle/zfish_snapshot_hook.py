"""Dump a live 3-D force-readout scene mid-run, for offline band-shift sweeps.

``streaming_sdf_forces_post_3d`` is a pure function of (body SDF tables, poses,
fluid fields, eps).  So if we capture every argument the live solver hands it at
one step, we can re-drive it offline at any ``eps_solver`` we like and watch the
viscous readout move — without re-running the coupled simulation once per point.
That is the 3-D twin of ``shift_sweep_2d.py``.

Install from a FARMS run by overriding ``_extra_run_patch`` in the SimConfig::

    def _extra_run_patch(self):
        return ("import lilytorch.validation.force_readout_oracle."
                "zfish_snapshot_hook as _z; _z.install(step=300, out='/path/snap.pt');")

The hook wraps ``FluidSolver.forces_method2_3d``, so it only fires on the
eulerian path — run the generator with ``force_method = "eulerian"``.

Env overrides: ``ZFISH_SNAP_STEP``, ``ZFISH_SNAP_OUT``, ``ZFISH_SNAP_STOP``.
"""
from __future__ import annotations

import os

import torch

_STATE = {"step": 300, "out": "/tmp/zfish_snap.pt", "stop": True, "done": False}


def _cpu(x):
    if torch.is_tensor(x):
        return x.detach().to("cpu").clone()
    return x


class SnapshotTaken(RuntimeError):
    """Raised to stop the run once the snapshot is on disk."""


def _capture(self, u, v, w, p, iteration):
    comp = self.composite_body
    static = getattr(comp, "_kernel_static_3d", None)
    step = getattr(comp, "_kernel_step", None)
    if static is None or step is None:
        raise RuntimeError(
            "no native streaming scene on the composite body — the snapshot "
            "hook only supports the native eulerian path")

    snap = {
        # static body geometry
        "F_flat": _cpu(static["F_flat"]),
        "F_offsets": _cpu(static["F_offsets"]),
        "body_shapes": _cpu(static["body_shapes"]),
        "body_meta": _cpu(static["body_meta"]),
        # per-step pose / crop
        "kin": _cpu(step["kin"]),
        "aabb_lo": _cpu(step["aabb_lo"]),
        "aabb_dim": _cpu(step["aabb_dim"]),
        "gx": _cpu(step["gx"]), "gy": _cpu(step["gy"]), "gz": _cpu(step["gz"]),
        "max_vol": int(step["max_vol"]),
        # fluid state
        "sdf_cc": _cpu(comp.sdf_val),
        "u": _cpu(u.contiguous()), "v": _cpu(v.contiguous()),
        "w": _cpu(w.contiguous()), "p": _cpu(p.contiguous()),
        # scalars the readout was called with
        "h": float(self.h), "h3": float(self.h3),
        "eps_body": float(comp.bodies[0].eps),
        "eps_solver": float(self.eps),
        "nu": float(self.nu), "rho": float(self.rho),
        "delta_order": int(self.force_delta_order),
        "interp_method": int(getattr(self, "_sdf_interp_method", 0)),
        "force_submethod": getattr(self, "force_submethod", "ndelta"),
        "ph_blend_cells": float(getattr(self, "force_ph_blend_cells", 1.5)),
        "n_bodies": len(comp.bodies),
        "iteration": int(iteration),
        "grid": (int(self.Nx), int(self.Ny), int(self.Nz)),
    }

    # Lagrangian triangulation, if the run carries one — lets the sweep compare
    # both readouts on the SAME frozen field (the live run only calls one).
    for key in ("tri_centroid_world", "tri_normal_world", "tri_area_world",
                "tri_offsets"):
        t = getattr(comp, key, None)
        if t is not None:
            snap[key] = _cpu(t)
    coms = [getattr(b, "com", None) for b in comp.bodies]
    if all(c is not None for c in coms):
        snap["com"] = _cpu(torch.stack([torch.as_tensor(c) for c in coms]))

    torch.save(snap, _STATE["out"])
    print(f"[zfish-snap] wrote {_STATE['out']} at iteration {iteration} "
          f"(eps_solver={snap['eps_solver']:.4e}, h={snap['h']:.4e}, "
          f"{snap['n_bodies']} bodies)", flush=True)


def install(step=None, out=None, stop=None):
    from lilytorch.src.solver import FluidSolver

    _STATE["step"] = int(os.environ.get("ZFISH_SNAP_STEP", step or 300))
    _STATE["out"] = os.environ.get("ZFISH_SNAP_OUT", out or _STATE["out"])
    if stop is not None:
        _STATE["stop"] = bool(stop)
    if "ZFISH_SNAP_STOP" in os.environ:
        _STATE["stop"] = os.environ["ZFISH_SNAP_STOP"] not in ("0", "", "false")

    original = FluidSolver.forces_method2_3d

    def wrapped(self, u, v, w, p, iteration):
        result = original(self, u, v, w, p, iteration)
        if not _STATE["done"] and iteration >= _STATE["step"]:
            _STATE["done"] = True
            _capture(self, u, v, w, p, iteration)
            if _STATE["stop"]:
                raise SnapshotTaken(
                    f"snapshot written at iteration {iteration}; stopping")
        return result

    FluidSolver.forces_method2_3d = wrapped
    print(f"[zfish-snap] armed: step={_STATE['step']} out={_STATE['out']} "
          f"stop={_STATE['stop']}", flush=True)
