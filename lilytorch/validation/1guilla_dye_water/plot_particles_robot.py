#!/usr/bin/env python3
"""
plot_particles_robot.py — Lagrangian dye-particle visualisation from HDF5 data.

Reads ``fields.h5`` (and optionally ``contours.h5``), seeds tracer particles,
advects them with saved velocity fields, and renders each frame as a PNG.

Usage
-----
  python plot_particles_robot.py --sim_dir /data/.../run/
  python plot_particles_robot.py --sim_dir /data/.../run/ --mode 3d_topview
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Optional, Sequence

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_interpolation import RegularGridInterpolator, RegularGridInterpolator3D
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

@dataclass
class ParticlePlotConfig:
    """All settings for particle seeding, advection, and rendering.

    Parameters are grouped into sections. Some are only relevant for a
    specific ``seed_mode`` or ``render_mode`` — see the inline notes.
    """

    # ── General ──────────────────────────────────────────────────────
    sim_dir: str = ""
    """Path to the simulation output directory.  Must contain
    ``fields.h5`` (velocity + SDF snapshots) and optionally
    ``contours.h5`` (body contour polygons)."""

    mode: str = "2d"
    """Visualisation mode:
    - ``"2d"``         – flat 2-D advection (uses u, v only).
    - ``"3d_topview"`` – 3-D advection projected onto the x-y plane.
    - ``"3d_full"``    – full 3-D matplotlib scatter view."""

    # ── Iteration range ──────────────────────────────────────────────
    it_start: int = 0
    """First HDF5 iteration index to process (inclusive)."""

    it_end: int = -1
    """Last iteration index to process (inclusive).  ``-1`` means the
    last available iteration in the file."""

    it_spacing: int = 1
    """Take every *n*-th iteration (stride).  E.g. ``2`` skips every
    other snapshot."""

    # ── Seeding ──────────────────────────────────────────────────────
    seed_mode: str = "tail_contour"
    """How new tracer particles are injected each frame:
    - ``"tail_contour"`` – at the trailing edge (min-x) of the body,
      extracted from ``contours.h5`` or from the SDF zero-level-set.
    - ``"body_boundary"``– uniformly around the full body boundary.
    - ``"line"``         – along a straight line between two endpoints
      (only seeded once, at frame 0)."""

    tail_body_index: int = -1
    """[tail_contour] Which body contour to use from ``contours.h5``.
    ``-1`` selects the last body (usually the tail)."""

    tail_center_idx: int = 85
    """[tail_contour] Index along the contour polygon that corresponds
    to the trailing-edge centre.  Points within ±tail_n_particles of
    this index are selected."""

    tail_n_particles: int = 12
    """[tail_contour] Half-width (in contour-point indices) around
    ``tail_center_idx`` used to extract seed points.  Also controls
    the number of SDF-fallback tail seeds (×2)."""

    line_start: tuple[float, ...] = (0.0, -0.05)
    """[line] Starting point of the seed line (x, y) or (x, y, z)."""

    line_end:   tuple[float, ...] = (0.0,  0.05)
    """[line] Ending point of the seed line."""

    line_n_particles: int = 30
    """[line] Number of particles placed along the seed line."""

    boundary_n_particles: int = 120
    """[body_boundary] Target number of seed points sampled uniformly
    around the body boundary in the x-y plane."""

    boundary_n_z_layers: int = 5
    """[body_boundary] Number of z-layers across the body thickness
    at which the 2-D boundary seeds are replicated (3-D only)."""

    seed_interval: int = 10
    """During sub-stepped advection between saved snapshots, inject a
    new batch of seeds every *seed_interval* solver sub-steps.  Lower
    values give a denser dye trail but cost more memory."""

    turb_diffusivity: float = 0.0
    """Effective turbulent diffusivity D_eff [m²/s] for stochastic
    random-walk dispersion.  Each particle receives an additional
    displacement  dx ~ N(0, sqrt(2·D_eff·dt))  per axis per sub-step.
    Set to 0 to disable (pure advection).  Typical values for a
    Smagorinsky LES at Cs=0.15 with h~5 mm: ~1e-4 to 1e-3."""

    # ── Rendering ────────────────────────────────────────────────────
    render_mode: str = "density"
    """How particles are drawn on the 2-D / top-view canvas:
    - ``"density"``  – 2-D histogram rendered as a pink density field.
    - ``"scatter"``  – individual dots (uses particle_color/size/alpha).
    - ``"both"``     – overlay scatter on top of the density field.
    (``3d_full`` mode always uses scatter.)"""

    density_bins: int = 300
    """[density] Number of histogram bins along each axis.  Higher
    values give a sharper image but can look noisy with few particles."""

    density_sigma: float = 0
    """[density] Gaussian blur sigma (in bin units) applied to the
    histogram before rendering.  ``0`` disables smoothing."""

    density_intensity: float = 10.0
    """[density] Manual scaling factor for dye density intensity (alpha multiplier)."""

    # ── Aesthetics ───────────────────────────────────────────────────
    particle_color: str = "#FF00A6"
    """[scatter] Hex colour for individual particle dots."""

    particle_size: float = 8.0
    """[scatter] Matplotlib marker size for each particle dot."""

    particle_alpha: float = 0.85
    """[scatter] Opacity of each particle dot (0 = invisible, 1 = opaque)."""

    body_color: str = "#C4AF0F"
    """Fill colour for the immersed body drawn from SDF or contours."""

    bg_color: str = "#ffffffdc"
    """Background colour of the figure and axes (hex, may include alpha
    nibble)."""

    bg_alpha: float = 1
    """Background opacity (0–1).  Useful for compositing over other
    images."""

    figsize: tuple[float, float] = (14, 7)
    """Figure size in inches ``(width, height)``."""

    dpi: int = 150
    """Output image resolution in dots-per-inch."""

    xlim: Optional[tuple[float, float]] = None
    """Override the x-axis limits ``(xmin, xmax)`` in metres.  ``None``
    uses the grid extent (excluding ghost cells)."""

    ylim: Optional[tuple[float, float]] = None
    """Override the y-axis limits ``(ymin, ymax)``.  Same convention as
    ``xlim``."""

    trail_length: int = 0
    """Maximum number of seeding batches to keep.  ``0`` means unlimited
    — particles are never discarded (except by domain / body culling).
    Useful to limit memory when many particles accumulate."""

    # ── Output ───────────────────────────────────────────────────────
    out_subdir: str = "particle_images"
    """Sub-directory (inside ``sim_dir``) where PNGs are saved."""

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """PyTorch device for interpolation and advection
    (``"cuda"`` or ``"cpu"``)."""

    dtype_str: str = "float32"
    """PyTorch dtype name (``"float32"`` or ``"float64"``)."""

    @property
    def dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype_str)


# ──────────────────────────────────────────────────────────────────────
#  HDF5 helpers
# ──────────────────────────────────────────────────────────────────────

def load_grids(h5path: str) -> dict[str, np.ndarray]:
    with h5py.File(h5path, "r") as f:
        return {name: f[f"grids/{name}"][:] for name in f["grids"]}


def load_snapshot(h5path: str, it: int, keys: Sequence[str] = ("u", "v")) -> dict[str, np.ndarray]:
    grp = f"fields/{it:06d}"
    with h5py.File(h5path, "r") as f:
        return {k: f[f"{grp}/{k}"][:] for k in keys if f"{grp}/{k}" in f}


def load_contours(cnt_h5: str, it: int) -> list[np.ndarray] | None:
    if not os.path.exists(cnt_h5):
        return None
    with h5py.File(cnt_h5, "r") as f:
        grp = f"{it:06d}"
        if grp not in f:
            return None
        g = f[grp]
        cnts = [g[f"body_{i}"][:] for i in range(len(g)) if f"body_{i}" in g]
    return cnts or None


def available_iterations(h5path: str) -> list[int]:
    with h5py.File(h5path, "r") as f:
        return sorted(int(k) for k in f["fields"] if k.isdigit())


# ──────────────────────────────────────────────────────────────────────
#  SDF-based seeding helpers
# ──────────────────────────────────────────────────────────────────────

def _sdf_boundary_xy(sdf: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Return (bx, by) coordinates of SDF zero-level-set boundary cells."""
    from scipy import ndimage
    if sdf.ndim == 3:
        sdf_2d = sdf[1:-1, 1:-1, :].min(axis=2)
    elif sdf.ndim == 2:
        sdf_2d = sdf[1:-1, 1:-1]
    else:
        return np.array([]), np.array([])
    xp, yp = x[1:-1], y[1:-1]
    inside = sdf_2d <= 0
    boundary = ndimage.binary_dilation(inside, iterations=1) & ~inside
    if not np.any(boundary):
        return np.array([]), np.array([])
    bi, bj = np.where(boundary)
    return xp[bi], yp[bj]


def _body_z_extent(sdf, z):
    """Return (zmin, zmax) of the body from SDF, or (None, None)."""
    if sdf is None or sdf.ndim != 3 or z is None:
        return None, None
    body_mask = (sdf[1:-1, 1:-1, :] <= 0).any(axis=(0, 1))
    if body_mask.any():
        idx = np.where(body_mask)[0]
        return float(z[idx[0]]), float(z[idx[-1]])
    return None, None


def _replicate_at_z_levels(pts_xy, sdf, z, n_z_layers):
    """Expand (2, N) points to (3, N*n_z_layers) across the body z-extent."""
    if z is None:
        return pts_xy
    zlo, zhi = _body_z_extent(sdf, z)
    if zlo is None:
        zmid = 0.5 * (float(z[1]) + float(z[-2]))
        zlo, zhi = zmid - 0.01, zmid + 0.01
    dz = zhi - zlo
    z_levels = np.linspace(zlo + 0.05 * dz, zhi - 0.05 * dz, n_z_layers)
    n = pts_xy.shape[1]
    layers = []
    for zl in z_levels:
        layer = np.vstack([pts_xy, np.full((1, n), zl)])
        layers.append(layer)
    return np.concatenate(layers, axis=1)


def _extract_body_tail_from_sdf(sdf, x, y, n_particles=8, z=None):
    """Seed *n_particles* points near the trailing edge (min-x boundary).

    Always selects only the ~8 boundary cells closest to the tail tip,
    then interpolates between them to reach *n_particles* total.  This
    keeps the seeds tightly clustered even when n_particles is large.
    """
    bx, by = _sdf_boundary_xy(sdf, x, y)
    if len(bx) == 0:
        ndim = 3 if (z is not None and sdf.ndim == 3) else 2
        return np.zeros((ndim, 0))
    # Pick a small fixed cluster of boundary cells at the tail tip
    n_anchor = min(8, len(bx))
    sel = np.argsort(bx)[:n_anchor]
    ax_, ay_ = bx[sel], by[sel]
    # Densify: interpolate along the anchor polyline to reach n_particles
    if n_particles > n_anchor:
        t_anc = np.linspace(0, 1, n_anchor)
        t_dense = np.linspace(0, 1, n_particles)
        ax_ = np.interp(t_dense, t_anc, ax_)
        ay_ = np.interp(t_dense, t_anc, ay_)
    pts_xy = np.stack([ax_, ay_], axis=0)
    if sdf.ndim == 3 and z is not None:
        return _replicate_at_z_levels(pts_xy, sdf, z, max(3, n_particles // 2))
    return pts_xy


def _extract_body_boundary_from_sdf(sdf, x, y, n_particles=120, z=None,
                                     n_z_layers=5):
    """Seed points uniformly around the entire body boundary."""
    bx, by = _sdf_boundary_xy(sdf, x, y)
    if len(bx) == 0:
        ndim = 3 if (z is not None and sdf.ndim == 3) else 2
        return np.zeros((ndim, 0))
    if len(bx) > n_particles:
        idx = np.linspace(0, len(bx) - 1, n_particles, dtype=int)
        bx, by = bx[idx], by[idx]
    pts_xy = np.stack([bx, by], axis=0)
    if sdf.ndim == 3 and z is not None:
        return _replicate_at_z_levels(pts_xy, sdf, z, n_z_layers)
    return pts_xy


# ──────────────────────────────────────────────────────────────────────
#  Simple seeding
# ──────────────────────────────────────────────────────────────────────

def seed_from_contour(contours, body_idx, center, half_width):
    cnt = contours[body_idx]
    lo, hi = max(0, center - half_width), min(cnt.shape[1], center + half_width)
    return cnt[:, lo:hi].copy()


def seed_from_line(start, end, n):
    s, e = np.array(start), np.array(end)
    t = np.linspace(0, 1, n)
    return s[:, None] + (e - s)[:, None] * t[None, :]


# ──────────────────────────────────────────────────────────────────────
#  Unified seeding helper
# ──────────────────────────────────────────────────────────────────────

def _seed_new_particles(cfg, snap, grids, contours_h5, it, frame_idx, ndim, zmid=0.0):
    """Return (ndim, N) numpy seed array, or None if no seeding this frame."""
    z = grids.get("z") if ndim == 3 else None
    sdf = snap.get("sdf")

    if cfg.seed_mode == "body_boundary" and sdf is not None:
        pts = _extract_body_boundary_from_sdf(
            sdf, grids["x"], grids["y"],
            n_particles=cfg.boundary_n_particles, z=z,
            n_z_layers=cfg.boundary_n_z_layers,
        )
        return pts if pts.shape[1] > 0 else None

    if cfg.seed_mode == "tail_contour":
        cnts = load_contours(contours_h5, it)
        if cnts is not None:
            pts2d = seed_from_contour(cnts, cfg.tail_body_index,
                                       cfg.tail_center_idx, cfg.tail_n_particles)
            if ndim == 3:
                return _replicate_at_z_levels(pts2d, sdf, z, n_z_layers=3)
            return pts2d
        if sdf is not None:
            pts = _extract_body_tail_from_sdf(
                sdf, grids["x"], grids["y"],
                n_particles=cfg.tail_n_particles * 2, z=z,
            )
            return pts if pts.shape[1] > 0 else None

    if cfg.seed_mode == "line" and frame_idx == 0:
        if ndim == 3:
            return seed_from_line((*cfg.line_start[:2], zmid),
                                  (*cfg.line_end[:2], zmid), cfg.line_n_particles)
        return seed_from_line(cfg.line_start, cfg.line_end, cfg.line_n_particles)

    return None


# ──────────────────────────────────────────────────────────────────────
#  Interpolation & advection
# ──────────────────────────────────────────────────────────────────────

def _build_interps(grids, device, dtype):
    """Build interpolators. Returns (coords, interps, ndim)."""
    ndim = 3 if "z" in grids else 2
    axes = ["x", "y", "z"][:ndim]
    coords = [torch.as_tensor(grids[ax], device=device, dtype=dtype).contiguous()
              for ax in axes]
    shape = [len(c) for c in coords]
    Interp = RegularGridInterpolator3D if ndim == 3 else RegularGridInterpolator
    interps = [Interp(tuple(coords), torch.zeros(*shape, device=device, dtype=dtype).contiguous())
               for _ in range(ndim)]
    return coords, interps, ndim


@torch.no_grad()
def advect_particles(particles, interps, dt, turb_diff=0.0):
    """Forward-Euler advection + optional stochastic turbulent diffusion.

    If turb_diff > 0, each particle receives an additional Wiener-process
    displacement  dx_i ~ N(0, sqrt(2 * D_eff * dt))  per spatial axis,
    modelling sub-grid turbulent dispersion.
    """
    for i, interp in enumerate(interps):
        vel = interp(*particles)
        particles[i].add_(vel, alpha=dt)
    if turb_diff > 0.0 and particles.shape[1] > 0:
        sigma = (2.0 * turb_diff * abs(dt)) ** 0.5
        particles.add_(torch.randn_like(particles) * sigma)
    return particles


# ──────────────────────────────────────────────────────────────────────
#  Culling
# ──────────────────────────────────────────────────────────────────────

def cull_out_of_domain(particles, grids):
    ndim = particles.shape[0]
    mask = torch.ones(particles.shape[1], dtype=torch.bool, device=particles.device)
    for i, ax in enumerate(["x", "y", "z"][:ndim]):
        g = grids[ax]
        mask &= (particles[i] >= float(g[1])) & (particles[i] <= float(g[-2]))
    return particles[:, mask]


@torch.no_grad()
def cull_inside_body(particles, sdf, grids, margin=0.0):
    if particles.shape[1] == 0:
        return particles
    device, dtype = particles.device, particles.dtype
    xg = torch.as_tensor(grids["x"], device=device, dtype=dtype)
    yg = torch.as_tensor(grids["y"], device=device, dtype=dtype)
    sdf_2d = sdf.min(axis=2) if sdf.ndim == 3 else sdf
    sdf_t = torch.as_tensor(np.ascontiguousarray(sdf_2d), device=device, dtype=dtype)
    ix = torch.searchsorted(xg, particles[0].contiguous()).clamp(0, xg.shape[0] - 1)
    iy = torch.searchsorted(yg, particles[1].contiguous()).clamp(0, yg.shape[0] - 1)
    return particles[:, sdf_t[ix, iy] > margin]


# ──────────────────────────────────────────────────────────────────────
#  Drawing
# ──────────────────────────────────────────────────────────────────────

def _draw_body_sdf(ax, sdf, x, y, z=None, color="#FFEA00"):
    sdf_2d = sdf.min(axis=2) if (sdf.ndim == 3 and z is not None) else sdf
    s = slice(1, -1)
    sdf_2d, xp, yp = sdf_2d[s, s], x[s], y[s]
    ax.contourf(xp, yp, sdf_2d.T, levels=[-1e10, 0.0], colors=[color], zorder=2)
    ax.contour(xp, yp, sdf_2d.T, levels=[0.0], colors=["#000000"], linewidths=[1.5], zorder=2)


def _draw_contours_2d(ax, contours, color="#FFEA00"):
    for cnt in contours:
        ax.fill(cnt[0], cnt[1], color=color, zorder=2)
        ax.plot(cnt[0], cnt[1], color="#000000", linewidth=1.5, zorder=2)


def _draw_dye_density(ax, px, py, xlim, ylim, bins=300, sigma=0.0,
                       intensity=10.0, zorder=1):
    """Render particles as a density field: fixed pink, alpha ∝ density (scaled by intensity)."""
    if len(px) == 0:
        return
    H, xe, ye = np.histogram2d(px, py, bins=bins, range=[list(xlim), list(ylim)])
    H = H.astype(np.float64)
    if sigma > 0:
        from scipy.ndimage import gaussian_filter
        H = gaussian_filter(H, sigma=sigma)
    pink = np.array([1.0, 0.41, 0.71])
    alpha = np.clip(H.T * intensity, 0.0, 1.0)
    rgba = np.zeros((*alpha.shape, 4))
    rgba[..., :3] = pink
    rgba[..., 3] = alpha
    ax.imshow(rgba, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]],
              aspect="auto", interpolation="bilinear", zorder=zorder)


# ──────────────────────────────────────────────────────────────────────
#  Timestep inference
# ──────────────────────────────────────────────────────────────────────

def _infer_dt(fields_h5, iters=None):
    sim_dir = os.path.dirname(fields_h5)

    # Try parameters.yaml
    params_path = os.path.join(sim_dir, "parameters.yaml")
    if os.path.isfile(params_path):
        try:
            with open(params_path) as fh:
                for line in fh:
                    m = re.match(r"\s*dt:\s*([\d.eE+\-]+)", line)
                    if m:
                        return float(m.group(1))
        except Exception:
            pass

    # Try config YAMLs
    for name in ("config.yaml", "config.yml", "bdim_config.yaml"):
        ypath = os.path.join(sim_dir, name)
        if os.path.isfile(ypath):
            try:
                import yaml
                with open(ypath) as fh:
                    dt = yaml.safe_load(fh).get("solver", {}).get("dt")
                if dt is not None:
                    return float(dt)
            except Exception:
                pass

    # Try HDF5 attribute
    try:
        with h5py.File(fields_h5, "r") as f:
            if "dt" in f.attrs:
                return float(f.attrs["dt"])
    except Exception:
        pass

    print("[WARN] Could not infer dt — using 0.001")
    return 0.001


# ──────────────────────────────────────────────────────────────────────
#  Main: unified top-view (handles both 2-D and 3-D)
# ──────────────────────────────────────────────────────────────────────

def run_topview(cfg: ParticlePlotConfig):
    """Unified 2-D / 3-D top-view particle advection and rendering."""
    fields_h5 = os.path.join(cfg.sim_dir, "fields.h5")
    contours_h5 = os.path.join(cfg.sim_dir, "contours.h5")
    grids = load_grids(fields_h5)
    iters = available_iterations(fields_h5)

    # Determine dimensionality
    ndim = 3 if ("z" in grids and cfg.mode != "2d") else 2
    vel_keys = tuple(["u", "v", "w"][:ndim]) + ("sdf",)

    it_end = cfg.it_end if cfg.it_end > 0 else iters[-1]
    sel = [it for it in iters if cfg.it_start <= it <= it_end][::cfg.it_spacing]

    device = torch.device(cfg.device)
    dtype = cfg.dtype

    if ndim == 3:
        coords, interps, _ = _build_interps(grids, device, dtype)
    else:
        x = torch.as_tensor(grids["x"], device=device, dtype=dtype)
        y = torch.as_tensor(grids["y"], device=device, dtype=dtype)
        dummy = torch.zeros(len(x), len(y), device=device, dtype=dtype)
        interps = [RegularGridInterpolator((x, y), dummy),
                    RegularGridInterpolator((x, y), dummy)]
        coords = [x, y]

    xlim = cfg.xlim or (float(coords[0][1]), float(coords[0][-2]))
    ylim = cfg.ylim or (float(coords[1][1]), float(coords[1][-2]))
    zmid = 0.5 * (float(coords[2][1]) + float(coords[2][-2])) if ndim == 3 else 0.0

    particles = torch.zeros((ndim, 0), device=device, dtype=dtype)
    img_dir = os.path.join(cfg.sim_dir, cfg.out_subdir)
    os.makedirs(img_dir, exist_ok=True)
    prefix = "particles_topview" if ndim == 3 else "particles"
    label = "3-D top-view" if ndim == 3 else "2-D"
    running_vmax = 0.0

    for frame_idx, it in enumerate(tqdm(sel, desc=f"{label} particles")):
        snap = load_snapshot(fields_h5, it, keys=vel_keys)

        # Update interpolator fields
        for i, key in enumerate(["u", "v", "w"][:ndim]):
            if key in snap:
                interps[i].F = torch.as_tensor(snap[key], device=device, dtype=dtype).contiguous()

        # Seed
        new_pts = _seed_new_particles(cfg, snap, grids, contours_h5, it, frame_idx, ndim, zmid)
        if new_pts is not None:
            if cfg.seed_mode == "line" and frame_idx == 0:
                particles = torch.as_tensor(new_pts, device=device, dtype=dtype)
            else:
                particles = torch.cat(
                    [particles, torch.as_tensor(new_pts, device=device, dtype=dtype)], dim=1)

        # Trim trail
        if cfg.trail_length > 0:
            max_pts = cfg.trail_length * cfg.tail_n_particles * 2 * (3 if ndim == 3 else 1)
            if particles.shape[1] > max_pts:
                particles = particles[:, -max_pts:]

        # Cull inside body
        if "sdf" in snap and particles.shape[1] > 0:
            particles = cull_inside_body(particles, snap["sdf"], grids)

        # ── Draw ────────────────────────────────────────────────────
        fig, ax = plt.subplots(1, 1, figsize=cfg.figsize, dpi=cfg.dpi)
        fig.patch.set_facecolor(cfg.bg_color)
        fig.patch.set_alpha(cfg.bg_alpha)

        ax.set_facecolor(cfg.bg_color)
        ax.patch.set_alpha(cfg.bg_alpha)

        if "sdf" in snap:
            _draw_body_sdf(ax, snap["sdf"], grids["x"], grids["y"],
                           grids.get("z") if ndim == 3 else None, color=cfg.body_color)
        else:
            cnts = load_contours(contours_h5, it)
            if cnts:
                _draw_contours_2d(ax, cnts, color=cfg.body_color)

        if particles.shape[1] > 0:
            px, py = particles[0].cpu().numpy(), particles[1].cpu().numpy()
            if cfg.render_mode in ("density", "both"):
                _draw_dye_density(
                    ax, px, py, xlim, ylim,
                    bins=cfg.density_bins, sigma=cfg.density_sigma,
                    intensity=cfg.density_intensity)
            if cfg.render_mode in ("scatter", "both"):
                ax.scatter(px, py, c=cfg.particle_color, s=cfg.particle_size,
                           alpha=cfg.particle_alpha, edgecolors="none", zorder=3)

        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_aspect("equal")
        ax.set_title(f"Simulation {'(top view) ' if ndim == 3 else ''}—  iter {it}", fontsize=12)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"{prefix}_{it:06d}.png"), dpi=cfg.dpi,
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        # ── Sub-stepped advection to next snapshot ──────────────────
        if frame_idx + 1 < len(sel) and particles.shape[1] > 0:
            next_it = sel[frame_idx + 1]
            solver_dt = _infer_dt(fields_h5, iters)
            n_solver_iters = next_it - it

            next_snap = load_snapshot(fields_h5, next_it, keys=vel_keys)
            vel_names = ["u", "v", "w"][:ndim]

            vel_curr = [torch.as_tensor(snap[k], device=device, dtype=dtype).contiguous()
                        for k in vel_names if k in snap]
            vel_next = [torch.as_tensor(next_snap[k], device=device, dtype=dtype).contiguous()
                        for k in vel_names if k in next_snap]
            vel_buf = [torch.empty_like(v) for v in vel_curr]

            # Seed endpoints for interpolation
            seed_curr_t = seed_next_t = None
            if cfg.seed_mode in ("tail_contour", "body_boundary") and "sdf" in snap:
                z_arg = grids.get("z") if ndim == 3 else None
                if cfg.seed_mode == "body_boundary":
                    kw = dict(n_particles=cfg.boundary_n_particles, z=z_arg,
                              n_z_layers=cfg.boundary_n_z_layers)
                    seed_curr = _extract_body_boundary_from_sdf(
                        snap["sdf"], grids["x"], grids["y"], **kw)
                    seed_next = (_extract_body_boundary_from_sdf(
                        next_snap.get("sdf", snap["sdf"]), grids["x"], grids["y"], **kw)
                        if "sdf" in next_snap else seed_curr)
                else:
                    kw = dict(n_particles=cfg.tail_n_particles * 2, z=z_arg)
                    seed_curr = _extract_body_tail_from_sdf(
                        snap["sdf"], grids["x"], grids["y"], **kw)
                    seed_next = (_extract_body_tail_from_sdf(
                        next_snap.get("sdf", snap["sdf"]), grids["x"], grids["y"], **kw)
                        if "sdf" in next_snap else seed_curr)
                n_common = min(seed_curr.shape[1], seed_next.shape[1])
                if n_common > 0:
                    seed_curr_t = torch.as_tensor(seed_curr[:, :n_common], device=device, dtype=dtype)
                    seed_next_t = torch.as_tensor(seed_next[:, :n_common], device=device, dtype=dtype)

            si = max(1, cfg.seed_interval)
            cull_every = max(1, min(50, n_solver_iters // 4))

            with torch.no_grad():
                for k in range(n_solver_iters):
                    alpha = (k + 1) / n_solver_iters
                    for vc, vn, buf, interp in zip(vel_curr, vel_next, vel_buf, interps):
                        torch.lerp(vc, vn, alpha, out=buf)
                        interp.F = buf

                    if seed_curr_t is not None and k % si == 0:
                        seed_pos = torch.lerp(seed_curr_t, seed_next_t, alpha)
                        particles = torch.cat([particles, seed_pos], dim=1)

                    particles = advect_particles(particles, interps, solver_dt,
                                                  turb_diff=cfg.turb_diffusivity)

                    if k % cull_every == 0:
                        particles = cull_out_of_domain(particles, grids)

            particles = cull_out_of_domain(particles, grids)

    print(f"[{label}] Saved {len(sel)} frames to {img_dir}")


# ──────────────────────────────────────────────────────────────────────
#  3-D full scatter view
# ──────────────────────────────────────────────────────────────────────

def run_3d_full(cfg: ParticlePlotConfig):
    """Full 3-D rendering using matplotlib's 3-D scatter."""
    fields_h5 = os.path.join(cfg.sim_dir, "fields.h5")
    contours_h5 = os.path.join(cfg.sim_dir, "contours.h5")
    grids = load_grids(fields_h5)
    iters = available_iterations(fields_h5)

    if "z" not in grids:
        print("[WARN] Data is 2-D — falling back to 2-D mode.")
        return run_topview(cfg)

    it_end = cfg.it_end if cfg.it_end > 0 else iters[-1]
    sel = [it for it in iters if cfg.it_start <= it <= it_end][::cfg.it_spacing]

    device, dtype = torch.device(cfg.device), cfg.dtype
    coords, interps, _ = _build_interps(grids, device, dtype)
    xmin, xmax = float(coords[0][1]), float(coords[0][-2])
    ymin, ymax = float(coords[1][1]), float(coords[1][-2])
    zmin, zmax = float(coords[2][1]), float(coords[2][-2])
    zmid = 0.5 * (zmin + zmax)

    particles = torch.zeros((3, 0), device=device, dtype=dtype)
    img_dir = os.path.join(cfg.sim_dir, cfg.out_subdir)
    os.makedirs(img_dir, exist_ok=True)

    for frame_idx, it in enumerate(tqdm(sel, desc="3-D full particles")):
        snap = load_snapshot(fields_h5, it, keys=("u", "v", "w", "sdf"))
        for i, key in enumerate(["u", "v", "w"]):
            if key in snap:
                interps[i].F = torch.as_tensor(snap[key], device=device, dtype=dtype).contiguous()

        new_pts = _seed_new_particles(cfg, snap, grids, contours_h5, it, frame_idx, 3, zmid)
        if new_pts is not None:
            if cfg.seed_mode == "line" and frame_idx == 0:
                particles = torch.as_tensor(new_pts, device=device, dtype=dtype)
            else:
                particles = torch.cat(
                    [particles, torch.as_tensor(new_pts, device=device, dtype=dtype)], dim=1)

        # Draw 3-D scatter
        fig = plt.figure(figsize=cfg.figsize, dpi=cfg.dpi)
        fig.patch.set_facecolor(cfg.bg_color)
        fig.patch.set_alpha(cfg.bg_alpha)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(cfg.bg_color)

        if particles.shape[1] > 0:
            px, py, pz = [particles[i].cpu().numpy() for i in range(3)]
            ax.scatter(px, py, pz, c=cfg.particle_color, s=cfg.particle_size,
                       alpha=cfg.particle_alpha, depthshade=True)

        if "sdf" in snap:
            sdf = snap["sdf"]
            mid_z = sdf.shape[2] // 2
            try:
                ax.contour(grids["x"][1:-1], grids["y"][1:-1], sdf[1:-1, 1:-1, mid_z].T,
                           levels=[0.0], colors=[cfg.body_color], linewidths=2,
                           zdir="z", offset=float(grids["z"][mid_z]))
            except Exception:
                pass

        ax.set_xlim(cfg.xlim or (xmin, xmax))
        ax.set_ylim(cfg.ylim or (ymin, ymax))
        ax.set_zlim(zmin, zmax)
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(f"3-D particles — iter {it}", fontsize=12)
        ax.view_init(elev=25, azim=-60)

        fig.tight_layout()
        fig.savefig(os.path.join(img_dir, f"particles_3d_{it:06d}.png"), dpi=cfg.dpi,
                    facecolor=fig.get_facecolor(), edgecolor="none")
        plt.close(fig)

        # Single-step advection (no sub-stepping for 3d_full)
        if frame_idx + 1 < len(sel) and particles.shape[1] > 0:
            dt_snap = (sel[frame_idx + 1] - it) * _infer_dt(fields_h5, iters)
            particles = advect_particles(particles, interps, dt_snap,
                                          turb_diff=cfg.turb_diffusivity)
            particles = cull_out_of_domain(particles, grids)

    print(f"[3-D full] Saved {len(sel)} frames to {img_dir}")


# ──────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(description="Lagrangian particle viz from HDF5.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--sim_dir", type=str, required=True,
                   help="Path to sim dir containing fields.h5.")
    p.add_argument("--mode", type=str, default="2d",
                   choices=["2d", "3d_topview", "3d_full"],
                   help="Visualisation mode.")

    g = p.add_argument_group("Iteration range")
    g.add_argument("--it_start", type=int, default=0,
                   help="First HDF5 iteration (inclusive).")
    g.add_argument("--it_end", type=int, default=-1,
                   help="Last iteration (inclusive). -1 = last available.")
    g.add_argument("--it_spacing", type=int, default=1,
                   help="Process every n-th iteration.")

    g = p.add_argument_group("Seeding")
    g.add_argument("--seed_mode", type=str, default="tail_contour",
                   choices=["tail_contour", "body_boundary", "line"],
                   help="Particle injection strategy.")
    g.add_argument("--boundary_n_particles", type=int, default=120,
                   help="[body_boundary] Target boundary seed count.")
    g.add_argument("--boundary_n_z_layers", type=int, default=5,
                   help="[body_boundary] Number of z replicas (3-D only).")
    g.add_argument("--tail_body_index", type=int, default=-1,
                   help="[tail_contour] Body index in contours.h5 (-1=last).")
    g.add_argument("--tail_center_idx", type=int, default=85,
                   help="[tail_contour] Contour index at trailing-edge centre.")
    g.add_argument("--tail_n_particles", type=int, default=12,
                   help="[tail_contour] Half-width in contour indices.")
    g.add_argument("--seed_interval", type=int, default=10,
                   help="Inject seeds every n solver sub-steps.")
    g.add_argument("--turb_diffusivity", type=float, default=0.0,
                   help="Turbulent diffusivity D_eff [m²/s] for stochastic dispersion. 0=off.")
    g.add_argument("--line_start", type=float, nargs="+", default=[0.0, -0.05],
                   help="[line] Start point of seed line.")
    g.add_argument("--line_end", type=float, nargs="+", default=[0.0, 0.05],
                   help="[line] End point of seed line.")
    g.add_argument("--line_n_particles", type=int, default=100,
                   help="[line] Particles along the seed line.")

    g = p.add_argument_group("Rendering")
    g.add_argument("--render_mode", type=str, default="density",
                   choices=["scatter", "density", "both"],
                   help="Particle drawing method (3d_full always uses scatter).")
    g.add_argument("--density_bins", type=int, default=300,
                   help="[density] Histogram bins per axis.")
    g.add_argument("--density_sigma", type=float, default=0,
                   help="[density] Gaussian blur sigma (bins). 0 = no blur.")
    g.add_argument("--density_intensity", type=float, default=0.05,
                   help="[density] Manual scaling factor for dye density intensity (alpha multiplier).")

    g = p.add_argument_group("Aesthetics")
    g.add_argument("--particle_color", type=str, default="#FF00A6",
                   help="[scatter] Hex colour for particle dots.")
    g.add_argument("--particle_size", type=float, default=8.0,
                   help="[scatter] Marker size for particle dots.")
    g.add_argument("--particle_alpha", type=float, default=1,
                   help="[scatter] Opacity per particle (0-1).")
    g.add_argument("--body_color", type=str, default="#C4AF0F",
                   help="Fill colour for the immersed body.")
    g.add_argument("--bg_color", type=str, default="#ffffffdc",
                   help="Background colour (hex, may include alpha).")
    g.add_argument("--bg_alpha", type=float, default=1,
                   help="Background opacity (0-1).")
    g.add_argument("--figsize", type=float, nargs=2, default=[14, 7],
                   help="Figure size in inches (width height).")
    g.add_argument("--dpi", type=int, default=150,
                   help="Output image resolution.")
    g.add_argument("--xlim", type=float, nargs=2, default=None,
                   help="Override x-axis limits (xmin xmax) [m].")
    g.add_argument("--ylim", type=float, nargs=2, default=None,
                   help="Override y-axis limits (ymin ymax) [m].")
    g.add_argument("--trail_length", type=int, default=0,
                   help="Max seeding batches to keep. 0 = unlimited.")

    g = p.add_argument_group("Output")
    g.add_argument("--out_subdir", type=str, default="particle_images",
                   help="Sub-directory inside sim_dir for PNG output.")
    g.add_argument("--device", type=str, default="cpu",
                   help="PyTorch device: 'cuda' or 'cpu'.")

    return p


def main():
    args = build_parser().parse_args()
    kw = {}
    for k, v in vars(args).items():
        if isinstance(v, list):
            kw[k] = tuple(v)
        elif k in ("xlim", "ylim") and v is not None:
            kw[k] = tuple(v)
        else:
            kw[k] = v
    cfg = ParticlePlotConfig(**kw)
    {"2d": run_topview, "3d_topview": run_topview, "3d_full": run_3d_full}[cfg.mode](cfg)


if __name__ == "__main__":
    main()





