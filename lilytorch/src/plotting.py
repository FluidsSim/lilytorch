import h5py
import logging
import os
import threading

import matplotlib
import torch
# matplotlib.use("Agg")          # non-interactive backend — thread-safe, no GUI
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
from matplotlib import cm
from scipy.interpolate import griddata

# VTK / PyVista is **not** thread-safe: the vtkFreeTypeTools singleton,
# OpenGL contexts, and internal caches are shared across threads.
# Concurrent Plotter instances cause heap corruption ("corrupted size
# vs. prev_size") and FreeType failures.  Serialise all VTK work.
_vtk_lock = threading.Lock()

matplotlib.rc('font', **{"size": 20})
plt.rcParams["figure.figsize"] = (15, 15)


def _save_figure(save_path, name, iteration, fmt="png", fig=None):
    """Save a figure into *save_path/name/name_ITER.fmt*.

    When *fig* is given the save is fully thread-safe (no pyplot global
    state involved).  Falls back to ``plt.gcf()`` for legacy callers.
    """
    if fig is None:
        fig = plt.gcf()
    folder = f"{save_path}/{name}"
    os.makedirs(folder, exist_ok=True)
    fig.savefig(
        f"{folder}/{name}_{iteration:06d}.{fmt}",
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
        edgecolor="none",
        dpi=150,
    )
    # If the figure is managed by pyplot, close it there;
    # otherwise just delete the canvas reference so it can be GC'd.
    try:
        plt.close(fig)
    except Exception:
        pass


def _vibrant_rdbu():
    """
    Highly saturated red–blue diverging colormap inspired by the
    Tekinalp et al. (2024) vortex-method visualisations
    (deep vivid blue ↔ black centre ↔ deep vivid red).
    """
    from matplotlib.colors import LinearSegmentedColormap
    cdict = {
        "red":   [(0.0, 0.05, 0.05), (0.35, 0.10, 0.10),
                  (0.5, 0.0,  0.0),  (0.65, 0.85, 0.85),
                  (1.0, 1.0,  1.0)],
        "green": [(0.0, 0.20, 0.20), (0.35, 0.10, 0.10),
                  (0.5, 0.0,  0.0),  (0.65, 0.08, 0.08),
                  (1.0, 0.15, 0.15)],
        "blue":  [(0.0, 1.0,  1.0),  (0.35, 0.85, 0.85),
                  (0.5, 0.0,  0.0),  (0.65, 0.10, 0.10),
                  (1.0, 0.05, 0.05)],
    }
    return LinearSegmentedColormap("vibrant_rdbu", cdict, N=512)

VIBRANT_RDBU = _vibrant_rdbu()
MUJOCO_DEFAULT_GEOM_RGBA = (0.5, 0.5, 0.5, 1.0)


def _resolve_body_color(body, index, body_colors, body_default_color):
    """Return a matplotlib-compatible RGBA tuple for one body.

    Priority: explicit *body_colors[index]* → ``body.mujoco_rgba``
    → *body_default_color* → MuJoCo's default geom grey.
    """
    from matplotlib.colors import to_rgba
    if body_colors is not None and index < len(body_colors):
        return to_rgba(body_colors[index])
    mj = getattr(body, "mujoco_rgba", None)
    if mj is not None:
        # MuJoCo rgba is [R, G, B, A] in 0‥1
        return tuple(float(c) for c in mj[:4])
    if body_default_color is not None:
        return to_rgba(body_default_color)
    return MUJOCO_DEFAULT_GEOM_RGBA


def build_body_mu0_rgba(
    bodies,
    grid_shape,
    eps,
    *,
    body_colors=None,
    body_default_color=None,
    sdf_vals=None,           # (B, *grid_shape) numpy array – batched per-body
                             # SDFs from comp.sdf_vals.  Used when individual
                             # body.sdf_val is not available (BDIMhandler path).
    slice_axis=None,         # None for 2-D; "xy", "xz", "yz" for 3-D slices
    slice_index=None,        # index along the sliced axis
):
    """Build an ``(Nx, Ny, 4)`` RGBA overlay image for body interiors.

    Each body's cell-centred SDF is converted to a smoothed Heaviside
    *mu0* (0 inside body → 1 in fluid).  The body interior
    ``(1 - mu0)`` is tinted with the body's MuJoCo / user colour.
    Where multiple bodies overlap, the deepest (smallest SDF) wins.

    SDF source priority (per body *bi*):

    1. ``sdf_vals[bi]``  – batched tensor from ``comp.sdf_vals``
    2. ``body.sdf_val``  – set by ``BodyAnalytical.update()``

    If neither is available the body is skipped.

    Parameters
    ----------
    bodies : list
        Body objects (used for colour resolution and optional ``sdf_val``).
    grid_shape : tuple
        ``(Nx, Ny)`` for 2-D or ``(Nx, Ny, Nz)`` for 3-D before slicing.
    eps : float
        BDIM smoothing half-width (same as the solver uses).
    body_colors, body_default_color
        Colour overrides (same semantics as ``plot_field_2d``).
    sdf_vals : np.ndarray, optional
        ``(B, Nx, Ny)`` or ``(B, Nx, Ny, Nz)`` pre-computed per-body SDF
        array (already on CPU as numpy).  Preferred over ``body.sdf_val``.
    slice_axis, slice_index
        For 3-D fields: which plane ("xy", "xz", "yz") and at what index.

    Returns
    -------
    rgba : np.ndarray, shape ``(Nx', Ny', 4)``, float32 in [0, 1]
        Ready for ``ax.imshow(rgba.transpose(1, 0, 2) …)``.
        Alpha is ``1-mu0`` inside bodies, 0 in fluid.
    """
    if bodies is None or len(bodies) == 0:
        return None

    Nx, Ny = grid_shape[0], grid_shape[1]

    # Pre-allocate correctly based on dimensionality
    if slice_axis is None:
        rgba = np.zeros((Nx, Ny, 4), dtype=np.float32)
        best_sdf = np.full((Nx, Ny), np.inf, dtype=np.float32)
    else:
        Nz = grid_shape[2]
        if slice_axis == "xy":
            s2d = (Nx, Ny)
        elif slice_axis == "xz":
            s2d = (Nx, Nz)
        elif slice_axis == "yz":
            s2d = (Ny, Nz)
        else:
            return None
        rgba = np.zeros((*s2d, 4), dtype=np.float32)
        best_sdf = np.full(s2d, np.inf, dtype=np.float32)

    for bi, body in enumerate(bodies):
        # ---- obtain per-body SDF ----
        sdf_np = None
        # Priority 1: batched sdf_vals from comp.sdf_vals
        if sdf_vals is not None and bi < sdf_vals.shape[0]:
            sdf_np = sdf_vals[bi].astype(np.float32)
        else:
            # Priority 2: individual body.sdf_val
            sdf_raw = getattr(body, "sdf_val", None)
            if sdf_raw is not None:
                sdf_np = np.asarray(
                    sdf_raw.detach().cpu().numpy() if hasattr(sdf_raw, "detach") else sdf_raw,
                    dtype=np.float32,
                )
        if sdf_np is None:
            continue

        # Slice for 3-D
        if slice_axis is not None and sdf_np.ndim == 3:
            if slice_axis == "xy":
                sdf_np = sdf_np[:, :, slice_index]
            elif slice_axis == "xz":
                sdf_np = sdf_np[:, slice_index, :]
            elif slice_axis == "yz":
                sdf_np = sdf_np[slice_index, :, :]

        # Smoothed Heaviside: mu0 ∈ [0,1], 0 = deep inside body, 1 = fluid
        mu0 = np.where(sdf_np >= eps, 1.0, np.where(
            sdf_np <= -eps, 0.0,
            0.5 * (1.0 + sdf_np / eps + np.sin(np.pi * sdf_np / eps) / np.pi)
        )).astype(np.float32)
        body_alpha = 1.0 - mu0   # body interior → 1, fluid → 0

        colour = _resolve_body_color(body, bi, body_colors, body_default_color)

        # Only overwrite cells where this body is deeper (smaller SDF)
        deeper = sdf_np < best_sdf
        inside = deeper & (body_alpha > 0)
        if not np.any(inside):
            continue
        best_sdf[inside] = sdf_np[inside]
        rgba[inside, 0] = colour[0]
        rgba[inside, 1] = colour[1]
        rgba[inside, 2] = colour[2]
        rgba[inside, 3] = body_alpha[inside] * colour[3]

    if np.max(rgba[:, :, 3]) == 0:
        return None  # nothing to draw
    return rgba


def plot_field_2d(
    field,              # 2-D numpy array (Nx, Ny)
    extent,             # (xmin, xmax, ymin, ymax) for imshow
    name,               # quantity string  (e.g. "curl")
    iteration,          # time-step index
    save_path,          # root folder for output
    *,
    vmin=None,
    vmax=None,
    bodies=None,        # list of body objects (optional, for contour overlay)
    sdf_2d=None,        # 2-D numpy array (Nx, Ny) – SDF; zero-contour drawn as body shape
    cmap=None,
    fmt="png",
    axis_labels=None,   # tuple (xlabel, ylabel); defaults to ("x [m]", "y [m]")
    body_colors=None,   # per-body colour list; each entry is an RGBA/RGB tuple
                        # or matplotlib colour string.  When ``None`` the
                        # function first tries ``body.mujoco_rgba`` (set
                        # from the SDF/MuJoCo visual material), then falls
                        # back to ``body_default_color``, then to MuJoCo's
                        # default geom grey.
    body_default_color=None,  # single fallback colour for all bodies
    body_fill_color=None,     # SDF contourf fill colour (default "#d0d0d0")
    body_mu0_rgba=None,       # pre-built RGBA overlay (Nx, Ny, 4) from
                              # build_body_mu0_rgba(); when provided the
                              # body interior is painted with per-body
                              # MuJoCo colours via mu0 blending.
    body_com_positions=None,  # list of numpy arrays – pre-snapshotted COM
                              # positions (thread-safe); when provided, used
                              # instead of reading body.com_pos (which may
                              # have been overwritten by the next solver step).
):
    """
    Single unified 2-D field plot.

    * Symmetric auto-range when *vmin*/*vmax* are ``None``.
    * Optional body contour scatter overlay – body is always shown even
      when it partially or fully exits the fluid domain.
    * White-background aesthetic with standard red/blue colormap.

    Per-body colouring
    ------------------
    When *body_mu0_rgba* is given (an ``(Nx, Ny, 4)`` RGBA array built by
    ``build_body_mu0_rgba``), each body's interior is painted using the
    smoothed Heaviside ``mu0`` blended with its MuJoCo / user colour.
    This replaces the single-grey SDF ``contourf`` fill.

    Alternatively, *body_colors* can override the contour-scatter colour.
    The function inspects each body's ``mujoco_rgba`` attribute (set from
    the SDF/MuJoCo ``<visual><material><diffuse>`` tag).  If that is also
    absent, *body_default_color* is used.  The ultimate fallback matches
    MuJoCo's default geom grey.
    """
    if cmap is None:
        cmap = "RdBu_r"

    # Resolve to a Colormap object so masked (NaN) cells — e.g. the
    # free-surface air half-space — render as a neutral colour rather than
    # the colormap's extreme.
    try:
        import matplotlib as _mpl
        _base_cmap = _mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    except Exception:  # pragma: no cover - older matplotlib
        from matplotlib import cm as _cm
        _base_cmap = _cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
    cmap = _base_cmap.copy()
    cmap.set_bad(color="#eeeeee")

    field_np = np.asarray(field)

    # ---- symmetric auto-range (only when vmin/vmax not set) ----
    if vmin is None or vmax is None:
        # Ghost cells are already stripped by the caller; ignore non-finite
        # (NaN-masked) cells so the range reflects the physical field only.
        abs_f  = np.abs(field_np)
        finite = abs_f[np.isfinite(abs_f)]
        if finite.size:
            limit = float(np.percentile(finite, 99.5))
            if limit == 0:
                limit = float(finite.max()) or 1.0
        else:
            limit = 1.0
        vmin, vmax = -limit, limit

    # ---- figure size from domain aspect ratio ----
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    scale_f = 25 / max(x_range, y_range)
    fig_w   = max(x_range * scale_f, 4)
    fig_h   = max(y_range * scale_f, 4)

    # Use the OO API (Figure + Canvas) instead of plt.subplots so that
    # plotting is thread-safe when _submit_io offloads to a thread pool.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(fig_w, fig_h))
    FigureCanvasAgg(fig)           # attach renderer so savefig works
    ax  = fig.add_subplot(111)

    # ---- white background styling ----
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # ---- heatmap ----
    im = ax.imshow(
        field_np.T,
        vmin=vmin, vmax=vmax,
        extent=extent,
        origin="lower",
        cmap=cmap,
        aspect="equal",
        interpolation=None,
    )
    cbar = fig.colorbar(im, ax=ax)

    # ---- body SDF fill + outline (always drawn when available) ----
    if sdf_2d is not None:
        sdf_np2 = np.asarray(sdf_2d)
        sdf_min = float(sdf_np2.min())
        if sdf_min < 0.0:
            sx = np.linspace(extent[0], extent[1], sdf_np2.shape[0])
            sy = np.linspace(extent[2], extent[3], sdf_np2.shape[1])
            SY, SX = np.meshgrid(sy, sx)
            if body_mu0_rgba is None or body_fill_color is not None:
                _fill_col = body_fill_color if body_fill_color is not None else "#d0d0d0"
                # Filled interior
                ax.contourf(SX, SY, sdf_np2, levels=[sdf_min, 0.0],
                            colors=[_fill_col], zorder=4)
            # Thick black outline
            ax.contour(SX, SY, sdf_np2, levels=[0.0], colors="black",
                       linewidths=2.5, zorder=5)

    # ---- body interior colouring via mu0 RGBA overlay (on top) ----
    if body_mu0_rgba is not None:
        ax.imshow(
            np.transpose(body_mu0_rgba, (1, 0, 2)),   # (Ny, Nx, 4)
            extent=extent,
            origin="lower",
            aspect="equal",
            interpolation="nearest",
            zorder=6,
        )

    # ---- body contours (always shown, even outside domain) ----
    view_xmin, view_xmax = extent[0], extent[1]
    view_ymin, view_ymax = extent[2], extent[3]

    if bodies is not None:
        for bi, body in enumerate(bodies):
            cnt = getattr(body, "cnt_update", None)
            mask = getattr(body, "mask", None)
            if cnt is not None and mask is not None:
                bx = cnt[0][mask].cpu().numpy()
                by = cnt[1][mask].cpu().numpy()
                # ---- resolve per-body colour ----
                #  priority: explicit body_colors > body.mujoco_rgba > body_default_color > MuJoCo default grey
                _bc = _resolve_body_color(body, bi, body_colors, body_default_color)
                ax.scatter(bx, by, c=[_bc], s=0.5, zorder=5,
                           edgecolors="none", linewidths=0)
                # expand view to include the full robot body
                if len(bx) > 0:
                    view_xmin = min(view_xmin, bx.min())
                    view_xmax = max(view_xmax, bx.max())
                if len(by) > 0:
                    view_ymin = min(view_ymin, by.min())
                    view_ymax = max(view_ymax, by.max())

                com = None
                if body_com_positions is not None and bi < len(body_com_positions):
                    com = body_com_positions[bi]  # already numpy
                else:
                    _cp = getattr(body, "com_pos", None)
                    if _cp is not None:
                        com = _cp.detach().cpu().numpy() if hasattr(_cp, 'cpu') else np.asarray(_cp)
                if com is not None:
                    cx = float(com[0])
                    cy = float(com[1])
                    ax.plot(cx, cy, "o", color="#FF4040",
                            markersize=3, zorder=6)
                    view_xmin = min(view_xmin, cx)
                    view_xmax = max(view_xmax, cx)
                    view_ymin = min(view_ymin, cy)
                    view_ymax = max(view_ymax, cy)

    # ---- standalone COM dots (when no bodies list but positions given) ----
    if bodies is None and body_com_positions is not None:
        for bi, com in enumerate(body_com_positions):
            if com is not None:
                cx = float(com[0])
                cy = float(com[1])
                ax.plot(cx, cy, "o", color="#FF4040",
                        markersize=3, zorder=6)
                view_xmin = min(view_xmin, cx)
                view_xmax = max(view_xmax, cx)
                view_ymin = min(view_ymin, cy)
                view_ymax = max(view_ymax, cy)

    # Set axis limits exactly to the field extent (no padding).
    # If a body extends outside the domain the view expands just enough
    # to keep it visible, but the heatmap fills the full plot area.
    ax.set_xlim(view_xmin, view_xmax)
    ax.set_ylim(view_ymin, view_ymax)

    _xlabel, _ylabel = axis_labels if axis_labels else ("x [m]", "y [m]")
    ax.set_xlabel(_xlabel)
    ax.set_ylabel(_ylabel)
    ax.set_title(name)
    _save_figure(save_path, name, iteration, fmt, fig=fig)


def _nearest_index(coord_1d, target=0.0):
    """Return the array index whose coordinate is closest to *target*."""
    return int(np.argmin(np.abs(coord_1d - target)))


def plot_field_3d_slices(
    field_3d,           # 3-D numpy array or dict of per-slice 3-D arrays
    coords,             # dict with keys "x", "y", "z" – 1-D numpy arrays
    name,               # quantity string
    iteration,
    save_path,
    *,
    vmin=None,
    vmax=None,
    bodies=None,
    sdf_3d=None,        # 3-D numpy array (Nx, Ny, Nz) – SDF; sliced per plane
    slice_indices=None,  # dict {"xy": k, "xz": j, "yz": i}  (None → origin)
    cmap=None,
    fmt="png",
    body_colors=None,
    body_default_color=None,
    body_fill_color=None,
    body_mu0_coloring=False,  # when True, build per-body mu0 RGBA overlays
    body_eps=None,            # BDIM eps for mu0 computation (required if body_mu0_coloring)
    sdf_vals=None,            # (B, Nx, Ny, Nz) numpy – batched per-body SDFs
    body_com_positions=None,  # list of numpy arrays – pre-snapshotted COM positions
):
    """
    For a 3-D field, produce three orthogonal mid-plane slices and save each
    as a separate 2-D plot (reuses `plot_field_2d`).

    Default slice positions pass through the coordinate origin (0, 0, 0)
    rather than the array midpoint, so that slices intersect the body
    even when the domain is not symmetric about the origin.
    """
    if isinstance(field_3d, dict):
        field_xy = np.asarray(field_3d["xy"])
        field_xz = np.asarray(field_3d["xz"])
        field_yz = np.asarray(field_3d["yz"])
    else:
        field_xy = field_xz = field_yz = np.asarray(field_3d)

    Nx, Ny, Nz = field_xy.shape
    x, y, z = coords["x"], coords["y"], coords["z"]
    si = slice_indices or {}
    k_xy = si.get("xy", _nearest_index(z, 0.0))
    j_xz = si.get("xz", _nearest_index(y, 0.0))
    i_yz = si.get("yz", _nearest_index(x, 0.0))

    sdf_np = np.asarray(sdf_3d) if sdf_3d is not None else None

    # ---- build per-body mu0 RGBA overlays for each slice ----
    mu0_xy = mu0_xz = mu0_yz = None
    if body_mu0_coloring and bodies is not None and body_eps is not None:
        mu0_xy = build_body_mu0_rgba(
            bodies, (Nx, Ny, Nz), body_eps,
            body_colors=body_colors, body_default_color=body_default_color,
            sdf_vals=sdf_vals,
            slice_axis="xy", slice_index=k_xy,
        )
        mu0_xz = build_body_mu0_rgba(
            bodies, (Nx, Ny, Nz), body_eps,
            body_colors=body_colors, body_default_color=body_default_color,
            sdf_vals=sdf_vals,
            slice_axis="xz", slice_index=j_xz,
        )
        mu0_yz = build_body_mu0_rgba(
            bodies, (Nx, Ny, Nz), body_eps,
            body_colors=body_colors, body_default_color=body_default_color,
            sdf_vals=sdf_vals,
            slice_axis="yz", slice_index=i_yz,
        )

    # Half-cell offset: imshow `extent` is the outer pixel boundary, not
    # the cell centre, so we expand by h/2 on each side.
    hx = 0.5 * float(x[1] - x[0]) if len(x) > 1 else 0.0
    hy = 0.5 * float(y[1] - y[0]) if len(y) > 1 else 0.0
    hz = 0.5 * float(z[1] - z[0]) if len(z) > 1 else 0.0

    # ---- XY slice (fixed z) ----
    extent_xy = (float(x[0]) - hx, float(x[-1]) + hx,
                 float(y[0]) - hy, float(y[-1]) + hy)
    sdf_xy = sdf_np[:, :, k_xy] if sdf_np is not None else None
    plot_field_2d(
        field_xy[:, :, k_xy], extent_xy, f"{name}_xy_k{k_xy}",
        iteration, save_path,
        vmin=vmin, vmax=vmax, bodies=bodies, sdf_2d=sdf_xy, cmap=cmap, fmt=fmt,
        axis_labels=("x [m]", "y [m]"),
        body_colors=body_colors, body_default_color=body_default_color,
        body_fill_color=body_fill_color, body_mu0_rgba=mu0_xy,
        body_com_positions=body_com_positions,
    )

    # ---- XZ slice (fixed y) ----
    extent_xz = (float(x[0]) - hx, float(x[-1]) + hx,
                 float(z[0]) - hz, float(z[-1]) + hz)
    sdf_xz = sdf_np[:, j_xz, :] if sdf_np is not None else None
    # For XZ slice, remap COM: axis0=x (index 0), axis1=z (index 2)
    _com_xz = None
    if body_com_positions is not None:
        _com_xz = [
            np.array([c[0], c[2]]) if c is not None else None
            for c in body_com_positions
        ]
    plot_field_2d(
        field_xz[:, j_xz, :], extent_xz, f"{name}_xz_j{j_xz}",
        iteration, save_path,
        vmin=vmin, vmax=vmax, bodies=None, sdf_2d=sdf_xz, cmap=cmap, fmt=fmt,
        axis_labels=("x [m]", "z [m]"),
        body_colors=body_colors, body_default_color=body_default_color,
        body_fill_color=body_fill_color, body_mu0_rgba=mu0_xz,
        body_com_positions=_com_xz,
    )

    # ---- YZ slice (fixed x) ----
    extent_yz = (float(y[0]) - hy, float(y[-1]) + hy,
                 float(z[0]) - hz, float(z[-1]) + hz)
    sdf_yz = sdf_np[i_yz, :, :] if sdf_np is not None else None
    # For YZ slice, remap COM: axis0=y (index 1), axis1=z (index 2)
    _com_yz = None
    if body_com_positions is not None:
        _com_yz = [
            np.array([c[1], c[2]]) if c is not None else None
            for c in body_com_positions
        ]
    plot_field_2d(
        field_yz[i_yz, :, :], extent_yz, f"{name}_yz_i{i_yz}",
        iteration, save_path,
        vmin=vmin, vmax=vmax, bodies=None, sdf_2d=sdf_yz, cmap=cmap, fmt=fmt,
        axis_labels=("y [m]", "z [m]"),
        body_colors=body_colors, body_default_color=body_default_color,
        body_fill_color=body_fill_color, body_mu0_rgba=mu0_yz,
        body_com_positions=_com_yz,
    )


def plot_field_3d(
    field_3d,           # 3-D numpy array (Nx, Ny, Nz)
    coords,             # dict  {"x": 1d, "y": 1d, "z": 1d}
    name,
    iteration,
    save_path,
    *,
    sdf_3d=None,        # optional SDF array for body isosurface (same shape)
    bodies=None,        # optional body list; used with sdf_vals for native colouring
    sdf_vals=None,      # optional (B, Nx, Ny, Nz) per-body SDF stack
    body_colors=None,
    body_default_color=None,
    iso_value=None,     # fixed isosurface threshold (overrides auto when set)
    iso_fraction=0.15,  # threshold = iso_fraction * max(|smoothed field|)
    smooth_sigma=0,   # Gaussian smoothing (in grid-cells) before isosurface extraction
    crop_boundary=0,    # number of cells to crop from each domain face before rendering
    realtime=None,      # physical simulation time in seconds (for annotation)
    window_size=(3840, 2160),
    image_scale=2,          # multiplier on window_size for the saved image
    fmt="png",
):
    """
    Render a 3-D field as isosurfaces with an optional opaque body surface
    (SDF = 0), using PyVista off-screen.

    For fields that contain both positive and negative values (e.g. vorticity
    component, pressure), dual ±threshold isosurfaces are drawn (red/blue).
    For non-negative fields (e.g. ``|curl|`` magnitude), a single isosurface is
    drawn at `threshold` coloured by value.

    ``crop_boundary`` removes cells from each face so that zero-padded
    boundary layers (from the vorticity stencil) and ghost-cell artifacts
    never reach the isosurface algorithm.

    Threshold strategy
    ------------------
    Vorticity fields from backward-difference stencils are dominated by
    near-zero noise: the 85th-percentile of ``|ω|`` can be 100,000× smaller
    than the peak.  A percentile-based threshold would trace isosurfaces
    through this noise.  Instead we use a **fraction of the peak amplitude**
    of the *smoothed* field (default 15 %).  This cleanly selects the
    physical vortex structures near the body.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("[plot_field_3d] pyvista not installed – skipping 3D render.")
        return

    # Serialise all VTK/PyVista work – VTK is not thread-safe.
    with _vtk_lock:
        _plot_field_3d_locked(
            pv, field_3d, coords, name, iteration, save_path,
            sdf_3d=sdf_3d, bodies=bodies, sdf_vals=sdf_vals,
            body_colors=body_colors, body_default_color=body_default_color,
            iso_value=iso_value, iso_fraction=iso_fraction,
            smooth_sigma=smooth_sigma, crop_boundary=crop_boundary,
            realtime=realtime,
            window_size=window_size, image_scale=image_scale, fmt=fmt,
        )


def _plot_field_3d_locked(
    pv, field_3d, coords, name, iteration, save_path, *,
    sdf_3d=None, bodies=None, sdf_vals=None,
    body_colors=None, body_default_color=None,
    iso_value=None, iso_fraction=0.15,
    smooth_sigma=0, crop_boundary=0,
    realtime=None,
    window_size=(3840, 2160), image_scale=2, fmt="png",
):
    """Inner implementation of plot_field_3d – must be called under _vtk_lock."""
    pv.OFF_SCREEN = True

    field_np = np.asarray(field_3d, dtype=np.float64)
    x = np.asarray(coords["x"])
    y = np.asarray(coords["y"])
    z = np.asarray(coords["z"])

    # ---- Crop boundary layers ----
    # The vorticity stencil ([2:-2]) zero-pads the outermost cells.
    # The sharp 0→nonzero transition creates spurious isosurfaces.
    # Cropping removes these cells from the rendering domain.
    c = int(crop_boundary)
    if c > 0:
        field_np = field_np[c:-c, c:-c, c:-c]
        x = x[c:-c]
        y = y[c:-c]
        z = z[c:-c]
        if sdf_3d is not None:
            sdf_3d = np.asarray(sdf_3d)[c:-c, c:-c, c:-c]
        if sdf_vals is not None:
            sdf_vals = np.asarray(sdf_vals)[:, c:-c, c:-c, c:-c]

    # ---- Gaussian smoothing suppresses grid-scale noise in derivative
    #      fields (vorticity) that would otherwise produce jagged,
    #      oscillatory isosurfaces.  sigma is in grid-cell units. ----
    if smooth_sigma and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        field_np = gaussian_filter(field_np, sigma=smooth_sigma)

    grid = pv.RectilinearGrid(x, y, z)
    grid.point_data[name] = field_np.flatten(order="F")

    # ---- classify sign character of the field ----
    fmin, fmax = float(field_np.min()), float(field_np.max())
    has_neg = fmin < -1e-12
    has_pos = fmax >  1e-12
    is_bipolar = has_neg and has_pos   # e.g. ωz, pressure
    # For non-negative fields like |curl|, we only draw a positive iso.

    # ---- determine isosurface threshold ----
    # Use the peak amplitude of the (already smoothed) field, excluding
    # the body interior when an SDF is available.
    if sdf_3d is not None:
        mask = np.asarray(sdf_3d).ravel() > 0
        abs_f = np.abs(field_np.ravel()[mask])
    else:
        abs_f = np.abs(field_np.ravel())
    peak = float(abs_f.max()) if abs_f.size > 0 else 0.0

    if iso_value is not None and iso_value > 0:
        # Fixed threshold requested by caller
        threshold = float(iso_value)
    elif peak < 1e-10:
        threshold = None
    else:
        threshold = iso_fraction * peak

    # ---- helper: remove tiny disconnected mesh fragments ----
    _min_cell_frac = 0.01   # drop components < 1% of total cells

    def _clean_iso(mesh, min_frac=_min_cell_frac):
        """Keep only connected components whose *cell* count exceeds
        *min_frac* of the total.  Strips noise speckle that survived
        Gaussian smoothing and the amplitude threshold."""
        if mesh is None or mesh.n_cells == 0:
            return mesh
        try:
            conn = mesh.connectivity(extraction_mode="all")
            region_ids = conn.cell_data.get("RegionId", None)
            if region_ids is None:
                return mesh
            from collections import Counter
            counts = Counter(region_ids)
            total = mesh.n_cells
            min_cells = max(1, int(total * min_frac))
            keep_ids = {rid for rid, cnt in counts.items()
                        if cnt >= min_cells}
            if len(keep_ids) == len(counts):
                return mesh          # nothing to remove
            # Build cell-based boolean mask and extract
            keep_mask = np.isin(region_ids, list(keep_ids))
            return conn.extract_cells(np.where(keep_mask)[0])
        except Exception:
            return mesh   # on any error, fall back to unfiltered mesh

    # ---- build scene (dark theme, Tekinalp et al. 2024 style) ----
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background("black")
    try:
        pl.enable_anti_aliasing("ssaa")   # screen-space AA – best quality
    except Exception:
        pass

    if threshold is not None:
        if is_bipolar:
            # Dual isosurfaces: +threshold (vivid red) and -threshold (vivid blue)
            try:
                iso_pos = _clean_iso(grid.contour([threshold], scalars=name))
                if iso_pos.n_points > 0:
                    pl.add_mesh(iso_pos, color="#FF2020", opacity=0.75,
                                smooth_shading=True, specular=0.5,
                                label=f"+{threshold:.2e}")
            except Exception:
                pass
            try:
                iso_neg = _clean_iso(grid.contour([-threshold], scalars=name))
                if iso_neg.n_points > 0:
                    pl.add_mesh(iso_neg, color="#1E90FF", opacity=0.75,
                                smooth_shading=True, specular=0.5,
                                label=f"\u2212{threshold:.2e}")
            except Exception:
                pass
        else:
            # Single isosurface at threshold, coloured hot-orange
            try:
                iso = _clean_iso(grid.contour([threshold], scalars=name))
                if iso.n_points > 0:
                    pl.add_mesh(iso, color="#FF6020", opacity=0.8,
                                smooth_shading=True, specular=0.5,
                                label=f"{name}={threshold:.2e}")
            except Exception:
                pass

    # ---- body surface from SDF ----
    # Prefer the union SDF used by the 2-D slice overlays so that the 3-D
    # body surface matches the slice plots for multi-link animats.
    body_surface_drawn = False
    if sdf_3d is not None:
        sdf_np = np.asarray(sdf_3d)
        grid.point_data["sdf"] = sdf_np.flatten(order="F")
        try:
            body_surf = grid.contour([0.0], scalars="sdf")
            if body_surf.n_points > 0:
                pl.add_mesh(body_surf, color="white", opacity=0.95,
                            smooth_shading=True, specular=0.6)
                body_surface_drawn = True
        except Exception:
            pass

    if not body_surface_drawn and bodies is not None and sdf_vals is not None:
        sdf_vals_np = np.asarray(sdf_vals, dtype=np.float64)
        for bi, body in enumerate(bodies):
            if bi >= sdf_vals_np.shape[0]:
                break
            body_sdf = sdf_vals_np[bi]
            if body_sdf.shape != field_np.shape:
                continue
            if float(body_sdf.min()) >= 0.0 or float(body_sdf.max()) <= 0.0:
                continue
            try:
                body_grid = pv.RectilinearGrid(x, y, z)
                body_grid.point_data["sdf"] = body_sdf.flatten(order="F")
                body_surf = body_grid.contour([0.0], scalars="sdf")
                if body_surf.n_points > 0:
                    colour = _resolve_body_color(body, bi, body_colors, body_default_color)
                    pl.add_mesh(
                        body_surf,
                        color=colour[:3],
                        opacity=max(0.0, min(1.0, 0.95 * colour[3])),
                        smooth_shading=True,
                        specular=0.6,
                    )
                    body_surface_drawn = True
            except Exception:
                pass

    # ---- domain outline (white tank lines) ----
    pl.add_mesh(grid.outline(), color="white", line_width=4.0, opacity=1.0)

    # ---- camera helpers ----
    bds = grid.bounds
    cx = (bds[0] + bds[1]) / 2
    cy = (bds[2] + bds[3]) / 2
    cz = (bds[4] + bds[5]) / 2
    Lmax = max(bds[1] - bds[0], bds[3] - bds[2], bds[5] - bds[4])
    focal = (cx, cy, cz)

    # ---- text annotation with field range + threshold ----
    if threshold is not None:
        info_lines = [f"{name}"]
        if realtime is not None:
            info_lines.append(f"t = {float(realtime):.4f} s")
        info_lines.append(f"range: [{fmin:.2e}, {fmax:.2e}]")
        if is_bipolar:
            info_lines.append(f"iso: ±{threshold:.2e}")
        else:
            info_lines.append(f"iso: {threshold:.2e}")
        info_text = "\n".join(info_lines)
        try:
            pl.add_text(
                info_text,
                position="upper_left",
                font_size=60,
                color="white",
                shadow=True,
            )
        except Exception:
            pass

    # ---- camera  (isometric, looking from upstream-above) ----
    cam_pos = (cx - 0.8 * Lmax, cy - 0.6 * Lmax, cz + 1.5 * Lmax)
    pl.camera_position = [cam_pos, focal, (0, 0, 1)]
    pl.camera.parallel_projection = True
    # Use the diagonal of the bounding box so elongated domains fill the frame
    Lx = bds[1] - bds[0]
    Ly = bds[3] - bds[2]
    Lz = bds[5] - bds[4]
    diag = (Lx**2 + Ly**2 + Lz**2) ** 0.5
    pl.camera.parallel_scale = diag * 0.38

    # ---- save high-resolution screenshot ----
    out_dir = os.path.join(save_path, f"{name}_3d")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_3d_{iteration:06d}.{fmt}")
    pl.screenshot(out_path, scale=image_scale)
    pl.close()


logger = logging.getLogger(__name__)


class PlottingMixin:
    """Mixin providing unified plotting and saving for FluidSolver (2-D and 3-D)."""

    # ------------------------------------------------------------------
    #   Default plot specifications.
    # Each entry: (name, field_lambda, vmin, vmax, body_contours)
    # field_lambda receives (solver, u, v, p, w_vel) and returns a CPU tensor/array.
    # vmin/vmax per-spec:  None or "yaml" → use the global output.vmin/vmax
    #                      float          → fixed limit for this field
    # The YAML output.vmin/vmax can be a number (fixed) or "auto" (auto-scale).
    # ``output.plot_specs`` can override these defaults with a list of
    # field names or dicts such as {"name": "pressure", "vmin": "auto"}.
    DEFAULT_PLOT_SPECS = [
        ("curl",       lambda s, u, v, p, w: (
            s.vorticity(u, v, w).cpu() if s.ndim == 2
            else PlottingMixin._curl_slice_fields(s, u, v, w)
        ), None, None, True),
        # ("v",          lambda s, u, v, p, w: v.cpu(), "auto", "auto", True),
        ("pressure",   lambda s, u, v, p, w: p.cpu(), "auto", "auto", True),
        # ("divergence", lambda s, u, v, p, w: s.divergence(u, v, w).cpu(),None, None, False),
    ]

    # Additional specs used only for 3-D isosurface rendering.
    # ``output.iso_3d_specs`` can override these defaults with field names
    # or dicts such as {"name": "omega_mag", "iso_value": 5.0}. When
    # ``output.iso_3d_value`` is set, it becomes the default threshold for
    # all configured 3-D isosurface fields; otherwise the automatic peak-
    # fraction thresholding in plot_field_3d is used.
    DEFAULT_3D_ISO_SPECS = [
        # ("omega_x",    lambda s, u, v, p, w: PlottingMixin._cached_vort(s, u, v, w)["omega_x"].cpu(), None, True),
        # ("omega_y",    lambda s, u, v, p, w: PlottingMixin._cached_vort(s, u, v, w)["omega_y"].cpu(), None, True),
        # ("omega_z",    lambda s, u, v, p, w: PlottingMixin._cached_vort(s, u, v, w)["omega_z"].cpu(), None, True),
        ("omega_mag",  lambda s, u, v, p, w: PlottingMixin._cached_vort(s, u, v, w)["omega_mag"].cpu(), None, True),
        ("vel_mag",    lambda s, u, v, p, w: PlottingMixin._vel_mag(s, u, v, w).cpu(),                  None, True),
        # ("pressure",   lambda s, u, v, p, w: p.cpu(),                                                  None, True),
    ]

    @staticmethod
    def _cached_vort(s, u, v, w):
        """Compute vorticity_components once per frame and cache on solver."""
        if not hasattr(s, "_vort_cache") or s._vort_cache_id != id(u):
            s._vort_cache = s.vorticity_components(u, v, w)
            s._vort_cache_id = id(u)
        return s._vort_cache

    @staticmethod
    def _curl_slice_fields(s, u, v, w):
        """Return the out-of-plane curl component for each orthogonal slice.

        XY uses ``omega_z`` as usual. For XZ, the 2-D-like scalar curl is
        ``dw/dx - du/dz = -omega_y``. For YZ, it is ``dw/dy - dv/dz = omega_x``.
        """
        vort = PlottingMixin._cached_vort(s, u, v, w)
        return {
            "xy": vort["omega_z"].cpu(),
            "xz": (-vort["omega_y"]).cpu(),
            "yz": vort["omega_x"].cpu(),
        }

    @staticmethod
    def _vel_mag(s, u, v, w):
        """Velocity magnitude (stagger offset negligible for visualisation)."""
        return (u**2 + v**2 + w**2).sqrt()

    @classmethod
    def _plot_spec_registry(cls):
        registry = {spec[0]: spec for spec in cls.DEFAULT_PLOT_SPECS}
        registry.update({
            "v": (
                "v",
                lambda s, u, v, p, w: v.cpu(),
                "auto",
                "auto",
                True,
            ),
            "divergence": (
                "divergence",
                lambda s, u, v, p, w: s.divergence(u, v, w).cpu(),
                None,
                None,
                False,
            ),
        })
        return registry

    @classmethod
    def _iso_3d_spec_registry(cls):
        registry = {spec[0]: spec for spec in cls.DEFAULT_3D_ISO_SPECS}
        registry.update({
            "omega_x": (
                "omega_x",
                lambda s, u, v, p, w: PlottingMixin._cached_vort(s, u, v, w)["omega_x"].cpu(),
                None,
                True,
            ),
            "omega_y": (
                "omega_y",
                lambda s, u, v, p, w: PlottingMixin._cached_vort(s, u, v, w)["omega_y"].cpu(),
                None,
                True,
            ),
            "omega_z": (
                "omega_z",
                lambda s, u, v, p, w: PlottingMixin._cached_vort(s, u, v, w)["omega_z"].cpu(),
                None,
                True,
            ),
            "pressure": (
                "pressure",
                lambda s, u, v, p, w: p.cpu(),
                None,
                True,
            ),
        })
        return registry

    @staticmethod
    def _normalize_iso_threshold(value):
        return None if value == "auto" else value

    def _resolve_plot_specs(self, plot_specs_cfg):
        if plot_specs_cfg is None:
            return list(self.DEFAULT_PLOT_SPECS)
        if isinstance(plot_specs_cfg, str):
            plot_specs_cfg = [plot_specs_cfg]

        registry = self._plot_spec_registry()
        resolved = []
        for entry in plot_specs_cfg:
            if isinstance(entry, str):
                name = entry
                overrides = {}
            elif isinstance(entry, dict):
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Each output.plot_specs entry must define a non-empty 'name'.")
                overrides = entry
            else:
                raise TypeError(
                    "output.plot_specs entries must be strings or dicts with a 'name' key."
                )

            if name not in registry:
                raise ValueError(
                    f"Unknown plot field '{name}'. Available fields: {sorted(registry)}"
                )

            base = registry[name]
            show_body = base[4] if "show_body" not in overrides else bool(overrides["show_body"])
            resolved.append((
                name,
                base[1],
                overrides.get("vmin", base[2]),
                overrides.get("vmax", base[3]),
                show_body,
            ))
        return resolved

    def _resolve_iso_3d_specs(self, iso_specs_cfg, *, global_iso_value=None):
        if self.ndim != 3:
            return []

        global_iso_value = self._normalize_iso_threshold(global_iso_value)
        if iso_specs_cfg is None:
            if global_iso_value is None:
                return list(self.DEFAULT_3D_ISO_SPECS)
            iso_specs_cfg = [spec[0] for spec in self.DEFAULT_3D_ISO_SPECS]
        if isinstance(iso_specs_cfg, str):
            iso_specs_cfg = [iso_specs_cfg]

        registry = self._iso_3d_spec_registry()
        resolved = []
        for entry in iso_specs_cfg:
            if isinstance(entry, str):
                name = entry
                overrides = {}
            elif isinstance(entry, dict):
                name = entry.get("name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Each output.iso_3d_specs entry must define a non-empty 'name'.")
                overrides = entry
            else:
                raise TypeError(
                    "output.iso_3d_specs entries must be strings or dicts with a 'name' key."
                )

            if name not in registry:
                raise ValueError(
                    f"Unknown 3D isosurface field '{name}'. Available fields: {sorted(registry)}"
                )

            base = registry[name]
            iso_value = overrides.get("iso_value")
            if "iso_value" not in overrides:
                iso_value = global_iso_value if global_iso_value is not None else base[2]
            iso_value = self._normalize_iso_threshold(iso_value)

            if len(base) > 3:
                show_body = base[3] if "show_body" not in overrides else bool(overrides["show_body"])
                resolved.append((name, base[1], iso_value, show_body))
            else:
                resolved.append((name, base[1], iso_value))
        return resolved

    # Maximum number of pending I/O futures before _submit_io blocks.
    # Each future can capture hundreds of MB of numpy arrays (3-D field
    # snapshots, per-body SDFs, …).  Without a cap the queue grows faster
    # than the 2-worker pool can drain it, causing OOM on large grids.
    _MAX_PENDING_IO = 10

    def _submit_io(self, fn, *args, **kwargs):
        """Submit *fn* to the background I/O pool and track the future."""
        # Reap already-finished futures to avoid unbounded growth
        self._io_futures = [f for f in self._io_futures if not f.done()]
        # Throttle: block until the queue drains below the cap so that
        # captured numpy arrays from old frames can be garbage-collected.
        while len(self._io_futures) >= self._MAX_PENDING_IO:
            # Wait for the oldest pending future to finish
            self._io_futures[0].result()
            self._io_futures = [f for f in self._io_futures if not f.done()]
        fut = self._io_executor.submit(fn, *args, **kwargs)
        self._io_futures.append(fut)
        return fut

    def flush_io(self):
        """Block until all pending background I/O tasks have completed."""
        for fut in self._io_futures:
            fut.result()          # re-raises any exception from the worker
        self._io_futures.clear()

    def _snapshot_body_sdf_vals_for_plots(self, crop_slices=None):
        """Snapshot per-body SDFs on CPU for plot colouring.

        Prefer ``composite_body._sdf_sparse`` when available so plotting
        matches the current per-step body update path. Fall back to the
        legacy dense ``composite_body.sdf_vals`` stack when sparse storage
        is unavailable.
        """
        comp = getattr(self, "composite_body", None)
        bodies = getattr(comp, "bodies", None) if comp is not None else None
        if comp is None or bodies is None or len(bodies) == 0:
            return None

        ndim = len(self.grid_shape)
        if crop_slices is None:
            crop_slices = (slice(None),) * ndim
        elif not isinstance(crop_slices, tuple):
            crop_slices = (crop_slices,) * ndim

        sdf_sparse = getattr(comp, "_sdf_sparse", None)
        if sdf_sparse is not None:
            full_shape = tuple(int(n) for n in self.grid_shape)
            crop_bounds = []
            cropped_shape = []
            for axis, sl in enumerate(crop_slices):
                start = 0 if sl.start is None else (full_shape[axis] + sl.start if sl.start < 0 else sl.start)
                stop = full_shape[axis] if sl.stop is None else (full_shape[axis] + sl.stop if sl.stop < 0 else sl.stop)
                start = max(0, min(full_shape[axis], start))
                stop = max(start, min(full_shape[axis], stop))
                crop_bounds.append((start, stop))
                cropped_shape.append(stop - start)

            sdf_vals_np = np.full((len(bodies), *cropped_shape), 1e4, dtype=np.float32)

            for body_i, sparse_entry in enumerate(sdf_sparse):
                if sparse_entry is None:
                    continue
                aabb, sdf_body = sparse_entry
                sdf_body_np = np.asarray(
                    sdf_body.detach().cpu().numpy() if hasattr(sdf_body, "detach") else sdf_body,
                    dtype=np.float32,
                )

                if aabb is None:
                    sdf_vals_np[body_i] = np.array(sdf_body_np[crop_slices], dtype=np.float32, copy=True)
                    continue

                axis_ranges = [
                    (aabb[2 * axis], aabb[2 * axis + 1])
                    for axis in range(ndim)
                ]
                dst_slices = []
                src_slices = []
                intersects = True
                for axis, (body_lo, body_hi) in enumerate(axis_ranges):
                    crop_lo, crop_hi = crop_bounds[axis]
                    src_lo = max(body_lo, crop_lo)
                    src_hi = min(body_hi, crop_hi)
                    if src_lo >= src_hi:
                        intersects = False
                        break
                    dst_slices.append(slice(src_lo - crop_lo, src_hi - crop_lo))
                    src_slices.append(slice(src_lo - body_lo, src_hi - body_lo))

                if not intersects:
                    continue

                sdf_vals_np[(body_i, *dst_slices)] = sdf_body_np[tuple(src_slices)]

            return sdf_vals_np

        sdf_vals = getattr(comp, "sdf_vals", None)
        if sdf_vals is None:
            return None
        sdf_vals_np = np.asarray(
            sdf_vals.detach().cpu().numpy() if hasattr(sdf_vals, "detach") else sdf_vals,
            dtype=np.float32,
        )
        return np.array(sdf_vals_np[(slice(None), *crop_slices)], dtype=np.float32, copy=True)

    def plotting_and_saving(self, u, v, p, iteration, *, w_vel=None, check_termination=True):
        """
        Unified plotting + data saving for 2-D and 3-D.
        Replaces the old ``plotting_debug`` and ``plotting_saving`` methods.

        Plotting and saving are offloaded to a background thread so the
        solver loop is not blocked by synchronous disk I/O.
        """
        if iteration % self.save_every != 0:
            if check_termination:
                return self.check_termination(iteration, u, v, p)
            return False

        # Invalidate vorticity cache so each frame recomputes it from the
        # current velocity tensors.  The cache key is id(u), which stays
        # constant when project() modif ies u in-place (kernel mode), so
        # without this reset the stale step-0 zeros would be returned
        # for every frame.
        self._vort_cache_id = -1

        # ---- snapshot tensors to CPU numpy *once* (still on main thread
        #      so the GPU transfer is overlapped with the previous kernel)
        # We clone/detach to decouple from the live computation graph.

        # ---- frame plots ----
        if self.save_frames:
            specs = getattr(self, "plot_specs", self.DEFAULT_PLOT_SPECS)
            bodies = getattr(self.composite_body, "bodies", None) if hasattr(self, "composite_body") else None

            # Snapshot per-body COM positions to CPU numpy *now* so the
            # background IO thread reads a frozen copy rather than the
            # live tensor (which the next solver step will overwrite).
            _com_positions = None
            if bodies is not None:
                _com_positions = []
                for _b in bodies:
                    _cp = getattr(_b, "com_pos", None)
                    if _cp is not None:
                        _com_positions.append(_cp.detach().cpu().numpy().copy())
                    else:
                        _com_positions.append(None)

            if self.ndim == 2:
                _phys_extent = (self.xmin, self.xmax, self.ymin, self.ymax)
                _crop_2d = (slice(1, -1), slice(1, -1))
                _body_sdf_vals_np = self._snapshot_body_sdf_vals_for_plots(_crop_2d)
                _sdf_2d = None
                if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                    _sdf_2d = self.composite_body.sdf_val.cpu().numpy().copy()[1:-1, 1:-1]
                _body_mu0_rgba = None
                if _body_sdf_vals_np is not None:
                    _body_mu0_rgba = build_body_mu0_rgba(
                        bodies,
                        _body_sdf_vals_np.shape[1:],
                        float(self.eps),
                        sdf_vals=_body_sdf_vals_np,
                    )

                for (name, field_fn, vmin, vmax, show_body) in specs:
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                    field_np = field_np[1:-1, 1:-1]  # strip ghost cells
                    eff_vmin = self.vmin if vmin is None else (None if vmin == "auto" else vmin)
                    eff_vmax = self.vmax if vmax is None else (None if vmax == "auto" else vmax)
                    save_path = self.save_path
                    self._submit_io(
                        plot_field_2d,
                        field_np, _phys_extent,
                        name, iteration, save_path,
                        vmin=eff_vmin, vmax=eff_vmax, bodies=None,
                        sdf_2d=_sdf_2d if show_body else None,
                        body_mu0_rgba=_body_mu0_rgba if show_body else None,
                        body_com_positions=_com_positions if show_body else None,
                    )
            else:
                # Crop ghost cells (index 0 and -1 on each axis) so plots
                # show only the physical domain, not BC-padded boundaries.
                _s = slice(1, -1)  # reusable ghost-cell crop slice
                coords = {
                    "x": self.x.cpu().numpy().copy()[_s],
                    "y": self.y.cpu().numpy().copy()[_s],
                    "z": self.z.cpu().numpy().copy()[_s],
                }
                _body_sdf_vals_np = self._snapshot_body_sdf_vals_for_plots((_s, _s, _s))
                # snapshot SDF for body-shape overlay on 2-D slice plots
                sdf_np = None
                if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                    sdf_np = self.composite_body.sdf_val.cpu().numpy().copy()[_s, _s, _s]
                for (name, field_fn, vmin, vmax, _show_body) in specs:
                    field = field_fn(self, u, v, p, w_vel)
                    if isinstance(field, dict):
                        field_np = {
                            key: (value.detach().cpu().numpy().copy() if hasattr(value, 'detach') else np.array(value))[_s, _s, _s]
                            for key, value in field.items()
                        }
                    else:
                        field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                        field_np = field_np[_s, _s, _s]  # strip ghost cells
                    eff_vmin = self.vmin if vmin is None else (None if vmin == "auto" else vmin)
                    eff_vmax = self.vmax if vmax is None else (None if vmax == "auto" else vmax)
                    save_path = self.save_path
                    _pass_bodies = bodies if _show_body else None
                    self._submit_io(
                        plot_field_3d_slices,
                        field_np, coords,
                        name, iteration, save_path,
                        vmin=eff_vmin, vmax=eff_vmax, bodies=_pass_bodies,
                        sdf_3d=sdf_np,
                        body_mu0_coloring=_show_body and _body_sdf_vals_np is not None,
                        body_eps=float(self.eps) if _body_sdf_vals_np is not None else None,
                        sdf_vals=_body_sdf_vals_np if _show_body else None,
                        body_com_positions=_com_positions if _show_body else None,
                    )

                # ---- HDF5 field export (3-D only, replaces old VTK) ----
                # Saves u, v, w, p, sdf — all derived fields (vorticity,
                # vel_mag, divergence) can be recomputed from these.
                # HDF5 stores the *full* arrays (including ghost cells)
                # so post-processing retains boundary information.
                if self.save:
                    h5_fields = {
                        "u": u.detach().cpu().numpy().copy(),
                        "v": v.detach().cpu().numpy().copy(),
                        "p": p.detach().cpu().numpy().copy(),
                    }
                    if w_vel is not None:
                        h5_fields["w"] = w_vel.detach().cpu().numpy().copy()
                    _sdf_full = None
                    if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                        _sdf_full = self.composite_body.sdf_val.cpu().numpy().copy()
                    if _sdf_full is not None:
                        h5_fields["sdf"] = _sdf_full

                    # Grids only on first snapshot (full coords incl. ghost cells)
                    grids = None
                    if iteration == 0 or not hasattr(self, '_grids_saved'):
                        grids = {
                            "x": self.x.cpu().numpy().copy(),
                            "y": self.y.cpu().numpy().copy(),
                            "z": self.z.cpu().numpy().copy(),
                        }
                        self._grids_saved = True

                    self._submit_io(
                        self._save_fields_h5,
                        h5_fields, grids, iteration,
                    )

                # ---- 3-D isosurface renders ----
                # sdf_np already snapshotted above for slice plots
                iso_specs = getattr(
                    self,
                    "iso_3d_specs",
                    self.DEFAULT_3D_ISO_SPECS if self.ndim == 3 else [],
                )
                for iso_entry in iso_specs:
                    name, field_fn = iso_entry[0], iso_entry[1]
                    iso_thresh = iso_entry[2] if len(iso_entry) > 2 else None
                    if iso_thresh == "vmax":
                        iso_thresh = getattr(self, "vmax", None)
                    field = field_fn(self, u, v, p, w_vel)
                    field_np = field.detach().cpu().numpy().copy() if hasattr(field, 'detach') else np.array(field)
                    field_np = field_np[_s, _s, _s]  # strip ghost cells
                    self._submit_io(
                        plot_field_3d,
                        field_np, coords,
                        name, iteration, self.save_path,
                        sdf_3d=sdf_np,
                        iso_value=iso_thresh,
                        bodies=bodies,
                        sdf_vals=_body_sdf_vals_np,
                        realtime=float(iteration * self.dt_np),
                    )

        # ---- raw data save (2-D) ----
        if self.save and self.ndim == 2:
            h5_fields = {
                "u": u.detach().cpu().numpy().copy(),
                "v": v.detach().cpu().numpy().copy(),
                "p": p.detach().cpu().numpy().copy(),
            }
            # SDF snapshot
            sdf_np = None
            if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
                sdf_np = self.composite_body.sdf_val.cpu().numpy().copy()
                h5_fields["sdf"] = sdf_np

            # Grids only on first snapshot
            grids = None
            if iteration == 0 or not hasattr(self, '_grids_saved'):
                grids = {
                    "x": self.x.cpu().numpy().copy(),
                    "y": self.y.cpu().numpy().copy(),
                }
                self._grids_saved = True

            self._submit_io(
                self._save_fields_h5,
                h5_fields, grids, iteration,
            )

            # 2-D body contours
            if hasattr(self, "composite_body") and hasattr(self.composite_body, "bodies"):
                cnt_arrays = [
                    body.cnt_update.cpu().numpy().copy()
                    for body in self.composite_body.bodies
                    if hasattr(body, "cnt_update")
                ]
                if cnt_arrays:
                    self._submit_io(
                        self._save_contours_h5, cnt_arrays, iteration,
                    )

        if check_termination:
            return self.check_termination(iteration, u, v, p)
        return False

    # keep old name as an alias so call-sites in alternative solver variants still work
    def plotting_debug(self, u, v, p, iteration, check_termination=True):
        return self.plotting_and_saving(u, v, p, iteration, check_termination=check_termination)

    def check_termination(self, iteration, u, v, p):
        # NaN in any velocity or pressure field
        has_nan = (torch.isnan(u).any() or torch.isnan(v).any()
                   or torch.isnan(p).any())
        if iteration == self.nt - 1 or has_nan:
                if has_nan:
                    logger.warning("Termination: NaN detected in velocity/pressure fields")
                else:
                    logger.info("Termination: reached max iterations (%d)", self.nt)
                terminate = True
        else:
            if hasattr(self.composite_body, "com_pos"):
                    terminate = not self.inside(self.composite_body.com_pos)
                    if terminate:
                        logger.warning(
                            "Termination condition met: body exited domain "
                            "(com_pos=%s, domain x=[%s, %s] y=[%s, %s])",
                            self.composite_body.com_pos.tolist(),
                            self.xmin, self.xmax, self.ymin, self.ymax,
                        )
            else:
                terminate = False
        return terminate

    def plotting_saving(self, u, v, p, iteration):
        """Legacy alias — delegates to the unified method."""
        self.plotting_and_saving(u, v, p, iteration, check_termination=False)

    def save_results(self, u, v, p, iteration, *, w_vel=None):
        """Legacy alias — delegates to the unified HDF5 pipeline."""
        if not self.save:
            return
        h5_fields = {
            "u": u.detach().cpu().numpy().copy(),
            "v": v.detach().cpu().numpy().copy(),
            "p": p.detach().cpu().numpy().copy(),
        }
        if w_vel is not None:
            h5_fields["w"] = w_vel.detach().cpu().numpy().copy()
        if hasattr(self, "composite_body") and hasattr(self.composite_body, "sdf_val"):
            h5_fields["sdf"] = self.composite_body.sdf_val.cpu().numpy().copy()
        grids = None
        if iteration == 0 or not hasattr(self, '_grids_saved'):
            grids = {"x": self.x.cpu().numpy().copy(),
                     "y": self.y.cpu().numpy().copy()}
            if self.z is not None:
                grids["z"] = self.z.cpu().numpy().copy()
            self._grids_saved = True
        self._submit_io(self._save_fields_h5, h5_fields, grids, iteration)

    # ------------------------------------------------------------------
    #   Unified HDF5 saving helpers (used by both 2-D and 3-D paths)
    # ------------------------------------------------------------------

    def _save_fields_h5(self, fields, grids, iteration):
        """Write velocity / pressure / SDF fields to a single HDF5 file.

        File layout::

            grids/x, grids/y [, grids/z]   – written once
            fields/000000/u
            fields/000000/v
            fields/000000/p
            fields/000000/w   (3-D only)
            fields/000000/sdf (when a body is present)
            ...

        All derived quantities (vorticity, velocity magnitude, divergence)
        can be recomputed from u, v, [w] and the grid coordinates.
        """
        h5_path = f'{self.save_path}/fields.h5'
        grp = f'fields/{iteration:06d}'
        lock = self._hdf5_lock
        with lock:
            with h5py.File(h5_path, 'a') as f:
                for name, arr in fields.items():
                    f.create_dataset(f'{grp}/{name}', data=arr)
                if grids is not None and 'grids' not in f:
                    for name, arr in grids.items():
                        f.create_dataset(f'grids/{name}', data=arr)

    def _save_contours_h5(self, cnt_arrays, iteration):
        """Write 2-D body contour data to a dedicated HDF5 file."""
        cnt_h5 = f'{self.save_path}/contours.h5'
        lock = self._hdf5_lock
        with lock:
            with h5py.File(cnt_h5, 'a') as f:
                for i, arr in enumerate(cnt_arrays):
                    f.create_dataset(f'{iteration:06d}/body_{i}', data=arr)

    def save_drags_h5(self, path=None):
        """Persist per-body force/torque histories to HDF5.

        Writes ``<save_path>/drags.h5`` with datasets:
            viscous_drags    (n_bodies, 3, nt)  viscous (skin) force  [N]
            pressure_drags   (n_bodies, 3, nt)  pressure (form) force [N]
            viscous_torques  (n_bodies, 3, nt)  viscous torque about COM  [N m]
            pressure_torques (n_bodies, 3, nt)  pressure torque about COM [N m]

        Metadata (dt, rho, nt, link names, geometry) is intentionally
        *not* written here — post-processing scripts can recover it from
        ``parameters.yaml``, ``simulation.hdf5`` and the cached SDFs.
        """
        if path is None:
            path = f'{self.save_path}/drags.h5'
        vd = self.viscous_drag_record.detach().cpu().numpy().copy()
        pd = self.pressure_drag_record.detach().cpu().numpy().copy()
        vt = self.viscous_torque_record.detach().cpu().numpy().copy()
        pt = self.pressure_torque_record.detach().cpu().numpy().copy()
        lock = self._hdf5_lock

        def _save(path, vd, pd, vt, pt):
            with lock:
                with h5py.File(path, 'w') as f:
                    f.create_dataset('viscous_drags',    data=vd)
                    f.create_dataset('pressure_drags',   data=pd)
                    f.create_dataset('viscous_torques',  data=vt)
                    f.create_dataset('pressure_torques', data=pt)

        self._submit_io(_save, path, vd, pd, vt, pt)

