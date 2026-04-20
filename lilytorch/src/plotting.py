import os
import threading

import matplotlib
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

def plot_time_histories(
    time: np.array,
    state: np.array,
    **kwargs,
):
    """
    Plot time histories of a vector of states
    time: array of times
    state: array of array of values
    kwargs: optional plotting properties
    """

    xlabel        = kwargs.pop('xlabel', "Time [s]")
    ylabel        = kwargs.pop('ylabel', "Activity [-]")
    title         = kwargs.pop('title', None)
    labels        = kwargs.pop('labels', None)
    colors        = kwargs.pop('colors', None)
    xlim          = kwargs.pop('xlim', [0,time[-1]])
    ylim          = kwargs.pop('ylim', None)
    offset        = kwargs.pop('offset', 0)
    savepath      = kwargs.pop('savepath', None)
    lw            = kwargs.pop('lw', 1.0)
    xticks        = kwargs.pop('xticks', None)
    yticks        = kwargs.pop('yticks', None)
    xticks_labels = kwargs.pop('xticks_labels', None)
    yticks_labels = kwargs.pop('xticks_labels', None)
    closefig      = kwargs.pop('closefig', True)

    ymin = np.min(state)
    ymax = np.max(state)
    if not ylim: ylim = [ymin-0.1*ymin, ymax+0.1*ymax]

    if title:
        plt.figure(title)
    for (idx, vector) in enumerate(state.transpose()):
        if not labels:
            label = None
        else:
            label = labels[idx]
        if not colors:
            color = None
        else:
            color = colors[idx]
        plt.plot(time, vector-offset*idx, label=label, color=color, linewidth=lw)
    if labels:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(False)
    if xticks:
        plt.xticks(xticks, labels=xticks_labels)
    if yticks:
        plt.yticks(yticks, labels=yticks_labels)
    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()

def plot_spk_train(
    pos,
    st,
    I='',
    color=None,
):
    """
    Plot spike train
    pos: array of positions
    st: array of arrays containing spike times
    I: optional xlim
    colors: optional uniform colour
    """
    for (i,spike_train) in enumerate(st):
        plt.plot(
            spike_train,
            pos[i]*np.ones_like(spike_train),
            marker='.',
            markersize=5,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle='None'
            )
        if len(I):
            plt.xlim(I)

def histogram_stair(
    data,
    bins,
    **kwargs,
):
    xlabel   = kwargs.pop('xlabel', None)
    ylabel   = kwargs.pop('ylabel', "# of occurrences")
    title    = kwargs.pop('title', None)
    xlim     = kwargs.pop('xlim', None)
    ylim     = kwargs.pop('ylim', None)
    savepath = kwargs.pop('savepath', None)
    closefig      = kwargs.pop('closefig', True)

    counts, bins = np.histogram(data, bins=bins)
    if title:
        plt.figure(title)
    plt.stairs(counts, bins)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xlim)
    plt.ylim(ylim)
    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()

def histogram_plot(
    fig,
    ax,
    x,
    y,
    **kwargs,
):
    xlabel   = kwargs.pop('xlabel', None)
    ylabel   = kwargs.pop('ylabel', "\# of occurrences")
    title    = kwargs.pop('title', None)
    color    = kwargs.pop('color', 'k')
    xlim     = kwargs.pop('xlim', None)
    ylim     = kwargs.pop('ylim', [np.min(y)-0.1*np.min(y), np.max(y)+0.1*np.max(y)+0.4])
    width    = kwargs.pop('width', 0.5)
    xticks   = kwargs.pop('xticks', None)
    savepath = kwargs.pop('savepath', None)
    closefig = kwargs.pop('closefig', True)
    yticks_labels = kwargs.pop('xticks_labels', [])

    xs = range(len(x))

    if title:
        ax.set_title(title)
    ax.bar(x, y, width, color=color)
    if xticks:
        plt.gca().set_xticks(xs)
        ax.set_xticklabels(xticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()
    for index, value in enumerate(y):
        plt.text(index, value,
                str(round(value,2)), ha = 'center')
    # ax.set_yticklabels(yticks_labels)


def stack_bar_chart(
    values,
    data,
    **kwargs,
):

    xlabel   = kwargs.pop('xlabel', None)
    width    = kwargs.pop('width', 0.5)
    xticks   = kwargs.pop('xticks', None)
    savepath = kwargs.pop('savepath', None)
    closefig = kwargs.pop('closefig', True)

    weight_counts = {
        "no conv": [d.count(-1) for d in data],
        "unclas": [d.count(0) for d in data],
        "lsw": [d.count(1) for d in data],
        "trot": [d.count(2) for d in data],
        "dsw": [d.count(3) for d in data],
        "pace": [d.count(4) for d in data]

    }

    fig, ax = plt.subplots()
    bottom = np.zeros(len(values))
    for boolean, weight_count in weight_counts.items():
        p = ax.bar(values, weight_count, width, label=boolean, bottom=bottom)
        bottom += weight_count
    plt.ylim([0, max(bottom)+1])
    ax.set_title("Gait classification")
    ax.legend(loc="upper right")
    plt.xlabel(xlabel)
    if xticks:
        # plt.gca().set_xticks([0,len(x)])
        plt.gca().set_xticklabels(xticks)

    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()



def patch_plot(
    data,
    npixel_x = 1,
    npixel_y = 30,
    gap      = 5,
    y_resize = 1000,
    cmap     = "cividis",
    **kwargs
    ):

    """
    foot_values = matrix
    npixel_x = number of x pixels to color (choose 1 for time plot)
    npixel_y = number of y pixels to color
    gap = number of gap pixels
    cmap = color map
    labels = labels of each bar on the y-axis (default = range values)
    """

    if data.shape[1] > 3000: # resize is dataset is too large
        data = data[:,::int(data.shape[1]/y_resize)]

    xlabel   = kwargs.pop('xlabel', None)
    ylabel   = kwargs.pop('ylabel', None)
    clabel   = kwargs.pop('clabel', None)
    title    = kwargs.pop('title', None)
    labels   = kwargs.pop('labels', [])
    savepath = kwargs.pop('savepath', None)
    k        = kwargs.pop('gap_value', np.mean(data))
    xticks   = kwargs.pop('xticks', None)
    vmin     = kwargs.pop('vmin', -99.0)
    cback    = kwargs.pop('cback', 'white')
    dpi      = kwargs.pop('dpi', 600)
    clim     = kwargs.pop('clim', [np.min(data),np.max(data)])
    closefig      = kwargs.pop('closefig', True)
    colorbar_on = kwargs.pop('colorbar_on', False)

    cmap     = getattr(plt.cm, cmap)
    cmap.set_under(cback)

    n = len(data)

    ex_ = np.insert(data, range(1,n+1), k*np.ones(len(data[0])), axis=0)
    ex_ = np.repeat(ex_, npixel_x, axis=1)
    ex_ = np.repeat(ex_, np.tile([npixel_y,gap], n), axis=0)

    if title:
        plt.figure(title)

    plt.imshow(
        ex_,
        cmap = cmap,
        vmin = vmin
    )
    plt.clim(clim)

    if len(labels):
        plt.yticks([(npixel_y+gap)*i+npixel_y/2-0.5 for i in range(n)], labels)
    else:
        plt.gca().set_yticks([])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xticks is not None:
        plt.gca().set_xticks([0,data.shape[1]])
        plt.gca().set_xticklabels(xticks)

    if colorbar_on:
        cbar = plt.colorbar()
        cbar.ax.set_title(clabel)

    if savepath:
        plt.savefig(savepath, dpi=dpi)


def plot2D(
    results,
    labels,
    n_data=300,
    log=False,
    cmap=None,
    sequence=None,
    interpolation=None,
    **kwargs
):
    """Plot result

    results - The results are given as a 2d array of dimensions [N, 3].

    labels - The labels should be a list of three string for the xlabel, the
    ylabel and zlabel (in that order).

    n_data - Represents the number of points used along x and y to draw the plot

    log - Set log to True for logarithmic scale.

    cmap - You can set the color palette with cmap. For example,
    set cmap='nipy_spectral' for high constrast results.

    """
    savepath = kwargs.pop('savepath', None)
    closefig      = kwargs.pop('closefig', True)

    x=results[0]
    y=results[1]
    z=results[2]
    xnew = np.linspace(min(x), max(x), n_data)
    ynew = np.linspace(min(y), max(y), n_data)
    grid_x, grid_y = np.meshgrid(xnew, ynew)
    results_interp = griddata(
        (x, y), z,
        (grid_x, grid_y),
        method='nearest',  # nearest, cubic
    )
    extent = (
        min(xnew), max(xnew),
        min(ynew), max(ynew)
    )
    imgplot = plt.imshow(
        results_interp,
        extent=extent,
        aspect='auto',
        origin='lower',
        interpolation=interpolation,
        norm=matplotlib.colors.LogNorm() if log else None
    )

    if cmap is not None:
        imgplot.set_cmap(cmap)
    cbar = plt.colorbar()
    cbar.set_label(labels[2])

    if sequence:
        sequence_interp = griddata(
            (x, y), sequence,
            (grid_x, grid_y),
            method='nearest',  # nearest, cubic
        )
        masked_data = np.ma.masked_where(sequence_interp==0, results_interp)
        plt.imshow(
            masked_data,
            extent=extent,
            aspect='auto',
            origin='lower',
            interpolation='none',
            cmap=matplotlib.cm.jet,
            norm=matplotlib.colors.LogNorm() if log else None
        )

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()

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

    field_np = np.asarray(field)

    # ---- symmetric auto-range (only when vmin/vmax not set) ----
    if vmin is None or vmax is None:
        # Ghost cells are already stripped by the caller, so use the
        # full field for the auto-range (99.5-th percentile symmetric).
        abs_f  = np.abs(field_np)
        limit  = float(np.percentile(abs_f, 99.5))
        if limit == 0:
            limit = float(abs_f.max()) or 1.0
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
    field_3d,           # 3-D numpy array (Nx, Ny, Nz) – already on CPU
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
    field_np = np.asarray(field_3d)
    Nx, Ny, Nz = field_np.shape
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
        field_np[:, :, k_xy], extent_xy, f"{name}_xy_k{k_xy}",
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
        field_np[:, j_xz, :], extent_xz, f"{name}_xz_j{j_xz}",
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
        field_np[i_yz, :, :], extent_yz, f"{name}_yz_i{i_yz}",
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
    window_size=(3840, 2160),
    image_scale=2,          # multiplier on window_size for the saved image
    fmt="png",
):
    """
    Render a 3-D field as isosurfaces with an optional opaque body surface
    (SDF = 0), using PyVista off-screen.

    For fields that contain both positive and negative values (e.g. vorticity
    component, pressure), dual ±threshold isosurfaces are drawn (red/blue).
    For non-negative fields (e.g. |curl| magnitude), a single isosurface is
    drawn at `threshold` coloured by value.

    ``crop_boundary`` removes cells from each face so that zero-padded
    boundary layers (from the vorticity stencil) and ghost-cell artifacts
    never reach the isosurface algorithm.

    Threshold strategy
    ------------------
    Vorticity fields from backward-difference stencils are dominated by
    near-zero noise: the 85th-percentile of |ω| can be 100,000× smaller
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
            window_size=window_size, image_scale=image_scale, fmt=fmt,
        )


def _plot_field_3d_locked(
    pv, field_3d, coords, name, iteration, save_path, *,
    sdf_3d=None, bodies=None, sdf_vals=None,
    body_colors=None, body_default_color=None,
    iso_value=None, iso_fraction=0.15,
    smooth_sigma=0, crop_boundary=0,
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
    body_surface_drawn = False
    if bodies is not None and sdf_vals is not None:
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

    if not body_surface_drawn and sdf_3d is not None:
        sdf_np = np.asarray(sdf_3d)
        grid.point_data["sdf"] = sdf_np.flatten(order="F")
        try:
            body_surf = grid.contour([0.0], scalars="sdf")
            if body_surf.n_points > 0:
                pl.add_mesh(body_surf, color="white", opacity=0.95,
                            smooth_shading=True, specular=0.6)
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
                font_size=35,
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

