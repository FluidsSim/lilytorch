import os

import matplotlib
matplotlib.use("Agg")          # non-interactive backend — thread-safe, no GUI
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.colors as colors
matplotlib.rc('font', **{"size":20})
plt.rcParams["figure.figsize"] = (15,15)

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
        norm=mpl.colors.LogNorm() if log else None
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
            cmap=mpl.cm.jet,
            norm=mpl.colors.LogNorm() if log else None
        )

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])

    if savepath:
        plt.savefig(savepath)
        if closefig:
            plt.close()

def _save_figure(save_path, name, iteration, fmt="png"):
    """Save the current pyplot figure into *save_path/name/name_ITER.fmt*."""
    folder = f"{save_path}/{name}"
    os.makedirs(folder, exist_ok=True)
    plt.savefig(
        f"{folder}/{name}_{iteration:06d}.{fmt}",
        bbox_inches="tight",
        facecolor=plt.gcf().get_facecolor(),
        edgecolor="none",
        dpi=150,
    )
    plt.close()


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
    cmap=None,
    fmt="png",
):
    """
    Single unified 2-D field plot.

    * Symmetric auto-range when *vmin*/*vmax* are ``None``.
    * Optional body contour scatter overlay – body is always shown even
      when it partially or fully exits the fluid domain.
    * Black-background aesthetic with vivid red/blue colormap inspired by
      Tekinalp et al. (2024).
    """
    if cmap is None:
        cmap = VIBRANT_RDBU

    field_np = np.asarray(field)

    # ---- symmetric auto-range ----
    if vmin is None or vmax is None:
        limit = max(abs(field_np.min()), abs(field_np.max())) / 2
        if limit == 0:
            limit = 1.0
        vmin, vmax = -limit, limit

    # ---- figure size from domain aspect ratio ----
    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    scale_f = 25 / max(x_range, y_range)
    fig_w   = max(x_range * scale_f, 4)
    fig_h   = max(y_range * scale_f, 4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # ---- dark theme styling ----
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.tick_params(colors="white", which="both")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")

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
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    cbar.outline.set_edgecolor("white")

    # ---- body contours (always shown, even outside domain) ----
    view_xmin, view_xmax = extent[0], extent[1]
    view_ymin, view_ymax = extent[2], extent[3]

    if bodies is not None:
        for body in bodies:
            cnt = getattr(body, "cnt_update", None)
            mask = getattr(body, "mask", None)
            if cnt is not None and mask is not None:
                bx = cnt[0][mask].cpu().numpy()
                by = cnt[1][mask].cpu().numpy()
                ax.scatter(bx, by, c="white", s=0.5, zorder=5,
                           edgecolors="none", linewidths=0)
                # expand view to include the full robot body
                if len(bx) > 0:
                    view_xmin = min(view_xmin, bx.min())
                    view_xmax = max(view_xmax, bx.max())
                if len(by) > 0:
                    view_ymin = min(view_ymin, by.min())
                    view_ymax = max(view_ymax, by.max())

                com = getattr(body, "com_pos", None)
                if com is not None:
                    cx = com[0].cpu().numpy()
                    cy = com[1].cpu().numpy()
                    ax.plot(cx, cy, "o", color="#FF4040",
                            markersize=3, zorder=6)
                    view_xmin = min(view_xmin, float(cx))
                    view_xmax = max(view_xmax, float(cx))
                    view_ymin = min(view_ymin, float(cy))
                    view_ymax = max(view_ymax, float(cy))

    # add a small margin so the body is not glued to the plot border
    pad_x = 0.02 * (view_xmax - view_xmin) if view_xmax > view_xmin else 0.01
    pad_y = 0.02 * (view_ymax - view_ymin) if view_ymax > view_ymin else 0.01
    ax.set_xlim(view_xmin - pad_x, view_xmax + pad_x)
    ax.set_ylim(view_ymin - pad_y, view_ymax + pad_y)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(name)
    _save_figure(save_path, name, iteration, fmt)


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
    slice_indices=None,  # dict {"xy": k, "xz": j, "yz": i}  (None → origin)
    cmap=None,
    fmt="png",
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

    # ---- XY slice (fixed z) ----
    extent_xy = (float(x[0]), float(x[-1]), float(y[0]), float(y[-1]))
    plot_field_2d(
        field_np[:, :, k_xy], extent_xy, f"{name}_xy_k{k_xy}",
        iteration, save_path,
        vmin=vmin, vmax=vmax, bodies=bodies, cmap=cmap, fmt=fmt,
    )

    # ---- XZ slice (fixed y) ----
    extent_xz = (float(x[0]), float(x[-1]), float(z[0]), float(z[-1]))
    plot_field_2d(
        field_np[:, j_xz, :], extent_xz, f"{name}_xz_j{j_xz}",
        iteration, save_path,
        vmin=vmin, vmax=vmax, bodies=None, cmap=cmap, fmt=fmt,
    )

    # ---- YZ slice (fixed x) ----
    extent_yz = (float(y[0]), float(y[-1]), float(z[0]), float(z[-1]))
    plot_field_2d(
        field_np[i_yz, :, :], extent_yz, f"{name}_yz_i{i_yz}",
        iteration, save_path,
        vmin=vmin, vmax=vmax, bodies=None, cmap=cmap, fmt=fmt,
    )


def plot_field_3d(
    field_3d,           # 3-D numpy array (Nx, Ny, Nz)
    coords,             # dict  {"x": 1d, "y": 1d, "z": 1d}
    name,
    iteration,
    save_path,
    *,
    sdf_3d=None,        # optional SDF array for body isosurface (same shape)
    iso_value=None,     # fixed isosurface threshold (overrides auto when set)
    iso_fraction=0.15,  # threshold = iso_fraction * max(|smoothed field|)
    smooth_sigma=2.5,   # Gaussian smoothing (in grid-cells) before isosurface extraction
    crop_boundary=3,    # number of cells to crop from each domain face before rendering
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
    if sdf_3d is not None:
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
                position="upper_right",
                font_size=10,
                color="white",
                shadow=True,
            )
        except Exception:
            pass

    # ---- camera  (isometric, looking from upstream-above) ----
    cam_pos = (cx - 0.8 * Lmax, cy - 0.6 * Lmax, cz + 1.5 * Lmax)
    pl.camera_position = [cam_pos, focal, (0, 0, 1)]
    pl.camera.parallel_projection = True
    pl.camera.parallel_scale = Lmax * 0.7

    # ---- save high-resolution screenshot ----
    out_dir = os.path.join(save_path, f"{name}_3d")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{name}_3d_{iteration:06d}.{fmt}")
    pl.screenshot(out_path, scale=image_scale)
    pl.close()


def save_vtk(
    fields,            # dict  {"u": array, "v": array, ...}
    coords,            # dict  {"x": 1d, "y": 1d, "z": 1d}
    iteration,
    save_path,
    *,
    name_prefix="flow",
):
    """
    Write a 3-D rectilinear-grid VTK file readable by ParaView.
    Requires PyVista (already available in the env).
    """
    try:
        import pyvista as pv
    except ImportError:
        print("[save_vtk] pyvista not installed – skipping VTK export.")
        return

    x = np.asarray(coords["x"])
    y = np.asarray(coords["y"])
    z = np.asarray(coords["z"])

    grid = pv.RectilinearGrid(x, y, z)

    for fname, fdata in fields.items():
        arr = np.asarray(fdata)
        # RectilinearGrid point data expects Fortran-order flattening
        grid.point_data[fname] = arr.flatten(order="F")

    vtk_dir = f"{save_path}/vtk"
    os.makedirs(vtk_dir, exist_ok=True)
    out_path = f"{vtk_dir}/{name_prefix}_{iteration:06d}.vtr"
    grid.save(out_path)


# ---------------------------------------------------------------------------
#   Legacy helpers  (kept for backward-compat, new code uses plot_field_*)
# ---------------------------------------------------------------------------

def save_fig_to_dedicated_folder(
    save_path   : str,
    quantity_str: str,
    iteration   : int,
    type        : str = "png"
):
    ''' Save plot to dedicated folder '''

    # Create folder for the current plot
    target_folder = f'{save_path}/{quantity_str}'
    os.makedirs(target_folder, exist_ok=True)

    # Save plot
    target_file   = f'{target_folder}/{quantity_str}_{iteration}.{type}'
    plt.savefig(target_file)
    plt.close()

    return

def plot_composite_countour(X,Y,properties,iteration,save_path,name):
    plt.figure(figsize=(20,10))
    for prop in properties:
        d = prop[0].cpu()
        plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_only(u,extent,iteration,save_path,name,vmin,vmax):
    if vmin is None:
        limit = max(abs(u.min()), abs(u.max()))
        vmin = -limit
        vmax = limit
    plt.figure(figsize=(20,10))
    plt.imshow(
        u.T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu,
        interpolation=None
    )
    plt.colorbar()
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_composite(X,Y,u,properties,extent,iteration,save_path,name,vmin,vmax):
    if vmin is None:
        limit = max(abs(u.min()), abs(u.max())) #/2
        vmin = -limit
        vmax = limit
    plt.figure(figsize=(20,10))
    for prop in properties:
        d = prop[0].cpu()
        plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
    plt.imshow(
        u.detach().numpy().T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu,
        interpolation=None
    )
    plt.colorbar()
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_composite_quiver(X,Y,u,bodies,normal_x,normal_y,extent,iteration,save_path,name,vmin,vmax,subsample_n = 2**4, scale=None, body_contours = True):



    if vmin is None:
        limit = max(abs(u.min()), abs(u.max()))/2
        vmin = -limit
        vmax = limit
    if scale:
        scale=1/scale


    x_range = extent[1] - extent[0]
    y_range = extent[3] - extent[2]
    scale = 25 / max(x_range, y_range)  # Adjust 10 for overall size
    fig_width = x_range * scale
    fig_height = y_range * scale

    # plt.figure(figsize=(5, 15))

    plt.figure(figsize=(fig_width, fig_height))
    if body_contours:
        for i, body in enumerate(bodies):
            # d = body.sdf.cpu()
            # plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
            plt.scatter(body.cnt_update[0][body.mask].cpu(), body.cnt_update[1][body.mask].cpu(), c="k",s=0.1)
            # plt.scatter(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), c='k', s=0.1)

            # plt.fill(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), color="#3E8854E1")

            plt.plot(body.com_pos[0].cpu(), body.com_pos[1].cpu(), 'ro', markersize=2)

            # plt.plot(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), 'k',linewidth=0.5)

            # plt.plot(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), 'k',linewidth=0.5)
    plt.imshow(
        u.cpu().detach().numpy().T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu,
        interpolation=None,
        aspect='equal',
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title(name)

    plt.colorbar()
    # q = plt.quiver(
    #     X[::subsample_n,::subsample_n].cpu(),
    #     Y[::subsample_n,::subsample_n].cpu(),
    #     normal_x[::subsample_n,::subsample_n].cpu(),
    #     normal_y[::subsample_n,::subsample_n].cpu(),
    #     color='g',
    #     scale=scale, scale_units='xy'
    # )
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)









def plot2d_imshow(X,Y,u,d,extent,iteration,save_path,name,vmin,vmax):
    if vmin is None:
        limit = max(abs(u.min()), abs(u.max()))/2
        vmin = -limit
        vmax = limit
    plt.figure(figsize=(20,10))
    ctr = plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
    plt.imshow(
        np.where(d<0,0,u).T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu
    )
    plt.colorbar()
    # plt.ylim([-0.25,0.25])
    # plt.xlim([0.58,1.13])
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_quiver(X,Y,u,d,normal_x,normal_y,extent,iteration,save_path,name,vmin,vmax,subsample_n = 2**4, scale=None):

    if vmin is None:
        limit = max(abs(u.min()), abs(u.max()))/2
        vmin = -limit
        vmax = limit
    if scale:
        scale=1/scale
    plt.figure(figsize=(20,10))
    ctr = plt.contour(X,Y,d, colors='k', levels=[0],linewidths=0.3)
    plt.imshow(
        u.T,
        vmin   = vmin,
        vmax   = vmax,
        extent = extent,
        origin = "lower",
        cmap = cm.RdBu
    )
    plt.colorbar()
    q = plt.quiver(
        X[::subsample_n,::subsample_n].cpu(),
        Y[::subsample_n,::subsample_n].cpu(),
        normal_x[::subsample_n,::subsample_n].cpu(),
        normal_y[::subsample_n,::subsample_n].cpu(),
        color='g',
        scale=scale, scale_units='xy'
    )
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)

def plot2d_imshow_simple(d,extent,iteration,save_path,name,vmin= -0.001,vmax= 0.001):
    plt.figure(figsize=(20,10))
    if vmin is None:
        limit = max(abs(d.min()), abs(d.max()))/2
        vmin = -limit
        vmax = limit
    plt.imshow(
        d.T,
        extent = extent,
        origin = "lower",
        vmin   = vmin,
        vmax   = vmax,
        cmap = cm.RdBu
    )
    plt.colorbar()
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)




def plot_ctrs(vars,bodies, extent, save_path, name, iteration,vmin= -0.001,vmax= 0.001, cmap = cm.get_cmap('viridis')):

    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    for i, body in enumerate(bodies):
        plt.scatter(body.cnt_update[0].cpu(), body.cnt_update[1].cpu(), c=vars[i].cpu(), cmap=cmap, norm=norm)
    plt.colorbar()
    plt.axis(extent)
    save_fig_to_dedicated_folder(save_path, name, iteration)
