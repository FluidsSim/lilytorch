"""Utility for computing a top-down camera configuration that fits a pool."""

from math import radians, tan


def top_down_camera_config(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float = 0.0,
    zmax: float = 0.0,
    *,
    fovy: float = 45.0,
    margin_factor: float = 0.08,
    min_margin: float = 0.01,
    overshoot: float = 1.30,
    max_width: int = 1280,
    max_height: int = 720,
) -> dict:
    """Compute camera parameters for a top-down view that fits the pool.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Horizontal extents of the simulation domain.
    zmin, zmax : float, optional
        Vertical extents (only used for the camera look-at centre).
    fovy : float
        Vertical field-of-view in degrees.
    margin_factor : float
        Fraction of the smallest pool dimension used as padding.
    min_margin : float
        Minimum padding (in world units).
    overshoot : float
        Safety factor applied to the computed distance.
    max_width, max_height : int
        Maximum pixel dimensions of the output frame (must not exceed the
        MuJoCo offscreen framebuffer set in the model XML).  The resolution
        is scaled to fill as much of this budget as possible while keeping
        the aspect ratio equal to the true padded-tank shape.  Both
        dimensions are rounded to the nearest even integer as required by
        most video codecs.

    Returns
    -------
    dict
        Keys: ``azimuth``, ``elevation``, ``distance``, ``offset``,
        ``resolution`` — ready to be merged into a ``CameraRecording``
        config dict.
    """
    pool_dx = xmax - xmin
    pool_dy = ymax - ymin
    pool_dz = zmax - zmin

    pool_dims = [d for d in (pool_dx, pool_dy, pool_dz) if d > 0]
    wt_cam = max(round(margin_factor * min(pool_dims), 4), min_margin) if pool_dims else min_margin

    padded_dx = pool_dx + 2 * wt_cam
    padded_dy = pool_dy + 2 * wt_cam
    half_fov = tan(radians(fovy / 2))

    # Largest uniform scale that keeps both axes within the framebuffer budget.
    # This fills the available resolution while preserving the true tank aspect
    # ratio.  Round to nearest even integer (required by most video codecs).
    scale = min(max_width / padded_dx, max_height / padded_dy)
    pix_dx = 2 * max(1, round(padded_dx * scale / 2))
    pix_dy = 2 * max(1, round(padded_dy * scale / 2))

    if padded_dx >= padded_dy:
        azimuth = 90
        cam_res = [pix_dx, pix_dy]
        dim_horiz, dim_vert = padded_dx, padded_dy
    else:
        azimuth = 0
        cam_res = [pix_dy, pix_dx]
        dim_horiz, dim_vert = padded_dy, padded_dx

    aspect_ratio = cam_res[0] / cam_res[1]
    d_for_vert = (dim_vert / 2) / half_fov
    d_for_horiz = (dim_horiz / 2) / (half_fov * aspect_ratio)
    distance = max(d_for_vert, d_for_horiz) * overshoot

    return {
        "azimuth":    azimuth,
        "elevation":  -90,
        "distance":   distance,
        "offset":     [(xmin + xmax) / 2,
                       (ymin + ymax) / 2,
                       (zmin + zmax) / 2],
        "resolution": cam_res,
    }


def side_camera_config(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    zmin: float,
    zmax: float,
    *,
    view_axis: str = "y",
    fovy: float = 45.0,
    margin_factor: float = 0.08,
    min_margin: float = 0.01,
    overshoot: float = 1.10,
    max_width: int = 1280,
    max_height: int = 720,
) -> dict:
    """Compute camera parameters for a horizontal side view of the tank.

    Parameters
    ----------
    xmin, xmax, ymin, ymax, zmin, zmax : float
        Full domain extents.
    view_axis : {'y', 'x'}
        Axis the camera looks *along*.

        * ``'y'`` — camera on the **+Y** side looking toward −Y; shows the
          **XZ** plane (swimming direction vs. tank depth).  Best for fish
          swimming along X in a shallow tank.
        * ``'x'`` — camera on the **+X** side looking toward −X; shows the
          **YZ** plane (lateral width vs. tank depth).
    fovy, margin_factor, min_margin, overshoot, max_width, max_height
        Same semantics as :func:`top_down_camera_config`.

    Returns
    -------
    dict
        Keys: ``azimuth``, ``elevation``, ``distance``, ``offset``,
        ``resolution`` — ready to be merged into a ``CameraRecording`` config.
    """
    pool_dx = xmax - xmin
    pool_dy = ymax - ymin
    pool_dz = zmax - zmin

    pool_dims = [d for d in (pool_dx, pool_dy, pool_dz) if d > 0]
    wt_cam = max(round(margin_factor * min(pool_dims), 4), min_margin) if pool_dims else min_margin

    if view_axis == "y":
        # Looking along Y: horizontal span = X, vertical span = Z.
        # azimuth=90 is consistent with the top-down camera's choice that
        # makes the X axis run left-right in the image; with elevation=0 it
        # places the camera on the +Y side looking toward −Y.
        dim_horiz_phys = pool_dx + 2 * wt_cam
        dim_vert_phys  = pool_dz + 2 * wt_cam
        azimuth = 90
    elif view_axis == "x":
        # Looking along X: horizontal span = Y, vertical span = Z.
        # azimuth=0 places the camera on the +X side looking toward −X.
        dim_horiz_phys = pool_dy + 2 * wt_cam
        dim_vert_phys  = pool_dz + 2 * wt_cam
        azimuth = 0
    else:
        raise ValueError(f"view_axis must be 'x' or 'y', got {view_axis!r}")

    half_fov = tan(radians(fovy / 2))

    scale   = min(max_width / dim_horiz_phys, max_height / dim_vert_phys)
    pix_h   = 2 * max(1, round(dim_horiz_phys * scale / 2))
    pix_v   = 2 * max(1, round(dim_vert_phys  * scale / 2))
    cam_res = [pix_h, pix_v]

    aspect_ratio = pix_h / pix_v
    d_for_vert   = (dim_vert_phys  / 2) / half_fov
    d_for_horiz  = (dim_horiz_phys / 2) / (half_fov * aspect_ratio)
    distance     = max(d_for_vert, d_for_horiz) * overshoot

    return {
        "azimuth":    azimuth,
        "elevation":  0,
        "distance":   distance,
        "offset":     [(xmin + xmax) / 2,
                       (ymin + ymax) / 2,
                       (zmin + zmax) / 2],
        "resolution": cam_res,
    }
