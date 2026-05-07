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
    landscape_res: tuple[int, int] = (1280, 720),
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
    landscape_res : tuple[int, int]
        Resolution ``(width, height)`` used when the wider pool axis is *x*;
        the values are swapped for portrait orientation.

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

    if padded_dx >= padded_dy:
        azimuth = 90
        cam_res = list(landscape_res)
        dim_horiz, dim_vert = padded_dx, padded_dy
    else:
        azimuth = 0
        cam_res = [landscape_res[1], landscape_res[0]]
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
