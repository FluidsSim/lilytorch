import numpy as np

def resample_closed_contour(points, spacing, keep_duplicate_endpoint=True):
    """
    Resample a closed contour for (approximately) uniform spacing.
    - points: (M,2) numpy array of (x,y). Can be closed (first==last) or open; treated as closed.
    - spacing: desired spacing between resampled points (float > 0).
    - keep_duplicate_endpoint: if True, return N+1 points with last == first (explicit closure).
                               if False, return N points (no duplicate at end).
    Returns:
    - new_pts: (N+1,2) or (N,2) array of resampled points.
    - actual_spacing: total_length / N  (the spacing actually used)
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("points must be an (M,2) array-like")

    if spacing <= 0:
        raise ValueError("spacing must be positive")

    # If not already closed, append first point for segment math
    if not np.allclose(pts[0], pts[-1]):
        pts_closed = np.vstack([pts, pts[0]])
    else:
        pts_closed = pts.copy()

    # segment vectors and lengths
    segs = pts_closed[1:] - pts_closed[:-1]
    seg_lens = np.hypot(segs[:,0], segs[:,1])
    total_length = seg_lens.sum()
    if total_length == 0:
        raise ValueError("zero-length contour")

    # choose number of equal intervals such that spacing ~ requested spacing
    N = max(3, int(round(total_length / spacing)))
    actual_spacing = total_length / N

    # cumulative distances along the closed polyline (start at 0, last = total_length)
    s = np.concatenate(([0.0], np.cumsum(seg_lens)))
    # x,y coordinates corresponding to s
    x = pts_closed[:,0]
    y = pts_closed[:,1]

    # target sample locations: include the final total_length so last interpolates to first point
    target_s = np.linspace(0.0, total_length, N+1)

    # np.interp requires strictly increasing x; s is non-decreasing
    xi = np.interp(target_s, s, x)
    yi = np.interp(target_s, s, y)
    new_pts = np.vstack([xi, yi]).T

    if not keep_duplicate_endpoint:
        return new_pts[:-1], actual_spacing  # return N points
    return new_pts, actual_spacing      # return N+1 points where last==first

# circle-like test
t = np.linspace(0, 2*np.pi, 30, endpoint=False)
circ = np.column_stack([np.cos(t), np.sin(t)])
pts, spacing_used = resample_closed_contour(circ, spacing=0.2, keep_duplicate_endpoint=True)
print("Returned points:", pts.shape)   # (N+1, 2)
print("Actual spacing used:", spacing_used)
# check seam spacing
d = np.hypot(*(pts[1:] - pts[:-1]).T)
print("min/max spacing:", d.min(), d.max())

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.plot(circ[:, 0], circ[:, 1], 'k--', label='Original Contour')
plt.plot(pts[:, 0], pts[:, 1], 'ro-', label='Resampled Points')
plt.axis('equal')
plt.legend()
plt.title('Resampled Closed Contour')
plt.show()