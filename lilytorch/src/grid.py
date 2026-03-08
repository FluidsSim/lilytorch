"""
lilytorch/src/grid.py
----------------------
Utility class for grid spacing metadata.
Supports both uniform and variable (non-uniform) spacing tensors.
"""

import torch


class GridSpacing:
    """
    Holds per-cell grid spacing tensors hx[i,j] and hy[i,j].

    On a uniform grid, hx and hy are constant tensors equal to dx, dy.
    On a non-uniform grid (e.g. two-level nested grid), they can vary spatially.

    Attributes
    ----------
    hx : torch.Tensor, shape (nx, ny)
        Grid spacing in the x-direction at each cell.
    hy : torch.Tensor, shape (nx, ny)
        Grid spacing in the y-direction at each cell.
    hxy : torch.Tensor, shape (nx, ny)
        Area element hx * hy at each cell.
    is_uniform : bool
        True if hx and hy are spatially constant.
    """

    def __init__(self, nx: int, ny: int, dx: float, dy: float,
                 device=torch.device("cpu"), dtype=torch.float32):
        self.nx = nx
        self.ny = ny
        self.device = device
        self.dtype = dtype

        self.hx = dx * torch.ones((nx, ny), device=device, dtype=dtype)
        self.hy = dy * torch.ones((nx, ny), device=device, dtype=dtype)
        self.hxy = self.hx * self.hy
        self.is_uniform = True

    def set_nonuniform(self, hx: torch.Tensor, hy: torch.Tensor):
        """
        Replace uniform spacing with spatially varying tensors.

        Parameters
        ----------
        hx : torch.Tensor, shape (nx, ny)
        hy : torch.Tensor, shape (nx, ny)
        """
        assert hx.shape == (self.nx, self.ny), f"hx shape mismatch: {hx.shape}"
        assert hy.shape == (self.nx, self.ny), f"hy shape mismatch: {hy.shape}"
        self.hx = hx.to(self.device)
        self.hy = hy.to(self.device)
        self.hxy = self.hx * self.hy
        self.is_uniform = False

    @property
    def h(self) -> float:
        """Returns scalar spacing (only valid for uniform grids)."""
        if not self.is_uniform:
            raise ValueError("Grid is non-uniform; use hx/hy tensors instead of h scalar.")
        return float(self.hx[0, 0])
