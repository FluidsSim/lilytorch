import torch
from torch import Tensor
import numpy as np

def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    """One-dimensional linear interpolation for monotonically increasing sample
    points.

    Returns the one-dimensional piecewise linear interpolant to a function with
    given discrete data points :math:`(xp, fp)`, evaluated at :math:`x`.

    Args:
        x: the :math:`x`-coordinates at which to evaluate the interpolated
            values.
        xp: the :math:`x`-coordinates of the data points, must be increasing.
        fp: the :math:`y`-coordinates of the data points, same length as `xp`.

    Returns:
        the interpolated values, same size as `x`.
    """
    m = (fp[1:] - fp[:-1]) / (xp[1:] - xp[:-1])
    b = fp[:-1] - (m * xp[:-1])

    indicies = torch.sum(torch.ge(x[:, None], xp[None, :]), 1) - 1
    indicies = torch.clamp(indicies, 0, len(m) - 1)

    return m[indicies] * x + b[indicies]


if __name__ == "__main__":

    import matplotlib.pyplot as plt

    def plot_interp():
        xp = torch.tensor([0.0, 0.01, 1.0, 2.0, 4.5, 5.0])
        fp = torch.sin(xp)
        x = torch.linspace(0, 5, 100)
        y = interp(x, xp, fp)
        plt.plot(xp.numpy(), fp.numpy(), 'o', label='Data points')
        plt.plot(x.numpy(), y.numpy(), '-', label='Interpolated')
        plt.legend()
        plt.title("1D Linear Interpolation")
        plt.xlabel("x")
        plt.ylabel("y")

        xp_np = xp.numpy()
        fp_np = fp.numpy()
        x_np = x.numpy()
        y_np_interp = np.interp(x_np, xp_np, fp_np)

        plt.figure()
        plt.plot(x_np, y_np_interp, '--', label='np.interp')
        plt.legend()
        plt.show()


    plot_interp()