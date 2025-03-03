import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from scipy.interpolate import make_interp_spline
from scipy.interpolate import CubicSpline

def main():

    ref_points_x = np.array([0.00, 44.5, 66.5, 110.0]) / 110.0
    ref_points_y = np.array([0.04, 0.05, 0.16,  0.24])

    # Create a cubic spline with specified boundary conditions
    spline = CubicSpline(ref_points_x, ref_points_y, bc_type=((1, 0.1), (1, 0.1)))
    x_fit = np.linspace(0, 1, 100)
    y_fit = spline(x_fit)

    plt.scatter(ref_points_x, ref_points_y, label='Data')
    plt.plot(x_fit, y_fit, label='Spline interpolation', color='red')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
