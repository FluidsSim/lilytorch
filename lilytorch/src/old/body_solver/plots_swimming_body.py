
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from lilytorch.body import Body
from scipy.interpolate import CubicSpline, interp1d
from matplotlib.animation import FuncAnimation

###############################################################################
# SPLINE ######################################################################
###############################################################################

def compute_cubic_spline(
    coordinates_x : np.ndarray,
    coordinates_y : np.ndarray,
):
    ''' Returns a cubic spline object from a list of points. '''

    # Combine x and y coordinates
    coordinates_xy = np.stack((coordinates_x, coordinates_y), axis=-1)

    # Compute arc lengths
    links_lengths = np.array(
        [
            np.linalg.norm( coordinates_xy[point] - coordinates_xy[point - 1] )
            for point in range(1, len(coordinates_xy))
        ]
    )
    arc_lengths = np.concatenate( [ [0], np.cumsum(links_lengths) ] ) / np.sum(links_lengths)

    # Separate x and y coordinates from the points
    x_coords = coordinates_xy[:, 0]
    y_coords = coordinates_xy[:, 1]

    # Create a cubic spline object
    cubic_spline_x = CubicSpline(arc_lengths, x_coords)
    cubic_spline_y = CubicSpline(arc_lengths, y_coords)

    return cubic_spline_x, cubic_spline_y

def spline_curve_length(
    spline_curve_x: CubicSpline,
    spline_curve_y: CubicSpline,
    s_start       : float,
    s_end         : float,
    num_samples   : int =100
):
    ''' Returns the length of the spline curve between x_start and x_end. '''

    s_samples = np.linspace(s_start, s_end, num_samples)
    samples_x = spline_curve_x(s_samples)
    samples_y = spline_curve_y(s_samples)

    # Integrate
    arc_length = np.sum(
        np.sqrt(
            np.diff(samples_x)**2 + np.diff(samples_y)**2
        )
    )

    return arc_length

def compute_coordinates_from_arc_lengths(
    spline_curve_x: CubicSpline,
    spline_curve_y: CubicSpline,
    arc_lengths   : np.ndarray[float],
):
    ''' Returns the point on the spline curve with the desired normalized arc length.'''
    return np.array(
        [
            [ spline_curve_x(arc_length), spline_curve_y(arc_length) ]
            for arc_length in arc_lengths
        ]
    )

###############################################################################
# FISH PLOTTING ###############################################################
###############################################################################

def _get_fish_thickness_model_spline(fish_length: float):
    '''
    Fish thickness model spline
    Derived from the mechanical model of the zebrafish body
    '''
    x_coords = np.array(
        [
            0.00000000e+00, 1.85375688e-05, 2.59863671e-04, 3.75253149e-04,
            6.61936962e-04, 9.68398441e-04, 1.58240186e-03, 1.83324935e-03,
            2.67368424e-03, 3.67668830e-03, 4.70110425e-03, 6.06038261e-03,
            6.71571663e-03, 7.27757934e-03, 9.24973080e-03, 1.08033482e-02,
            1.23470686e-02, 1.34687836e-02, 1.40113190e-02, 1.41112078e-02,
            1.48027059e-02, 1.53315544e-02, 1.62264019e-02, 1.74619712e-02,
            1.80000000e-02,
        ]
    )
    y_coords = np.array(
        [
            0.00000000e+00, 9.15521925e-05, 2.58186783e-04, 3.88766211e-04,
            4.69319851e-04, 6.04953773e-04, 6.78922684e-04, 7.72618010e-04,
            8.85581041e-04, 9.20080493e-04, 9.46684097e-04, 9.85263519e-04,
            1.03368572e-03, 1.02831663e-03, 7.72560102e-04, 5.57949118e-04,
            3.94504830e-04, 2.58659862e-04, 2.68833330e-04, 2.52653865e-04,
            2.47203824e-04, 2.26617545e-04, 1.24073370e-04, 1.29116038e-04,
            0.00000000e+00,
        ]
    )

    # Rescale
    scaling   = fish_length / x_coords[-1]
    x_coords *= scaling
    y_coords *= scaling

    # Compute thickness spline
    # thickness_spline = CubicSpline(x_coords_scaled, y_coords_scaled)
    thickness_spline = CubicSpline(x_coords, y_coords)

    return thickness_spline

def get_fish_thickness_from_model(
    normalized_arclengths: np.ndarray,
    fish_length          : float,
    thickness_spline     : callable,
):
    '''
    Fish thickness model
    Derived from the mechanical model of the zebrafish body
    '''

    if thickness_spline is None:
        thickness_spline = _get_fish_thickness_model_spline(fish_length)

    # Compute thickness
    arclength        = normalized_arclengths * fish_length
    thickness_values = thickness_spline(arclength)

    return  thickness_values

def get_fish_thickness_from_gazzola(
    normalized_arclengths: np.ndarray,
    length               : float = 0.018,
):
    """
    Fish thickness
    See Gazzola et al. 2011
    """

    L = length
    s = normalized_arclengths * L
    sb, st, wh, wt = 0.07*L ,0.95*L ,0.07*L ,0.01*L

    # Conditions
    cond1 = ( 0 <= s) & (s <  sb)
    cond2 = (sb <= s) & (s <  st)
    cond3 = (st <= s) & (s <=  L)

    thickness = np.zeros_like(s)
    thickness[cond1] = np.sqrt(2*wh*s[cond1]-s[cond1]**2)
    thickness[cond2] = wh-(wh-wt)*(((s[cond2]-sb)/(st-sb))**2)
    thickness[cond3] = wt*(L-s[cond3])/(L-st)

    return thickness

def get_fish_boundaries(
    positions_x   : np.ndarray,
    positions_y   : np.ndarray,
    thickness_f   : callable,
    thickness_args: tuple = None,
    n_points      : int = 100,
):
    ''' Returns the fish body shape given the midline points and thickness function. '''

    if thickness_args is None:
        thickness_args = ()

    # Compute spline
    (
        spline_x,
        spline_y,
    ) = compute_cubic_spline(
        coordinates_x = positions_x,
        coordinates_y = positions_y,
    )

    # Generate midline points and thickness
    s_values         = np.linspace(0, 1, n_points)
    x_midline        = spline_x(s_values)
    y_midline        = spline_y(s_values)
    thickness_values = thickness_f(s_values, *thickness_args)

    # Calculate the fish body shape by offsetting along a perpendicular vector
    x_upper = np.zeros(n_points)
    y_upper = np.zeros(n_points)
    x_lower = np.zeros(n_points)
    y_lower = np.zeros(n_points)

    for i in range(n_points - 1):
        # Tangent vector
        dx = x_midline[i+1] - x_midline[i]
        dy = y_midline[i+1] - y_midline[i]

        # Normalize
        length = np.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length

        # Perpendicular vector
        perp_x = -dy
        perp_y = dx

        # Calculate upper and lower points by offsetting along the perpendicular
        thickness_v = thickness_values[i]
        x_upper[i] = (x_midline[i] + perp_x * thickness_v)
        y_upper[i] = (y_midline[i] + perp_y * thickness_v)
        x_lower[i] = (x_midline[i] - perp_x * thickness_v)
        y_lower[i] = (y_midline[i] - perp_y * thickness_v)

    # Close the fish body shape by connecting the endpoints
    x_upper[-1] = (x_midline[-1] + perp_x * thickness_values[-1])
    y_upper[-1] = (y_midline[-1] + perp_y * thickness_values[-1])
    x_lower[-1] = (x_midline[-1] - perp_x * thickness_values[-1])
    y_lower[-1] = (y_midline[-1] - perp_y * thickness_values[-1])

    return (
        x_midline, y_midline,
        x_lower, y_lower,
        x_upper, y_upper,
    )

def plot_fish_configuration(
    axis       : plt.Axes,
    positions_x: np.ndarray,
    positions_y: np.ndarray,
    decorate   : bool = True,
    axis_lines : list = None,
    from_model : bool = True,
):
    ''' Plot fish configuration '''

    # Get thickness spline
    body_length      = 0.018

    if from_model:
        thickness_f      = get_fish_thickness_from_model
        thickness_spline = _get_fish_thickness_model_spline(body_length)
        thickness_args   = (body_length, thickness_spline)
    else:
        thickness_f      = get_fish_thickness_from_gazzola
        thickness_spline = None
        thickness_args   = (body_length,)

    (
        x_midline, y_midline,
        x_lower, y_lower,
        x_upper, y_upper,
    ) = get_fish_boundaries(
        positions_x    = positions_x,
        positions_y    = positions_y,
        thickness_f    = thickness_f,
        thickness_args = thickness_args,
        n_points       = 100,
    )

    # Plot the fish body shape
    if axis_lines is None:
        body_points   , = axis.plot(positions_x, positions_y, 'o', label="Fish Body Points", markersize=2, color='red')
        midline       , = axis.plot(x_midline, y_midline, 'k--', label="Midline")
        upper_boundary, = axis.plot(x_upper, y_upper, 'b-', label="Upper Boundary")
        lower_boundary, = axis.plot(x_lower, y_lower, 'b-', label="Lower Boundary")
        fish_body       = axis.fill_between(x_midline, y_lower, y_upper, color='lightblue', alpha=0.7, label="Fish Body")
    else:
        body_points, midline, upper_boundary, lower_boundary, fish_body = axis_lines
        body_points.set_data(positions_x, positions_y)
        midline.set_data(x_midline, y_midline)
        upper_boundary.set_data(x_upper, y_upper)
        lower_boundary.set_data(x_lower, y_lower)
        fish_body.remove()
        fish_body = axis.fill_between(x_midline, y_lower, y_upper, color='lightblue', alpha=0.7, label="Fish Body")

    axis_lines = [body_points, midline, upper_boundary, lower_boundary, fish_body]

    if decorate:
        axis.legend()

    return axis_lines

###############################################################################
# EXAMPLE #####################################################################
###############################################################################
def main():

    freq_t      = 3.0
    freq_x      = 1.0
    times       = np.linspace(0, 10, 1000)

    # Amplitude envelope
    bl = 0.018
    c2 = +0.28 * bl
    c1 = -0.13 * bl
    c0 = +0.05 * bl
    x  = np.linspace(0, bl, 16)
    s  = x / bl

    amp_envelope = c2*(s**2)+c1*s+c0

    # Evolve x and y positions
    positions_x_evolution = np.array( [ x for t in times ])
    positions_y_evolution = np.array(
        [
            amp_envelope * np.sin( 2*np.pi*(freq_x*s - freq_t * t))
            for t in times
        ]
    )

    # Plot fish configuration
    fig, ax = plt.subplots(figsize=(10, 5))

    axis_lines = plot_fish_configuration(
        axis        = ax,
        positions_x = positions_x_evolution[0],
        positions_y = positions_y_evolution[0],
    )

    def update(frame):
        nonlocal axis_lines
        axis_lines = plot_fish_configuration(
            axis        = ax,
            positions_x = positions_x_evolution[frame],
            positions_y = positions_y_evolution[frame],
            decorate    = False,
            axis_lines  = axis_lines
        )
        ax.set_title(f"Time: {times[frame]:.2f} s")

        return axis_lines

    ani = FuncAnimation(fig, update, frames=len(times), repeat=False)

    plt.show()


if __name__ == "__main__":
    main()
