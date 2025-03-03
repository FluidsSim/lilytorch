
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from scipy.signal import find_peaks
from scipy.interpolate import CubicSpline, interp1d

from matplotlib.collections import PolyCollection
from matplotlib.animation import FuncAnimation

from lilytorch.scripts.zebrafish_files.interpolate_experimental_angles import create_fictive_schooling_trace
from lilytorch.scripts.zebrafish_files.load_data import get_experimental_signal

###############################################################################
# SPLINE ######################################################################
###############################################################################

def compute_body_interpolation(
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

    # Cubic spline interpolation
    # interp_x = CubicSpline(arc_lengths, x_coords)
    # interp_y = CubicSpline(arc_lengths, y_coords)

    # Linear interpolation
    interp_x = lambda s: np.interp(s, arc_lengths, x_coords)
    interp_y = lambda s: np.interp(s, arc_lengths, y_coords)

    return interp_x, interp_y

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

def get_fish_half_thickness_model(
    normalized_arclengths: np.ndarray,
    fish_length          : float,
):
    '''
    Fish thickness model spline
    Derived from the mechanical model of the zebrafish body
    '''
    s_coords_ref = np.array(
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
    y_coords_ref = np.array(
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
    scaling       = fish_length / s_coords_ref[-1]
    s_coords_ref *= scaling
    y_coords_ref *= scaling

    # Compute thickness spline
    thickness_spline = interp1d(s_coords_ref, y_coords_ref, kind='linear')

    # Compute thickness
    arclengths       = normalized_arclengths * fish_length
    thickness_values = thickness_spline(arclengths)

    return  thickness_values

def get_fish_half_thickness_liu(
    normalized_arclengths: np.ndarray,
    length               : float,
):
    """
    Fish thickness
    See Liu et al. 2021
    # NOTE: Equation defines the thickness
    """

    L = length
    s = normalized_arclengths
    x = s * L

    s1, s2, s3, s4, w1, w2 = 0.54, 0.72, 0.83, 0.85, 0.16, 0.004

    s_star  = x / (s4 * L)

    coef = np.array( [      0.2969, -0.1260,  -0.3516,    0.2843,   -0.1015 ] )
    vals = np.array( [ s_star**0.5, s_star, s_star**2, s_star**3, s_star**4 ] )

    # Conditions
    cond1 = ( 0 <= s) & (s <  s3)
    cond2 = (s3 <= s) & (s <=  1)

    thickness = np.zeros_like(s)
    thickness[cond1] = 5 * (s4 * w1) * np.sum( coef * vals.T, axis=1 )[cond1]
    thickness[cond2] = w2

    # Normalize
    actual_max       = np.max(thickness[cond1])
    theory_max       = 0.16
    thickness[cond1] = thickness[cond1] * theory_max / actual_max

    # Scale by length
    thickness = thickness * L / 2

    return thickness

def get_fish_half_thickness_gazzola(
    normalized_arclengths: np.ndarray,
    length               : float,
):
    """
    Fish thickness
    See Gazzola et al. 2011
    # NOTE: Equation defines the half-thickness
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
    positions_x: np.ndarray,
    positions_y: np.ndarray,
    thickness_f: callable = get_fish_half_thickness_gazzola,
    n_points   : int = 100,
    fish_length: float = 0.018,

):
    # Compute spline
    (
        spline_x,
        spline_y
    ) = compute_body_interpolation(
        coordinates_x = positions_x,
        coordinates_y = positions_y,
    )

    # Generate midline points and thickness
    s_values         = np.linspace(0, 1, n_points)
    x_midline        = spline_x(s_values)
    y_midline        = spline_y(s_values)
    thickness_values = thickness_f(s_values, fish_length)

    # thk_mod = get_fish_half_thickness_model(s_values, fish_length)
    # thk_liu = get_fish_half_thickness_liu(s_values, fish_length)
    # thk_gaz = get_fish_half_thickness_gazzola(s_values, fish_length)

    # plt.figure()
    # plt.plot(s_values * fish_length, +thk_mod, label='mod', color='red')
    # plt.plot(s_values * fish_length, -thk_mod, label= None, color='red')
    # plt.plot(s_values * fish_length, +thk_liu, label='liu', color='green')
    # plt.plot(s_values * fish_length, -thk_liu, label= None, color='green')
    # plt.plot(s_values * fish_length, +thk_gaz, label='gaz', color='blue')
    # plt.plot(s_values * fish_length, -thk_gaz, label= None, color='blue')
    # plt.axis('equal')
    # plt.legend()

    # Calculate the fish body shape by offsetting along a perpendicular vector
    x_upper = []
    y_upper = []
    x_lower = []
    y_lower = []

    for i in range(len(s_values) - 1):
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
        x_upper.append(x_midline[i] + perp_x * thickness_v)
        y_upper.append(y_midline[i] + perp_y * thickness_v)
        x_lower.append(x_midline[i] - perp_x * thickness_v)
        y_lower.append(y_midline[i] - perp_y * thickness_v)

    # Close the fish body shape by connecting the endpoints
    x_upper.append(x_midline[-1] + perp_x * thickness_values[-1])
    y_upper.append(y_midline[-1] + perp_y * thickness_values[-1])
    x_lower.append(x_midline[-1] - perp_x * thickness_values[-1])
    y_lower.append(y_midline[-1] - perp_y * thickness_values[-1])

    return (
        x_midline, y_midline,
        x_lower, y_lower,
        x_upper, y_upper,
    )

def plot_fish_configuration(
    positions_x: np.ndarray,
    positions_y: np.ndarray,
    decorate   : bool = True,
    fish_length: float = 0.018,
):
    ''' Plot fish configuration '''

    (
        x_midline, y_midline,
        x_lower, y_lower,
        x_upper, y_upper,
    ) = get_fish_boundaries(
        positions_x = positions_x,
        positions_y = positions_y,
        fish_length = fish_length,
    )

    # Plot the fish body shape
    lines_points  = plt.plot(positions_x, positions_y, 'o')
    lines_midline = plt.plot(x_midline, y_midline, 'k--')
    lines_upper   = plt.plot(x_upper, y_upper, 'b-')
    lines_lower   = plt.plot(x_lower, y_lower, 'b-')
    fish_body_fill = plt.fill_between(x_midline, y_lower, y_upper, color='lightblue', alpha=0.7)

    if decorate:
        plt.axis('equal')
        plt.xlabel("X")
        plt.ylabel("Y")

        x_min, x_max = np.amin(positions_x), np.amax(positions_x)
        y_min, y_max = np.amin(positions_y), np.amax(positions_y)
        x_mid, y_mid = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
        p_range      = np.amax( [x_max - x_min, y_max - y_min] )
        p_tol        = 0.6 * p_range

        plt.xlim(x_mid - p_tol, x_mid + p_tol)
        plt.ylim(y_mid - p_tol, y_mid + p_tol)

        plt.title("Fish Body with Arclength-Dependent Thickness")

    return lines_points, lines_midline, lines_upper, lines_lower, fish_body_fill

def update_fish_configuration(
    ax            : plt.Axes,
    lines_points  : list,
    lines_midline : list,
    lines_upper   : list,
    lines_lower   : list,
    fish_body_fill: PolyCollection,
    positions_x   : np.ndarray,
    positions_y   : np.ndarray,
    fish_length   : float = 0.018,
):
    ''' Plot fish configuration '''

    (
        x_midline, y_midline,
        x_lower, y_lower,
        x_upper, y_upper,
    ) = get_fish_boundaries(
        positions_x = positions_x,
        positions_y = positions_y,
        fish_length = fish_length,
    )
    # Update the fish body shape
    lines_points[0].set_data(positions_x, positions_y)
    lines_midline[0].set_data(x_midline, y_midline)
    lines_upper[0].set_data(x_upper, y_upper)
    lines_lower[0].set_data(x_lower, y_lower)

    # Update the fill between the upper and lower boundaries
    # path.vertices[:, 1]
    x_path = np.concatenate([x_midline, x_midline[::-1]])
    y_path = np.concatenate([y_lower, y_upper[::-1]])
    verts  = np.array([x_path, y_path]).T

    fish_body_fill.set_verts([verts])

    return

###############################################################################
# MAIN ########################################################################
###############################################################################
def main_analytical():

    timestep = 0.001
    duration = 30.0

    frequency     = 3.5
    wavefrequency = 0.95

    times   = np.arange(0, duration, timestep)
    n_steps = len(times)

    # Envelope
    ref_points_x = np.array([0.0, 44.5, 66.5, 110.0]) / 110.0
    ref_points_y = np.array([0.04, 0.05, 0.16, 0.24])
    envelope_fun = CubicSpline(ref_points_x, ref_points_y, bc_type=((1, 0.1), (1, 0.1)))

    # From Di Santo et al. 2021
    body_length  = 0.018
    positions_x  = np.linspace(0, 1, 100) * body_length
    positions_y  = np.zeros_like(positions_x)
    s_vals       = positions_x / body_length
    amplitudes_y = body_length * envelope_fun(s_vals)

    # Evolve x and y according to the time

    positions_x_evolution = np.array([positions_x] * n_steps)

    positions_y_evolution = np.array(
        [
            [
                amp * np.sin(
                    2 * np.pi * frequency * time -
                    2 * np.pi * wavefrequency * s_vals[p_ind]
                )

                for p_ind, amp in enumerate(amplitudes_y)
            ]
            for time in times
        ]
    )

    # Plot fish configuration
    fig, ax = plt.subplots(figsize=(10, 5))

    (
        lines_points,
        lines_midline,
        lines_upper,
        lines_lower,
        fish_body_fill
    ) = plot_fish_configuration(
        positions_x = positions_x,
        positions_y = positions_y,
        decorate    = True,
    )

    def update(frame):
        update_fish_configuration(
            ax,
            lines_points,
            lines_midline,
            lines_upper,
            lines_lower,
            fish_body_fill,
            positions_x = positions_x_evolution[frame],
            positions_y = positions_y_evolution[frame],
        )
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        ax.set_title(f"Time: {times[frame]:.2f} s")

        return ax, lines_points, lines_midline, lines_upper, lines_lower, fish_body_fill

    ani = FuncAnimation(
        fig,
        update,
        frames   = np.arange(0, n_steps, 10),
        repeat   = False,
        interval = 1
    )

    plt.show()

def main_experimental():

    folder_name = 'lilytorch/scripts/zebrafish_files/data'
    file_name   = 'kinematics_recording.csv'

    save_data = True
    plot_data = False

    # KI reference: 12100 (Containing chunk: 11300)
    # KI reference: 12260 (Containing chunk: 12400)

    target_fish     = 'Fish3'
    start_recording = 11300 # 12100
    end_recording   = 12400 # 12260

    # Get the frequency-scaled signal
    timestep       = 0.001
    total_duration = 30
    freq_scaling   = 0.30
    filter_freqs   = [1.0, 10]

    # Get the signal
    scaled_signals_df = get_experimental_signal(
        folder_name     = folder_name,
        file_name       = file_name,
        target_fish     = target_fish,
        start_recording = start_recording,
        end_recording   = end_recording,
        timestep        = timestep,
        total_duration  = total_duration,
        freq_scaling    = freq_scaling,
        save_data       = save_data,
        plot_data       = plot_data,
        filter_freqs    = filter_freqs,
    )

    times   = scaled_signals_df['time'].values
    n_steps = len(times)

    # Fish properties
    # positions_x_evolution = np.array([positions_x] * n_steps)
    positions_x_evolution = scaled_signals_df.filter(regex='x_').values
    positions_y_evolution = scaled_signals_df.filter(regex='y_').values

    positions_x_evolution[:] = np.mean(positions_x_evolution, axis=0)

    # Plot fish configuration
    fig, ax = plt.subplots(figsize=(10, 5))

    (
        lines_points,
        lines_midline,
        lines_upper,
        lines_lower,
        fish_body_fill
    ) = plot_fish_configuration(
        positions_x = positions_x_evolution[0],
        positions_y = positions_y_evolution[0] * 0,
        decorate    = True,
        fish_length = 1.0,
    )

    def update(frame):
        update_fish_configuration(
            ax,
            lines_points,
            lines_midline,
            lines_upper,
            lines_lower,
            fish_body_fill,
            positions_x = positions_x_evolution[frame],
            positions_y = positions_y_evolution[frame],
            fish_length = 1.0,
        )
        # ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y_min, y_max)
        ax.set_title(f"Time: {times[frame]:.2f} s")

        return ax, lines_points, lines_midline, lines_upper, lines_lower, fish_body_fill

    ani = FuncAnimation(
        fig,
        update,
        frames   = np.arange(0, n_steps, 5),
        repeat   = False,
        interval = 1
    )

    plt.show()



if __name__ == "__main__":
    main_analytical()
    # main_experimental()
