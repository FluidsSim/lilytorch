import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

MODEL_POINTS_POSITIONS = np.array(
    [
        0.000,
        0.003,
        0.004,
        0.005,
        0.006,
        0.007,
        0.008,
        0.009,
        0.010,
        0.011,
        0.012,
        0.013,
        0.014,
        0.015,
        0.016,
        0.017,
        0.018,
    ]
)
MODEL_BODY_LENGTH          = MODEL_POINTS_POSITIONS[-1]
MODEL_POINTS_POSITIONS_REL = MODEL_POINTS_POSITIONS / MODEL_BODY_LENGTH

N_POINTS_MODEL   = len(MODEL_POINTS_POSITIONS_REL)
N_POINTS_PASSIVE = 2
N_POINTS_ACTIVE  = N_POINTS_MODEL - N_POINTS_PASSIVE

def compute_angles_from_coordinates(
    coordinates_xy: np.ndarray,
) -> np.ndarray:
    ''' Compute joint angles from point coordinates '''

    link_vectors = (
        coordinates_xy[:, +1:, :2] -
        coordinates_xy[:, :-1, :2]
    )
    links_vectors_norm = np.linalg.norm(link_vectors, axis=2)

    assert not np.any(links_vectors_norm == 0), 'Zero norm for link vectors'

    link_vectors = link_vectors / links_vectors_norm[:, :, np.newaxis]
    vects_1      = link_vectors[:, :-1]
    vects_2      = link_vectors[:, +1:]

    cos_angles = np.sum(vects_1 * vects_2, axis=2)
    sin_angles = np.cross(vects_1, vects_2, axis=2)

    return np.arctan2(sin_angles, cos_angles)

def compute_coordinates_from_arc_lengths(
    target_arclen: np.ndarray[float],
    points_arclen: np.ndarray[float],
    points_pos_x : np.ndarray[float],
    points_pos_y : np.ndarray[float],
    debug_plot   : bool = False,
):
    """
    Return points on the spline curve for desired arc lengths.

    Outside the interpolation interval, extrapolation is linear using
    the endpoint tangent (slope).
    """

    # Check they are sorted
    assert np.all(np.diff(points_arclen) > 0), 'points_arclen not sorted'
    assert np.all(np.diff(target_arclen) > 0), 'target_arclen not sorted'

    # Get spline
    spline_x = CubicSpline(points_arclen, points_pos_x, extrapolate=False)
    spline_y = CubicSpline(points_arclen, points_pos_y, extrapolate=False)

    s_min = points_arclen[0]
    s_max = points_arclen[-1]

    # Endpoint values
    x_min = spline_x(s_min)
    x_max = spline_x(s_max)
    y_min = spline_y(s_min)
    y_max = spline_y(s_max)

    # Endpoint slopes
    dx_min = spline_x(s_min, 1)
    dx_max = spline_x(s_max, 1)
    dy_min = spline_y(s_min, 1)
    dy_max = spline_y(s_max, 1)

    # Target points
    n_target     = len(target_arclen)
    target_pos_x = np.zeros(n_target)
    target_pos_y = np.zeros(n_target)

    for i, s in enumerate(target_arclen):

        if s < s_min:
            x = x_min + dx_min * (s - s_min)
            y = y_min + dy_min * (s - s_min)

        elif s > s_max:
            x = x_max + dx_max * (s - s_max)
            y = y_max + dy_max * (s - s_max)

        else:
            x = spline_x(s)
            y = spline_y(s)

        target_pos_x[i] = x
        target_pos_y[i] = y

    # DEBUG PLOT
    if debug_plot:
        plt.figure()
        plt.plot(points_pos_x, points_pos_y, 'o', label='original')
        plt.plot(target_pos_x, target_pos_y, '-', label='spline')
        plt.legend()
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Cubic spline interpolation with linear extrapolation')

    return target_pos_x, target_pos_y

def run_angles_extrapolation(
    speed_type: str,
):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    files_dict = {
        'slow' : 'ep248_Cl2_slow_fish13_XY_BL.csv',
        'fast' : 'ep223_Cl1_fast_fish13_XY_BL.csv',
    }
    file_path = os.path.join(current_dir, files_dict[speed_type])

    # Load the data
    positions_df = pd.read_csv(file_path)

    x_cols = [col for col in positions_df.columns if col.startswith('x')]
    y_cols = [col for col in positions_df.columns if col.startswith('y')]

    times    = positions_df['time_ms'].values / 1000.0
    x_vals   = positions_df[x_cols].values
    y_vals   = positions_df[y_cols].values
    pos_vals = np.stack([x_vals, y_vals], axis=2)

    n_steps  = pos_vals.shape[0]
    n_points = pos_vals.shape[1]

    # Get links lengths
    links_lengths = np.mean(
        np.linalg.norm(
            np.diff(pos_vals, axis=1),
            axis=2
        ),
        axis=0
    )
    links_cumsum    = np.cumsum(links_lengths)
    links_fraction  = links_cumsum / links_cumsum[-1]
    points_fraction = np.concatenate( [ [0], links_fraction ] )

    # Get points arclengths
    arc_pos_0 = MODEL_POINTS_POSITIONS_REL[1]
    arc_pos_1 = MODEL_POINTS_POSITIONS_REL[N_POINTS_ACTIVE - 1]

    points_arclen = arc_pos_0 + points_fraction * (arc_pos_1 - arc_pos_0 )

    ###########################################################################
    # MAP DLC POINTS TO MODEL
    ###########################################################################

    model_arclen = MODEL_POINTS_POSITIONS_REL
    n_model      = len(model_arclen)

    model_pos_x = np.zeros( (n_steps, n_model) )
    model_pos_y = np.zeros( (n_steps, n_model) )

    n_plots      = 5
    n_plots_jump = max(1, n_steps // n_plots)

    for step in range(n_steps):
        (
            model_pos_x[step, :],
            model_pos_y[step, :]
        ) = compute_coordinates_from_arc_lengths(
            target_arclen = model_arclen,
            points_arclen = points_arclen,
            points_pos_x  = x_vals[step, :],
            points_pos_y  = y_vals[step, :],
            debug_plot    = (step % n_plots_jump == 0),
        )

    # Compute original angles
    # original_angles = compute_angles_from_coordinates(
    #     coordinates_xy = pos_vals
    # )

    # Compute model angles
    model_pos    = np.stack([model_pos_x, model_pos_y], axis=2)
    model_angles = compute_angles_from_coordinates(
        coordinates_xy = model_pos
    )

    # Save as XSLX file
    model_angles_dict = (
        { 'time': times.tolist() } |
        {
            f'Joint_{j}': model_angles[:, j].tolist()
            for j in range(n_model - 2)
        }
    )

    # Convert to DataFrame
    columns_order = ['time'] + [ f'Joint_{a}' for a in range(n_model - 2) ]

    model_angles_df = pd.DataFrame(model_angles_dict)
    model_angles_df = model_angles_df[columns_order]

    # Save as XLSX
    model_angles_path = file_path.replace('XY_BL.csv', 'model_angles.xlsx')
    model_angles_df.to_excel(model_angles_path, index=False)

    # Filter angles
    from scipy.signal import butter, filtfilt

    def butter_lowpass_filter(sig, cut_lp, fs, order=5):
        f_nyq = 0.5 * fs
        wn    = cut_lp / f_nyq
        b, a  = butter(order, wn, btype='low', analog=False)
        sig_f = filtfilt(b, a, sig)
        return sig_f

    # Apply low-pass filter to model angles
    fcut_lp    = 30.0
    timestep   = times[1] - times[0]
    f_sampling = 1.0 / timestep

    filtered_angles = np.zeros_like(model_angles)
    for j in range(model_angles.shape[1]):
        filtered_angles[:, j] = butter_lowpass_filter(
            sig    = model_angles[:, j],
            cut_lp = fcut_lp,
            fs     = f_sampling
        )

    # Plot angles
    plt.figure()
    plt.plot(model_angles)
    plt.title('Model Joint Angles')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (rad)')
    plt.grid()

    # Plot filtered angles
    plt.figure()
    plt.plot(filtered_angles)
    plt.title('Filtered Model Joint Angles')
    plt.xlabel('Time Step')
    plt.ylabel('Angle (rad)')
    plt.grid()

    # Plot positions
    n_steps = model_pos.shape[0]
    steps   = range(0, n_steps, max(1, n_steps // 20))
    plt.figure()
    for step in steps:
        plt.plot(
            model_pos[step, :, 0],
            model_pos[step, :, 1],
            'o-',
            label=f'Step {step}'
        )
    plt.legend()
    plt.title('Model Points Positions')
    plt.axis('equal')
    plt.grid()

    plt.show()

    return


if __name__ == '__main__':
    run_angles_extrapolation('slow')
    # run_angles_extrapolation('fast')
















