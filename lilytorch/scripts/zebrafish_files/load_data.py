"""
Extract the DLC data in world coordinates
"""

import os

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
from scipy.signal import butter, filtfilt, find_peaks

SAMPLING_RATE = 240

###############################################################################
# PLOTTING ####################################################################
###############################################################################

def plot_individuals_intervals(
    data_episode,
    individuals,
):
    ''' Plot the intervals of good points for each individual '''

    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111)

    colors = plt.cm.viridis(np.linspace(0, 1, len(individuals)))

    for i, ind in enumerate(individuals):
        good_points_limits =  [
            data_episode[ind]['first_good_point'],
            data_episode[ind]['last_good_point'],
        ]

        good_chunks_limits = [
            [ chunk[0], chunk[-1] ]
            for chunk in data_episode[ind]['good_chunks']
        ]

        best_chunk_limits = [
            data_episode[ind]['best_chunk_start'],
            data_episode[ind]['best_chunk_end'],
        ]

        y_good = np.ones(2) * i - 0.1
        y_best = np.ones(2) * i + 0.1

        ax.plot( good_points_limits, y_good, lw = 10, c = colors[i] )
        ax.plot(  best_chunk_limits, y_best, lw = 10, c = 'g' )

        for chunk_limits in good_chunks_limits:
            y_chunk = np.ones(2) * i
            ax.plot(chunk_limits, y_chunk, lw = 5, c = 'b', marker='o')


    plt.ylim(-1, len(individuals))
    plt.yticks(np.arange(len(individuals)), individuals)

    return fig

def plot_points_positions(
    x_coords_det,
    y_coords_det,
    points_names,
    sampling_rate,
):
    ''' Plot the detrended data '''

    steps_recording = x_coords_det.shape[1]
    n_points        = len(points_names)

    steps         = np.arange(0, steps_recording, 1)
    times         = steps / sampling_rate
    colors_steps  = plt.cm.viridis(np.linspace(0, 1, steps_recording))
    colors_points = plt.cm.viridis(np.linspace(0, 1, n_points))

    # Over space
    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111)
    ax.set_aspect('equal', adjustable='datalim')

    for step in steps:
        plt.plot(
            x_coords_det[:, step],
            y_coords_det[:, step],
            'o-',
            c = colors_steps[step],
        )

    plt.xlabel('Normalized x-coord')
    plt.ylabel('Normalized y-coord')

    # Over time
    fig = plt.figure(figsize=(10, 10))
    ax  = fig.add_subplot(111)

    for point in range(n_points):
        plt.plot(
            times,
            y_coords_det[point, :],
            'o-',
            c     = colors_points[point],
            label = points_names[point],
        )

    plt.xlabel('Time (s)')
    plt.ylabel('Normalized y-coord')
    plt.legend()

    return

###############################################################################
# DATA LOADING ################################################################
###############################################################################

def load_kinematics_data_single_episode(
    folder_name: str,
    file_name  : str,
    plot_data  : bool,
) -> dict:
    ''' Get kinematics data from a single episode in the database '''

    file_path = os.path.join(folder_name, file_name)

    print(f"Processing the dataset {file_path}")

    df_episode = pd.read_csv(file_path, header=[0,1,2,3], index_col=0)

    # Remove 'scorer' level
    df_episode.columns = df_episode.columns.droplevel('scorer')

    # Individuals
    exclude_individuals = ['single']
    individuals         = df_episode.columns.get_level_values('individuals')
    _, idx              = np.unique(individuals, return_index=True)
    individuals         = individuals[np.sort(idx)]
    individuals         = [i for i in individuals if i not in exclude_individuals]

    # Point names
    exclude_points = ['Wall1', 'Wall2']
    points_names   = df_episode.columns.get_level_values('bodyparts')
    _, idx         = np.unique(points_names, return_index=True)
    points_names   = points_names[np.sort(idx)]
    points_names   = [p for p in points_names if p not in exclude_points]


    # COLLECT ALL DATA
    data_episode = {}

    for individual in individuals:

        # Get the data for the individual
        df_individual   = df_episode.xs(individual, level='individuals', axis=1)
        data_individual = {}

        for point_index, point_name in enumerate(points_names):

            # Get the data for the body part
            point_vals = df_individual.xs(point_name, level='bodyparts', axis=1)

            # Get the x, y, and likelihood values
            data_individual[point_name] = {
                'name'       : point_name,
                'index'      : point_index,
                'x'          : point_vals['x'].values,
                'y'          : point_vals['y'].values,
                'likelyhood' : point_vals['likelihood'].values,
            }

        # Get average likelyhood
        average_likelihood = np.mean(
            [
                data_individual[point_name]['likelyhood']
                for point_name in points_names
            ],
            axis=0,
        )

        # Get good points
        good_points = np.where(average_likelihood > 0.95)[0]

        first_good_point = good_points[0]
        last_good_point  = good_points[-1]

        # Get largest chunk of good points
        good_chunks = np.split(good_points, np.where(np.diff(good_points) != 1)[0]+1)
        good_chunks = [chunk for chunk in good_chunks if len(chunk) > 10]
        good_chunks = sorted(good_chunks, key=lambda x: len(x), reverse=True)

        # Get start and end of the largest chunk
        best_chunk_start  = good_chunks[0][0]
        best_chunk_end    = good_chunks[0][-1]

        data_individual['good_points']      = good_points
        data_individual['good_chunks']      = good_chunks
        data_individual['first_good_point'] = first_good_point
        data_individual['last_good_point']  = last_good_point
        data_individual['best_chunk_start'] = best_chunk_start
        data_individual['best_chunk_end']   = best_chunk_end

        data_episode[individual] = data_individual


    # Plot the individuals intervals
    if plot_data:
        fig = plot_individuals_intervals(
            data_episode,
            individuals,
        )

    return data_episode, points_names

###############################################################################
# DATA PROCESSING #############################################################
###############################################################################

def interpolate_signal_over_time(times, signal, times_sampled):
    ''' Interpolates signal. '''

    # Cubic spline interpolation
    func_interpolate    = CubicSpline(times, signal)
    interpolated_values = func_interpolate(times_sampled)

    # Linear interpolation
    # interpolated_values = np.interp(times_sampled, times, signal)

    return interpolated_values

def filter_signal(
    signal    : np.ndarray,
    signal_dt : float,
    fcut_hp   : float = None,
    fcut_lp   : float = None,
    filt_order: int = 5,
    pad_type  : str = 'odd',
):
    ''' Butterwort, zero-phase filtering '''

    # Nyquist frequency
    fnyq = 0.5 / signal_dt

    # Filters
    if fcut_hp is not None:
        num, den = butter(filt_order, fcut_hp/fnyq,  btype= 'highpass')
        signal  = filtfilt(num, den, signal, padtype= pad_type)

    if fcut_lp is not None:
        num, den = butter(filt_order, fcut_lp/fnyq, btype= 'lowpass' )
        signal  = filtfilt(num, den, signal, padtype= pad_type)

    return signal

def get_frequency_scaled_signal(
    signal        : np.ndarray,
    timestep      : float,
    total_duration: float,
    freq_scaling  : float,
):
    ''' Get frequency-scaled signal. '''

    # Flatten signal
    signal = signal.flatten()

    # INTERPOLATE
    n_samples         = len(signal)
    sampling_interval = 1 / SAMPLING_RATE
    duration_scaling  = 1 / freq_scaling
    sampling_times    = np.arange(n_samples) * sampling_interval * duration_scaling

    t_start = sampling_times[0]
    t_end   = sampling_times[-1] + sampling_interval

    times_interpolated = np.arange( t_start, t_end, timestep)

    signal_interpolated = interpolate_signal_over_time(
        times         = sampling_times,
        signal        = signal,
        times_sampled = times_interpolated,
    )

    # REPEAT SIGNAL
    times_repeated   = np.arange(0, total_duration + timestep, timestep)
    n_times_repeated = len(times_repeated)

    signal_repeats   = int( np.ceil( total_duration / times_interpolated[-1] ) )
    signal_repeated  = np.tile(signal_interpolated, signal_repeats)
    signal_repeated  = signal_repeated[:n_times_repeated]

    return times_repeated, signal_repeated

def compute_frequency_from_signal(
    signals_df: pd.DataFrame,
    timestep  : float,
    verbose   : bool = True,
    max_freq  : float = 20.0,
):
    ''' Compute frequency from signal. '''

    y_coords = signals_df.filter(regex='y_').values.T

    points_mean_frequencies = []

    for y_coord in y_coords:

        # Find peaks
        min_width = 1 / max_freq / timestep
        peaks, _  = find_peaks(y_coord, width = min_width)

        # Compute frequency
        peak_intervals = np.diff(peaks) * timestep
        frequencies    = 1 / peak_intervals
        mean_freq      = np.mean(frequencies)
        points_mean_frequencies.append(mean_freq)

    if not verbose:
        return points_mean_frequencies

    print("Cycle frequencies for each y_coord signal:")
    for i, mean_freq in enumerate(points_mean_frequencies):
        print(f"y_coord {i}: {mean_freq}")

    print(f'Average cycle frequency: {np.mean(points_mean_frequencies)}')

    return points_mean_frequencies

###############################################################################
# DATA EXTRACTION #############################################################
###############################################################################

def get_target_signal_from_episode_data(
    data_episode   : dict,
    target_fish    : str,
    start_recording: int,
    end_recording  : int,
    plot_data      : bool,
) -> pd.DataFrame:
    ''' Get the target signal from the data '''

    data_fish       = data_episode[target_fish]
    steps_recording = end_recording - start_recording

    # Get points names from the data
    points_names      = [
        data_fish[key]['name']
        for key in data_fish.keys()
        if key not in [
                 'good_points',     'good_chunks',
            'first_good_point', 'last_good_point',
            'best_chunk_start',  'best_chunk_end',
        ]
    ]

    # Get the x and y coordinates
    x_coords = np.array( [ data_fish[p]['x'] for p in points_names ] )
    y_coords = np.array( [ data_fish[p]['y'] for p in points_names ] )

    # Target inds
    x_coords = x_coords[:, start_recording:end_recording]
    y_coords = y_coords[:, start_recording:end_recording]

    ### Detrend x_coords
    mean_x_coords = np.mean( x_coords - x_coords[0, :], axis=1 )
    x_coords_det = np.array(
        [mean_x_coords] * steps_recording
    ).T

    ### Detrend y_coords
    steps        = np.arange(steps_recording)
    steps_det    = np.stack([steps] * len(points_names), axis=0)
    y_coords_det = np.zeros_like(y_coords)

    # Over time
    fitting_line_t = np.polynomial.Polynomial.fit(
        steps_det.flatten(),
        y_coords.flatten(),
        deg = 3,
    )

    for i in range(x_coords_det.shape[0]):
        y_coords_det[i, :] = y_coords[i, :] - fitting_line_t(steps)

    # Over space
    fitting_line_x = np.polynomial.Polynomial.fit(
        x_coords_det.flatten(),
        y_coords_det.flatten(),
        deg = 1,
    )

    for i in range(x_coords_det.shape[0]):
        y_coords_det[i, :] = y_coords_det[i, :] - fitting_line_x(x_coords_det[i, :])

    # Normalize coords
    body_length = np.amax(x_coords_det) - np.amin(x_coords_det)
    x_coords_det = x_coords_det / body_length
    y_coords_det = y_coords_det / body_length

    data_detrended = np.concatenate(
        (x_coords_det, y_coords_det),
        axis = 0,
    ).T

    df_detrended = pd.DataFrame(
        data_detrended,
        columns = [
            [ f'x_{p}' for p in points_names ] +
            [ f'y_{p}' for p in points_names ]
        ],
    )

    # Plot the detrended data
    if plot_data:
        plot_points_positions(
            x_coords_det  = x_coords_det,
            y_coords_det  = y_coords_det,
            points_names  = points_names,
            sampling_rate = SAMPLING_RATE,
        )

    return df_detrended

def get_experimental_signal(
    folder_name    : str,
    file_name      : str,
    target_fish    : str,
    start_recording: int,
    end_recording  : int,
    timestep       : float,
    total_duration : float,
    freq_scaling   : float,
    save_data      : bool,
    plot_data      : bool,
    filter_freqs   : tuple = (1.0, 10.0),
):

    # Load the kinematics data
    data_episode, points_names = load_kinematics_data_single_episode(
        folder_name = folder_name,
        file_name   = file_name,
        plot_data   = plot_data,
    )

    # Get the target signal
    target_signals_df = get_target_signal_from_episode_data(
        data_episode    = data_episode,
        target_fish     = target_fish,
        start_recording = start_recording,
        end_recording   = end_recording,
        plot_data       = plot_data,
    )

    original_chunk_duration = ( end_recording - start_recording ) / SAMPLING_RATE
    scaled_chunk_duration   = original_chunk_duration / freq_scaling
    print(f"Original chunk duration: {original_chunk_duration:.2f} s")
    print(f"Scaled chunk duration  : {  scaled_chunk_duration:.2f} s")

    # Create a dataframe with the same headers as the target_signals_df
    scaled_signals_dict = {}

    for column in target_signals_df.columns:

        column_name = column[0]

        scaled_times, scaled_signal = get_frequency_scaled_signal(
            signal         = target_signals_df[column_name].values,
            timestep       = timestep,
            total_duration = total_duration,
            freq_scaling   = freq_scaling,
        )

        if column_name.startswith('x_'):
            scaled_signals_dict[column_name] = scaled_signal
            continue

        # Filter the signal
        scaled_signal_f = filter_signal(
            signal    = scaled_signal,
            signal_dt = timestep,
            fcut_hp   = filter_freqs[0],
            fcut_lp   = filter_freqs[1],
        )

        scaled_signals_dict[column_name] = scaled_signal_f

    scaled_signals_dict['time'] = scaled_times
    scaled_signals_df           = pd.DataFrame(scaled_signals_dict)
    scaled_signals_df           = (
        scaled_signals_df[
            ['time'] +
            [col for col in scaled_signals_df.columns if col != 'time']
        ]
    )

    # Save the scaled signals as csv
    if save_data:
        scaled_signals_df.to_csv(
            os.path.join(folder_name, 'kinematics_signals.csv'),
            index=False,
        )

    # Plot the scaled signals
    if plot_data:
        plot_points_positions(
            x_coords_det  = scaled_signals_df.filter(regex='x_').values.T,
            y_coords_det  = scaled_signals_df.filter(regex='y_').values.T,
            points_names  = points_names,
            sampling_rate = 1 / timestep,
        )

    return scaled_signals_df

###############################################################################
# MAIN ########################################################################
###############################################################################

def main():

    folder_name = 'lilytorch/scripts/zebrafish_files/data'
    file_name   = 'kinematics_recording.csv'

    save_data = False
    plot_data = True

    # KI reference: 12100 (Containing chunk: 11300)
    # KI reference: 12260 (Containing chunk: 12400)

    target_fish     = 'Fish3'
    start_recording = 11300 # 12100
    end_recording   = 12400 # 12260

    # Get the frequency-scaled signal
    timestep       = 0.001
    total_duration = 30
    freq_scaling   = 0.25

    # Load the signal
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
    )

    print('Loaded the data and extracted the target signal')

    # Compute the frequency from the signal
    points_mean_frequencies = compute_frequency_from_signal(
        signals_df = scaled_signals_df,
        timestep   = timestep,
        verbose    = False,
        max_freq   = 20.0,
    )

    plt.show()

    return


if __name__ == '__main__':
    main()