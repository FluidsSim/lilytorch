
import os
import sys
import inspect
import numpy as np
import random
from lilytorch.util.paths import save_path
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.sensors.sensor_convention import sc
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 16})
from lilytorch.util.mp_util import sweep_1d

data_dir = os.path.join(save_path, "pinned_2guilla_exp_5")

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, CURRENTDIR)
from phase_analysis import get_schooling_data
import yaml




def get_swimmer_data(
    animat_id    : int,
    animats_list : dict,
):
    ''' Get swimmer data from HDF5 file '''

    # States
    sensors_data  = animats_list[animat_id]['sensors']
    joints_states = sensors_data['joints']['array']
    links_states  = sensors_data['links']['array']

    # Positions
    joints_pos  = joints_states[:, :, sc.joint_position]
    links_pos_x = links_states[:, :, sc.link_com_position_x]
    links_pos_y = links_states[:, :, sc.link_com_position_y]
    links_pos   = np.stack((links_pos_x, links_pos_y), axis=2)

    # Data
    swimmer_data = {
        'joints_pos': joints_pos,
        'links_pos' : links_pos,
    }
    return swimmer_data


def main(data_dir, color):
    ''' Analyze simulation data from HDF5 file '''

    file_path = os.path.join(data_dir, "output", "simulation.hdf5")

    animat_config_path0 = os.path.join(data_dir, "animat_config_0.yaml")
    animat_config_path1 = os.path.join(data_dir, "animat_config_1.yaml")

    with open(animat_config_path0, 'r') as f:
        animat_config0 = yaml.unsafe_load(f)
    with open(animat_config_path1, 'r') as f:
        animat_config1 = yaml.unsafe_load(f)

    freq0 = animat_config0["extensions"][0]["config"]["freq"]
    freq1 = animat_config1["extensions"][0]["config"]["freq"]

    # Load simulation data
    simulation_data = hdf5_to_dict(file_path)
    times           = simulation_data['times']
    animats_list    = simulation_data['animats']

    # Get swimmers data
    swimmer_data_1 = get_swimmer_data(
        animat_id    = 0,
        animats_list = animats_list,
    )
    swimmer_data_2 = get_swimmer_data(
        animat_id    = 1,
        animats_list = animats_list,
    )

    # Get schooling data
    times, freq1_computed, freq2_computed, phases = get_schooling_data(
        times        = times,
        joints_pos_1 = swimmer_data_1['joints_pos'],
        joints_pos_2 = swimmer_data_2['joints_pos'],
        links_pos_1  = swimmer_data_1['links_pos'],
        links_pos_2  = swimmer_data_2['links_pos'],
        plotting     = True,
        data_dir     = data_dir,
        freq0         = freq0,
        freq1         = freq1,
    )


    # Remove variables with times between 1 and 19
    mask = (times > 15) & (times < 35)
    times = times[mask]
    phases = phases[mask]
    freq1_computed = freq1_computed[mask]
    freq2_computed = freq2_computed[mask]

    # # plt.plot(times, freq1_computed, label='Swimmer 1', alpha=0.7, color=color)
    # plt.plot(times, freq2_computed, label='Swimmer 2', alpha=0.7, color=color, linestyle='--')
    # plt.axhline(y=freq0, color=color, linestyle=':')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')
    # plt.title('Computed Frequencies')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()

    # plt.plot(times, phases)
    # plt.xlabel('Time (s)')
    # plt.ylabel('Phase Difference (rad)')
    # plt.title('Phase Difference Between Swimmers')
    # plt.grid(True)
    # plt.tight_layout()


    # # Plot
    # plot_schooling_data(schooling_data)

    return times, phases, freq1, freq1_computed, freq2_computed


if __name__ == "__main__":

    dirs = os.listdir(data_dir)
    all_phases = []
    all_freqs = []
    all_freqs_computed_1 = []
    all_freqs_computed_2 = []
    # Use a continuous colormap (e.g., viridis) for better color distinction
    colors = plt.cm.viridis(np.linspace(0, 1, len(dirs)))

    # output = sweep_1d(main, subdirs, num_process=16)
    # subdirs = [d for d in dirs if os.path.isdir(os.path.join(data_dir, d))]

    for subdir in dirs:
        print(f'Analyzing data in {os.path.join(data_dir, subdir)}...')
        try:
            times, phases, freq1, freq1_computed, freq2_computed = main(os.path.join(data_dir, subdir), colors[dirs.index(subdir)])
            all_phases.append(phases)
            all_freqs.append(freq1)
            all_freqs_computed_1.append(freq1_computed)
            all_freqs_computed_2.append(freq2_computed)

        except Exception as e:
            print(f"Skipping {subdir} due to error: {e}")
            all_phases.append(np.full_like(times if 'times' in locals() else np.array([np.nan]), np.nan))
            all_freqs.append(np.nan)
            all_freqs_computed_1.append(np.full_like(times if 'times' in locals() else np.array([np.nan]), np.nan))
            all_freqs_computed_2.append(np.full_like(times if 'times' in locals() else np.array([np.nan]), np.nan))


    # Reorder all_phases and all_freqs according to increasing all_freqs
    sorted_indices       = np.argsort(all_freqs)
    all_phases           = np.array(all_phases)[sorted_indices]
    all_freqs_sorted            = np.array(all_freqs)[sorted_indices]
    all_freqs_computed_1 = np.array(all_freqs_computed_1)[sorted_indices]
    all_freqs_computed_2 = np.array(all_freqs_computed_2)[sorted_indices]
    all_phases_converted = (all_phases + 2 * np.pi) % (2 * np.pi) # wrap to [0, 2pi]




    plt.figure()
    nbins = 10
    bins = np.linspace(0, 2*np.pi, nbins)
    zbin = []
    for i, phases in enumerate(all_phases_converted):
        hist, bin_edges = np.histogram(phases, bins=bins, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(centers, hist + i*0.1, label=f'Freq: {all_freqs_sorted[i]:.2f} Hz', color=colors[sorted_indices[i]])
        zbin.append(hist)
    zbin = np.array(zbin)


    # plt.xlabel('Phase Difference (rad)')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Phase Differences per Simulation')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()

    # Remove simulations with NaN frequencies or all-NaN phases
    valid_mask = (~np.isnan(all_freqs_sorted)) & (~np.all(np.isnan(all_phases_converted), axis=1))
    valid_freqs = all_freqs_sorted[valid_mask]
    valid_phases = all_phases_converted[valid_mask]
    valid_colors = colors[sorted_indices][valid_mask]

    all_freqs_1_computed_mean = np.mean(all_freqs_computed_1, axis=1)[valid_mask]
    all_freqs_2_computed_mean = np.mean(all_freqs_computed_2, axis=1)[valid_mask]


    if len(valid_freqs) == 0:
        print("No valid simulations to plot.")
    else:
        x = valid_freqs # center around 0.7 Hz
        y = centers
        zbin_valid = []
        for phases in valid_phases:
            if np.all(np.isnan(phases)):
                zbin_valid.append(np.zeros_like(centers))
            else:
                hist, _ = np.histogram(phases[~np.isnan(phases)], bins=bins, density=False)
                hist = hist / hist.sum() if hist.sum() > 0 else hist  # Normalize to sum to 1 zbin_valid.append(hist)
                zbin_valid.append(hist)
        z = np.array(zbin_valid)

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        # Right subplot: phase histogram heatmap
        X, Y = np.meshgrid(x, y)
        extent = [x[0], x[-1], 0, 2*np.pi]
        aspect = 'auto'
        im = axs[1].imshow(
            z.T,
            extent=extent,
            origin='lower',
            aspect=aspect,
            cmap='viridis',
            interpolation='nearest'
        )
        cbar = fig.colorbar(im, ax=axs[1], label='Density')
        axs[1].set_xlabel('Leader Set Frequency (Hz)')
        axs[1].set_ylabel('Phase Difference (rad)')
        axs[1].set_ylim(0, 2*np.pi)
        axs[1].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        axs[1].set_yticklabels(['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        axs[1].set_title('Phase Histogram')

        # Left subplot: mean computed frequencies
        axs[0].plot(all_freqs_sorted, all_freqs_1_computed_mean, marker='o', label='Follower', color='C0')
        axs[0].plot(all_freqs_sorted, all_freqs_2_computed_mean, marker='o', label='Leader', color='C1')
        axs[0].set_xlabel('Leader Set Frequency (Hz)')
        axs[0].set_ylabel('Mean Computed Frequency (Hz)')
        axs[0].set_title('Mean Computed Frequencies')
        axs[0].grid(True, alpha=0.3)
        axs[0].legend()

        plt.tight_layout()
        os.makedirs(os.path.join(CURRENTDIR, "figures"), exist_ok=True)
        plt.savefig(os.path.join(CURRENTDIR, "figures", "phase_and_freq_subplots.svg"), bbox_inches='tight')
