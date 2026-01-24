
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

data_dir = os.path.join(save_path, "pinned_2guilla_exp_2")

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
    mask = (times > 10) & (times < 19)
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

    return times, phases, freq1


if __name__ == "__main__":

    dirs = os.listdir(data_dir)
    all_phases = []
    all_freqs = []
    # Use a continuous colormap (e.g., viridis) for better color distinction
    colors = plt.cm.viridis(np.linspace(0, 1, len(dirs)))

    # output = sweep_1d(main, subdirs, num_process=16)
    # subdirs = [d for d in dirs if os.path.isdir(os.path.join(data_dir, d))]

    for subdir in dirs:
        print(f'Analyzing data in {os.path.join(data_dir, subdir)}...')
        times, phases, freq1 = main(os.path.join(data_dir, subdir), colors[dirs.index(subdir)])
        all_phases.append(phases)
        all_freqs.append(freq1)
    all_phases = np.array(all_phases)


    # Reorder all_phases and all_freqs according to increasing all_freqs
    sorted_indices = np.argsort(all_freqs)
    all_phases     = all_phases[sorted_indices]
    all_freqs      = np.array(all_freqs)[sorted_indices]

    all_phases_converted = (all_phases + 2 * np.pi) % (2 * np.pi)

    # Convert phases from [-pi, pi] to [0, 2pi]
    # all_phases = (all_phases + 2 * np.pi) % (2 * np.pi)




    # idxs = range(19) #range(10,11)
    # plt.figure()
    # for i, phases in enumerate(all_phases_converted[idxs]):
    #     plt.plot(times, phases, label=f'Simulation {i+1}', color=colors[i], alpha=0.7)
    # plt.xlabel('Time Index')
    # plt.ylabel('Phase Difference (rad)')
    # plt.title('Phase Difference Over Time for All Simulations')
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.show()





    plt.figure()
    nbins = 20
    bins = np.linspace(0, 2*np.pi, nbins)
    zbin = []
    for i, phases in enumerate(all_phases_converted):
        hist, bin_edges = np.histogram(phases, bins=bins, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(centers, hist + i*0.1, label=f'Freq: {all_freqs[i]:.2f} Hz', color=colors[sorted_indices[i]])
        zbin.append(hist)
    zbin = np.array(zbin)


    plt.xlabel('Phase Difference (rad)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Phase Differences per Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




    x = all_freqs - 0.7 # center around 0.7 Hz -
    y = centers
    z = zbin
    plt.figure(figsize=(8,6))
    plt.imshow(z.T, aspect='auto', origin='lower', extent=[x[0], x[-1], y[0], y[-1]], cmap='viridis', interpolation='None')
    plt.colorbar(label='Density')
    plt.xlabel('Leader frequency (Hz)')
    plt.ylabel('Phase Difference (rad)')
    plt.tight_layout()
    plt.ylim(0, 2*np.pi)
    plt.yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi], ['0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    plt.savefig(os.path.join(CURRENTDIR, "figures", "phase_histogram.png"), bbox_inches='tight')








