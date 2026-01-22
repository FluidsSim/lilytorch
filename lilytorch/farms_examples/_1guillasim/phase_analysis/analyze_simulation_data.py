
import os
import sys
import inspect
import numpy as np
import random
from lilytorch.util.paths import save_path
from farms_core.io.hdf5 import hdf5_to_dict
from farms_core.sensors.sensor_convention import sc
import matplotlib.pyplot as plt

data_dir = os.path.join(save_path, "pinned_2guilla_exp")

CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, CURRENTDIR)
from phase_analysis import get_schooling_data, plot_schooling_data
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

    animat_id_0 = 1
    animat_id_1 = 0

    # Load simulation data
    simulation_data = hdf5_to_dict(file_path)

    times           = simulation_data['times']

    animats_list    = simulation_data['animats']
    n_animats       = len(animats_list)

    assert animat_id_0 < n_animats, f'animat_id_0 {animat_id_0} >= n_animats {n_animats}'
    assert animat_id_1 < n_animats, f'animat_id_1 {animat_id_1} >= n_animats {n_animats}'

    # Get swimmers data
    swimmer_data_1 = get_swimmer_data(
        animat_id    = animat_id_0,
        animats_list = animats_list,
    )
    swimmer_data_2 = get_swimmer_data(
        animat_id    = animat_id_1,
        animats_list = animats_list,
    )

    # Get schooling data
    times, freq1_computed, freq2_computed, phases = get_schooling_data(
        times        = times,
        joints_pos_1 = swimmer_data_1['joints_pos'],
        joints_pos_2 = swimmer_data_2['joints_pos'],
        links_pos_1  = swimmer_data_1['links_pos'],
        links_pos_2  = swimmer_data_2['links_pos'],
        linearize    = True,
        plotting     = False,
    )

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
    colors = plt.cm.tab10(np.linspace(0, 1, len(dirs)))
    all_phases = []
    all_freqs = []
    for subdir in dirs:
        print(f'Analyzing data in {os.path.join(data_dir, subdir)}...')
        times, phases, freq1 = main(os.path.join(data_dir, subdir), colors[dirs.index(subdir)])
        all_phases.append(phases)
        all_freqs.append(freq1)
    all_phases = np.array(all_phases)

    from IPython import embed; embed()

    # Plot histogram of phase differences for each simulation vs its frequency

    # Reorder all_phases and all_freqs according to increasing all_freqs
    sorted_indices = np.argsort(all_freqs)
    all_phases     = all_phases[sorted_indices]
    all_freqs      = np.array(all_freqs)[sorted_indices]

    plt.figure()
    nbins = 30
    bins = np.linspace(-np.pi, np.pi, nbins)
    for i, phases in enumerate(all_phases):
        hist, bin_edges = np.histogram(phases, bins=bins, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(centers, hist + i*0.1, label=f'Freq: {all_freqs[i]:.2f} Hz', color=colors[sorted_indices[i]])


    plt.xlabel('Phase Difference (rad)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Phase Differences per Simulation')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()




