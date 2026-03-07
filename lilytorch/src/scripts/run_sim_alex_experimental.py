
from lilytorch.solver import FluidSolver
from lilytorch.util.yaml_operations import yaml2pyobject

SOLVER_TYPE_PARAMS = {
    'implicit': {
        'convection_method': 'implicit',
        'dt'               : 1e-3,
    },
    'abdquickest': {
        'convection_method': 'abdquickest',
        'dt'               : 1e-4,
    }
}


def get_speed_from_frequency_empirical(frequency, body_length):
    ''' Get speed of fluid from frequency '''

    # Empirical data
    f0, v0 = [ 15.0, 3.90 ] # Jensen et al. 2023 (BL / s)
    f1, v1 = [  8.0, 2.10 ] # Mwaffo et al. 2017 (BL / s)

    # Extrapolate (y = mx + c)
    slope     = (v1 - v0) / (f1 - f0)
    intercept = v0 - slope * f0
    speed_bl  = slope * frequency + intercept

    return speed_bl * body_length

def get_speed_from_frequency_theoretical(frequency, body_length):
    ''' Get speed of fluid from frequency '''

    # Brainbridge 1958
    speed_bl = (3 * frequency - 4) / 4

    return speed_bl * body_length

def get_speed_from_frequency(
    frequency,
    body_length,
    empirical,
):
    ''' Get speed of fluid from frequency '''

    if empirical:
        speed = get_speed_from_frequency_empirical(frequency, body_length)
    else:
        speed = get_speed_from_frequency_theoretical(frequency, body_length)

    return speed

def main(
    empirical = True,
    method    = "abdquickest",
):

    empirical_str = 'empirical' if empirical else 'theoretical'
    print(f"Running simulations with {empirical_str} speed and {method} method")

    # Overall parameters
    body_length = 0.018
    duration    = 30
    frequency   = 3.5

    # Load parameters
    pars = yaml2pyobject("lilytorch/src/scripts/configs/zebrafish_experimental.yaml")

    # Control parameters
    freq_scaling = 0.25
    frame_start  = 11300 # 12100 - 11300
    frame_end    = 12400 # 12260 - 12400

    pars["body"]["control"]["body_length"]     = body_length
    pars["body"]["control"]["folder_name"]     = "lilytorch/scripts/zebrafish_files/data"
    pars["body"]["control"]["file_name"]       = "kinematics_recording.csv"
    pars["body"]["control"]["save_data"]       = False
    pars["body"]["control"]["plot_data"]       = False
    pars["body"]["control"]["target_fish"]     = 'Fish3'
    pars["body"]["control"]["start_recording"] = frame_start
    pars["body"]["control"]["end_recording"]   = frame_end
    pars["body"]["control"]["timestep"]        = 0.001
    pars["body"]["control"]["total_duration"]  = 30
    pars["body"]["control"]["freq_scaling"]    = freq_scaling
    pars["body"]["control"]["filter_freqs"]    = [1.0, 10.0]
    pars["body"]["control"]["xshift"]          = 0.0
    pars["body"]["control"]["yshift"]          = 0.0

    # Fluid parameters
    speed  = get_speed_from_frequency(
        frequency   = frequency,
        body_length = body_length,
        empirical   = empirical,
    )

    pars['boundary_conditions']['BC_values_u'] = [speed, 0, 0, 0]

    # Solver parameters ("abdquickest" or "implicit")
    solver_type = method
    solver_mult = 2.0
    solver_n    = 512
    solver_xmin = -0.004
    solver_xmax = 3 * body_length

    solver_pars = SOLVER_TYPE_PARAMS[solver_type]
    time_step   = solver_pars["dt"] * solver_mult

    pars["solver"]['N']    = solver_n
    pars["solver"]['xmin'] = solver_xmin
    pars["solver"]['xmax'] = solver_xmax
    pars["solver"]['ymin'] = - (solver_xmax - solver_xmin) / 2
    pars["solver"]['ymax'] = + (solver_xmax - solver_xmin) / 2

    pars["solver"]["convection_method"] = solver_pars["convection_method"]
    pars["solver"]["dt"]                = time_step
    pars["solver"]["nt"]                = round(duration / time_step)

    # Save parameters
    save_folder_path = "/data/pazzagli/simulation_results/fluid_solver/"
    save_folder_name = (
        f'{ round(duration) }s_'                             # e.g. 30s
        f'{ str(round(frequency *  10)).zfill(3) }Hz_'       # e.g. 035Hz
        f'{method}_'                                         # e.g. implicit
        'exp_'                                               # e.g. exp
        f'{frame_start}_{frame_end}_'                        # e.g. 11300_12400
        f'scaled_{str(round(freq_scaling * 100)).zfill(3)}_' # e.g. scaled_025
        'large_grid_'                                        # e.g. large_grid
        f'{empirical_str}_speed'                             # e.g. theoretical_speed
    )

    save_interval = 0.010

    pars["output"]["save_frames"] = True
    pars["output"]["save_uv"]     = True
    pars["output"]["save_path"]   = f"{save_folder_path}/{save_folder_name}_"
    pars["output"]["save_every"]  = round(save_interval / time_step)

    # # RESUME SIMULATION
    # pars["solver"]["starting_iteration"]      = 121950 - round(save_interval / time_step) * 5
    # pars["solver"]["starting_iteration_path"] = (
    #     f'{save_folder_path}/'
    #     '30s_035Hz_abdquickest_exp_11300_12400_scaled_025_large_grid_theoretical_speed_2025-02-04T08:17:01.774271/'
    # )

    # Run simulation
    solver = FluidSolver(pars)
    solver.run_sim()


if __name__ == "__main__":
    # # Wait for 5 hours to run the simulation
    # import time
    # wait_time = 4 * 60 * 60
    # print(f"Waiting for {wait_time / 3600} hours before starting the simulation")
    # time.sleep(wait_time)

    # Run simulations
    # TODO: Repeat for the experimental envelope from Di Santo et al. 2021

    for empirical in [ True, False ]:
        for method in [  "implicit", "abdquickest" ]:

            main(
                empirical = empirical,
                method    = method
            )











