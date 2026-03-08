
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
    frequency   = 3.5,
    empirical   = True,
    method      = "abdquickest",
):

    empirical_str = 'empirical' if empirical else 'theoretical'
    print(f"Running simulations at {frequency} Hz with {empirical_str} speed and {method} method")

    # Overall parameters
    body_length = 0.018
    duration    = 30

    # Load parameters
    pars = yaml2pyobject("lilytorch/src/scripts/configs/zebrafish_analytical.yaml")

    # Control parameters
    pars["body"]["control"]["f"]             = frequency
    pars["body"]["control"]["wavefrequency"] = 0.95

    # From Di Santo et al. 2021
    pars["body"]["control"]["A"]  = body_length
    pars["body"]["control"]["L"]  = body_length
    pars["body"]["control"]["c1"] = +0.05
    pars["body"]["control"]["c2"] = -0.13
    pars["body"]["control"]["c3"] = +0.28
    pars["body"]["control"]["sb"] = 0.07
    pars["body"]["control"]["st"] = 0.95
    pars["body"]["control"]["wh"] = 0.07
    pars["body"]["control"]["wt"] = 0.01


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
        f'{ round(duration) }s_'                            # e.g. 30s
        f'{ str(round(frequency *  10)).zfill(3) }Hz_'      # e.g. 035Hz
        f'{method}_'                                        # e.g. implicit
        'large_grid_'                                       # e.g. large_grid
        f'{empirical_str}_speed'                            # e.g. theoretical_speed
    )

    save_interval = 0.010

    pars["output"]["save_frames"] = True
    pars["output"]["save_uv"]     = True
    pars["output"]["save_path"]   = f"{save_folder_path}/{save_folder_name}"
    pars["output"]["save_every"]  = round(save_interval / time_step)

    # Run simulation
    solver = FluidSolver(pars)
    solver.run_sim()

    return


if __name__ == "__main__":

    # Wait for 2 hours to run the simulation
    # import time
    # wait_time = 3 * 3600
    # print(f"Waiting for {wait_time / 3600} hours before starting the simulation")
    # time.sleep(wait_time)

    # Run simulations
    # TODO: Repeat for the experimental envelope from Di Santo et al. 2021

    for frequency in [ 3.0, 4.0,]: # 3.5, 3.0, 4.0,
        for empirical in [ True, False ]:
            for method in [  "implicit", "abdquickest" ]:

                if frequency == 3.0 and empirical == True and method == "implicit":
                    continue

                main(
                    frequency = frequency,
                    empirical = empirical,
                    method    = method
                )

