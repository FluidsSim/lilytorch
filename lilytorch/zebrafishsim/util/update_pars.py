
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

N_JOINTS_AXIS = 15

def _plot_muscle_params(
    muscle_pars: dict,
):
    ''' Plot the muscle parameters '''
    joints = np.arange(N_JOINTS_AXIS)
    alphas = np.array([muscle_pars[joint][1]['alpha'] for joint in joints])
    betas  = np.array([muscle_pars[joint][1]['beta']  for joint in joints])
    deltas = np.array([muscle_pars[joint][1]['delta'] for joint in joints])

    def _subplot(value_new, var_name, sub_plot_ind):
        plt.subplot(3, 1, sub_plot_ind)
        plt.plot(joints, value_new, 'o-', label=f'{var_name}')
        plt.ylabel(var_name)
        plt.yscale('log')
        plt.legend()

    plt.figure(figsize=(12, 8))
    _subplot(alphas, 'Alpha', 1)
    _subplot( betas,  'Beta', 2)
    _subplot(deltas, 'Delta', 3)

    plt.xlabel('Joint Index')
    plt.tight_layout()
    plt.show()

def load_muscle_parameters_from_file(
    parameters_file: str,
    folder_name    : str = 'muscle_parameters',
):
    ''' Load the muscle parameters '''

    # Load the parameter
    muscle_params_df = pd.read_csv(f'{folder_name}/{parameters_file}')

    # Example
    # muscle_parameters_options = [
    #     [['joint_0'],  {'alpha': 6.9-08,  'beta': 8.3e-08, 'delta': 2.0-09 }],
    #     ...
    #     [['joint_14'], {'alpha': 7.1-09,  'beta': 8.4e-09, 'delta': 2.2-10 }]
    # ]

    muscle_parameters_options = [
        [
            [f'joint_{i}'],
            {
                'alpha'  : muscle_params_df.loc[i, 'alpha'],
                'beta'   : muscle_params_df.loc[i, 'beta'],
                'delta'  : muscle_params_df.loc[i, 'delta'],
                'gamma'  : 1.0,
                'epsilon': 0,
            }
        ]
        for i in range(N_JOINTS_AXIS)
    ]

    return muscle_parameters_options

def scale_muscle_parameters(
    muscle_parameters: list,
    muscle_factors   : np.ndarray = np.ones(N_JOINTS_AXIS),
    head_joints      : int = 1,
    tail_joints      : int = 3,
):
    ''' Scale the muscle parameters '''

    # Target joints
    head_inds = np.arange(head_joints)
    tail_inds = np.arange(N_JOINTS_AXIS - tail_joints, N_JOINTS_AXIS)

    alphas = np.array([muscle_parameters[joint][1]['alpha'] for joint in range(N_JOINTS_AXIS)])
    betas  = np.array([muscle_parameters[joint][1]['beta']  for joint in range(N_JOINTS_AXIS)])
    deltas = np.array([muscle_parameters[joint][1]['delta'] for joint in range(N_JOINTS_AXIS)])

    # Cap head joints
    first_non_head_joint = head_joints
    for head_i in head_inds:
        muscle_parameters[head_i][1]['alpha'] = alphas[first_non_head_joint]
        muscle_parameters[head_i][1]['beta']  =  betas[first_non_head_joint]
        muscle_parameters[head_i][1]['delta'] = deltas[first_non_head_joint]

    # Cap tail joints
    last_non_tail_joint = N_JOINTS_AXIS - tail_joints - 1
    for tail_i in tail_inds:
        muscle_parameters[tail_i][1]['alpha'] = alphas[last_non_tail_joint]
        muscle_parameters[tail_i][1]['beta']  =  betas[last_non_tail_joint]
        muscle_parameters[tail_i][1]['delta'] = deltas[last_non_tail_joint]

    # Joint stiffness scaling
    for joint_i, joint_factor in enumerate(muscle_factors):
        muscle_parameters[joint_i][1]['alpha'] *= joint_factor
        muscle_parameters[joint_i][1]['beta']  *= joint_factor
        muscle_parameters[joint_i][1]['delta'] *= joint_factor**0.5

    return muscle_parameters

def update_muscle_param(animat_options, pars):

    # Load muscle parameters
    muscle_tag  = animat_options['muscle_parameters_tag']
    muscle_file = f'muscle_parameters_{muscle_tag}.csv'

    # _C0
    # muscle_tags=muscle_tag.split("_")
    # N_JOINTS_AXIS=15
    # G0         = 2 * np.pi / N_JOINTS_AXIS
    # FN         = float(muscle_tags[1])/1000.0
    # ZC         = float(muscle_tags[3])/1000.0
    # GAMMA      = float(muscle_tags[5])/1000.0
    # MUSCLE_SUM = 1.0

    # import test_analytic_muscle_pars
    # test_analytic_muscle_pars.main(
    #     target_g0    = G0,
    #     target_fn    = FN,
    #     target_zc    = ZC,
    #     muscle_gamma = GAMMA,
    #     muscle_sum   = MUSCLE_SUM,
    # )
    # muscle_file = f'muscle_parameters_analytic_{muscle_tag}.csv'
    # _C1


    muscle_parameters = load_muscle_parameters_from_file(
        parameters_file = muscle_file,
        folder_name     = 'muscle_parameters',
    )

    # Scale muscle parameters
    muscle_factors = np.ones(N_JOINTS_AXIS)
    head_joints    = 1
    tail_joints    = 3
    plot_muscles   =  False

    muscle_parameters = scale_muscle_parameters(
        muscle_parameters = muscle_parameters,
        muscle_factors    = muscle_factors,
        head_joints       = head_joints,
        tail_joints       = tail_joints,
    )

    if plot_muscles:
        _plot_muscle_params(muscle_parameters)

    # Assign muscle parameters
    for joint_i, muscle_pars in enumerate(muscle_parameters):
        animat_options["control"]["muscles"][joint_i]["alpha"] = muscle_pars[1]['alpha']
        animat_options["control"]["muscles"][joint_i]["beta"]  = muscle_pars[1]['beta']
        animat_options["control"]["muscles"][joint_i]["gamma"] = muscle_pars[1]['gamma']
        animat_options["control"]["muscles"][joint_i]["delta"] = muscle_pars[1]['delta']

        animat_options["control"]["muscles"][joint_i]["delta"] *= pars.damping_factor



    return

# _C0
# def update_drag_param(animat_options):
#     ''' Update the drag parameters '''

#     linear_coeff_x = np.array( [ 8.8970e-05, 1.8425e-05, 2.1206e-05, 2.3562e-05, 2.3562e-05, 2.3562e-05, 3.0189e-05, 1.5586e-05, 1.3854e-05, 1.1257e-05, 8.6590e-06, 5.1836e-06, 2.9452e-06, 2.9452e-06, 2.0617e-06, 3.7110e-06 ] )
#     linear_coeff_y = np.array( [ 1.7241e-03, 1.2945e-03, 1.4992e-03, 1.8143e-03, 1.8143e-03, 1.8143e-03, 2.6114e-03, 1.6648e-03, 1.6648e-03, 1.6648e-03, 1.6648e-03, 6.6523e-04, 4.3982e-04, 5.7727e-04, 8.7965e-04, 1.9792e-03 ] )
#     linear_coeff_z = np.array( [ 1.1921e-03, 1.4011e-03, 1.6658e-03, 1.8143e-03, 1.8143e-03, 1.8143e-03, 2.1414e-03, 2.0386e-03, 1.8121e-03, 1.4723e-03, 1.1325e-03, 4.8381e-04, 2.1991e-04, 2.8863e-04, 3.0788e-04, 3.8485e-04 ] )

#     linear_drag_coeffs = np.array( [ linear_coeff_x, linear_coeff_y, linear_coeff_z ] ).T

#     angular_coeff_x = np.array( [ 5.5217e-13, 1.6381e-13, 2.3030e-13, 3.1416e-13, 3.1416e-13, 3.1416e-13, 6.5745e-13, 1.0313e-13, 8.0122e-14, 5.3103e-14, 3.3573e-14, 8.7960e-15, 3.1063e-15, 3.1063e-15, 2.0381e-15, 2.3600e-14 ] )
#     angular_coeff_y = np.array( [ 9.5793e-12, 6.3104e-12, 9.3503e-12, 1.3556e-11, 1.3556e-11, 1.3556e-11, 2.7741e-11, 1.5098e-11, 1.3421e-11, 1.0904e-11, 8.3879e-12, 8.1499e-13, 1.9628e-13, 3.9667e-13, 9.3556e-13, 3.4024e-12 ] )
#     angular_coeff_z = np.array( [ 6.6746e-12, 1.0747e-11, 1.6662e-11, 2.2368e-11, 2.2368e-11, 2.2368e-11, 3.8231e-11, 4.3470e-11, 3.7887e-11, 3.0542e-11, 2.4321e-11, 1.5308e-12, 2.6730e-13, 8.2437e-13, 4.3659e-12, 1.2303e-11 ] )

#     angular_drag_coeffs = np.array( [ angular_coeff_x, angular_coeff_y, angular_coeff_z ] ).T

#     for link_ind, link in enumerate( animat_options["morphology"]["links"] ):
#         link["drag_coefficients"][0] = - linear_drag_coeffs[link_ind]
#         link["drag_coefficients"][1] = - angular_drag_coeffs[link_ind]

# _C1

# _C0
# def update_drag_param(animat_options):
#     ''' Update the drag parameters '''

#     linear_coeff_x = np.array( [ 8.8970e-05, 1.8425e-05, 2.1206e-05, 2.3562e-05, 2.3562e-05, 2.3562e-05, 3.0189e-05, 1.5586e-05, 1.3854e-05, 1.1257e-05, 8.6590e-06, 5.1836e-06, 2.9452e-06, 2.9452e-06, 2.0617e-06, 3.7110e-06 ] )
#     linear_coeff_y = np.array( [ 1.7241e-03, 1.2945e-03, 1.4992e-03, 1.8143e-03, 1.8143e-03, 1.8143e-03, 2.6114e-03, 1.6648e-03, 1.6648e-03, 1.6648e-03, 1.6648e-03, 6.6523e-04, 4.3982e-04, 5.7727e-04, 8.7965e-04, 1.9792e-03 ] )
#     linear_coeff_z = np.array( [ 1.1921e-03, 1.4011e-03, 1.6658e-03, 1.8143e-03, 1.8143e-03, 1.8143e-03, 2.1414e-03, 2.0386e-03, 1.8121e-03, 1.4723e-03, 1.1325e-03, 4.8381e-04, 2.1991e-04, 2.8863e-04, 3.0788e-04, 3.8485e-04 ] )

#     linear_drag_coeffs = np.array( [ linear_coeff_x, linear_coeff_y, linear_coeff_z ] ).T

#     angular_coeff_x = np.array( [ 5.5217e-13, 1.6381e-13, 2.3030e-13, 3.1416e-13, 3.1416e-13, 3.1416e-13, 6.5745e-13, 1.0313e-13, 8.0122e-14, 5.3103e-14, 3.3573e-14, 8.7960e-15, 3.1063e-15, 3.1063e-15, 2.0381e-15, 2.3600e-14 ] )
#     angular_coeff_y = np.array( [ 9.5793e-12, 6.3104e-12, 9.3503e-12, 1.3556e-11, 1.3556e-11, 1.3556e-11, 2.7741e-11, 1.5098e-11, 1.3421e-11, 1.0904e-11, 8.3879e-12, 8.1499e-13, 1.9628e-13, 3.9667e-13, 9.3556e-13, 3.4024e-12 ] )
#     angular_coeff_z = np.array( [ 6.6746e-12, 1.0747e-11, 1.6662e-11, 2.2368e-11, 2.2368e-11, 2.2368e-11, 3.8231e-11, 4.3470e-11, 3.7887e-11, 3.0542e-11, 2.4321e-11, 1.5308e-12, 2.6730e-13, 8.2437e-13, 4.3659e-12, 1.2303e-11 ] )

#     angular_drag_coeffs = np.array( [ angular_coeff_x, angular_coeff_y, angular_coeff_z ] ).T

#     for link_ind, link in enumerate( animat_options["morphology"]["links"] ):
#         link["drag_coefficients"][0] = - linear_drag_coeffs[link_ind]
#         link["drag_coefficients"][1] = - angular_drag_coeffs[link_ind]

# _C1

def update_drag_param(animat_options):
    for link in animat_options["morphology"]["links"]:
        link["drag_coefficients"][0][1] = 0.3*link["drag_coefficients"][0][1]
        link["drag_coefficients"][0][2] = -0.7