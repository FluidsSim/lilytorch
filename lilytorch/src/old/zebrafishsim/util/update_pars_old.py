
import csv
import numpy

def update_muscle_param(animat_options, pars):

    with open('muscle_parameters/muscle_parameters_FN_5000_ZC_750_G0_419.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            joint_i = int(row[''])
            joint_name="joint_"+str(joint_i)
            # if joint_name in animat_options["control"]["muscles"]:
            animat_options["control"]["muscles"][joint_i]["alpha"] = float(row['alpha'])
            animat_options["control"]["muscles"][joint_i]["beta"] = float(row['beta'])
            animat_options["control"]["muscles"][joint_i]["gamma"] = 1
            animat_options["control"]["muscles"][joint_i]["delta"] = float(row['delta'])
            if joint_i==12:
                mult=2
                sqrt_mult=numpy.sqrt(mult)
                animat_options["control"]["muscles"][joint_i]["alpha"] *= mult
                animat_options["control"]["muscles"][joint_i]["beta"] *= mult
                animat_options["control"]["muscles"][joint_i]["delta"] *= sqrt_mult
            if joint_i==13:
                mult=5
                sqrt_mult=numpy.sqrt(mult)
                animat_options["control"]["muscles"][joint_i]["alpha"] *= mult
                animat_options["control"]["muscles"][joint_i]["beta"] *= mult
                animat_options["control"]["muscles"][joint_i]["delta"] *= sqrt_mult
            if joint_i==14 or joint_i==0:
                mult=15
                sqrt_mult=numpy.sqrt(mult)
                animat_options["control"]["muscles"][joint_i]["alpha"] *= mult
                animat_options["control"]["muscles"][joint_i]["beta"] *= mult
                animat_options["control"]["muscles"][joint_i]["delta"] *= sqrt_mult


            # animat_options["control"]["muscles"][joint_i]["alpha"] = 4.0e-07
            # animat_options["control"]["muscles"][joint_i]["beta"] = 1.0e-12
            # animat_options["control"]["muscles"][joint_i]["gamma"] = 1.0e6
            # animat_options["control"]["muscles"][joint_i]["delta"] = 1.0e-08

            animat_options["control"]["muscles"][joint_i]["alpha"] *= pars.stiffness_factor
            animat_options["control"]["muscles"][joint_i]["beta"] *= pars.stiffness_factor
            animat_options["control"]["muscles"][joint_i]["delta"] *= pars.damping_factor

def update_drag_param(animat_options):
    # return
    for link in animat_options["morphology"]["links"]:
        link["drag_coefficients"][0][1] = 0.1*link["drag_coefficients"][0][1]
        # link["drag_coefficients"][0][2] = -0.01

def update_swimming_mode(arena_options, swimming_mode=True):
    arena_options.water.drag = swimming_mode
