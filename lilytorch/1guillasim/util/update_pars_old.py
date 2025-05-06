
def update_muscle_param(animat_options):
    for joint in animat_options["control"]["muscles"]:
        joint["alpha"] = 0.9
        joint["beta"] = 0.001
        joint["gamma"] = 1600
        joint["delta"] = 0.1

def update_drag_param(animat_options):
    # return
    for link in animat_options["morphology"]["links"]:

        link["drag_coefficients"][0][0] = -0.0
        link["drag_coefficients"][0][1] = -0.4
        link["drag_coefficients"][0][2] = -0.4
        link["drag_coefficients"][1][0] = -0.01
        link["drag_coefficients"][1][1] = -0.01
        link["drag_coefficients"][1][2] = -0.01
