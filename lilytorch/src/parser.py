''' Functions to pars arguments of the optimizations with the meuromechanical models '''

import argparse

def parse_arguments():
    ''' Parse arguments of the optimization '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        action  = 'store',
        type    = str,
        help    = "Yaml file to parse",
        default = 'lilytorch/src/scripts/configs/zebrafish_analytical.yaml'
    )

    return vars(parser.parse_args())
