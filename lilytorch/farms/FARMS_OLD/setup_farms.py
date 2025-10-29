"""Clone all FARMS repos"""

import os
import sys
from subprocess import check_call
try:
    from git import Repo
except ImportError:
    check_call([sys.executable, '-m', 'pip', 'install', 'GitPython'])
    from git import Repo

def pip_install(
    package_name: str,
    local_package: bool = False,
):
    ''' Install a package using pip '''
    print(f'Installing {package_name}')
    pip_install = ['pip', 'install']
    if local_package:
        pip_install.append('-e')
    check_call(pip_install + [package_name])
    print(f'Completed installation of {package_name}\n')
    return

def main():
    """Main"""

    # Install MuJoCo
    pip_install('mujoco')

    # Install dm_control
    pip_install('dm_control')

    # Farms pylog
    pip_install('git+https://gitlab.com/FARMSIM/farms_pylog.git')

    # FARMS
    for package in ['farms_core', 'farms_mujoco', 'farms_sim', 'farms_amphibious']:
        pip_install(package, local_package=True)

if __name__ == '__main__':
    main()
