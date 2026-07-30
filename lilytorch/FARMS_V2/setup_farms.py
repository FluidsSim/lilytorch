"""Install FARMS repos"""

import os
import sys
from subprocess import check_call
try:
    import uv
except ImportError:
    print('Installing uv')
    check_call([sys.executable, '-m', 'pip', 'install', 'uv'])
try:
    from git import Repo
except ImportError:
    print('Installing GitPython')
    check_call(['uv', 'pip', 'install', 'GitPython'])
    from git import Repo


def main():
    """Main"""
    pip_install = ['uv', 'pip', 'install', '--no-build-isolation']
    for package, install in [
            ['farms_core', True],
            ['farms_mujoco', True],
            ['farms_sim', True],
            ['farms_amphibious', True],
    ]:
        print(f'Setting up {package}')
        if install:
            requirements = f'{package}/requirements.txt'
            if os.path.isfile(requirements):
                print(f'Installing {package} dependencies')
                check_call(pip_install + ['-r', requirements])
            print(f'Installing {package}')
            check_call(pip_install + ['-e', package, '-v'])  # vvv
        print(f'Completed setup for {package}\n')


if __name__ == '__main__':
    main()
