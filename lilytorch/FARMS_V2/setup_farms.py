"""Clone FARMS repos"""

import os
import sys
from subprocess import check_call
try:
    import uv
except ImportError:
    check_call([sys.executable, '-m', 'pip', 'install', 'uv'])
try:
    from git import Repo, GitCommandError
except ImportError:
    print('Installing GitPython')
    check_call(['uv', 'pip', 'install', 'GitPython'])
    from git import Repo, GitCommandError


def main():
    """Main"""
    pip_install = ['uv', 'pip', 'install', '--no-build-isolation']
    for package, branch, install in [
            ['farms_core', 'amphibious_v0.2', True],
            ['farms_mujoco', 'amphibious_v0.2', True],
            ['farms_sim', 'amphibious_v0.2', True],
            ['farms_amphibious', 'amphibious_v0.2', True],
    ]:
        print(f'Setting up {package}')
        if not os.path.isdir(package):
            repo = Repo.clone_from(
                f'git@github.com:farmsim/{package}.git',
                package,
                branch=branch,
            )
        repo = Repo(package)
        current_branch = repo.active_branch.name
        if current_branch != branch:
            print(f'Checking out branch {branch} for {package}')
            try:
                repo.git.checkout(branch)
            except GitCommandError:
                print(f'Branch {branch} not found locally, trying to fetch it from origin')
                repo.remotes.origin.fetch()
                repo.git.checkout(branch)
        print(f'Active branch of {package}: {repo.active_branch.name}')
        # print(f'Pulling latest version of {package}')
        # repo.remotes.origin.pull()
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
