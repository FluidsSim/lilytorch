
import os
import lilytorch
import datetime

lilytorch_repo_root = os.path.join(os.path.dirname(os.path.dirname(lilytorch.__file__)), "lilytorch")
farms_examples_path = os.path.join(lilytorch_repo_root, "farms_examples")
sdfs_path           = os.path.join(farms_examples_path, "sdfs")
save_path           = "/data/andreaferrario/ns_data/"
today               = datetime.datetime.now()
todaystr            = today.isoformat()
output_folder       = os.path.join(save_path, todaystr)

__all__ = [
    'lilytorch_repo_root',
    'farms_examples_path',
    'sdfs_path',
    'output_folder',
    'todaystr',
    'save_path',
]
