
import os
import lilytorch
import datetime

lilytorch_repo_root = os.path.join(os.path.dirname(os.path.dirname(lilytorch.__file__)), "lilytorch")
farms_examples_path = os.path.join(lilytorch_repo_root, "farms_examples")
sdfs_path           = os.path.join(farms_examples_path, "sdfs")
save_path           = "/data/andreaferrario/ns_data/"

def gen_new_folder(stack_folder=""):
    """Generate a new folder for saving data based on the current date and time."""
    today               = datetime.datetime.now()
    todaystr            = today.isoformat()
    output_folder       = os.path.join(save_path, stack_folder, todaystr)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder

__all__ = [
    'lilytorch_repo_root',
    'farms_examples_path',
    'sdfs_path',
    'gen_new_folder',
    'save_path',
]
