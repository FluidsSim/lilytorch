"""Portable repository- and output-folder helpers."""

import datetime
import os
from pathlib import Path

# Derived from this file's location (<root>/util/paths.py) rather than from
# lilytorch.__file__, which is None whenever the package is picked up as a
# namespace package -- as happens when the working directory sits above the
# editable-install checkout and shadows it.
lilytorch_repo_root = str(Path(__file__).resolve().parent.parent)
examples_path = os.path.join(lilytorch_repo_root, "examples")
sdfs_path = os.path.join(examples_path, "sdfs")


def get_output_root():
    """Return the configured output root, resolved at call time."""
    return str(Path(
        os.environ.get("LILYTORCH_OUTPUT_DIR", Path.cwd() / "lilytorch_output")
    ).expanduser().resolve())


def gen_new_folder(stack_folder=""):
    """Generate a new folder for saving data based on the current date and time."""
    today = datetime.datetime.now()
    todaystr = today.isoformat()
    output_root = Path(get_output_root())
    stack_folder = Path(stack_folder).expanduser() if stack_folder else output_root
    if not stack_folder.is_absolute():
        stack_folder = output_root / stack_folder
    output_folder = stack_folder / todaystr
    os.makedirs(output_folder, exist_ok=True)
    return str(output_folder)


def __getattr__(name):
    # ``save_path`` is the former name of the output root. Resolving it here
    # rather than binding it at module import keeps get_output_root()'s
    # call-time semantics, so LILYTORCH_OUTPUT_DIR and the working directory
    # are still honoured when a caller imports the name.
    if name == "save_path":
        return get_output_root()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "lilytorch_repo_root",
    "examples_path",
    "sdfs_path",
    "gen_new_folder",
    "get_output_root",
    "save_path",
]
