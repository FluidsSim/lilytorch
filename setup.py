
import glob
import os

from setuptools import setup, find_packages
import numpy as np


def _kernel_extensions():
    """Build the ``lilytorch.src._C`` CUDA extension.

    Skipped (returns []) when:
      * ``LILYTORCH_NO_CUDA=1`` is set, or
      * CUDA is not available at build time.
    """
    if os.environ.get("LILYTORCH_NO_CUDA", "0") == "1":
        return []
    try:
        import torch  # noqa: F401
        from torch.utils.cpp_extension import (
            CppExtension,
            CUDAExtension,
            CUDA_HOME,
        )
    except Exception as exc:
        raise RuntimeError(
            "PyTorch must be installed in the target environment before "
            "building lilytorch's native kernels. Install torch first, then "
            "rerun `pip install -e . --no-build-isolation` or "
            "`python setup.py build_ext --inplace`."
        ) from exc

    torch_path = os.path.realpath(torch.__file__)
    if "pip-build-env-" in torch_path:
        raise RuntimeError(
            "lilytorch's native kernels must be built against the same "
            "PyTorch installation used at runtime. pip build isolation is "
            f"currently using a temporary PyTorch at {torch_path}. Rerun "
            "`pip install -e . --no-build-isolation` after installing torch "
            "in the target environment, or rebuild in place with "
            "`python setup.py build_ext --inplace`."
        )

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    Extension = CUDAExtension if use_cuda else CppExtension

    debug_mode = os.environ.get("DEBUG", "0") == "1"
    # ``at::parallel_for`` is a header-only template that selects its
    # parallel implementation via ``#ifdef INTRA_OP_PARALLEL``, which is
    # defined in ATen/ParallelOpenMP.h only when ``_OPENMP`` is set --
    # i.e. only when the calling translation unit is compiled with
    # ``-fopenmp``.  Without it, every ``at::parallel_for`` call in our
    # kernels silently falls back to a serial loop, even though
    # ``torch.set_num_threads(N)`` is in effect. So OpenMP linkage IS
    # required for our ``.cpp`` sources to actually multithread.
    use_openmp = os.environ.get("LILYTORCH_NO_OPENMP", "0") != "1"
    extra_compile_args = {
        "cxx": [
            "-O0" if debug_mode else "-O3",
            "-fdiagnostics-color=always",
        ],
        "nvcc": ["-O0" if debug_mode else "-O3"],
    }
    extra_link_args = []
    if use_openmp:
        extra_compile_args["cxx"].append("-fopenmp")
        extra_link_args.append("-fopenmp")
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    here = os.path.dirname(os.path.abspath(__file__))
    csrc = os.path.join("lilytorch", "src", "csrc")
    abs_csrc = os.path.join(here, csrc)
    sources = sorted(glob.glob(os.path.join(abs_csrc, "*.cpp")))
    if use_cuda:
        sources += sorted(glob.glob(os.path.join(abs_csrc, "cuda", "*.cu")))

    # Convert all absolute paths to relative paths (relative to setup.py dir)
    sources = [os.path.relpath(src, here) for src in sources]

    return [
        Extension(
            name="lilytorch.src._C",
            sources=sources,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]


def _cmdclass():
    try:
        from torch.utils.cpp_extension import BuildExtension
        return {"build_ext": BuildExtension}
    except Exception:
        return {}


setup(
    name="lilytorch",
    version="0.1.0",
    author="Andrea Ferrario",
    author_email="ferrarioa5@gmail.com",
    description="GPU-accelerated CFD with immersed-boundary methods, built on PyTorch",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    include_package_data=True,
    include_dirs=[np.get_include(), "lilytorch"],
    python_requires=">=3.9",
    zip_safe=False,
    ext_modules=_kernel_extensions(),
    cmdclass=_cmdclass(),
)
