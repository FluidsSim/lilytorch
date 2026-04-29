
import glob
import os

from setuptools import setup, find_packages
import numpy as np


def _kernel_extensions():
    """Build the ``lilytorch.src.kernels._C`` CUDA extension.

    Skipped (returns []) when:
      * ``LILYTORCH_NO_CUDA=1`` is set, or
      * torch is not importable, or
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
    except Exception:
        return []

    use_cuda = torch.cuda.is_available() and CUDA_HOME is not None
    Extension = CUDAExtension if use_cuda else CppExtension

    debug_mode = os.environ.get("DEBUG", "0") == "1"
    # CPU parallelism is provided by ATen's intra-op thread pool
    # (``at::parallel_for``) — no OpenMP linkage is required for the
    # ``.cpp`` sources. The ``-fopenmp`` / ``-lgomp`` flags were dropped
    # so the extension builds cleanly across PyTorch versions and on
    # toolchains without an OpenMP runtime (e.g. default macOS clang).
    extra_compile_args = {
        "cxx": [
            "-O0" if debug_mode else "-O3",
            "-fdiagnostics-color=always",
        ],
        "nvcc": ["-O0" if debug_mode else "-O3"],
    }
    extra_link_args = []
    if debug_mode:
        extra_compile_args["cxx"].append("-g")
        extra_compile_args["nvcc"].append("-g")
        extra_link_args.extend(["-O0", "-g"])

    here = os.path.dirname(os.path.abspath(__file__))
    csrc = os.path.join(here, "lilytorch", "src", "kernels", "csrc")
    sources = sorted(glob.glob(os.path.join(csrc, "*.cpp")))
    if use_cuda:
        sources += sorted(glob.glob(os.path.join(csrc, "cuda", "*.cu")))

    return [
        Extension(
            name="lilytorch.src.kernels._C",
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