
from setuptools import setup, find_packages


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
    python_requires=">=3.9",
    zip_safe=False,
)
