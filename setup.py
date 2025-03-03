
import setuptools
import numpy as np  # pylint: disable=wrong-import-position

# setuptools.dist.Distribution().fetch_build_eggs(['ns_core'])

with open("README.md", "r") as fh:
    description = fh.read()
  
setuptools.setup(
    name="lilytorch",
    version="0.0.1",
    author="Andrea Ferrario",
    author_email="ferrarioa5@gmail.com",
    packages=["lilytorch"],
    description="Lilytorch",
    long_description=description,
    include_package_data=True,
    include_dirs=[np.get_include(), 'lilytorch'],
    long_description_content_type="text/markdown",
    python_requires='>=3.8',
    zip_safe=False,    
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'torch'
    ],
)