from setuptools import find_packages, setup

# Metadata and dependencies are defined in pyproject.toml.
# Keep setup.py side-effect free for reproducible builds.
setup(
    include_package_data=True,
    packages=find_packages(exclude=("tests",)),
)
