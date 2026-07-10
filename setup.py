from setuptools import setup, find_packages

setup(
    name="terramind-mask2former",
    version="0.1.0",
    description="Multi-task Mask2Former segmentation heads on a frozen TerraMind foundation-model backbone",
    packages=find_packages(include=["data", "data.*", "engine", "engine.*", "losses", "losses.*", "models", "models.*"]),
    python_requires=">=3.10",
)
