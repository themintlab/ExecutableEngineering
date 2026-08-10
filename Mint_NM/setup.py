from pathlib import Path
from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="Mint_NM",
    version="0.1.29",
    description="Tools for ENGPHYS 3NM4",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/themintlab/ExecutableEngineer/tree/main/Mint_NM",
    author="Michael Welland",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "matplotlib",
        "networkx",
        "ipywidgets",
        "IPython",
        "pyppeteer",
        "nbconvert",
        "scipy",
        "plotly",
        "pandas",
    ],
    python_requires=">=3.9",
    license="CC-BY-4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "Operating System :: OS Independent",
    ],
)
