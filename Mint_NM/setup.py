from setuptools import find_packages, setup

setup(
    name="Mint_NM",
    version="0.1.29",
    description="Tools for ENGPHYS 3NM4",
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
    ],
    python_requires=">=3.9",
    license="CC-BY-4.0",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Other/Proprietary License",
        "Operating System :: OS Independent",
    ],
)
