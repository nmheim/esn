#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="esn_dev",
    version="0.1",
    description="Scalable Echo State Networks",
    author="Jacob Ungar Felding / Niklas Heim",
    packages=find_packages(),
    install_requires=[
        "joblib",
        "matplotlib",
        "numpy",
        "scipy",
        "sklearn",
        "cmocean",
        "IMED"
    ],
    extras_require={
        "test":["pytest","pytest-cov"]
    },
    license="MIT"
)
