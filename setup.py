#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name="esn",
    version="0.1",
    description="Echo State Networks",
    author="Niklas Heim",
    packages=find_packages(),
    install_requires=[
        "numpy",
    ],
    extras_require={
        "test":["pytest","pytest-cov"]
    },
    license="MIT"
)
