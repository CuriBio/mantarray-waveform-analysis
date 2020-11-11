# -*- coding: utf-8 -*-
"""Setup configuration."""
import os

from setuptools import Extension
from setuptools import find_packages
from setuptools import setup

# In order to locally Cythonize files, make sure to have installed the Python dev module: sudo apt-get install python3.7-dev

try:
    from Cython.Build import cythonize
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

ext = ".pyx" if USE_CYTHON else ".c"
extensions = [
    Extension(
        "mantarray_waveform_analysis.compression_cy",
        [os.path.join("src", "mantarray_waveform_analysis", "compression_cy") + ext],
    )
]

if USE_CYTHON:
    # cythonizing compression_cy.pyx with kwarg annotate=True will help when optimizing the code by enabling generation of the html annotation file
    extensions = cythonize(extensions, annotate=False)


setup(
    name="mantarray_waveform_analysis",
    version="0.5.8",
    description="Tools for analyzing waveforms produced by a Mantarray Instrument",
    url="https://github.com/CuriBio/mantarray-waveform-analysis",
    author="Curi Bio",
    author_email="contact@curibio.com",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.19.1",
        "scipy>=1.5.2",
        "nptyping>=1.3.0",
        "attrs>=19.3.0",
    ],
    zip_safe=False,
    include_package_data=True,
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering",
    ],
    ext_modules=extensions,
)
