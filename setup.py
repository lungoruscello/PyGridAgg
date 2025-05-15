import re

from setuptools import setup, find_packages

import pygridagg as agg

with open("README.md", 'r') as f:
    long_description = f.read()

# for Pypi, strip out the project icon from the header line
long_description = re.sub(r'PyGridAgg <img.*?>', 'PyGridAgg', long_description)

setup(
    name="PyGridAgg",
    version=agg.__version__,
    description=agg.__about__,
    url=agg.__url__,
    author=agg.__author__,
    license=agg.__license__,
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=['numpy', 'matplotlib'],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent"],
    python_requires='>=3.6',
    package_data={
        'pygridagg': ['examples/data/quakes_jpn.npy'],
    }

)
