import re

from setuptools import setup, find_packages

import pygridagg as pga


def clean_readme_for_pypi():
    with open("README.md", 'r') as f:
        readme_lines = f.readlines()

    # strip out the project icon from the title line
    readme_lines[0] = re.sub(r'PyGridAgg <img.*?>', 'PyGridAgg', readme_lines[0])

    # remove the two shields
    filtered_lines = [l for l in readme_lines if not l.startswith("[![PyPI Latest Release]")]
    filtered_lines = [l for l in filtered_lines if not l.startswith("[![MIT License]")]

    # concatenate
    long_description = "".join(filtered_lines)

    return long_description


setup(
    name="PyGridAgg",
    version=pga.__version__,
    description=pga.__about__,
    url=pga.__url__,
    author=pga.__author__,
    license=pga.__license__,
    long_description=clean_readme_for_pypi(),
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
