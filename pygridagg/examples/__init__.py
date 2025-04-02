from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / 'data'


def load_japanese_earthquake_data():
    """
    Return example data on the locations and magnitudes of earthquakes
    in and around Japan between 2010 and 2023. Earthquake locations
    are given in degrees longitude and latitude.

    Data source: USGS Earthquake catalogue
    (https://www.usgs.gov/programs/earthquake-hazards)
    """
    with open(DATA_DIR / 'quakes_jpn.npy', mode='rb') as f:
        data = np.load(f)  # type: ignore

    coords, magnitude = data[:, :2], data[:, 2]
    return coords, magnitude
