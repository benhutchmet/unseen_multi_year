#%%
"""
process_ERA5_wind_gen.py
=======================

This script processes daily ERA5 10m wind speed data into an estimate of wind
power generation for the UK.

Usage:
------

    $ python process_ERA5_wind_gen.py

Arguments:
----------

    None

Returns:
--------

    dataframes : pd.DataFrame : processed dataframes for the given variable, country, and initialisation year which are saved to the /gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs folder.

Author:
-------

    Ben W. Hutchins, University of Reading, 2025
"""

# Local imports
import os
import sys
import glob
import time

# Third-party imports
import iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Specific imports
from tqdm import tqdm

# Imports from Hannah's functions
sys.path.append("/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/")

# Importing the necessary functions from the module
from European_hourly_hub_height_winds_2023_model import (
    country_mask,
    load_power_curves,
    load_wind_farm_location,
    load_wind_speed_and_take_to_hubheight,
    convert_to_wind_power
)

# Define the main function
def main():
    # Start the timer
    start_time = time.time()

    # Set up the hardcoded variables
    test_ERA5_file_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_wind_test_1960_1961.nc"
    test_dps_file_path = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1961-r9i1p1f2/day/sfcWind/gn/files/d20200417/sfcWind_day_HadGEM3-GC31-MM_dcppA-hindcast_s1961-r9i1p1f2_gn_19720101-19720330.nc"
    onshore_pc_path = "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/power_onshore.csv"
    offshore_pc_path = "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/power_offshore.csv"
    path_to_farm_locations_ons = (
    "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/United_Kingdomwindfarm_dist_ons_2021.nc"
    )
    path_to_farm_locations_ofs = (
        "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/United_Kingdomwindfarm_dist_ofs_2021.nc"
    )
    eu_grid = {
        "lon1": -40,  # degrees east
        "lon2": 30,
        "lat1": 30,  # degrees north
        "lat2": 80,
    }

    # Set up the fname
    fname = "ERA5_UK_wind_power_generation_1960_1961.csv"
    fpath = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/Hannah_wind"

    # if the directory does not exist, create it
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    # If the full path already exists, then exit
    full_path = os.path.join(fpath, fname)

    if os.path.exists(full_path):
        print(f"File {full_path} already exists. Exiting.")
        return None

    # If the test file path ERA5 exists, then load it
    if os.path.exists(test_ERA5_file_path):
        print(f"Loading test ERA5 file from {test_ERA5_file_path}")
        ERA5_cube = iris.load_cube(test_ERA5_file_path)
    else:
        print(f"Test ERA5 file not found at {test_ERA5_file_path}")

    # If the test file path dps exists, then load it
    if os.path.exists(test_dps_file_path):
        print(f"Loading test DPS file from {test_dps_file_path}")
        dps_cube = iris.load_cube(test_dps_file_path)
    else:
        print(f"Test DPS file not found at {test_dps_file_path}")

    # Intersect the cubes
    ERA5_cube = ERA5_cube.intersection(
        longitude=(-180, 180), latitude=(0, 90)
    )
    dps_cube = dps_cube.intersection(
        longitude=(-180, 180), latitude=(0, 90)
    )

    # Limit the depresys cube to the first time step
    dps_cube = dps_cube[0, :, :]

    # Intersect to the European grid
    ERA5_cube = ERA5_cube.intersection(
        longitude=(eu_grid["lon1"], eu_grid["lon2"]),
        latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    )

    # Same for the depresys cube
    dps_cube = dps_cube.intersection(
        longitude=(eu_grid["lon1"], eu_grid["lon2"]),
        latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    )

    # Regrid the ERA5 data to the DePreSys grid
    ERA5_cube_rg = ERA5_cube.regrid(
        dps_cube, iris.analysis.Linear()
    )

    # If any values are negative, set them to zero
    if np.any(ERA5_cube_rg.data < 0):
        print("Negative values found in ERA5 data. Setting them to zero.")
        ERA5_cube_rg.data[ERA5_cube_rg.data < 0] = 0

    # print the min, max, and mean of the ERA5 data
    print(f"ERA5 data min: {np.min(ERA5_cube_rg.data):.2f} m/s")
    print(f"ERA5 data max: {np.max(ERA5_cube_rg.data):.2f} m/s")
    print(f"ERA5 data mean: {np.mean(ERA5_cube_rg.data):.2f} m/s")

    end_time = time.time()

    # Print the time taken to run the script
    print(f"Time taken to run the script: {end_time - start_time:.2f} seconds")

    return None

if __name__ == "__main__":
    # Start the main function
    main()

# %%
