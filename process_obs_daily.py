#!/usr/bin/env python

"""
process_daily_unseen.py
=======================

This script processes daily obs data into a dataframe for a given variable and country combination.

Usage:
------

    $ python process_daily_UNSEEN.py --variable "t2m" --country "United Kingdom"

Arguments:
----------

    --variable : str : variable name (e.g. tas, pr, psl)
    --country : str : country name (e.g. United Kingdom, France, Germany)

Returns:
--------

    dataframes : pd.DataFrame : processed dataframes for the given variable, country, and initialisation year which are saved to the /gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs folder.

Author:
-------

    Ben W. Hutchins, University of Reading, 2024
    
"""

# Local imports
import os
import sys
import time
import argparse

# Third-party imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry
import cartopy.io.shapereader as shpreader
import iris

# Specific imports
from tqdm import tqdm
from datetime import datetime, timedelta

# load the dictionaries
import dictionaries as dic

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import create_masked_matrix
from unseen_analogs_functions import regrid_obs_to_model
    
# Define the main function
def main():
    """
    Main function for processing daily model data into dataframes for a given
    initialisation year, variable, member, and country combination.
    """

    start_time = time.time()

    # Set up the output directory for the dfs
    output_dir_dfs = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs"

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process daily model data into dataframes for a given initialisation year, variable, member, and country combination.")
    parser.add_argument("--variable", type=str, help="Variable name (e.g. tas, pr, psl)")
    parser.add_argument("--country", type=str, help="Country name (e.g. United Kingdom, France, Germany)")

    # Parse the arguments
    args = parser.parse_args()

    # if country contains a _
    # e.g. United_Kingdom
    # replace with a space
    if "_" in args.country and args.country == "United_Kingdom":
        args.country = args.country.replace("_", " ")

    # Print the arguyments
    print("=====================================")
    print("Processing daily ERA5 data for the following arguments:")
    print(f"Variable: {args.variable}")
    print(f"Country: {args.country}")
    print("=====================================")

    # if country has a space, replace with _
    country = args.country.replace(" ", "_")

    # set up the current day
    current_day = datetime.now().strftime("%Y-%m-%d")

    # Set up the name for the df
    df_name = f"ERA5_{args.variable}_{country}_1960-2018_daily_{current_day}.csv"

    # print the path to the df
    print(f"Output path: {os.path.join(output_dir_dfs, df_name)}")

    # If the df already exists raise an error
    if os.path.exists(os.path.join(output_dir_dfs, df_name)):
        print(f"The dataframe {df_name} already exists.")
        # return

    # Load and process the observed data to compare against
    # Set up the path to the observed data
    base_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/"

    # if the variable is tas
    if args.variable == "tas":
        obs_path = os.path.join(base_path, "ERA5_t2m_daily_1950_2020.nc")
        test_file_path = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1961-r9i1p1f2/day/tas/gn/files/d20200417/tas_day_HadGEM3-GC31-MM_dcppA-hindcast_s1961-r9i1p1f2_gn_19720101-19720330.nc"
    elif args.variable == "sfcWind":
        obs_path = os.path.join(base_path, "ERA5_wind_daily_1952_2020.nc")
        test_file_path = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1961-r9i1p1f2/day/sfcWind/gn/files/d20200417/sfcWind_day_HadGEM3-GC31-MM_dcppA-hindcast_s1961-r9i1p1f2_gn_19720101-19720330.nc"
    else:
        raise ValueError("Variable not recognised.")

    # Assert that the obs path exists
    assert os.path.exists(obs_path), "Observations path does not exist."

    # set up the obs variab;e
    if args.variable == "tas":
        obs_variable = "t2m"
    elif args.variable == "sfcWind":
        obs_variable = "si10"
    else:
        raise ValueError("Variable not recognised.")

    # load the obs cube test
    obs_cube_test = iris.load_cube(obs_path)

    # load the model cube test (for regridding)
    model_cube_test = iris.load_cube(test_file_path)

    # constrain to the relevant years
    obs_cube_test = obs_cube_test.extract(
        iris.Constraint(time=lambda cell: 1960 <= cell.point.year <= 2018)
    )

    # # perform the intersection
    # obs_cube_test = obs_cube_test.intersection(
    #     latitude=(30, 80),
    #     longitude=(-40, 30),
    # )

    # # print the model cube test dimensions
    # print("Model cube test dimensions:")
    # print(model_cube_test)

    # # Select the first member and time from the model cube
    # model_cube_regrid = model_cube_test[0, :, :]

    # ensure that hadgem is in -180 to 180
    model_cube_test = model_cube_test.intersection(longitude=(-180, 180), latitude=(0, 90))

    # Europe grid to subset to
    eu_grid = {
        "lon1": -40,  # degrees east
        "lon2": 30,
        "lat1": 30,  # degrees north
        "lat2": 80,
    }
    
    # subset the ERA5 data to the EU grid
    obs_cube_test = obs_cube_test.intersection(
        longitude=(eu_grid["lon1"], eu_grid["lon2"]),
        latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    )

    # subset the HadGEM data to the EU grid
    model_cube_test = model_cube_test.intersection(
        longitude=(eu_grid["lon1"], eu_grid["lon2"]),
        latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    )

    # print the model cube regrid dimensions
    # print("Model cube regrid dimensions:")
    # print(model_cube_regrid)

    # model_cube_regrid.coord("latitude").units = obs_cube_test[0].coord("latitude").units
    # model_cube_regrid.coord("longitude").units = obs_cube_test[0].coord("longitude").units

    # # and for the attributes
    # model_cube_regrid.coord("latitude").attributes = obs_cube_test[0].coord("latitude").attributes
    # model_cube_regrid.coord("longitude").attributes = obs_cube_test[0].coord("longitude").attributes

    obs_cube_regrid = obs_cube_test.regrid(model_cube_test, iris.analysis.Linear())

    # longitude : -45 to 40.219 by 0.2812508 degrees_east
    #  latitude : 89.78487 to 29.92973 by -0.2810101 degrees_north
    # for obs

    # if the country is in "United Kingdom" "United_Kingdom"
    if args.country == "United Kingdom" or args.country == "United_Kingdom":
        # Create the mask matrix for the UK
        MASK_MATRIX = create_masked_matrix(
            country=args.country,
            cube=obs_cube_regrid,
        )

        obs_data = obs_cube_regrid.data

        # print the obs data
        print("Obs data:")
        print(obs_data)

        # Apply the mask to the observed and model data
        obs_values = obs_data * MASK_MATRIX
        # model_values = model_cube.data * MASK_MATRIX

        # print the obs values
        print("Obs values:")
        print(obs_values)

        # Where there are zeros we want to set these to NaNs
        obs_values = np.where(obs_values == 0, np.nan, obs_values)
        # model_values = np.where(model_values == 0, np.nan, model_values)

        # prit the obs values
        print("Obs values:")
        print(obs_values)

        # print the obs values shape
        print("Obs values shape:")
        print(obs_values.shape)

        # Take the Nanmean of the data
        # over lat and lon dims
        obs_mean = np.nanmean(obs_values, axis=(1, 2))
    elif args.country == "North Sea":
        print("Taking gridbox average for the North Sea")

        # Set up the gridbox
        gridbox = dic.north_sea_kay

        # Subset to the north sea region
        obs_cube_regrid = obs_cube_regrid.intersection(
            longitude=(gridbox["lon1"], gridbox["lon2"]),
            latitude=(gridbox["lat1"], gridbox["lat2"]),
        )

        # print the obs cube regrid
        print(obs_cube_regrid)

        # print the lats and lons of the obs cube regrid
        print(obs_cube_regrid.coord("latitude").points)
        print(obs_cube_regrid.coord("longitude").points)

        # Take the mean over lat and lon
        obs_mean = obs_cube_regrid.collapsed(["latitude", "longitude"], iris.analysis.MEAN).data
    elif args.country == "UK_wind_box":
        print("Taking gridbox average for the UK wind box")

        # set up the gridbox
        gridbox = dic.wind_gridbox

        # subset to the wind gridbox
        obs_cube_regrid = obs_cube_regrid.intersection(
            longitude=(gridbox["lon1"], gridbox["lon2"]),
            latitude=(gridbox["lat1"], gridbox["lat2"]),
        )

        # Take the mean over lat and lon
        obs_mean = obs_cube_regrid.collapsed(["latitude", "longitude"], iris.analysis.MEAN).data
    else:
        raise ValueError("Country not recognised.")

    # print the obs mean
    print("Obs mean:")
    print(obs_mean)

    dates = obs_cube_regrid.coord("time").points

    # Set up the dataframe
    obs_df = pd.DataFrame(
        {
            "time": dates,
            "data": obs_mean,
        }
    )

    # if the path to the file exists, raise an error
    if os.path.exists(os.path.join(output_dir_dfs, df_name)):
        raise ValueError(f"The dataframe {df_name} already exists.")
    else:
        print(f"Saving the dataframe to {os.path.join(output_dir_dfs, df_name)}")
        obs_df.to_csv(os.path.join(output_dir_dfs, df_name), index=False)

    # end teh timer
    end_time = time.time()

    # Print the time taken
    print(f"Time taken to load the data: {end_time - start_time} seconds")

    return

if __name__ == "__main__":
    main()