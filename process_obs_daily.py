#!/usr/bin/env python
#%%
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
import iris.coord_categorisation

# Third-party imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry
import cartopy.io.shapereader as shpreader
import iris
import iris.coord_categorisation
import cftime

# Specific imports
from tqdm import tqdm
from iris.util import unify_time_units, equalise_attributes, describe_diff, equalise_cubes
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

    # Check if running in IPython
    if "ipykernel_launcher" in sys.argv[0]:
        # Manually set arguments for IPython
        args = parser.parse_args(["--variable", "sfcWind", "--country", "UK_wind_box"])
    else:
        # Parse arguments normally
        args = parser.parse_args()

    # # Parse the arguments
    # args = parser.parse_args()

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
    df_name = f"ERA5_{args.variable}_{country}_1960-2025_daily_{current_day}.csv"

    # print the path to the df
    print(f"Output path: {os.path.join(output_dir_dfs, df_name)}")

    # If the df already exists raise an error
    if os.path.exists(os.path.join(output_dir_dfs, df_name)):
        print(f"The dataframe {df_name} already exists.")
        # return

    # Load and process the observed data to compare against
    # Set up the path to the observed data
    base_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/"

    # Set up the remaining years
    remaining_years = [str(year) for year in range(2021, 2025 + 1)]

    # Set up the path to the observed data
    remaining_files_dir = os.path.join(base_path, "year_month")

    # Set up an empty cubelist
    obs_cubelist_first = []
    obs_cubelist_u10 = []
    obs_cubelist_v10 = []

    # Loop over the remaining years
    for year in tqdm(remaining_years):
        for month in ["01", "02", "12"]:
            # if the year is 2025 and the month is 12, then skip
            if year == "2025" and month == "12":
                continue
            
            # Set up the fname this
            fname_this = f"ERA5_EU_T_U10_V10_msl{year}_{month}.nc"

            # if the variable is tas
            if args.variable == "tas":
                # Set up the path to the observed data
                obs_path_this = os.path.join(remaining_files_dir, fname_this)

                # Load the observed data
                obs_cube_this = iris.load_cube(obs_path_this, "t2m")

                # Append to the cubelist
                obs_cubelist_first.append(obs_cube_this)
            elif args.variable == "sfcWind":
                # Set up the path to the observed data
                obs_path_this = os.path.join(remaining_files_dir, fname_this)

                # Load the observed data
                obs_cube_u10 = iris.load_cube(obs_path_this, "u10")
                obs_cube_v10 = iris.load_cube(obs_path_this, "v10")

                # Append to the cubelist
                obs_cubelist_u10.append(obs_cube_u10)
                obs_cubelist_v10.append(obs_cube_v10)

    # convert the list to a cube list
    obs_cubelist_first = iris.cube.CubeList(obs_cubelist_first)
    obs_cubelist_u10 = iris.cube.CubeList(obs_cubelist_u10)
    obs_cubelist_v10 = iris.cube.CubeList(obs_cubelist_v10)

    # Concatenate the cubelist
    if args.variable == "tas":
        print("obs cubelist:", obs_cubelist_first)

        removed_attrs = equalise_attributes(obs_cubelist_first)

        obs_cube = obs_cubelist_first.concatenate_cube()
    elif args.variable == "sfcWind":
        # removed the attributes
        removed_attrs_u10 = equalise_attributes(obs_cubelist_u10)
        removed_attrs_v10 = equalise_attributes(obs_cubelist_v10)

        obs_cube_u10 = obs_cubelist_u10.concatenate_cube()
        obs_cube_v10 = obs_cubelist_v10.concatenate_cube()

        # Calculate the wind speed from the data
        # Calculate wind speed
        windspeed_10m = (obs_cube_u10 ** 2 + obs_cube_v10 ** 2) ** 0.5
        windspeed_10m.rename("si10")

        # rename as obs cube
        obs_cube = windspeed_10m

    # print the obs cube
    print(obs_cube)

    # Re assign the variable
    obs_cube_recent = obs_cube

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

    # # constrain to the relevant years
    obs_cube_test = obs_cube_test.extract(
        iris.Constraint(time=lambda cell: 1960 <= cell.point.year)
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
    model_cube_regrid = model_cube_test[0, :, :]

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
        longitude=(eu_grid["lon1"], eu_grid["lon2"]),l
        latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    )

    # subset the recent obs data to the EU grid
    obs_cube_recent = obs_cube_recent.intersection(
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

    model_cube_regrid.coord("latitude").units = obs_cube_test[0].coord("latitude").units
    model_cube_regrid.coord("longitude").units = obs_cube_test[0].coord("longitude").units

    # and for the attributes
    model_cube_regrid.coord("latitude").attributes = obs_cube_test[0].coord("latitude").attributes
    model_cube_regrid.coord("longitude").attributes = obs_cube_test[0].coord("longitude").attributes

    # regrid the obs cube test
    obs_cube_test_regrid = obs_cube_test.regrid(model_cube_test, iris.analysis.Linear())

    # regrid the recent period obs cube to the model cube
    obs_cube_recent_regrid = obs_cube_recent.regrid(model_cube_test, iris.analysis.Linear())

    # print the obs cube regrid
    print("Obs cube recent regrid dimensions:")
    print(obs_cube_recent_regrid)

    # print the obs cube test regrid
    print("Obs cube test regrid dimensions:")
    print(obs_cube_test_regrid)

    # print the time units of obs cube test regrid
    print("Obs cube test regrid time units:")
    print(obs_cube_test_regrid.coord("time").units)

    # print the time units of obs cube recent regrid
    print("Obs cube recent regrid time units:")
    print(obs_cube_recent_regrid.coord("time").units)

    # # create a cubelist
    # obs_cubelist = iris.cube.CubeList([obs_cube_test_regrid, obs_cube_recent_regrid])

    # # unify the time unites
    # unify_time_units(obs_cubelist)

    # # print the time units of obs cube test regrid
    # print("Obs cubelist time units:")
    # print(obs_cubelist[0].coord("time").units)

    # # print the obs cubelist time units [1]
    # print("Obs cubelist time units [1]:")
    # print(obs_cubelist[1].coord("time").units)

    # # print the first time value in the obs cubelist [0]
    # print("First time value in obs cubelist [0]:")
    # print(obs_cubelist[0].coord("time").points[0])

    # # print the final time value in the obs cubelist [0]
    # print("Final time value in obs cubelist [0]:")
    # print(obs_cubelist[0].coord("time").points[-1])

    # # print the first time value in the obs cubelist [1]
    # print("First time value in obs cubelist [1]:")
    # print(obs_cubelist[1].coord("time").points[0])

    # # print the final time value in the obs cubelist [1]
    # print("Final time value in obs cubelist [1]:")
    # print(obs_cubelist[1].coord("time").points[-1])

    # # extract the times from the first cube
    # times_long = obs_cubelist[0].coord("time").points
    # times_short = obs_cubelist[1].coord("time").points

    # # convert from days since 1952-01-01 00:00:00 to datetime
    # times_long_dt = cftime.num2date(times_long, obs_cubelist[0].coord("time").units.origin)
    # times_short_dt = cftime.num2date(times_short, obs_cubelist[1].coord("time").units.origin)

    # # print the first time value in the obs cubelist [0]
    # print("First time value in obs cubelist [0]:")
    # print(times_long_dt[0])
    # # print the final time value in the obs cubelist [0]
    # print("Final time value in obs cubelist [0]:")
    # print(times_long_dt[-1])

    # # print the first time value in the obs cubelist [1]
    # print("First time value in obs cubelist [1]:")
    # print(times_short_dt[0])
    # # print the final time value in the obs cubelist [1]
    # print("Final time value in obs cubelist [1]:")
    # print(times_short_dt[-1])

    # # print the obs cube list
    # print("Obs cubelist dimensions:")
    # print(obs_cubelist)

    # # print the first obs cube
    # print("First obs cube dimensions:")
    # print(obs_cubelist[0])

    # # print the second obs cube
    # print("Second obs cube dimensions:")
    # print(obs_cubelist[1])

    # # print the metadata for time for the first obs cube
    # print("Metadata for time for the first obs cube:"
    #         f"\n{obs_cubelist[0].coord('time').units}")
    # # print the metadata for time for the second obs cube
    # print("Metadata for time for the second obs cube:"
    #         f"\n{obs_cubelist[1].coord('time').units}")

    # # remove problematic attributes
    # removed_attrs = equalise_attributes(obs_cubelist)

    # # unify the time units
    # unify_time_units(obs_cubelist)

    # # loop over the cubes and print the units
    # for cube in obs_cubelist:
    #     print("Cube units:")
    #     print(cube.units)

    #     # if the variable is sfcWind
    #     if args.variable == "sfcWind":
    #         # set the units to m.s-1
    #         cube.units = "m.s-1"
    #     else:
    #         # set the units to K
    #         cube.units = "K"

    # # loop over the cubes and print the units
    # for cube in obs_cubelist:
    #     print("Cube units:")
    #     print(cube.units)

    # # loop over the cube and print the time attributes
    # for cube in obs_cubelist:
    #     print("Cube time attributes:")
    #     print(cube.coord("time").var_name)

    #     # if var_name is not time, set it to time
    #     if cube.coord("time").var_name != "time":
    #         cube.coord("time").var_name = "time"

    # # Align the time coordinate points
    # for cube in obs_cubelist:
    #     time_coord = cube.coord("time")
        
    #     # Convert time points to a consistent format (e.g., midnight)
    #     new_time_points = np.floor(time_coord.points)  # Round down to the nearest day
    #     time_coord.points = new_time_points

    #     # Remove bounds to avoid mismatches
    #     if time_coord.has_bounds():
    #         time_coord.bounds = None

    # # loop over the cube
    # for cube in obs_cubelist:
    #     # print the time attributes
    #     print("Cube time attributes:")
    #     print(cube.coord("time"))
    #     # print(cube.coord("latitude"))
    #     # print(cube.coord("longitude"))

    # # Make sure the cell methods are empty
    # for cube in obs_cubelist:
    #     if cube.cell_methods is not None:
    #         cube.cell_methods = None

    # # eqaulise the attributes
    # more_removed_attrs = equalise_attributes(obs_cubelist)

    # equalise_cubes(obs_cubelist, apply_all=True)

    # # describe the difference between the two cubes
    # describe_diff(obs_cubelist[0], obs_cubelist[1])

    # # loop over the cubes
    # for cube in obs_cubelist:
    #     if cube.long_name != None:
    #         # set the long name to the variable name
    #         cube.long_name = None
    #     elif cube.standard_name != None:
    #         # set the standard name to the variable name
    #         cube.standard_name = None
    #     elif cube.var_name != None:
    #         # set the variable name to the variable name
    #         cube.var_name = None
    #     else:
    #         print("No long name, standard name or variable name found.")

    # # print the cube long name
    # print("Cube long name:")
    # print(obs_cubelist[0].long_name)
    # # print the cube standard name
    # print("Cube standard name:")
    # print(obs_cubelist[0].standard_name)
    # # print the cube variable name
    # print("Cube variable name:")
    # print(obs_cubelist[0].var_name)
    # # do the same for cube 1
    # print("Cube long name:")
    # print(obs_cubelist[1].long_name)
    # # print the cube standard name
    # print("Cube standard name:")
    # print(obs_cubelist[1].standard_name)
    # # print the cube variable name
    # print("Cube variable name:")
    # print(obs_cubelist[1].var_name)

    # # concatenate the cubelist
    # obs_cube_full = obs_cubelist.concatenate_cube()

    # # print the obs cube regrid
    # print("Obs cube regrid dimensions:")
    # print(obs_cube_full)

    # sys.exit()

    # # extract the times
    # times = obs_cube_full.coord("time").points

    # # convert to dt
    # times_dt = cftime.num2date(times, obs_cube_full.coord("time").units.origin)

    # # print the first time value in the obs cube regrid
    # print("First time value in obs cube regrid:")
    # print(times_dt[0])

    # # print the final time value in the obs cube regrid
    # print("Final time value in obs cube regrid:")
    # print(times_dt[-1])

    # longitude : -45 to 40.219 by 0.2812508 degrees_east
    #  latitude : 89.78487 to 29.92973 by -0.2810101 degrees_north
    # for obs

    # if the country is in "United Kingdom" "United_Kingdom"
    if args.country == "United Kingdom" or args.country == "United_Kingdom":
        # Create the mask matrix for the UK
        MASK_MATRIX = create_masked_matrix(
            country=args.country,
            cube=obs_cube_full,
        )

        obs_data = obs_cube_full.data

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
        obs_cube_full = obs_cube_full.intersection(
            longitude=(gridbox["lon1"], gridbox["lon2"]),
            latitude=(gridbox["lat1"], gridbox["lat2"]),
        )

        # print the obs cube regrid
        print(obs_cube_full)

        # print the lats and lons of the obs cube regrid
        print(obs_cube_full.coord("latitude").points)
        print(obs_cube_full.coord("longitude").points)

        # Take the mean over lat and lon
        obs_mean = obs_cube_full.collapsed(["latitude", "longitude"], iris.analysis.MEAN).data
    elif args.country == "UK_wind_box":
        print("Taking gridbox average for the UK wind box")

        # set up the gridbox
        gridbox = dic.wind_gridbox

        # obs cube test regrid intersection
        obs_cube_test_regrid = obs_cube_test_regrid.intersection(
            longitude=(gridbox["lon1"], gridbox["lon2"]),
            latitude=(gridbox["lat1"], gridbox["lat2"]),
        )

        # obs cube recent regrid intersection
        obs_cube_recent_regrid = obs_cube_recent_regrid.intersection(
            longitude=(gridbox["lon1"], gridbox["lon2"]),
            latitude=(gridbox["lat1"], gridbox["lat2"]),
        )

        # # subset to the wind gridbox
        # obs_cube_full = obs_cube_full.intersection(
        #     longitude=(gridbox["lon1"], gridbox["lon2"]),
        #     latitude=(gridbox["lat1"], gridbox["lat2"]),
        # )

        # Take the mean over lat and lon
        # obs_mean = obs_cube_full.collapsed(["latitude", "longitude"], iris.analysis.MEAN).data

        # take the mean over the lat and lon for obs cube test regrid
        obs_mean_test_vals = obs_cube_test_regrid.collapsed(
            ["latitude", "longitude"], iris.analysis.MEAN
        ).data
        # take the mean over the lat and lon for obs cube recent regrid
        obs_mean_recent_vals = obs_cube_recent_regrid.collapsed(
            ["latitude", "longitude"], iris.analysis.MEAN
        ).data
    else:
        raise ValueError("Country not recognised.")

    # # print the obs mean
    # print("Obs mean:")
    # print(obs_mean)

    # dates = obs_cube_full.coord("time").points

    # extract the dates from obs mean test vals
    dates_test = cftime.num2date(
        obs_cube_test_regrid.coord("time").points,
        obs_cube_test_regrid.coord("time").units.origin,
    )

    # convert to datetime
    dates_test_dt = pd.to_datetime(dates_test.astype(str))

    # do the same for obs mean recent vals
    dates_recent = cftime.num2date(
        obs_cube_recent_regrid.coord("time").points,
        obs_cube_recent_regrid.coord("time").units.origin,
    )

    # convert to datetime
    dates_recent_dt = pd.to_datetime(dates_recent.astype(str))

    # Set up the dataframe
    obs_df_past = pd.DataFrame(
        {
            "dates": dates_test_dt,
            "obs_mean": obs_mean_test_vals,
        }
    )

    # Set up the dataframe
    obs_df_recent = pd.DataFrame(
        {
            "dates": dates_recent_dt,
            "obs_mean": obs_mean_recent_vals,
        }
    )

    # Set the dates as the index in the dataframe
    obs_df_past.set_index("dates", inplace=True)
    obs_df_recent.set_index("dates", inplace=True)

    # Calculate daily means for the recent period
    obs_df_recent_daily = obs_df_recent.resample("D").mean()

    # join the past and recent daily dataframes by index
    obs_df = pd.concat([obs_df_past, obs_df_recent_daily], axis=0)

    # print the head of the obs df
    print("Obs df:")
    print(obs_df.head())
    # print the tail of the obs df
    print("Obs df:")
    print(obs_df.tail())

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
# %%
