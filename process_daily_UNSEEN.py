#!/usr/bin/env python

"""
process_daily_UNSEEN.py
=======================

This script processes daily model data into dataframes for a given
initialisation year, variable, and country combination.

Usage:
------

    $ python process_daily_UNSEEN.py --variable tas --country "United Kingdom" --init_year 1960

Arguments:
----------

    --variable : str : variable name (e.g. tas, pr, psl)
    --country : str : country name (e.g. United Kingdom, France, Germany)
    --init_year : int : initialisation year (e.g. 1960)

Returns:
--------

    dataframes : pd.DataFrame : processed dataframes for the given variable, country, and initialisation year which are saved to a /gws/ folder.

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

# # Specific imports
# from ncdata.iris_xarray import cubes_to_xarray, cubes_from_xarray

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
import functions as funcs

# Define the main function
def main():
    # Start a timer
    start = time.time()

    # Set up the hard-coded args
    model = "HadGEM3-GC31-MM"
    experiment = "dcppA-hindcast"
    freq = "day"
    months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process daily UNSEEN data.")
    parser.add_argument("--variable", type=str, help="Variable name (e.g. tas, pr, psl)")
    parser.add_argument("--country", type=str, help="Country name (e.g. United Kingdom, France, Germany)")
    parser.add_argument("--init_year", type=int, help="Initialisation year (e.g. 1960)")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("=============================================")
    print("Processing UNSEEN data for the following:")
    print("Variable: {}".format(args.variable))
    print("Country: {}".format(args.country))
    print("Initialisation year: {}".format(args.init_year))
    print("=============================================")

    # if country contains a _
    # e.g. United_Kingdom
    # replace with a space
    if "_" in args.country:
        args.country = args.country.replace("_", " ")

    # Set up the output directory for the dfs
    output_dir_dfs = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs"

    # Set up the name for the model df
    model_df_name = f"{model}_{experiment}_{args.variable}_{args.country}_{args.init_year}_{freq}.csv"

    # Set up the name for the observations
    obs_name = f"ERA5_{args.variable}_{args.country}_{args.init_year}_{freq}.csv"

    # If the paths exist
    if os.path.exists(os.path.join(output_dir_dfs, model_df_name)) and os.path.exists(os.path.join(output_dir_dfs, obs_name)):
        print("=============================================")
        print("Dataframes already exist.")
        print("=============================================")
        return
    
    # print that we are creating the dfs
    print("=============================================")
    print("Creating dataframes.")
    print("=============================================")

    # Set up the path to the observed data
    base_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/"

    # if the variable is tas
    if args.variable == "tas":
        obs_path = os.path.join(base_path, "ERA5_t2m_daily_1950_2020.nc")
    elif args.variable == "sfcWind":
        obs_path = os.path.join(base_path, "ERA5_wind_daily_1960_2020.nc")
    else:
        raise ValueError("Variable not recognised.")
    
    # Assert that the obs path exists
    assert os.path.exists(obs_path), "Observations path does not exist."

    # Load the model data
    model_ds = funcs.load_model_data_xarray(
        model_variable=args.variable,
        model=model,
        experiment=experiment,
        start_year=args.init_year,
        end_year=args.init_year,
        first_fcst_year=int(args.init_year) + 1,
        last_fcst_year=int(args.init_year) + 10,
        months=months,
        frequency=freq,
        parallel=False,
    )

    # print the model data
    print("Model data:")
    print(model_ds)

    # Modify the member coordinate before conversion to iris
    # Modify member coordiante before conbersion to iris
    model_ds["member"] = model_ds["member"].str[1:-6].astype(int)

    # convert to an iris cube
    model_cube = model_ds[args.variable].squeeze().to_iris()

    # set up the obs variab;e
    if args.variable == "tas":
        obs_variable = "t2m"
    elif args.variable == "sfcWind":
        obs_variable = "si10"
    else:
        raise ValueError("Variable not recognised.")
    
    # load the obs cube test
    obs_cube_test = iris.load_cube(obs_path)

    # constrain to the relevant year
    obs_cube_test = obs_cube_test.extract(iris.Constraint(time=lambda cell: cell.point.year == args.init_year))

    # perform the intersection
    obs_cube_test = obs_cube_test.intersection(
        latitude=(30, 80),
        longitude=(-40, 30),
    )

    # print(obs_cube_test.coord('latitude').attributes.get('axis'))
    # print(obs_cube_test.coord('longitude').attributes.get('axis'))

    # # print the obs cube axis=x
    # print(obs_cube_test.coord(axis='x'))
    # print(obs_cube_test.coord(axis='y'))

    # sys.exit()

    # # Load the observed data
    # obs_ds = xr.open_mfdataset(
    #     obs_path,
    #     combine="by_coords",
    #     parallel=False,
    #     engine="netcdf4",
    # )

    # # restrict the observed data to the initialisation year
    # obs_ds = obs_ds.sel(time=slice(f"{args.init_year}-01-01", f"{args.init_year}-12-31"))

    # # If expver is present in the observations
    # if "expver" in obs_ds.coords:
    #     # Combine the first two expver variables
    #     obs_ds = obs_ds.sel(expver=1).combine_first(obs_ds.sel(expver=5))



    # # Convert to an iris cube
    # obs_cube = obs_ds[obs_variable].squeeze().to_iris()

    # # print the obs cube
    # print("Observed data:")
    # print(obs_cube)

    # # print the model cube
    # print("Model data:")
    # print(model_cube)

    # # print the lats and lons of the obs_cube and model_cube
    # print("Model cube lats and lons:")
    # print(model_cube.coord("latitude").points.min())
    # print(model_cube.coord("latitude").points.max())
    # print(model_cube.coord("longitude").points.min())
    # print(model_cube.coord("longitude").points.max())
    # print("Obs cube lats and lons:")
    # print(obs_cube.coord("latitude").points.min())
    # print(obs_cube.coord("latitude").points.max())
    # print(obs_cube.coord("longitude").points.min())
    # print(obs_cube.coord("longitude").points.max())

    # # subset the model cube and obs cube to the same region
    # obs_cube = obs_cube.intersection(
    #     latitude=(30, 80),
    #     longitude=(-40, 30),
    # )
    model_cube = model_cube.intersection(
        latitude=(30, 80),
        longitude=(-40, 30),
    )

    # print the model cube dimensions
    print("Model cube dimensions post intersection:")
    print(model_cube)

    # Select the first member and time from the model cube
    model_cube_regrid = model_cube[0, 0, :, :]

    # print the model cube regrid dimensions
    print("Model cube regrid dimensions:")
    print(model_cube_regrid)

    model_cube_regrid.coord("latitude").units = obs_cube_test[0].coord("latitude").units
    model_cube_regrid.coord("longitude").units = obs_cube_test[0].coord("longitude").units

    # and for the attributes
    model_cube_regrid.coord("latitude").attributes = obs_cube_test[0].coord("latitude").attributes
    model_cube_regrid.coord("longitude").attributes = obs_cube_test[0].coord("longitude").attributes

    # obs_cube.coord("latitude").units = model_cube_regrid.coord("latitude").units
    # obs_cube.coord("longitude").units = model_cube_regrid.coord("longitude").units

    # # and for the attributes
    # obs_cube.coord("latitude").attributes = model_cube_regrid.coord("latitude").attributes
    # obs_cube.coord("longitude").attributes = model_cube_regrid.coord("longitude").attributes

    # print the model cube regrid dimensions
    print("Model cube regrid dimensions:")
    print(model_cube_regrid)

    # print the model cube regrid lats and lons
    print(model_cube_regrid.coord("latitude"))
    print(model_cube_regrid.coord("longitude"))

    # print the min values
    print(model_cube_regrid.coord("latitude").points.min())
    print(model_cube_regrid.coord("latitude").points.max())

    # print the min values
    print(model_cube_regrid.coord("longitude").points.min())
    print(model_cube_regrid.coord("longitude").points.max())

    # print the shapes of the model and obs cubes
    print("Model cube regrid lat shapes:")
    print(model_cube_regrid.coord("latitude").shape)
    print("Model cube regrid lon shapes:")
    print(model_cube_regrid.coord("longitude").shape)

    # # print the obs cube dimensions
    # print("Obs cube dimensions:")
    # print(obs_cube[0])

    # # print the shapes of the obs cube
    # print("Obs cube lat shapes:")
    # print(obs_cube[0].coord("latitude").shape)
    # print("Obs cube lon shapes:")
    # print(obs_cube[0].coord("longitude").shape)

    # # print the min values
    # print(obs_cube[0].coord("latitude").points.min())
    # print(obs_cube[0].coord("latitude").points.max())

    # # print the min values
    # print(obs_cube[0].coord("longitude").points.min())
    # print(obs_cube[0].coord("longitude").points.max())

    # print(obs_cube.coord('latitude').attributes.get('axis'))
    # print(obs_cube.coord('longitude').attributes.get('axis'))

    # obs_lat = obs_cube.coord('latitude')
    # obs_lon = obs_cube.coord('longitude')

    # obs_lat.standard_name = 'latitude'
    # obs_lat.long_name = 'latitude'

    # # remove the coord from the cube
    # obs_cube.remove_coord('latitude')
    # obs_cube.add_dim_coord(obs_lat, 1)
    # obs_cube.remove_coord('longitude')
    # obs_cube.add_dim_coord(obs_lon, 2)

    # # if the lats and lons are not the same

    # print(obs_cube.coord('latitude').attributes.get('axis'))
    # print(obs_cube.coord('longitude').attributes.get('axis'))

    # # print the obs cube axis=x
    # print(obs_cube.coord(axis='x'))
    # print(obs_cube.coord(axis='y'))

    # # get the x and y dims for the obs cube
    LONS,LATS = iris.analysis.cartography.get_xy_grids(obs_cube_test)

    # print these
    print("Observed data:")
    print(LONS)
    print(LATS)

    # get the x and y dims for the model cube
    LONS_MODEL,LATS_MODEL = iris.analysis.cartography.get_xy_grids(model_cube_regrid)

    # print these
    print("Model data:")
    print(LONS_MODEL)
    print(LATS_MODEL)

    # regrid the obs cube to the model cube
    obs_cube_regrid = obs_cube_test.regrid(model_cube_regrid, iris.analysis.Linear())

    # print the obs cube
    print("Observed data regrid:")
    print(obs_cube_regrid)

    # print the model cube
    print("Model data:")
    print(model_cube)

    # create the mask
    MASK_MATRIX = funcs.create_masked_matrix(
        country=args.country,
        cube=model_cube,
    )

    # Apply the mask to the observed and model data
    obs_values = obs_cube_regrid.data * MASK_MATRIX
    model_values = model_cube.data * MASK_MATRIX

    # Where there are zeros we want to set these to NaNs
    obs_values = np.where(obs_values == 0, np.nan, obs_values)
    model_values = np.where(model_values == 0, np.nan, model_values)

    # Take the Nanmean of the data
    # over lat and lon dims
    obs_mean = np.nanmean(obs_values, axis=(1, 2))
    model_mean = np.nanmean(model_values, axis=(2, 3))

    # print the times points for the obscube
    print("Observed data time points:")
    print(obs_cube_regrid.coord("time").points)

    # print the times points for the modelcube
    print("Model data time points:")
    print(model_cube.coord("time").points)

    # print that the sctipt is finished
    print("=============================================")
    print("Script finished.")
    print("=============================================")

    # print the time taken
    print("Time taken: {} seconds.".format(time.time() - start))

    return

if __name__ == "__main__":
    main()