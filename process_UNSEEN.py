#!/usr/bin/env python

"""
process_UNSEEN.py
=================

This script takes as input the variable name (e.g. "tas"), the country (e.g.
"United Kingdom"), the season (e.g. "ONDJFM"), the first year (e.g. 1960) and
the last year (e.g. 2014) and then loads the model and observed data for the
period, variable and region specified. It then performs the fidelity testing
(mean, sigma, skewness, kurtosis) for the model and observed data and saves the
resulting plot to the output directory.

Usage:
------

    $ python process_UNSEEN.py --variable tas --country "United Kingdom" --season ONDJFM --first_year 1960 --last_year 2014

Arguments:
----------

    --variable: str
        The variable name (e.g. "tas").

    --country: str
        The country name (e.g. "United Kingdom").

    --season: str
        The season name (e.g. "ONDJFM").

    --first_year: int
        The first year of the period (e.g. 1960).

    --last_year: int
        The last year of the period (e.g. 2014).

Returns:
--------

    A plot of the fidelity testing results for the model and observed data.
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

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
import functions as funcs


# Define the main function
def main():
    # Start the timer
    start = time.time()

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process UNSEEN data.")
    parser.add_argument("--variable", type=str, help="The variable name (e.g. tas).")
    parser.add_argument("--country", type=str, help="The country name (e.g. United Kingdom).")
    parser.add_argument("--season", type=str, help="The season name (e.g. ONDJFM).")
    parser.add_argument("--first_year", type=int, help="The first year of the period (e.g. 1960).")
    parser.add_argument("--last_year", type=int, help="The last year of the period (e.g. 2014).")

    # Parse the arguments
    args = parser.parse_args()

    # print the arguments
    print(f"Variable: {args.variable}")
    print(f"Country: {args.country}")
    print(f"Season: {args.season}")
    print(f"First year: {args.first_year}")
    print(f"Last year: {args.last_year}")

    # set up the obs variable depending on the variable
    if args.variable == "tas":
        obs_var = "t2m"
    elif args.variable == "sfcWind":
        obs_var = "si10"
    else:
        raise ValueError("Variable not recognised")
    
    # Set up the months depending on the season
    if args.season == "DJF":
        months = [12, 1, 2]
    elif args.season == "MAM":
        months = [3, 4, 5]
    elif args.season == "JJA":
        months = [6, 7, 8]
    elif args.season == "SON":
        months = [9, 10, 11]
    elif args.season == "ONDJFM":
        months = [10, 11, 12, 1, 2, 3]
    else:
        raise ValueError("Season not recognised")

    # set up the hard coded args
    model = "HadGEM3-GC31-MM"
    experiment = "dcppA-hindcast"
    freq = "Amon"

    # Set up the path to the ERA5 data
    # if the variable is tas
    if args.variable == "tas":
        # already regridded!
        obs_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/t2m_ERA5_regrid_HadGEM.nc"
    # if the variable is sfcWind
    elif args.variable == "sfcWind":
        # needs regridding
        obs_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/surface_wind_ERA5.nc"
    else:
        raise ValueError("Variable not recognised") 

    # Load the model ensemble
    model_ds = funcs.load_model_data_xarray(
        model_variable=args.variable,
        model=model,
        experiment=experiment,
        start_year=args.first_year,
        end_year=args.last_year,
        first_fcst_year=int(args.first_year) + 1,
        last_fcst_year=int(args.first_year) + 2,
        months=months,
        frequency=freq,
        parallel=False,
    )

    # Get the size of the model data in bytes
    size_in_bytes = model_ds[args.variable].size * model_ds[args.variable].dtype.itemsize

    # Convert bytes to gigabytes
    size_in_gb = size_in_bytes / (1024 ** 3)

    # Print the size
    print(f"Model data size: {size_in_gb} GB")

    # Modify member coordiante before conbersion to iris
    model_ds['member'] = model_ds['member'].str[1:-6].astype(int)

    # convert to an iris cube
    model_cube = model_ds[args.variable].squeeze().to_iris()

    # Load the observed data
    obs_ds = xr.open_mfdataset(
        obs_path,
        combine="by_coords",
        parallel=False,
        engine="netcdf4",
    )

    # Restrict the time to the region we are interested in
    obs_ds = obs_ds.sel(time=slice(f"{int(args.first_year)}-{months[0]}-01", f"{int(args.last_year) + 1}-{months[-1]}-31"))

    # If expver is present in the observations
    if "expver" in obs_ds.coords:
        # Combine the first two expver variables
        obs_ds = obs_ds.sel(expver=1).combine_first(obs_ds.sel(expver=5))

    # Get the size of the observed data in bytes
    size_in_bytes = obs_ds[obs_var].size * obs_ds[obs_var].dtype.itemsize

    # Convert bytes to gigabytes
    size_in_gb = size_in_bytes / (1024 ** 3)

    # Print the size
    print(f"Observed data size: {size_in_gb} GB")

    # convert to an iris cube
    obs_cube = obs_ds[obs_var].squeeze().to_iris()

    # if the lats and lons are not the same
    if not model_cube.coord("latitude").shape == obs_cube.coord("latitude").shape or not model_cube.coord("longitude").shape == obs_cube.coord("longitude").shape:
        print("Regridding model data")
        # regrid the obs cube to the model cube
        obs_cube = obs_cube.regrid(model_cube, iris.analysis.Linear())

    # create the mask
    MASK_MATRIX = funcs.create_masked_matrix(
        country=args.country,
        cube=model_cube,
    )

    # Apply the mask to the observed data
    obs_values = obs_cube.data * MASK_MATRIX
    model_values = model_cube.data * MASK_MATRIX

    # Where there are zeros in the mask we want to set these to Nans
    obs_values_masked = np.where(MASK_MATRIX == 0, np.nan, obs_values)
    model_values_masked = np.where(MASK_MATRIX == 0, np.nan, model_values)

    # Take the Nanmean of the data
    # TODO: 

    # print the shape of the mask matrix
    print(f"Mask matrix shape: {MASK_MATRIX.shape}")

    # print the sum of the mask matrix
    print(f"Mask matrix sum: {np.sum(MASK_MATRIX)}")

    print("----------------")
    # print the amount of time taken
    print(f"Time taken to load model data: {time.time() - start}")
    print("----------------")
    print("Script complete")

# Run the main function
if __name__ == "__main__":
    main()