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

    # print that the sctipt is finished
    print("=============================================")
    print("Script finished.")
    print("=============================================")

    # print the time taken
    print("Time taken: {} seconds.".format(time.time() - start))

    return

if __name__ == "__main__":
    main()