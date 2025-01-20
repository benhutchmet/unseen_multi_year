#!/usr/bin/env python

"""
process_daily_unseen.py
=======================

This script processes daily model data into dataframes for a given
initialisation year, variable, member, and country combination.

Usage:
------

    $ python process_daily_UNSEEN.py --variable tas --country "United Kingdom" --init_year 1960 --member 1

Arguments:
----------

    --variable : str : variable name (e.g. tas, pr, psl)
    --country : str : country name (e.g. United Kingdom, France, Germany)
    --init_year : int : initialisation year (e.g. 1960)
    --member : int : ensemble member number (e.g. 1)

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
from functions import load_model_data_xarray, create_masked_matrix

# Define a function to get the member string
def get_member_string(member, model):
    """
    Returns the appropriate member string based on the member value.
    
    Parameters:
    member (int): The member number (1 to 10).
    model (str): The model name.
    
    Returns:
    str: The corresponding member string.
    """
    # check the model is HadGEM3-GC31-MM
    if model != "HadGEM3-GC31-MM":
        raise ValueError("Model must be HadGEM3-GC31-MM")

    if 1 <= member <= 10 and model == "HadGEM3-GC31-MM":
        return f'r{member}i1p1f2'
    else:
        raise ValueError("Member must be between 1 and 10")
    
# Define the main function
def main():
    """
    Main function for processing daily model data into dataframes for a given
    initialisation year, variable, member, and country combination.
    """

    start_time = time.time()

    # Set up the hard-coded arguments
    model="HadGEM3-GC31-MM"
    experiment="dcppA-hindcast"
    freq="day"
    months=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    # Set up the output directory for the dfs
    output_dir_dfs = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs"

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process daily model data into dataframes for a given initialisation year, variable, member, and country combination.")
    parser.add_argument("--variable", type=str, help="Variable name (e.g. tas, pr, psl)")
    parser.add_argument("--country", type=str, help="Country name (e.g. United Kingdom, France, Germany)")
    parser.add_argument("--init_year", type=int, help="Initialisation year (e.g. 1960)")
    parser.add_argument("--member", type=int, help="Ensemble member number (e.g. 1)")

    # Parse the arguments
    args = parser.parse_args()

    # if country contains a _
    # e.g. United_Kingdom
    # replace with a space
    if "_" in args.country:
        args.country = args.country.replace("_", " ")

    # Print the arguyments
    print("=====================================")
    print("Processing daily UNSEEN data for the following arguments:")
    print(f"Variable: {args.variable}")
    print(f"Country: {args.country}")
    print(f"Initialisation year: {args.init_year}")
    print(f"Member: {args.member}")
    print("=====================================")

    # if country has a space, replace with _
    country = args.country.replace(" ", "_")

    # Set up the name for the df
    df_name = f"{model}_{experiment}_{args.variable}_{country}_{args.init_year}_{args.member}_{freq}.csv"

    # print the path to the df
    print(f"Output path: {os.path.join(output_dir_dfs, df_name)}")

    # If the df already exists raise an error
    if os.path.exists(os.path.join(output_dir_dfs, df_name)):
        print(f"The dataframe {df_name} already exists.")
        return

    # Get the member string
    member_str = get_member_string(args.member, model)

    # Load the model data
    model_ds = load_model_data_xarray(
        model_variable=args.variable,
        model=model,
        experiment=experiment,
        start_year=args.init_year,
        end_year=args.init_year,
        first_fcst_year=args.init_year + 1,
        last_fcst_year=args.init_year + 10,
        months=months,
        member=member_str,
        frequency=freq,
        parallel=False,
    )

    # print the model ds
    print(f"Dimensions of the model dataset: {model_ds.dims}")

    # # convert modify the member coordinate
    # model_ds["member"] = model_ds["member"].str[1:-6].astype(int)

    # convert to an iris cube
    model_cube = model_ds[args.variable].squeeze().to_iris()

    # Make sure cube is on the correct grid system
    model_cube = model_cube.intersection(longitude=(-180, 180))

    # if the country is in "United Kingdom" "United_Kingdom"
    if args.country == "United Kingdom" or args.country == "United_Kingdom":
        # Create the mask matrix for the UK
        MASK_MATRIX = create_masked_matrix(
            country=args.country,
            cube=model_cube,
        )

        model_data = model_cube.data

        # Apply the mask to the model cube
        model_values = model_data * MASK_MATRIX

        # Where there are zeros in the mask we want to set these to Nans
        model_values_masked = np.where(MASK_MATRIX == 0, np.nan, model_values)

        # Take the Nanmean of the data
        model_values = np.nanmean(model_values_masked, axis=(1, 2))
    elif args.country == "North Sea":
        print("Taking gridbox average for the North Sea")

        # Set up the gridbox
        gridbox = dic.north_sea_kay

        # Subset to the north sea region
        model_cube = model_cube.intersection(
            longitude=(gridbox["lon1"], gridbox["lon2"]),
            latitude=(gridbox["lat1"], gridbox["lat2"]),
        )

        # print the model cube
        print(model_cube)

        # print the lats and lons of the model cube
        print(model_cube.coord("latitude").points)
        print(model_cube.coord("longitude").points)

        # Take the mean over lat and lon
        model_values = model_cube.collapsed(["latitude", "longitude"], iris.analysis.MEAN).data
    else:
        raise ValueError("Country not recognised")

    model_df = pd.DataFrame()

    # Extract the ini years, member and lead times
    init_years = model_cube.coord("init").points
    members = model_cube.coord("member").points
    lead_times = model_cube.coord("lead").points

    # loop through the inits, members and leadtimes
    for i, init_year in enumerate(init_years):
        for m, member in enumerate(members):
            for l, lead_time in enumerate(lead_times):
                # get the model data
                model_data = model_values[l]

                # set up the model df this
                model_df_this = pd.DataFrame(
                    {
                        "init_year": [init_year],
                        "member": [member],
                        "lead": [lead_time],
                        "data": [model_data],
                    },
                )

                # concat to the model df
                model_df = pd.concat([model_df, model_df_this])

    # if the path to the file exists, raise an error
    if os.path.exists(os.path.join(output_dir_dfs, df_name)):
        raise ValueError(f"The dataframe {df_name} already exists.")
    else:
        print(f"Saving the dataframe to {os.path.join(output_dir_dfs, df_name)}")
        model_df.to_csv(os.path.join(output_dir_dfs, df_name), index=False)

    # end teh timer
    end_time = time.time()

    # Print the time taken
    print(f"Time taken to load the data: {end_time - start_time} seconds")

    return

if __name__ == "__main__":
    main()