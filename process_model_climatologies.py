"""
process_model_climatologies.py
=========================

This script processes the daily mean climatology for the model data (and maybe the obs?).

Assumes the climatology over all forecast years.

Usage:
------

    $ python process_model_climatologies.py \
        --variable <variable> \
        --season <season> \
        --region <region> \
        --start_year <start_year> \
        --end_year <end_year>

    $ python process_model_climatologies.py \
        --variable tas \
        --season DJF \
        --region NA \
        --start_year 1960 \
        --end_year 2018

Arguments:
----------

    --variable : str : variable name
    --season : str : season name (DJF, MAM, JJA, SON)
    --region : str : region name (NA, EU, AS)
    --start_year : int : start year of the climatology
    --end_year : int : end year of the climatology

Returns:
    None

    Saves output to specified df.

"""

# Local imports
import os
import sys
import glob
import time
import argparse

# Third-party imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import iris

# Specific imports
from tqdm import tqdm
from datetime import datetime, timedelta

# Define the main function
def main():
    # Set up the start time
    start_time = time.time()

    # Set up the hard coded variables
    temp_res = "day"
    clim_output_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_clim/"
    model = "HadGEM3-GC31-MM"
    arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/"
    winter_years = np.arange(1, 11 + 1, 1)

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process model climatologies.")
    parser.add_argument("--variable", type=str, required=True, help="Variable name")
    parser.add_argument("--season", type=str, required=True, help="Season name (DJF, MAM, JJA, SON)")
    parser.add_argument("--region", type=str, required=True, help="Region name (NA, EU, AS)")
    parser.add_argument("--start_year", type=int, required=True, help="Start year of the climatology")
    parser.add_argument("--end_year", type=int, required=True, help="End year of the climatology")

    # Print the arguments
    args = parser.parse_args()

    # print the arguments
    print("-------------------")
    print("Arguments:")
    print(f"Variable: {args.variable}")
    print(f"Season: {args.season}")
    print(f"Region: {args.region}")
    print(f"Start Year: {args.start_year}")
    print(f"End Year: {args.end_year}")
    print("-------------------")

    # if args.season is not DJF
    if args.season != "DJF":
        raise NotImplementedError(
            f"Season {args.season} is not implemented. Only DJF is implemented."
        )

    # Set up the fname for the climatology
    clim_fname = f"climatology_{model}_{args.variable}_{args.season}_{args.region}_{args.start_year}_{args.end_year}_{temp_res}.npy"

    # Set up the full path to the climatology
    clim_path = os.path.join(clim_output_dir, clim_fname)

    # Check if the climatology file exists
    if os.path.exists(clim_path):
        print(f"Climatology file already exists: {clim_path}")
        return None
    
    # print that the climatology file does not exist
    print(f"Climatology file does not exist: {clim_path}")

    # Set up the years to extract
    years = np.arange(args.start_year, args.end_year + 1)

    # Set up a list for the files which dont exist
    missing_files = []

    # Loop over the years
    # to check whether files exist
    for year in tqdm(years, desc="Checking files", unit="year"):
        # Set up the file name
        file_name = f"{model}_{args.variable}_{args.region}_{year}_{args.season}_{temp_res}_*.npy"

        # glob the file name
        file_path = os.path.join(arrs_dir, file_name)

        # glob the file path
        file_paths = glob.glob(file_path)

        if len(file_paths) == 0:
            print(f"File {file_name} does not exist")
            missing_files.append(file_name)
            continue

    print("Missing files:")
    print(missing_files)
    
    # Set up the test array to load
    test_file_path = os.path.join(
        arrs_dir,
        f"{model}_{args.variable}_{args.region}_{years[0]}_{args.season}_{temp_res}_*.npy"
    )

    # glob the test file path
    test_file_paths = glob.glob(test_file_path)

    # if the test file does not exist
    if len(test_file_paths) == 0:
        print(f"Test file {test_file_path} does not exist")
        raise FileNotFoundError(
            f"Test file {test_file_path} does not exist"
        )
    elif len(test_file_paths) > 1:
        print(f"Test file {test_file_path} does not exist")
        raise FileExistsError(
            f"Test file {test_file_path} does not exist"
        )
    
    # Set up the test file path
    test_file_path = test_file_paths[0]
    
    # Load the test file as an array
    test_arr = np.load(test_file_path)
    
    # Set up an empty list to store the indices
    winter_indices = []

    # Loop over the years
    for winter_year in winter_years:
        # Set up the indices
        indices_this = np.arange(
            30 + ((winter_year - 1) * 360),
            30 + 90 + ((winter_year - 1) * 360) + 1,
            1
        )

        # print the winter year and the first lead this
        print(winter_year, indices_this[0], indices_this[-1])

        # Append the indices to the list
        winter_indices.extend(indices_this)

    # Set up the model array to load into
    model_full_arr = np.zeros((
        len(years),
        test_arr.shape[1],
        len(winter_indices),
        test_arr.shape[3],
        test_arr.shape[4],
    ))
    
    # Loop over the years
    for i, year in tqdm(enumerate(years), desc="Loading files", unit="year"):
        # Form the file path
        file_path = os.path.join(arrs_dir, f"{model}_{args.variable}_{args.region}_{year}_{args.season}_{temp_res}_*.npy")

        # glob the file path
        file_paths = glob.glob(file_path)

        # if the file path is empty then print an error message
        if len(file_paths) == 0:
            print(f"File {file_path} does not exist")
            sys.exit()
        elif len(file_paths) > 1:
            print(f"File {file_path} greater than 1")
            sys.exit()

        # load the array
        arr_this_year = np.load(file_paths[0])

        # if the 2th dimension does not have shape 3750
        # then print an error message
        if arr_this_year.shape[2] != 3750:
            print(f"Array {file_path} does not have the correct shape")
            print("subsetting the array")
            arr_this_year = arr_this_year[:, :, 0:3750, :, :]

        # Subset the array to only include the winter days
        arr_this_year = arr_this_year[:, :, winter_indices, :, :]
        
        # Load in the array
        model_full_arr[i, :, :, :, :] = arr_this_year

    # print the shape of the model full array
    print(f"Model full array shape: {model_full_arr.shape}")

    # Take the mean over the first three dimensions
    model_clim = np.mean(model_full_arr, axis=(0, 1, 2))

    # print the shape of the model climatology
    print(f"Model climatology shape: {model_clim.shape}")

    # Save the climatology to a .npy file
    np.save(clim_path, model_clim)
    print(f"Climatology saved to {clim_path}")

    # print the time taken
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("Script completed successfully.")

    return None

if __name__ == "__main__":
    main()
