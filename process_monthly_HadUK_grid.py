"""
process_monthly_HadUK_grid.py
=============================

This script processes the monthly gridded HadUK-Grid data for the UK.

This is done by:

1. Extracting the data from the netCDF files for a specific year.
2. Applying the country mask to the data.
3. Taking the spatial average of the data for the UK.
4. Saving the data to a CSV file.

Usage:
======

    $ python process_monthly_HadUK_grid.py --year 2019 --variable tas

Arguments:
==========

    --year: The year to process the data for.
    --variable: The variable to process the data for. This can be one of the following:
        - tas: Mean air temperature (°C)
        - pr: Precipitation (mm)
        - sun: Sunshine duration (hours)

Example:
========

    $ python process_monthly_HadUK_grid.py --year 2019 --variable tas

Author:
=======
Ben Hutchins, March 2025

"""

# Local imports
import os
import sys
import glob
import time
import argparse

# Third-party imports
import numpy as np
import pandas as pd
import iris
import cftime

# Specific imports
from tqdm import tqdm
from iris.util import equalise_attributes

# Import the dictionaries
import dictionaries as dic

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import create_masked_matrix

def convert_360_day_calendar_month(time_value, obs_cube):
    return cftime.num2date(time_value, units=obs_cube.coord("time").units.name, calendar="gregorian").strftime("%Y-%m")

# Define the main function
def main():
    # Set up the start timer
    start_time = time.time()

    # Set up the hard-coded variables
    data_dir = "/badc/ukmo-hadobs/data/insitu/MOHC/HadOBS/HadUK-Grid/v1.1.0.0/60km/"
    time_res = "mon"
    output_dir_dfs_haduk = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/HadUK-Grid"

    # if the directory does not exist, create it
    if not os.path.exists(output_dir_dfs_haduk):
        os.makedirs(output_dir_dfs_haduk)

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process the monthly gridded HadUK-Grid data for the UK.")
    parser.add_argument("--year", type=int, help="The year to process the data for.")
    parser.add_argument("--variable", type=str, help="The variable to process the data for.")

    # Parse the arguments
    args = parser.parse_args()

    # Print the year and variable
    print("=======================================")
    print(f"Year: {args.year}")
    print(f"Variable: {args.variable}")
    print("=======================================")

    # Set up the df name
    df_name = f"HadUK-Grid_60km_{args.variable}_{args.year}_UK_landmask.csv"

    # form the output path
    output_path = os.path.join(output_dir_dfs_haduk, df_name)

    # print the path
    print("=======================================")
    print("Output path:")
    print(output_path)
    print("=======================================")

    # if the file already exists, then return
    if os.path.exists(output_path):
        print("=======================================")
        print(f"File already exists: {output_path}")
        print("=======================================")
        return None

    # Set up the data directory
    data_dir_full = f"{data_dir}/{args.variable}/{time_res}/v????????/"

    # Set up the file pattern
    file_pattern = f"{args.variable}_hadukgrid_uk_60km_{time_res}_{args.year}01-{args.year}12.nc"

    # Find the files
    files = glob.glob(os.path.join(data_dir_full, file_pattern))

    # if there are no files
    if len(files) == 0:
        print("No files found.")
        return None
    
    # print the files
    print("=======================================")
    print("Files found:")
    for file in files:
        print(file)
    print("=======================================")

    # assert that the len of files is 1
    assert len(files) == 1, "More than one file found."

    # Load the data
    obs_cube = iris.load_cube(files[0], f"{args.variable}")

    # Create a masked matrix
    MASK_MATRIX = create_masked_matrix(
        country="United Kingdom",
        cube=obs_cube,
    )

    # Extract the data
    obs_cube_values = obs_cube.data

    # print the obs_cube_values
    print("=======================================")
    print("Shape of data:")
    print(obs_cube_values.shape)
    print("=======================================")

    # print the actual values of obs cube values
    print("=======================================")
    print("Data values:")
    print(obs_cube_values)
    print("=======================================")

    # Multiply this by the mask matrix
    obs_cube_values_masked = obs_cube_values * MASK_MATRIX

    # Where the mask is zero, set the value to NaN
    obs_cube_values_masked = np.where(MASK_MATRIX == 0, np.nan, obs_cube_values_masked)

    # Where the mask is greater than 1e18, set the value to NaN
    obs_cube_values_masked = np.where(obs_cube_values_masked > 1e18, np.nan, obs_cube_values_masked)

    # print the shape of the masked data
    print("=======================================")
    print("Shape of masked data:")
    print(obs_cube_values_masked.shape)
    print("=======================================")

    # Take the spatial average
    obs_cube_values_masked_avg = np.nanmean(obs_cube_values_masked, axis=(1, 2))

    # extract the time points
    time_points = obs_cube.coord("time").points

    # # convert the time points
    # time_points = cftime.num2date(time_points, units=obs_cube.coord("time").units.name, calendar="gregorian")

    # print the time points
    print("=======================================")
    print("Time points:")
    print(time_points)
    print("=======================================")

    #     time:axis = "T" ;
    # time:bounds = "time_bnds" ;
    # time:units = "hours since 1800-01-01 00:00:00" ;
    # time:standard_name = "time" ;
    # time:calendar = "gregorian" ;

    # print the time points
    print("=======================================")
    print("Time points:")
    print(time_points)
    print("=======================================")

    year = args.year
    months = [f"{year}-{str(month).zfill(2)}" for month in range(1, 13)]


    # Set up the dataframe
    df = pd.DataFrame(
        {
            "time": time_points,
            f"{args.variable}": obs_cube_values_masked_avg,
        }
    )

    # convert the time points
    df["time"] = df["time"].apply(convert_360_day_calendar_month, obs_cube=obs_cube)

    # print the df
    print("=======================================")
    print("DataFrame:")
    print(df)
    print("=======================================")

    # Save the dataframe
    df.to_csv(output_path, index=False)

    # print that the dataframe has been saved to a csv
    print("=======================================")
    print("DataFrame saved to CSV at: ", output_path)
    print("=======================================")

    # Set up the end time
    end_time = time.time()

    # Print the time taken
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Time taken: {(end_time - start_time) / 60:.2f} minutes")

    print("=======================================")
    print("Exiting...")
    print("=======================================")

    return None

# If name is main
if __name__ == "__main__":
    main()