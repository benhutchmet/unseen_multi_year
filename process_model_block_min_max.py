"""
process_model_block_min_max.py
==========================

This script processes the model data into the correct array format for a given block minima or maxima dataframe.

Hard coded to extract the spatial fields for block maxima demand net wind from the model

Usage:
------
    $ python process_model_block_min_max.py \
        --variable <variable> \
        --season <season> \
        --region <region>

    $ python process_model_block_min_max.py \
        --variable tas \
        --season DJF \
        --region NA

Arguments:
    --variable : str : variable name
    --season : str : season name (DJF, MAM, JJA, SON)
    --region : str : region name (NA, EU, AS)

Returns:
    None

    Saves output to specified df.

"""

# Local imports
import os
import sys
import glob
import time
import json
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
    start_time = time.time()

    # Set up the hard coded variables
    maxima_dnw_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/block_maxima_model_demand_net_wind_30-06-2025_2020-2024.csv"
    arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/"
    subset_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/subset/"
    model = "HadGEM3-GC31-MM"
    temp_res = "day"
    model_years = np.arange(1960, 2018 + 1, 1)
    members_list = np.array([10, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    # if the subset dir does not exist, create it
    if not os.path.exists(subset_dir):
        os.makedirs(subset_dir)
        print(f"Created directory {subset_dir}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process model data into the correct array format for a given block minima or maxima dataframe."
    )
    parser.add_argument(
        "--variable", type=str, required=True, help="Variable name (e.g. tas)"
    )
    parser.add_argument(
        "--season",
        type=str,
        required=True,
        help="Season name (e.g. DJF, MAM, JJA, SON)",
    )
    parser.add_argument(
        "--region", type=str, required=True, help="Region name (e.g. NA, EU, AS)"
    )
    args = parser.parse_args()

    # Print the args
    print("--------------------")
    print("Arguments passed to the script:")
    print("Variable: ", args.variable)
    print("Season: ", args.season)
    print("Region: ", args.region)
    print("--------------------")

    # Set up the current date
    current_date = datetime.now()

    # Set up the output file path
    # FIXME: Hard-coded for low wind
    # output_file_path = os.path.join(
    #     subset_dir,
    #     f"HadGEM3-GC31-MM_{args.variable}_{args.region}_1960-2018_{args.season}_{temp_res}_DnW_subset_{current_date.strftime('%Y-%m-%d')}.npy",
    # )
    output_file_path = os.path.join(
        subset_dir,
        f"HadGEM3-GC31-MM_{args.variable}_{args.region}_1960-2018_{args.season}_{temp_res}_DnW_subset_{current_date.strftime('%Y-%m-%d')}.npy",
    )

    # set up the output file name
    # for the index list
    # output_file_name_index_list = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_1960-2018_{args.season}_{temp_res}_DnW_subset_index_list_{current_date.strftime('%Y-%m-%d')}.json"
    # FIXME: Hard-coded for low wind
    output_file_name_index_list = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_1960-2018_{args.season}_{temp_res}_DnW_subset_index_list_{current_date.strftime('%Y-%m-%d')}.json"
    output_file_path_index_list = os.path.join(
        subset_dir,
        output_file_name_index_list,
    )

    # if the output file already exists, raise an error
    if os.path.exists(output_file_path):
        raise FileExistsError(
            f"Output file {output_file_path} already exists. Please delete it before running the script."
        )
    else:
        print(f"Output file {output_file_path} does not exist. Proceeding...")

    # if the output file for the index list already exists, raise an error
    if os.path.exists(output_file_path_index_list):
        print(
            "Warning: Output file for index list already exists. Proceeding to overwrite it..."
        )

    # Load the block maxima dataframe
    df = pd.read_csv(maxima_dnw_path)

    # print the head of the df
    print("Head of the dataframe:")
    print(df.head())

    # print the tail of the df
    print("Tail of the dataframe:")
    print(df.tail())

    # Print the shape of the df
    print("Shape of the dataframe: ", df.shape)

    # # print the time taken
    # print("Time taken to load the dataframes: ", time.time() - start_time)
    # sys.exit()

    # Set up the test file path
    test_file_path = os.path.join(
        arrs_dir,
        f"HadGEM3-GC31-MM_{args.variable}_{args.region}_1960_{args.season}_day.npy",
    )

    # Glob the test file path
    test_file_paths = glob.glob(test_file_path)

    # if the test file path is empty, raise an error
    if len(test_file_paths) == 0:
        raise FileNotFoundError(
            f"Test file {test_file_path} does not exist. Please check the path."
        )
    elif len(test_file_paths) > 1:
        raise FileExistsError(
            f"More than 1 test file {test_file_path} found. Please check the path."
        )

    # Load the test file
    test_file = np.load(test_file_paths[0])

    # print the shape of the test file
    print("Shape of the test file: ", test_file.shape)

    # df = df_low_wind
    # df = df_higher_wind

    # Set up the shape for the subset array
    subset_arr_full = np.zeros(
        (
            len(df),
            test_file.shape[3],
            test_file.shape[4],
        )
    )

    # print the shape of subset_arr_full
    print("Shape of the subset array: ", subset_arr_full.shape)

    # sys.exit()

    # Set up the dictionary to append the data to
    data_dict = {
        "init_year": [],
        "member": [],
        "lead": [],
        "effective_dec_year": [],
    }

    # strip the effective dec year column
    # to the first 4 characters
    df["effective_dec_year"] = df["effective_dec_year"].astype(str).str[:4]

    # # print that we are testing
    # print("------------------------------------------------")
    # print("Testing the script...")
    # print("Limiting the script to 10 iterations for testing purposes...")

    # # subset the dataframe to the first 10 rows
    # df = df.iloc[:10, :]

    # # print that the dataframe has been subsetted
    # print("Dataframe subsetted to 10 rows.")
    # print("Shape of the dataframe: ", df.shape)
    # print("--------------------------------")

    # Loop through the dataframe
    for i, row in tqdm(
        enumerate(df.iterrows()), desc="Processing dataframe", total=len(df)
    ):
        # Get the row
        row = row[1]

        # Get the init year
        init_year = int(row["init_year"])
        member = int(row["member"])
        lead = int(row["lead"])

        # Include the effective dec year
        effective_dec_year = row["effective_dec_year"]

        # Strip the first 4 characters and format as an int
        effective_dec_year_int = int(effective_dec_year)

        # set up the file to extract
        model_data_path = os.path.join(
            arrs_dir,
            f"{model}_{args.variable}_{args.region}_{init_year}_{args.season}_{temp_res}.npy",
        )

        # glob the model data path
        model_data_paths = glob.glob(model_data_path)

        # if the model data path is empty, raise an error
        if len(model_data_paths) == 0:
            raise FileNotFoundError(
                f"Model data file {model_data_path} does not exist. Please check the path."
            )
        elif len(model_data_paths) > 1:
            raise FileExistsError(
                f"Model data file {model_data_path} does not exist. Please check the path."
            )

        # load the data
        model_data = np.load(model_data_paths[0])

        # find the index of the member
        member_index = np.where(members_list == member)[0][0]

        # find the index of the lead
        lead_index = int(lead) - 1

        # extract the data for this member
        model_data_this = model_data[0, member_index, lead_index, :, :]

        # append the data to the subset array
        subset_arr_full[i, :, :] = model_data_this

        # append the data to the dictionary
        data_dict["init_year"].append(init_year)
        data_dict["member"].append(member)
        data_dict["lead"].append(lead)
        data_dict["effective_dec_year"].append(effective_dec_year_int)

    # Print the shape of the subset array
    print("Shape of the subset array: ", subset_arr_full.shape)

    # print the values of the subset array
    print("Values of the subset array: ")
    print(subset_arr_full)

    # Print the values of the dictionary
    print("Values of the dictionary: ")
    print(data_dict)

    # Save the subset array to a file
    np.save(output_file_path, subset_arr_full)

    # Save the dictionary to a file
    with open(output_file_path_index_list, "w") as f:
        json.dump(data_dict, f)

    # print the time taken to run the script
    print("Time taken to run the script: ", time.time() - start_time)

    return None


if __name__ == "__main__":
    main()
