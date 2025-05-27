#!/usr/bin/env python
"""

process_delta_p_obs.py
======================

This script loads in the obs PSL spatial arrays for the winter(s) and then
processes the delta P index for these, for comparison against the wind speed.

Example usage:

    python process_delta_p_obs.py --year 2020 --region NA

Arguments:
    --year: The year for which to process the delta P observations.
    --region: The region for which to process the delta P observations (e.g., 'NA' for North Atlantic).

Returns:
    Saves the processed delta P index to a specified output file.
    
"""

# %%
# Local imports
import os
import sys
import glob
import time
import argparse
import warnings

# Third-party imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry
import cartopy.io.shapereader as shpreader
import iris
import cftime
import seaborn as sns

# Specific imports
from tqdm import tqdm

# Import the dictionaries
import dictionaries as dicts

# Define the main function
def main():
    # Start a timer
    start_time = time.time()

    sys.argv = [
        "process_delta_p_obs.py",  # Script name
        "--year", "1961",          # Specify the year
        "--region", "NA"           # Specify the region
    ]


    # hardcode the variable
    variable = "psl"
    time_freq = "day"
    obs_save_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/obs/"
    metadata_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/"
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"

    # Set up the n and s box
    n_box = dicts.uk_n_box_corrected
    s_box = dicts.uk_s_box_corrected

    # Set up the argparse parser
    parser = argparse.ArgumentParser(description="Process delta P observations.")
    parser.add_argument("--year", type=int, required=True, help="Year to process delta P observations for.")
    parser.add_argument("--region", type=str, required=True, help="Region to process delta P observations for (e.g., 'NA').")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print(f"Processing delta P observations for all years, region: {args.region}")

    # loop over the year
    for year_this in tqdm(range(2019, 2024 + 1)):
        # Print the year
        print(f"Processing year: {year_this}")

        # Set up the output fname
        output_fname = f"delta_p_obs_{year_this}_DJF.csv"

        # Set up the obs path
        obs_output_path = os.path.join(obs_save_path, output_fname)

        # Check if the obs file already exists
        if os.path.exists(obs_output_path):
            print(f"Obs file {obs_output_path} already exists. Skipping processing.")
            continue
        
        # Set up the fname
        # if the year is 2020 or less
        if year_this <= 2019:
            fname = f"ERA5_{variable}_{args.region}_{year_this}_DJF_{time_freq}.npy"
            fname_times = f"ERA5_{variable}_{args.region}_{year_this}_DJF_{time_freq}_times.npy"

            # glob the files
            obs_files = glob.glob(os.path.join(obs_save_path, fname))

            # Assert that the file has len 1
            assert len(obs_files) == 1, f"Expected one file, found {len(obs_files)} files."
        else:
            fname = f"ERA5_{variable}_{args.region}_{year_this}_DJF_{time_freq}_????????_??????.npy"
            fname_times = f"ERA5_{variable}_{args.region}_{year_this}_DJF_{time_freq}_times_????????_??????.npy"

            # glob the files
            obs_files = glob.glob(os.path.join(obs_save_path, fname))

            # Assert that the file has len 1
            assert len(obs_files) == 1, f"Expected one file, found {len(obs_files)} files."

        # Load the obs data
        obs_data = np.load(obs_files[0])

        # print the ful path to the obs times files
        print(f"Obs times files: {os.path.join(metadata_dir, fname_times)}")

        # load in the times
        obs_times_files = glob.glob(os.path.join(metadata_dir, fname_times))

        # Set up the lats fname
        # HadGEM3-GC31-MM_psl_NA_1962_DJF_day_lats.npy
        lats_fname = f"HadGEM3-GC31-MM_{variable}_{args.region}_{year_this}_DJF_{time_freq}_lats.npy"
        lats_files = glob.glob(os.path.join(metadata_dir, lats_fname))
        assert len(lats_files) == 1, f"Expected one lats file, found {len(lats_files)} files."

        # Load the lats
        lats = np.load(lats_files[0])

        # Set up the lons fname
        # HadGEM3-GC31-MM_psl_NA_1962_DJF_day_lons.npy
        lons_fname = f"HadGEM3-GC31-MM_{variable}_{args.region}_{year_this}_DJF_{time_freq}_lons.npy"
        lons_files = glob.glob(os.path.join(metadata_dir, lons_fname))
        assert len(lons_files) == 1, f"Expected one lons file, found {len(lons_files)} files."
        # Load the lons
        lons = np.load(lons_files[0])

        # Load the times
        if len(obs_times_files) == 1:
            obs_times = np.load(obs_times_files[0])
        else:
            raise ValueError(f"Expected one times file, found {len(obs_times_files)} files.")
        
        # if the year is not 2019
        if year_this != 2019:
            # Convert the times
            obs_times_dt = pd.to_datetime(
                obs_times, origin='1900-01-01 00:00:00', unit='h'
            )
        else:
            # For 2019, the times are already in datetime format
            start_date = pd.to_datetime("2019-12-01 00:00:00")
            end_date = pd.to_datetime("2020-02-29 23:00:00")

            # Create a date range for the times
            obs_times_dt = pd.date_range(start=start_date, end=end_date, freq='D')

        # print the first time
        print(f"First time in obs: {obs_times_dt[0]}")

        # Print the last time
        print(f"Last time in obs: {obs_times_dt[-1]}")

        # Extract the n_box lats and lons
        lat1_box_n, lat2_box_n = n_box["lat1"], n_box["lat2"]
        lon1_box_n, lon2_box_n = n_box["lon1"], n_box["lon2"]

        # Extract the s_box lats and lons
        lat1_box_s, lat2_box_s = s_box["lat1"], s_box["lat2"]
        lon1_box_s, lon2_box_s = s_box["lon1"], s_box["lon2"]

        # # Print the box coordinates
        # print(f"North box coordinates: lat1={lat1_box_n}, lat2={lat2_box_n}, lon1={lon1_box_n}, lon2={lon2_box_n}")
        # print(f"South box coordinates: lat1={lat1_box_s}, lat2={lat2_box_s}, lon1={lon1_box_s}, lon2={lon2_box_s}")

        # # print the lats
        # print(f"Lats: {lats}")
        # # print the lons
        # print(f"Lons: {lons}")

        # Find the indices of the lats which correspond to the gridbox
        lat1_idx_n = np.argmin(np.abs(lats - lat1_box_n))
        lat2_idx_n = np.argmin(np.abs(lats - lat2_box_n))
        lon1_idx_n = np.argmin(np.abs(lons - lon1_box_n))
        lon2_idx_n = np.argmin(np.abs(lons - lon2_box_n))

        lat1_idx_s = np.argmin(np.abs(lats - lat1_box_s))
        lat2_idx_s = np.argmin(np.abs(lats - lat2_box_s))
        lon1_idx_s = np.argmin(np.abs(lons - lon1_box_s))
        lon2_idx_s = np.argmin(np.abs(lons - lon2_box_s))

        # calculate the gridbox mean n
        obs_n_box = obs_data[:, lat1_idx_n:lat2_idx_n + 1, lon1_idx_n:lon2_idx_n + 1]
        obs_n_box_mean = np.mean(obs_n_box, axis=(1, 2))

        # calculate the gridbox mean s
        obs_s_box = obs_data[:, lat1_idx_s:lat2_idx_s + 1, lon1_idx_s:lon2_idx_s + 1]
        obs_s_box_mean = np.mean(obs_s_box, axis=(1, 2))

        # Calculate the delta P index
        delta_p_index = obs_n_box_mean - obs_s_box_mean

        print(f"Delta P index shape: {delta_p_index.shape}")
        # print(f"Delta P index: {delta_p_index}")

        # prtin the shape of obs time dt
        print(f"Obs times dt shape: {obs_times_dt.shape}")

        # Create a DataFrame with the delta P index and times
        df_delta_p = pd.DataFrame({
            "time": obs_times_dt,
            "delta_p_index": delta_p_index
        })

        # Save the df
        df_delta_p.to_csv(obs_output_path, index=False)
        print(f"Saved delta P index to {obs_output_path}")

    # stop the time
    end_time = time.time()

    # Print the time taken
    print(f"Time taken to process obs: {end_time - start_time:.2f} seconds")

    return None

if __name__ == "__main__":
    # Call the main function
    main()
# %%
