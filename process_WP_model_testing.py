# %%
"""
process_WP_model_testing.py
=======================

This script processes daily DePreSys model data into an estimate of UK
wind power generation/capacity factors.

Usage:
------

    $ python process_WP_model_testing.py

Arguments:
----------

    None

Returns:
--------

    dataframes : pd.DataFrame : processed dataframes for the given variable, country, and initialisation year which are saved to the /gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs folder.

Author:
-------

    Ben W. Hutchins, University of Reading, 2025
"""

# Local imports
import os
import sys
import glob
import time

# Third-party imports
import iris
import iris.analysis
import iris.analysis.cartography
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry

# Specific imports
from tqdm import tqdm
from iris import cube
from scipy.stats import linregress

# Specific imports from local modules
from process_ERA5_wind_gen import (
    process_area_weighted_mean,
    convert_wind_speed_to_power_generation,
)

# more specific imports from other modules
from plot_dnw_circ import (
    load_obs_data,
)

# Imports from Hannah's functions
sys.path.append(
    "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/"
)

# Importing the necessary functions from the module
from European_hourly_hub_height_winds_2023_model import (
    load_power_curves,
)


# Define a function for loading the wind speed and taking it to hub height
def load_wind_speed_and_take_to_hubheight_model(
    model_data: np.ndarray,
    land_mask: np.ndarray,
    height_of_wind_speed: float = 10.0,
) -> np.ndarray:
    """
    Load wind speed data and take it to hub height.

    Args:
        model_data (np.ndarray): Wind speed data from the model.
        land_mask (np.ndarray): Land mask for the region.
        height_of_wind_speed (float): Height of the wind speed data in meters.

    Returns:
        np.ndarray: Wind speed data at hub height.
    """

    # If there are any values below 0 in the model data
    if np.any(model_data < 0):
        print("Warning: Negative wind speed values found in model data. Setting them to 0.")
        # Set negative values to 0
        model_data[model_data < 0] = 0.0

    # Perform the correction to hub height
    correction_hubheight = land_mask * (71.0 / height_of_wind_speed) ** (
        1.0 / 7.0
    ) + abs(land_mask - 1) * (92.0 / height_of_wind_speed) ** (1.0 / 7.0)

    # Calculate the speed at hub height
    ERA5_cube_hubheight = model_data * correction_hubheight

    return ERA5_cube_hubheight


# Define the main function to process the data
def main():
    # Start a timer to measure the execution time
    start_time = time.time()

    # Set up a path of the test file
    arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/"
    subset_arrs_dir = (
        "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/subset_WP/"
    )
    test_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/HadGEM3-GC31-MM_sfcWind_Europe_2018_DJF_day.npy"
    lats_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_Europe_2018_DJF_day_lats.npy"
    lons_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_Europe_2018_DJF_day_lons.npy"
    members_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_Europe_2018_DJF_day_members.npy"
    test_dps_file_path = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1961-r9i1p1f2/day/sfcWind/gn/files/d20200417/sfcWind_day_HadGEM3-GC31-MM_dcppA-hindcast_s1961-r9i1p1f2_gn_19720101-19720330.nc"

    onshore_pc_path = "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/power_onshore.csv"
    offshore_pc_path = "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/power_offshore.csv"
    path_to_farm_locations_ons = "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/United_Kingdomwindfarm_dist_ons_2021.nc"
    path_to_farm_locations_ofs = "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/United_Kingdomwindfarm_dist_ofs_2021.nc"
    eu_grid = {
        "lon1": -40,  # degrees east
        "lon2": 30,
        "lat1": 30,  # degrees north
        "lat2": 80,
    }

    # Hard code the country to UK
    COUNTRY = "United Kingdom"

    # Set up the test years
    test_years = np.arange(1960, 2018 + 1, 1)  # Full period now

    # Set up the specifications for loading the model data
    season = "DJF"
    time_freq = "day"
    len_winter_days = 5776 + 90
    winter_arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/obs/"

    # Set up the constant period
    constant_years = np.arange(1970, 2017 + 1, 1)  # 1970 to 2018 inclusive

    # Set up the constraints
    constraint_lon = (-6.25, 4.58)  # degrees east
    constraint_lat = (50.28, 59.72)  # degrees north

    # Load the test data
    test_data = np.load(test_file_path)
    lats = np.load(lats_file_path)
    lons = np.load(lons_file_path)
    members = np.load(members_file_path)

    # Load obs data for sfcWind
    # Do the same for wind speed
    obs_wind_arr, obs_wind_wmeans = load_obs_data(
        variable="sfcWind",
        region="Europe",
        season=season,
        time_freq=time_freq,
        winter_years=(1960, 2024),
        winter_dim_shape=len_winter_days,
        lat_shape=63,  # Europe region
        lon_shape=49,  # Europe region
        arrs_dir=winter_arrs_dir,
    )

    # Set up the valid dec years obs
    valid_dec_years_obs = np.arange(1960, 2024 + 1, 1)  # 1960 to 2024 inclusive

    # Print the shape of the obs wind array
    print(f"Shape of obs wind array: {obs_wind_arr.shape}")
    # Print the shape of the obs wind wmeans
    print(f"Shape of obs wind wmeans: {obs_wind_wmeans.shape}")

    # Print the min and max of the obs wind array
    print(f"Min obs wind speed: {np.min(obs_wind_arr)} m/s ")
    print(f"Max obs wind speed: {np.max(obs_wind_arr)} m/s ")
    print(f"Mean obs wind speed: {np.mean(obs_wind_arr)} m/s ")

    # # Calculate the obs_wind_clim
    # obs_wind_clim = np.mean(obs_wind_arr, axis=0)

    # # Print the shape of the obs wind climatology
    # print(f"Shape of obs wind climatology: {obs_wind_clim.shape}")

    # Set up the fname for the data anomalies plus obs
    current_date = time.strftime("%Y%m%d")
    fname_data_anoms_plus_obs = f"HadGEM3-GC31-MM_sfcWind_Europe_20250625_DJF_day_NO_drift_bc_anoms_1960-2018.npy"

    # Set up the path to the data anomalies plus obs
    path_data_anoms_plus_obs = os.path.join(
        arrs_dir,
        fname_data_anoms_plus_obs,
    )

    # Set up a fname for the data anomalies dt
    fname_data_anoms_plus_obs_dt = f"HadGEM3-GC31-MM_sfcWind_Europe_{current_date}_DJF_day_drift_bc_anoms_1960-2018_NO_DETREND.npy"

    # Set up the path to the data anomalies plus obs dt
    path_data_anoms_plus_obs_dt = os.path.join(
        arrs_dir,
        fname_data_anoms_plus_obs_dt,
    )

    # # If the path exists, then return
    if os.path.exists(path_data_anoms_plus_obs_dt):
        print(f"Path {path_data_anoms_plus_obs_dt} already exists. Exiting.")

        # Load the data
        data_anoms_plus_obs_dt = np.load(path_data_anoms_plus_obs_dt)

        print(f"Loaded data anomalies plus obs dt from {path_data_anoms_plus_obs_dt}")

        # Print the shape of the data anomalies plus obs dt
        print(f"Shape of data anomalies plus obs dt: {data_anoms_plus_obs_dt.shape}")

        # Print the mean, min and max of the data anomalies plus obs dt
        print(f"Mean of data anomalies plus obs dt: {np.mean(data_anoms_plus_obs_dt)}")
        print(f"Min of data anomalies plus obs dt: {np.min(data_anoms_plus_obs_dt)}")
        print(f"Max of data anomalies plus obs dt: {np.max(data_anoms_plus_obs_dt)}")

        # if the subset_arrs_dir does not exist, create it
        if not os.path.exists(subset_arrs_dir):
            os.makedirs(subset_arrs_dir)
            print(f"Created directory {subset_arrs_dir}")

        # Loop over the years and save the data anomalies plus obs dt
        for year_idx in tqdm(range(data_anoms_plus_obs_dt.shape[0])):
            year_this = test_years[year_idx]

            # Define the filename for the current year
            fname_this = f"HadGEM3-GC31-MM_sfcWind_Europe_{year_this}_DJF_day_drift_bc_anoms_1960-2018_dt.npy"

            # Set up the path to the file
            path_this = os.path.join(subset_arrs_dir, fname_this)

            # If the path does not exist, save the data anomalies plus obs dt
            if not os.path.exists(path_this):
                # Save the data anomalies plus obs dt for this year
                np.save(path_this, data_anoms_plus_obs_dt[year_idx, :, :, :, :, :])
                print(
                    f"Saved data anomalies plus obs dt for year {year_this} to {path_this}"
                )
            else:
                print(
                    f"Path {path_this} already exists. Skipping saving for year {year_this}"
                )

        sys.exit()

    #     sys.exit()

    # If the path exists, save the data anomalies plus obs
    if os.path.exists(path_data_anoms_plus_obs):
        print(f"Path {path_data_anoms_plus_obs} already exists. Loading data...")
        
        # Load the data
        data_anoms_plus_obs = np.load(path_data_anoms_plus_obs)

        # Print the shape of the data anomalies plus obs
        print(f"Shape of data anomalies plus obs: {data_anoms_plus_obs.shape}")

        # Directory to save the split arrays
        output_dir = "/home/users/benhutch/unseen_multi_year/split_data_anomalies"
        os.makedirs(output_dir, exist_ok=True)

        # Assuming `data_anoms_plus_obs_dt` is the array with shape (59, 10, 91, 11, 63, 49)
        for year_idx in tqdm(range(data_anoms_plus_obs.shape[0])):
            # Extract the array for the current year
            year_data = data_anoms_plus_obs[year_idx]

            # Define the filename for the current year
            fname = os.path.join(
                output_dir, f"data_anoms_plus_obs_year_{year_idx + 1}_no_dt.npy"
            )

            # Save the array to the file
            np.save(fname, year_data)

            print(f"Saved year {year_idx + 1} data to {fname}")

        # Print the amount of time taken
        end_time = time.time()

        print(f"Execution time: {end_time - start_time:.2f} seconds")

        # Exit the script
        sys.exit()

        print(f"Loaded data anomalies plus obs from {path_data_anoms_plus_obs}")
    else:
        print(f"Path {path_data_anoms_plus_obs} does not exist. Processing data...")
        # Set up the winter years for subsetting the data
        winter_years = np.arange(1, 11 + 1, 1)

        # Set up an empty list to store the indices
        winter_indices = []

        # Loop over the years
        for winter_year in winter_years:
            # Set up the indices
            indices_this = np.arange(
                30 + ((winter_year - 1) * 360), # i: 30 -> lead: 31
                30 + 90 + ((winter_year - 1) * 360), # i: 120 - 1 -> lead: 120
                1,
            )

            # print the winter year and the first lead this
            print(winter_year, indices_this[0], indices_this[-1])

            # Append the indices to the list
            winter_indices.extend(indices_this)

        # Set up the test array
        test_array = np.zeros(
            (
                len(test_years),
                test_data.shape[1],
                len(winter_indices),
                test_data.shape[3],
                test_data.shape[4],
            )
        )

        # Loop over the test years and print them
        for year in test_years:
            # Foprm the year as a string
            year_str = str(year)

            # Set up the fname
            fname_this = f"HadGEM3-GC31-MM_sfcWind_Europe_{year_str}_DJF_day.npy"

            # Set up the path to the file
            path_this = os.path.join(arrs_dir, fname_this)

            # If the file exists, load it
            if os.path.exists(path_this):
                # Load the data
                test_array_this = np.load(path_this)

                # Subset the data for the winter indices
                test_array_this_subset = test_array_this[:, :, winter_indices, :, :]

                # Store the data in the test array
                test_array[year - test_years[0], :, :, :, :] = test_array_this_subset
            else:
                raise FileNotFoundError(
                    f"File {path_this} does not exist. Please check the path and file name."
                )

        # Print the shape of the test array
        print(f"Shape of test array: {test_array.shape}")

        # # Print the values of the test array
        # print(f"Values of test array: {test_array}")

        # Reshape the test array to the desired shape
        reshaped_test_arr = test_array.reshape(
            len(test_years),
            len(members),
            len(winter_years),
            int(len(winter_indices) / len(winter_years)),
            test_array.shape[3],
            test_array.shape[4],
        )

        # Print the shape of the reshaped array
        print(f"Shape of reshaped array: {reshaped_test_arr.shape}")

        # Set up the shape of the ensemble mean by wyear to append to
        wyear_climatologies = np.zeros(
            (
                len(winter_years),
                len(lats),
                len(lons),
            )
        )

        # Loop over the lats
        for ilat, lat in tqdm(enumerate(lats)):
            for ilon, lon in enumerate(lons):
                for iwyear, wyear in enumerate(winter_years):
                    # Subset the test data for this lat, lon, and member
                    subset_data = reshaped_test_arr[:, :, iwyear, :, ilat, ilon]

                    # Set up the effective dec years this
                    effective_dec_years_this = test_years + (wyear - 1)

                    # Find the indices of the effective dec years this
                    # which are in the constant years
                    indices_this = np.where(
                        np.isin(effective_dec_years_this, constant_years)
                    )[0]

                    # Extract these indices from the subset data
                    subset_data_constant = subset_data[indices_this, :, :]

                    # Calculate the winter mean, ensemble mean, and climatology
                    winter_mean_this = np.mean(subset_data_constant)

                    # Store the winter mean in the climatology array
                    wyear_climatologies[iwyear, ilat, ilon] = winter_mean_this

        # Print the shape of the climatology array
        print(f"Shape of winter year climatologies: {wyear_climatologies.shape}")

        # Print the mean, min and max of the climatology array
        print(f"Mean of winter year climatologies: {np.mean(wyear_climatologies)}")
        print(f"Min of winter year climatologies: {np.min(wyear_climatologies)}")
        print(f"Max of winter year climatologies: {np.max(wyear_climatologies)}")

        # Loop over the winter years and print the climatology values
        for iwyear, wyear in enumerate(winter_years):
            print(
                f"Winter year {wyear} climatology mean: {np.mean(wyear_climatologies[iwyear, :, :])}"
            )
            print(
                f"Winter year {wyear} climatology min: {np.min(wyear_climatologies[iwyear, :, :])}"
            )
            print(
                f"Winter year {wyear} climatology max: {np.max(wyear_climatologies[iwyear, :, :])}"
            )

        # # Print the values of the climatology array
        # print(f"Values of winter year climatologies: {wyear_climatologies}")

        # Calculate the anomalies from the mean
        # Swap round the 2th and 3th dimensions
        test_data = np.swapaxes(reshaped_test_arr, 2, 3)

        # Print the shape of the test data
        print(f"Shape of test data: {test_data.shape}")

        # print the mean, min amd max of the test data
        print(f"Mean of test data: {np.mean(test_data)}")
        print(f"Min of test data: {np.min(test_data)}")
        print(f"Max of test data: {np.max(test_data)}")

        # Now remove the climatologies from the test data
        # data_anoms = (
        #     test_data - wyear_climatologies[np.newaxis, np.newaxis, np.newaxis, :, :, :]
        # )
        data_anoms = test_data

        # Print the shape of the data anomalies
        print(f"Shape of data anomalies: {data_anoms.shape}")

        # Print the mean, min and max of the data anomalies
        print(f"Mean of data anomalies: {np.mean(data_anoms)}")
        print(f"Min of data anomalies: {np.min(data_anoms)}")
        print(f"Max of data anomalies: {np.max(data_anoms)}")

        # Set up a new array to store the anomalies
        data_anoms_plus_obs = np.zeros_like(data_anoms, dtype=np.float32)

        # Lopo over the winter years
        for iwyear, wyear in tqdm(enumerate(winter_years)):
            # Subset the anomalies for this winter year
            data_anoms_this = data_anoms[:, :, :, iwyear, :, :]

            # Print the shape of the data anomalies for this winter year
            # print(
            #     f"Shape of data anomalies for winter year {wyear}: {data_anoms_this.shape}"
            # )

            # # Set up the effective dec years this
            # effective_dec_years_this = test_years + (wyear - 1)

            # # Find the common years between the effective dec years and the obs years
            # common_years = np.intersect1d(effective_dec_years_this, valid_dec_years_obs)

            # # Find the indices of the common years in the effective dec years
            # indices_this_model = np.where(
            #     np.isin(effective_dec_years_this, common_years)
            # )[0]

            # # Find the indices of the common years in the obs years
            # indices_this_obs = np.where(np.isin(valid_dec_years_obs, common_years))[0]

            # # # subset the data anomalies to these indices
            # # data_anoms_this = data_anoms_this[indices_this_model, :, :, :, :]

            # # Subset the obs wind data wmeans to these indices
            # obs_wmeans_this = obs_wind_wmeans[indices_this_obs, :, :]

            # # Quantify the time mean of obs wind wmeans
            # obs_wmeans_this = np.mean(obs_wmeans_this, axis=0)

            # # Add the obs wind wmeans to the data anomalies
            # data_anoms_plus_obs_this = (
            #     data_anoms_this
            #     + obs_wmeans_this[np.newaxis, np.newaxis, np.newaxis, :, :]
            # )

            # # Print the shape of the data anomalies plus obs for this winter year
            # print(
            #     f"Shape of data anomalies plus obs for winter year {wyear}: {data_anoms_plus_obs_this.shape}"
            # )
            # # Print the shape of obs wmeans this
            # print(f"Shape of obs wmeans this: {obs_wmeans_this.shape}")
            # # Print the shape of data anoms this
            # print(f"Shape of data anoms this: {data_anoms_this.shape}")

            # # print the shape of data_anoms_plus_obs
            # print(
            #     f"Shape of data_anoms_plus_obs_this: {data_anoms_plus_obs_this.shape}"
            # )
            # # print the shape of data_anoms_plus_obs
            # print(f"Shape of data_anoms_plus_obs: {data_anoms_plus_obs.shape}")

            # Store the data anomalies plus obs in the new array
            # data_anoms_plus_obs[:, :, :, iwyear, :, :] = data_anoms_plus_obs_this

            # Second option for no bias correction
            data_anoms_plus_obs[:, :, :, iwyear, :, :] = data_anoms_this

        # Save the data anomalies plus obs to the path
        np.save(path_data_anoms_plus_obs, data_anoms_plus_obs)
        print(f"Saved data anomalies plus obs to {path_data_anoms_plus_obs}")

    # Print the shape of the data anomalies plus obs
    print(f"Shape of data anomalies plus obs: {data_anoms_plus_obs.shape}")
    # Print the mean, min and max of the data anomalies plus obs
    print(f"Mean of data anomalies plus obs: {np.mean(data_anoms_plus_obs)}")
    print(f"Min of data anomalies plus obs: {np.min(data_anoms_plus_obs)}")
    print(f"Max of data anomalies plus obs: {np.max(data_anoms_plus_obs)}")

    # Print the numbver of values in the data anomalies plus obs
    num_values = np.prod(data_anoms_plus_obs.shape)
    print(f"Number of values in data anomalies plus obs: {num_values}")

    # Count the number of values below 0 in the data anomalies plus obs
    num_below_zero = np.sum(data_anoms_plus_obs < 0)
    print(f"Number of values below 0 in data anomalies plus obs: {num_below_zero}")

    # -----------------------------------
    # FIXME: Now just save without detrending in this case
    # -----------------------------------

    # Directory to save the split arrays
    output_dir = "/home/users/benhutch/unseen_multi_year/split_data_anomalies"
    os.makedirs(output_dir, exist_ok=True)

    # Assuming `data_anoms_plus_obs_dt` is the array with shape (59, 10, 91, 11, 63, 49)
    for year_idx in range(data_anoms_plus_obs.shape[0]):
        # Extract the array for the current year
        year_data = data_anoms_plus_obs[year_idx]

        # Define the filename for the current year
        fname = os.path.join(
            output_dir, f"data_anoms_plus_obs_year_{year_idx + 1}_no_dt_no_drift_bc.npy"
        )

        # Save the array to the file
        np.save(fname, year_data)

        print(f"Saved year {year_idx + 1} data to {fname}")

    # End the timer and print the execution time
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")

    sys.exit()


    # -------------------------------
    # Now perform pivot detrending on the model array
    # Shape: (59, 10, 91, 11, 63, 49)
    # -------------------------------

    winter_years = np.arange(1, 11 + 1, 1)  # 1 to 11 inclusive

    # Bin these by effective dec year
    model_eff_dec_years = np.arange(
        1960, 2018 + len(winter_years) + 1, 1
    )  # 1960 to 2018 + len(winter_years) inclusive

    # Print the min and max of the effective dec years
    print(f"Min effective dec year: {np.min(model_eff_dec_years)}")
    print(f"Max effective dec year: {np.max(model_eff_dec_years)}")

    # Set up a dictionary with keys as the model effective dec years and values as empty arrays
    model_eff_dec_years_dict = {}

    for eff_dec_year in model_eff_dec_years:
        # Initialize the dictionary with empty arrays
        model_eff_dec_years_dict[eff_dec_year] = []

    # Loop over the winter years
    for iwyear, wyear in tqdm(enumerate(winter_years)):
        # For each effective dec year, store the data in the dictionary
        for i_init_year, init_year in enumerate(test_years):
            # Set up the effective dec year this
            effective_dec_year_this_val = init_year + (wyear - 1)

            # Print the effective dec year this
            print(f"Effective dec year this: {effective_dec_year_this_val}")

            # Print trhe init year and the winter year
            print(f"Init year: {init_year}, Winter year: {wyear}")

            # Subset the data anomalies plus obs for this init year and winter year
            data_anoms_plus_obs_this = data_anoms_plus_obs[
                i_init_year, :, :, iwyear, :, :
            ]

            # Print the shape of the data anomalies plus obs this
            print(
                f"Shape of data anomalies plus obs this: {data_anoms_plus_obs_this.shape}"
            )

            # Store the data in the dictionary
            model_eff_dec_years_dict[effective_dec_year_this_val].append(
                data_anoms_plus_obs_this
            )

    # Initialize a dictionary to store the mean values for each year
    mean_values_by_year = {}

    # Iterate through the dictionary
    for year, arrays in model_eff_dec_years_dict.items():
        if arrays:  # Check if there are arrays for this year
            # Stack the arrays along a new axis and compute the mean
            mean_values_by_year[year] = np.mean(np.stack(arrays, axis=0), axis=0)
        else:
            # Handle the case where there are no arrays for the year
            mean_values_by_year[year] = (
                None  # Or np.zeros((10, 91, 63, 49)) if you prefer
            )

    # Example: Print the shape of the mean values for each year
    for year, mean_array in mean_values_by_year.items():
        if mean_array is not None:
            print(f"Year: {year}, Mean shape: {mean_array.shape}")
        else:
            print(f"Year: {year}, No data available")

    valid_years = np.arange(1960, 2028 + 1, 1)  # 1960 to 2018 inclusive

    # Set up an array to store the data for the valid years
    data_anoms_plus_obs_valid_years = np.zeros(
        (
            len(valid_years),
            len(members),
            91,
            len(lats),
            len(lons),
        )
    )

    # Loop over trhe dictionary and store the data in the array
    for i_year, year in tqdm(enumerate(valid_years)):
        # Check if the year is in the dictionary
        if year in mean_values_by_year:
            # Store the data in the array
            data_anoms_plus_obs_valid_years[i_year, :, :, :, :] = mean_values_by_year[
                year
            ]
        else:
            print(f"Year {year} not found in the dictionary. Skipping.")

    # Print the shape of the data anomalies plus obs valid years
    print(
        f"Shape of data anomalies plus obs valid years: {data_anoms_plus_obs_valid_years.shape}"
    )

    # Print the mean, min and max of the data anomalies plus obs valid years
    print(
        f"Mean of data anomalies plus obs valid years: {np.mean(data_anoms_plus_obs_valid_years)}"
    )
    print(
        f"Min of data anomalies plus obs valid years: {np.min(data_anoms_plus_obs_valid_years)}"
    )
    print(
        f"Max of data anomalies plus obs valid years: {np.max(data_anoms_plus_obs_valid_years)}"
    )

    # Find the indices of valid years which is the same as the observed years
    valid_years_indices = np.where(np.isin(valid_years, valid_dec_years_obs))[0]

    # Subset the data anomalies plus obs valid years to these indices
    data_anoms_plus_obs_valid_years = data_anoms_plus_obs_valid_years[
        valid_years_indices, :, :, :, :
    ]
    valid_years_model_subset = valid_years[valid_years_indices]

    # Print the shape of the data anomalies plus obs valid years after subsetting
    print(
        f"Shape of data anomalies plus obs valid years after subsetting: {data_anoms_plus_obs_valid_years.shape}"
    )

    # Create a np zeros like for data_anoms_plus_obs_dt
    data_anoms_plus_obs_dt = np.zeros_like(data_anoms_plus_obs, dtype=np.float32)

    # (69, 10, 91, 63, 49)
    # Loop over the lats and lons
    for ilat, lat in tqdm(enumerate(lats)):
        for ilon, lon in enumerate(lons):
            # Set up the data for this lat and lon
            data_this_lat_lon = data_anoms_plus_obs_valid_years[:, :, :, ilat, ilon]

            # Extract the values for data_anoms_plus_obs
            data_this_lat_lon_for_detrend = data_anoms_plus_obs[:, :, :, :, ilat, ilon]

            # Subset the observed wind data for this lat and lon
            obs_wind_this_lat_lon = obs_wind_wmeans[:, ilat, ilon]

            # Take the mean over dimensions -1, -2
            data_this_lat_lon_mean_ts = np.mean(data_this_lat_lon, axis=(-1, -2))

            model_slope_this, model_intercept_this, _, _, _ = linregress(
                valid_years_model_subset, data_this_lat_lon_mean_ts
            )

            obs_slope_this, obs_intercept_this, _, _, _ = linregress(
                valid_dec_years_obs, obs_wind_this_lat_lon
            )

            # Calculate the trend line
            model_trend_this = (
                model_slope_this * valid_years_model_subset + model_intercept_this
            )
            obs_trend_this = obs_slope_this * valid_dec_years_obs + obs_intercept_this

            # Determine the final point for the trend line for the model
            final_point_model = model_trend_this[-1]
            # Determine the final point for the trend line for the obs
            final_point_obs = obs_trend_this[-1]

            # Calculate the model bias
            model_bias = final_point_model - final_point_obs

            # Add the model bias to the data for this lat and lon
            final_point_model_bc = final_point_model - model_bias

            # # Print the shape of the trend
            # print(f"Shape of data this lat lon: {data_this_lat_lon.shape}")
            # print(f"Shape of data this lat lon for detrend: {data_this_lat_lon_for_detrend.shape}")
            # print(f"Shape of obs wind this lat lon: {obs_wind_this_lat_lon.shape}")

            # # Print the shape of the trend lines
            # print(f"Shape of model trend this: {model_trend_this.shape}")
            # print(f"Shape of obs trend this: {obs_trend_this.shape}")

            # # Print the final points for the trend lines
            # print(f"Final point model: {final_point_model}")
            # print(f"Final point obs: {final_point_obs}")
            # print(f"Final point model bias corrected: {final_point_model_bc}")

            # Loop over the winter years
            for iwyear, wyear in tqdm(enumerate(winter_years)):
                for i_init_year, init_year in enumerate(test_years):
                    # Set up the effective dec year this
                    effective_dec_year_this_val = init_year + (wyear - 1)

                    # Subset the data_this_lat_lon_for_detrend
                    data_this_lat_lon_for_detrend_this = data_this_lat_lon_for_detrend[
                        i_init_year, :, :, iwyear
                    ]

                    # Find the trend value at the effective dec year this
                    trend_value_this = (
                        model_slope_this * effective_dec_year_this_val
                        + model_intercept_this
                    )

                    # # Print the trend value for this effective dec year
                    # print(
                    #     f"Trend value for effective dec year {effective_dec_year_this_val}: {trend_value_this}"
                    # )

                    # # Print the shape of the data for this lat and lon
                    # print(f"Shape of data this lat lon for detrend this: {data_this_lat_lon_for_detrend_this.shape}")

                    # # Print the values of this
                    # print(f"Values of data this lat lon for detrend this: {data_this_lat_lon_for_detrend_this}")
                    # print(f"Mean of data this lat lon for detrend this: {np.mean(data_this_lat_lon_for_detrend_this)}")
                    # print(f"Min of data this lat lon for detrend this: {np.min(data_this_lat_lon_for_detrend_this)}")
                    # print(f"Max of data this lat lon for detrend this: {np.max(data_this_lat_lon_for_detrend_this)}")

                    # # print the final point model BC
                    # print(f"Final point model BC: {final_point_model_bc}")

                    # Remove the trend value from the data for this lat and lon
                    data_this_lat_lon_dt = (
                        final_point_model_bc
                        - trend_value_this
                        + data_this_lat_lon_for_detrend_this
                    )

                    # # Print the data this lat lon dt
                    # print(f"Data this lat lon dt post trend corr: {data_this_lat_lon_dt}")
                    # print(f"mean of data this lat lon dt post trend corr: {np.mean(data_this_lat_lon_dt)}")
                    # print(f"min of data this lat lon dt post trend corr: {np.min(data_this_lat_lon_dt)}")
                    # print(f"max of data this lat lon dt post trend corr: {np.max(data_this_lat_lon_dt)}")

                    # Store the data in the data_anoms_plus_obs_dt array
                    data_anoms_plus_obs_dt[i_init_year, :, :, iwyear, ilat, ilon] = (
                        data_this_lat_lon_dt
                    )

    # Print the shape of the data anomalies plus obs dt
    print(f"Shape of data anomalies plus obs dt: {data_anoms_plus_obs_dt.shape}")
    # Print the mean, min and max of the data anomalies plus obs dt
    print(f"Mean of data anomalies plus obs dt: {np.mean(data_anoms_plus_obs_dt)}")
    print(f"Min of data anomalies plus obs dt: {np.min(data_anoms_plus_obs_dt)}")
    print(f"Max of data anomalies plus obs dt: {np.max(data_anoms_plus_obs_dt)}")

    # Save the data anomalies plus obs dt to the path
    np.save(path_data_anoms_plus_obs_dt, data_anoms_plus_obs_dt)

    # Directory to save the split arrays
    output_dir = "/home/users/benhutch/unseen_multi_year/split_data_anomalies"
    os.makedirs(output_dir, exist_ok=True)

    # Assuming `data_anoms_plus_obs_dt` is the array with shape (59, 10, 91, 11, 63, 49)
    for year_idx in range(data_anoms_plus_obs_dt.shape[0]):
        # Extract the array for the current year
        year_data = data_anoms_plus_obs_dt[year_idx]

        # Define the filename for the current year
        fname = os.path.join(
            output_dir, f"data_anoms_plus_obs_dt_year_{year_idx + 1}.npy"
        )

        # Save the array to the file
        np.save(fname, year_data)

        print(f"Saved year {year_idx + 1} data to {fname}")

    # End the timer and print the execution time
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")

    sys.exit()

    # print the shape of the test data
    print(f"Shape of test data: {test_data.shape}")

    # print the values of the test data
    print(f"Values of test data: {test_data}")

    # Extract and plot the first member's data
    first_member_data = test_data[0, 0, 0, :, :]

    # fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': ccrs.PlateCarree()})

    # ax.pcolormesh(lons, lats, first_member_data, cmap='viridis', transform=ccrs.PlateCarree())
    # ax.set_title('First Member Data')
    # ax.coastlines()
    # plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', label='Wind Speed (m/s)')
    # plt.show()

    # print the min and max lat lon
    print(f"Min latitude: {np.min(lats)}, Max latitude: {np.max(lats)}")
    print(f"Min longitude: {np.min(lons)}, Max longitude: {np.max(lons)}")

    # Load the power curves
    pc_winds, pc_power_ons, pc_power_ofs = load_power_curves(
        path_onshore_curve=onshore_pc_path,
        path_offshore_curve=offshore_pc_path,
    )

    # Make a country mask for the UK
    countries_shp = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )

    # Set up the land shapereader
    # Initialize the mask with the correct shape
    MASK_MATRIX_TMP = np.zeros((len(lats), len(lons)))
    country_shapely = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME"][0:14] == COUNTRY:
            # Convert the shapefile geometry to shapely geometry
            country_shapely.append(shapely.geometry.shape(country.geometry))

    # Loop over the latitude and longitude points
    for l in range(len(lats)):
        for j in range(len(lons)):
            point = shapely.geometry.Point(lons[j], lats[l])
            for country in country_shapely:
                if country.contains(point):
                    MASK_MATRIX_TMP[l, j] = 1.0

    # Reshape the mask to match the shape of the data
    MASK_MATRIX_RESHAPE = MASK_MATRIX_TMP

    # # plot the mask matrix reshape
    # fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection':ccrs.PlateCarree()})

    # ax.pcolormesh(lons, lats, MASK_MATRIX_RESHAPE, cmap='viridis', transform=ccrs.PlateCarree())
    # ax.set_title('Country Mask for UK')
    # ax.coastlines()
    # plt.colorbar(ax.collections[0], ax=ax, orientation='vertical', label='Mask Value')

    # Process the area weighted means for the wind farm locations
    rg_farm_locations = process_area_weighted_mean(
        path_to_farm_locations_ons=path_to_farm_locations_ons,
        path_to_farm_locations_ofs=path_to_farm_locations_ofs,
        cube=iris.load_cube(test_dps_file_path),
    )

    # Print the shape of the rg_farm_locations
    print(f"Shape of rg_farm_locations: {rg_farm_locations.shape}")
    # Print the type of the rg_farm_locations
    print(f"Type of rg_farm_locations: {type(rg_farm_locations)}")

    # Limit the cube to the constrain region
    rg_farm_locations = rg_farm_locations.intersection(
        longitude=constraint_lon,
        latitude=constraint_lat,
    )

    # Load the wind speed data and take it to hub height
    ws_hh = load_wind_speed_and_take_to_hubheight_model(
        model_data=test_data,
        land_mask=MASK_MATRIX_RESHAPE,
        height_of_wind_speed=10.0,  # Assuming the wind speed is at 10m
    )

    # print the shape of the ws_hh
    print(f"Shape of ws_hh: {ws_hh.shape}")
    # print the min and max of the ws_hh
    print(f"Min wind speed at hub height: {np.min(ws_hh)} m/s")
    print(f"Max wind speed at hub height: {np.max(ws_hh)} m/s")
    # print the mean and std of the ws_hh
    print(f"Mean wind speed at hub height: {np.mean(ws_hh)} m/s")
    print(f"Std wind speed at hub height: {np.std(ws_hh)} m/s")

    # reshpe to the test data shape
    test_ws_hh = ws_hh[0, :, :, :, :]

    # Convert wind speed to power generation using the power curves
    p_hh_total_GW = convert_wind_speed_to_power_generation(
        ERA5_cube_hubheight=test_ws_hh,
        pc_winds=pc_winds,
        pc_power_ons=pc_power_ons,
        pc_power_ofs=pc_power_ofs,
        land_mask=MASK_MATRIX_RESHAPE,
        farm_locations=rg_farm_locations.data,
    )

    # Print the shape of the power generation data
    print(f"Shape of power generation data: {p_hh_total_GW.shape}")
    print(f"Type of power generation data: {type(p_hh_total_GW)}")

    # extract the data
    p_hh_total_GW_vals = np.array(p_hh_total_GW.data)

    # Print the shape of the power generation data values
    print(f"Shape of power generation data values: {p_hh_total_GW_vals.shape}")
    # Print the type of the power generation data values
    print(f"Type of power generation data values: {type(p_hh_total_GW_vals)}")
    # print the values
    print(f"Values of power generation data: {p_hh_total_GW_vals}")

    # print the min and max of the power generation data
    print(f"Min power generation: {np.min(p_hh_total_GW_vals)} GW")
    print(f"Max power generation: {np.max(p_hh_total_GW_vals)} GW")
    # print the mean and std of the power generation data
    print(f"Mean power generation: {np.mean(p_hh_total_GW_vals)} GW")
    print(f"Std power generation: {np.std(p_hh_total_GW_vals)} GW")

    # Print the min and max of the power generation data
    print(f"Min power generation: {np.min(p_hh_total_GW_vals)} GW")
    print(f"Max power generation: {np.max(p_hh_total_GW_vals)} GW")
    # Print the mean and std of the power generation data
    print(f"Mean power generation: {np.mean(p_hh_total_GW_vals)} GW")
    print(f"Std power generation: {np.std(p_hh_total_GW_vals)} GW")

    # # Plot the rg_farm_locations
    # fig, ax = plt.subplots(
    #     figsize=(10, 10), subplot_kw={"projection": ccrs.PlateCarree()}
    # )
    # ax.pcolormesh(
    #     rg_farm_locations.coord("longitude").points,
    #     rg_farm_locations.coord("latitude").points,
    #     rg_farm_locations.data,
    #     cmap="viridis",
    #     transform=ccrs.PlateCarree(),
    # )
    # ax.set_title("RG Farm Locations")
    # ax.coastlines()
    # plt.colorbar(
    #     ax.collections[0], ax=ax, orientation="vertical", label="RG Farm Locations"
    # )
    # plt.show()

    total_gen_MW = p_hh_total_GW / 1000.0  # Convert from GW to MW

    # Get the capacity factor
    capacity_factor = total_gen_MW / (np.sum(rg_farm_locations.data) / 1000000.0)

    # Flatten the capacity factor array
    capacity_factor_flat = capacity_factor.flatten()

    # Print the mean, max, and min of the capacity factor
    print(f"Mean capacity factor: {np.mean(capacity_factor_flat)}")
    print(f"Max capacity factor: {np.max(capacity_factor_flat)}")
    print(f"Min capacity factor: {np.min(capacity_factor_flat)}")

    # Print the 0.25, 0.5, and 0.75 quantiles of the capacity factor
    print(f"0.25 quantile: {np.quantile(capacity_factor_flat, 0.25)}")
    print(f"0.5 quantile: {np.quantile(capacity_factor_flat, 0.5)}")
    print(f"0.75 quantile: {np.quantile(capacity_factor_flat, 0.75)}")

    return None


if __name__ == "__main__":
    main()
# %%
