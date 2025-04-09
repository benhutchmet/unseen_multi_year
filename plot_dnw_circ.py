"""
plot_dnw_circ.py
=================

This script processes the obs and model data and plots the composites.

"""

# %%
# Local imports
import os
import sys
import glob
import time

# Third-party imports
import iris
import cftime
import shapely.geometry
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

# Specific imports
from tqdm import tqdm
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta


# Set up a function for loading the data
def load_obs_data(
    variable: str,
    region: str,
    season: str,
    time_freq: str,
    winter_years: tuple,
    winter_dim_shape: int,
    lat_shape: int,
    lon_shape: int,
    arrs_dir: str,
) -> np.ndarray:
    """
    Loads the observed data array to be used for the composites.

    Args:
    ======
        variable (str): The variable to load.
        region (str): The region to load.
        season (str): The season to load.
        time_freq (str): The time frequency to load.
        winter_years (tuple): The years to load.
        winter_dim_shape (int): The shape of the winter dimension.
        lat_shape (int): The shape of the latitude dimension.
        lon_shape (int): The shape of the longitude dimension.
        arrs_dir (str): The directory to load the data from.

    Returns:
    ========
        np.ndarray: The loaded data array
                    Has shape: (winter_dim_shape, lat_shape, lon_shape)
    """
    # Set up the years array
    years_arr = np.arange(winter_years[0], winter_years[1] + 1, 1)

    # Set up the first dim ticker
    first_dim_ticker = 0

    # Set up the data array to append to
    data_arr = np.zeros((winter_dim_shape, lat_shape, lon_shape))

    # Loop through the years
    for year in tqdm(years_arr, desc="Loading data"):
        # Set up the filename
        fname_this = f"ERA5_{variable}_{region}_{year}_{season}_{time_freq}.npy"

        # if the path does not exist then raise an error
        if not os.path.exists(os.path.join(arrs_dir, fname_this)):
            raise FileNotFoundError(f"File {fname_this} does not exist.")

        # Load the data
        data_this = np.load(os.path.join(arrs_dir, fname_this))

        # Append the arr to the all arr
        if data_this.size != 0:
            data_arr[first_dim_ticker : first_dim_ticker + data_this.shape[0], :, :] = (
                data_this
            )
            first_dim_ticker += data_this.shape[0]
        else:
            raise ValueError(f"Data array is empty for {fname_this}")

    return data_arr


# Write a function which extracts the data from the obs df
# for the correct dates
def extract_obs_data(
    obs_df: pd.DataFrame,
    variable: str,
    region: str,
    time_freq: str,
    season: str,
    lat_shape: int,
    lon_shape: int,
    arrs_dir: str,
    metadata_dir: str,
) -> np.ndarray:
    """
    Extracts the obs data for specific times and appends to an array.

    Args:
    ======
        obs_df (pd.DataFrame): The obs dataframe.
        variable (str): The variable to load.
        region (str): The region to load.
        time_freq (str): The time frequency to load.
        season (str): The season to load.
        lat_shape (int): The shape of the latitude dimension.
        lon_shape (int): The shape of the longitude dimension.
        arrs_dir (str): The directory to load the data from.
        metadata_dir (str): The directory to load the metadata from.

    Returns:
    ========
        np.ndarray: The loaded data array
                    Has shape: (len(obs_df), lat_shape, lon_shape)

    """

    # Psl calendar
    # time:calendar = "gregorian" ;
    # time:axis = "T" ;
    # time:units = "hours since 1900-01-01" ;

    # tas calendar
    # time:units = "days since 1950-01-01 00:00:00" ;
    # time:calendar = "proleptic_gregorian" ;

    # sfcWind calendar
    # time:units = "days since 1952-01-01 00:00:00" ;
    # time:calendar = "proleptic_gregorian" ;

    # if the variable is psl then set the calendar to "gregorian"
    if variable == "psl":
        calendar = "gregorian"
        units = "hours since 1900-01-01"
    # if the variable is tas or sfcWind then set the calendar to "proleptic_gregorian"
    elif variable == "tas":
        calendar = "proleptic_gregorian"
        units = "days since 1950-01-01 00:00:00"
    elif variable == "sfcWind":
        calendar = "proleptic_gregorian"
        units = "days since 1952-01-01 00:00:00"
    else:
        raise ValueError(
            f"Variable {variable} not recognised. Must be psl, tas or sfcWind."
        )

    # Set up the array to append to
    data_arr = np.zeros((len(obs_df), lat_shape, lon_shape))

    # extract the list of dates from the obs df
    dates = obs_df["time"].values

    # Format these using datetime
    dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

    # Format these using cftime
    dates = [cftime.datetime(date.year, date.month, date.day) for date in dates]

    # print the dates
    print("Dates: ", dates)

    # Set up an empty dates list
    dates_list = []

    # Loop through the dates
    for i, date in tqdm(enumerate(dates), desc="Loading data"):
        # Extract the year as an int
        year_this = int(date.year)
        month_this = int(date.month)
        day_this = int(date.day)

        # if variable is psl then set the month to 1
        if variable == "psl":
            # format the date to extract
            time_to_extract_this = cftime.DatetimeGregorian(
                year_this, month_this, day_this, hour=11, calendar=calendar
            )
        # if variable is tas or sfcWind then set the month to 1
        elif variable == "tas" or variable == "sfcWind":
            # format the date to extract
            time_to_extract_this = cftime.DatetimeProlepticGregorian(
                year_this, month_this, day_this, hour=0, calendar=calendar
            )
        else:
            raise ValueError(
                f"Variable {variable} not recognised. Must be psl, tas or sfcWind."
            )

        # set up the year to extract
        if month_this == 12:
            year_to_extract_this = year_this
        else:
            year_to_extract_this = year_this - 1

        # Set up the filename
        fname_this = (
            f"ERA5_{variable}_{region}_{year_to_extract_this}_{season}_{time_freq}.npy"
        )

        # If the file does not exist then raise an error
        if not os.path.exists(os.path.join(arrs_dir, fname_this)):
            raise FileNotFoundError(f"File {fname_this} does not exist.")

        # Set up the fname for the times
        times_fname = f"ERA5_{variable}_{region}_{year_to_extract_this}_{season}_{time_freq}_times.npy"

        # If the file does not exist then raise an error
        if not os.path.exists(os.path.join(metadata_dir, times_fname)):
            raise FileNotFoundError(
                f"File {os.path.join(metadata_dir, times_fname)} does not exist."
            )

        # load the data for this
        data_this = np.load(os.path.join(arrs_dir, fname_this))

        # load the times for this
        times_this = np.load(os.path.join(metadata_dir, times_fname))

        # convert the times to cftime
        times_this_cf = cftime.num2date(
            times_this,
            units=units,
            calendar=calendar,
        )

        # find the index of this time in the tyimes_this_cf
        time_index = np.where(times_this_cf == time_to_extract_this)[0][0]

        # extract the data for this time
        data_this_time = data_this[time_index, :, :]

        # extract the data for this time and append
        data_arr[i, :, :] = data_this_time

        # append the date extracted to the list
        dates_list.append(time_to_extract_this)

    return data_arr, dates_list

# Define a function to plot all of the data
def plot_data_postage_stamp(
    subset_arr: np.ndarray,
    clim_arr: np.ndarray,
    dates_list: List[cftime.DatetimeProlepticGregorian],
    variable: str,
    region: str,
    season: str,
    lats_path: str,
    lons_path: str,
):
    """
    Plots the data as a postage stamp plot.
    
    Args:
    ======

        subset_arr (np.ndarray): The data array to plot.
        clim_arr (np.ndarray): The climatology array to plot.
        dates_list (List[cftime.DatetimeProlepticGregorian]): The list of dates to plot.
        variable (str): The variable to plot.
        region (str): The region to plot.
        season (str): The season to plot.
        lats_path (str): The path to the latitude data.
        lons_path (str): The path to the longitude data.
        
    Returns:
    =========

        None
        
    """

    # Extract the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # if variable is psl
    if variable == "psl":
        cmap = "bwr"
        levels = np.array(
            [
                -20,
                -18,
                -16,
                -14,
                -12,
                -10,
                -8,
                -6,
                -4,
                -2,
                2,
                4,
                6,
                8,
                10,
                12,
                14,
                16,
                18,
                20,
            ]
        )
    elif variable == "tas":
        cmap = "bwr"
        levels = np.array(
            [
                -10,
                -8,
                -6,
                -4,
                -2,
                2,
                4,
                6,
                8,
                10,
            ]
        )

        # Set up the x and y
        x, y = lons, lats

        # Set up the countries shapefile
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
            country_shapely.append(country.geometry)

        # Loop over the latitude and longitude points
        for i in range(len(lats)):
            for j in range(len(lons)):
                point = shapely.geometry.Point(lons[j], lats[i])
                for country in country_shapely:
                    if country.contains(point):
                        MASK_MATRIX_TMP[i, j] = 1.0

        # Reshape the mask to match the shape of the data
        MASK_MATRIX_RESHAPED = MASK_MATRIX_TMP

        # print the shape of the mask
        print(f"Shape of the mask: {MASK_MATRIX_RESHAPED.shape}")

        # print teh sum of the mask
        print(f"Sum of the mask: {np.sum(MASK_MATRIX_RESHAPED)}")
    elif variable == "sfcWind":
        cmap = "PRGn"
        levels = np.array(
            [
                -10,
                -8,
                -6,
                -4,
                -2,
                0,
                2,
                4,
                6,
                8,
                10,
            ]
        )
    else:
        raise ValueError(
            f"Variable {variable} not recognised. Must be psl, tas or sfcWind."
        )

    # Set up a figure with 6 nrows and 10 ncols
    fig, axes = plt.subplots(
        nrows=6,
        ncols=10,
        figsize=(20, 12),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="compressed",
    )

    # Flatten the axes
    axes_flat = axes.flatten()

    anoms_mean_min = []
    anoms_mean_max = []

    # Loop over the axes
    for i, ax in enumerate(axes_flat):
        # if i is greater than the length of the dates list then break
        if i >= len(dates_list):
            break

        # Set up the title
        ax.set_title(f"{dates_list[i].strftime('%Y-%m-%d')}", fontsize=8)

        # Calculate the anoms this
        anoms_this = subset_arr[i, :, :] - clim_arr

        # # print the min and max of the anoms
        # print(f"Min of anoms this: {np.min(anoms_this)}")
        # print(f"Max of anoms this: {np.max(anoms_this)}")
        if variable == "psl":
            # Plot the data
            im = ax.contourf(
                lons,
                lats,
                anoms_this / 100,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels,
                extend="both",
            )
        elif variable == "tas":
            # Apply the mask to the temperature data
            anoms_this = np.ma.masked_where(MASK_MATRIX_RESHAPED == 0, anoms_this)

            # Plot the data
            im = ax.contourf(
                lons,
                lats,
                anoms_this,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels,
                extend="both",
            )
        else:
            # Plot the data
            im = ax.contourf(
                lons,
                lats,
                anoms_this,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels,
                extend="both",
            )

        # Add coastlines
        ax.coastlines()

        # Add gridlines
        ax.gridlines()

        # append the min and max of the anoms to the list
        anoms_mean_min.append(np.min(anoms_this))
        anoms_mean_max.append(np.max(anoms_this))

    # print the mean minima
    print(f"Mean minima: {np.mean(anoms_mean_min)}")
    # print the mean maxima
    print(f"Mean maxima: {np.mean(anoms_mean_max)}")

    # remove the plots
    for ax in axes_flat[len(dates_list) :]:
        fig.delaxes(ax)

    # Set up the colorbar
    cbar = fig.colorbar(im, ax=axes_flat, orientation="horizontal", pad=0.01)

    # Set the colorbar ticks
    cbar.set_ticks(levels)

    # Set up the colorbar label
    cbar.set_label(f"{variable} Anomalies", fontsize=12)

    return None

# Define the main function
def main():
    start_time = time.time()

    # Set up the hard coded variables
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"
    obs_df_fname = "block_maxima_obs_demand_net_wind.csv"
    winter_arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/obs/"
    metadata_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/"
    arrs_persist_dir = "/home/users/benhutch/unseen_multi_year/data"
    season = "DJF"
    time_freq = "day"
    len_winter_days = 5324

    # If the path esists, load in the obs df
    if os.path.exists(os.path.join(dfs_dir, obs_df_fname)):
        obs_df = pd.read_csv(os.path.join(dfs_dir, obs_df_fname), index_col=0)
    else:
        raise FileNotFoundError(f"File {obs_df_fname} does not exist in {dfs_dir}")

    # Load the psl data for the north atlantic region
    obs_psl_arr = load_obs_data(
        variable="psl",
        region="NA",
        season=season,
        time_freq=time_freq,
        winter_years=(1960, 2018),
        winter_dim_shape=len_winter_days,
        lat_shape=90,  # NA region
        lon_shape=96,  # NA region
        arrs_dir=winter_arrs_dir,
    )

    # Do the same for temperature
    obs_temp_arr = load_obs_data(
        variable="tas",
        region="Europe",
        season=season,
        time_freq=time_freq,
        winter_years=(1960, 2018),
        winter_dim_shape=len_winter_days,
        lat_shape=63,  # Europe region
        lon_shape=49,  # Europe region
        arrs_dir=winter_arrs_dir,
    )

    # Do the same for wind speed
    obs_wind_arr = load_obs_data(
        variable="sfcWind",
        region="Europe",
        season=season,
        time_freq=time_freq,
        winter_years=(1960, 2018),
        winter_dim_shape=len_winter_days,
        lat_shape=63,  # Europe region
        lon_shape=49,  # Europe region
        arrs_dir=winter_arrs_dir,
    )

    # Calculate the psl climatology
    obs_psl_clim = np.mean(obs_psl_arr, axis=0)
    obs_tas_clim = np.mean(obs_temp_arr, axis=0)
    obs_wind_clim = np.mean(obs_wind_arr, axis=0)

    # print the head of the dfs
    print("Head of the obs df:")
    print(obs_df.head())

    # print the tail of the dfs
    print("Tail of the obs df:")
    print(obs_df.tail())

    # Set up fnames for the psl data
    psl_fname = f"ERA5_psl_NA_1960-2018_{season}_{time_freq}.npy"
    psl_times_fname = f"ERA5_psl_NA_1960-2018_{season}_{time_freq}_times.npy"

    # set up fnames for the temperature data
    temp_fname = f"ERA5_tas_Europe_1960-2018_{season}_{time_freq}.npy"
    temp_times_fname = f"ERA5_tas_Europe_1960-2018_{season}_{time_freq}_times.npy"

    # set up fnames for the wind data
    wind_fname = f"ERA5_sfcWind_Europe_1960-2018_{season}_{time_freq}.npy"
    wind_times_fname = f"ERA5_sfcWind_Europe_1960-2018_{season}_{time_freq}_times.npy"

    # if the psl files do not exist then  create them
    if not os.path.exists(
        os.path.join(arrs_persist_dir, psl_fname)
    ) and not os.path.exists(os.path.join(arrs_persist_dir, psl_times_fname)):
        # Call the function to load the obs data
        psl_subset, psl_dates_list = extract_obs_data(
            obs_df=obs_df,
            variable="psl",
            region="NA",
            time_freq=time_freq,
            season=season,
            lat_shape=90,
            lon_shape=96,
            arrs_dir=winter_arrs_dir,
            metadata_dir=metadata_dir,
        )

        # Save the data to the arrs_persist_dir
        np.save(os.path.join(arrs_persist_dir, psl_fname), psl_subset)
        np.save(os.path.join(arrs_persist_dir, psl_times_fname), psl_dates_list)

    # if the temperature files do not exist then  create them
    if not os.path.exists(
        os.path.join(arrs_persist_dir, temp_fname)
    ) and not os.path.exists(os.path.join(arrs_persist_dir, temp_times_fname)):
        # Call the function to load the obs data
        temp_subset, temp_dates_list = extract_obs_data(
            obs_df=obs_df,
            variable="tas",
            region="Europe",
            time_freq=time_freq,
            season=season,
            lat_shape=63,
            lon_shape=49,
            arrs_dir=winter_arrs_dir,
            metadata_dir=metadata_dir,
        )

        # Save the data to the arrs_persist_dir
        np.save(os.path.join(arrs_persist_dir, temp_fname), temp_subset)
        np.save(os.path.join(arrs_persist_dir, temp_times_fname), temp_dates_list)

    # if the wind files do not exist then  create them
    if not os.path.exists(
        os.path.join(arrs_persist_dir, wind_fname)
    ) and not os.path.exists(os.path.join(arrs_persist_dir, wind_times_fname)):
        # Call the function to load the obs data
        wind_subset, wind_dates_list = extract_obs_data(
            obs_df=obs_df,
            variable="sfcWind",
            region="Europe",
            time_freq=time_freq,
            season=season,
            lat_shape=63,
            lon_shape=49,
            arrs_dir=winter_arrs_dir,
            metadata_dir=metadata_dir,
        )

        # Save the data to the arrs_persist_dir
        np.save(os.path.join(arrs_persist_dir, wind_fname), wind_subset)
        np.save(os.path.join(arrs_persist_dir, wind_times_fname), wind_dates_list)

    # load the psl data
    psl_subset = np.load(os.path.join(arrs_persist_dir, psl_fname))
    psl_dates_list = np.load(
        os.path.join(arrs_persist_dir, psl_times_fname), allow_pickle=True
    )

    # load the temperature data
    temp_subset = np.load(os.path.join(arrs_persist_dir, temp_fname))
    temp_dates_list = np.load(
        os.path.join(arrs_persist_dir, temp_times_fname), allow_pickle=True
    )

    # load the wind data
    wind_subset = np.load(os.path.join(arrs_persist_dir, wind_fname))
    wind_dates_list = np.load(
        os.path.join(arrs_persist_dir, wind_times_fname), allow_pickle=True
    )

    # plot all of the data for psl
    plot_data_postage_stamp(
        subset_arr=psl_subset,
        clim_arr=obs_psl_clim,
        dates_list=psl_dates_list,
        variable="psl",
        region="NA",
        season=season,
        lats_path=os.path.join(
            metadata_dir,
            "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
        ),
        lons_path=os.path.join(
            metadata_dir,
            "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
        ),
    )

    # Plot the postage stamps for tas
    plot_data_postage_stamp(
        subset_arr=temp_subset,
        clim_arr=obs_tas_clim,
        dates_list=temp_dates_list,
        variable="tas",
        region="Europe",
        season=season,
        lats_path=os.path.join(
            metadata_dir,
            "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
        ),
        lons_path=os.path.join(
            metadata_dir,
            "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
        ),
    )

    # Plot the postage stamps for wind speed
    plot_data_postage_stamp(
        subset_arr=wind_subset,
        clim_arr=obs_wind_clim,
        dates_list=wind_dates_list,
        variable="sfcWind",
        region="Europe",
        season=season,
        lats_path=os.path.join(
            metadata_dir,
            "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
        ),
        lons_path=os.path.join(
            metadata_dir,
            "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
        ),
    )

    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print("Done!")

    return None


if __name__ == "__main__":
    main()

# %%
