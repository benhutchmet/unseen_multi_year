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
import json

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
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime, timedelta
from scipy.stats import pearsonr, linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Import dictionaries
import dictionaries as dicts

# Local imports
import process_dnw_gev as pdg_funcs


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
    data_arr_full = np.zeros((winter_dim_shape, lat_shape, lon_shape))

    print(f"Data array shape: {data_arr_full.shape}")

    data_arr_wmeans = np.zeros((len(years_arr), lat_shape, lon_shape))

    # Loop through the years
    for year in tqdm(years_arr, desc="Loading data"):
        # Set up the filename
        fname_this = f"ERA5_{variable}_{region}_{year}_{season}_{time_freq}.npy"

        # if the path does not exist then raise an error
        if not os.path.exists(os.path.join(arrs_dir, fname_this)):
            raise FileNotFoundError(f"File {fname_this} does not exist.")

        # Load the data
        data_this = np.load(os.path.join(arrs_dir, fname_this))

        # # Print the shape of data this
        # print(f"Shape of data this: {data_this.shape}")

        # # print the first dim ticker
        # print(f"First dim ticker: {first_dim_ticker}")

        # Append the arr to the all arr
        if data_this.size != 0:
            # Check if we have enough space in the target array
            end_index = first_dim_ticker + data_this.shape[0]

            if end_index > data_arr_full.shape[0]:
                print(f"Error: Not enough space in data_arr_full")
                print(f"data_arr_full shape: {data_arr_full.shape}")
                print(f"data_this shape: {data_this.shape}")
                print(f"first_dim_ticker: {first_dim_ticker}")
                print(f"end_index: {end_index}")
                print(f"Available space: {data_arr_full.shape[0] - first_dim_ticker}")
                print(f"Required space: {data_this.shape[0]}")
                print(f"Current year: {year}")
                print(f"File: {fname_this}")
                raise ValueError("Target array is too small for the data")

            if data_this.shape[0] == 0:
                print(f"Warning: data_this has 0 time steps for file {fname_this}")
                continue

            try:
                data_arr_full[first_dim_ticker:end_index, :, :] = data_this
                first_dim_ticker += data_this.shape[0]

                # Take the mean over the first dimension and append to the data arr wmeans
                data_arr_wmeans[year - winter_years[0], :, :] = np.mean(
                    data_this, axis=0
                )
            except ValueError as e:
                print(f"Error broadcasting array for year {year}")
                print(
                    f"data_arr_full slice shape: {data_arr_full[first_dim_ticker:end_index, :, :].shape}"
                )
                print(f"data_this shape: {data_this.shape}")
                print(f"first_dim_ticker: {first_dim_ticker}")
                print(f"end_index: {end_index}")
                print(f"File: {fname_this}")
                raise e

        else:
            raise ValueError(f"Data array is empty for {fname_this}")

    return data_arr_full, data_arr_wmeans


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
    lats_path: str,
    lons_path: str,
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
        lats_path (str): The path to the latitude data.
        lons_path (str): The path to the longitude data.

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

    # extract the unique years in the dates list
    unique_dec_years = np.arange(1960, 2024 + 1, 1)

    # print the dates
    print("Dates: ", dates)

    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # If the variable is temperature, then apply a detrend to the field
    if variable == "tas":
        # Set up an array to append the data
        data_arr_detrended = np.zeros((len(unique_dec_years), lat_shape, lon_shape))

        # loop over the unique dec years
        for i, unique_dec_year in enumerate(unique_dec_years):
            # Set up the fname this
            fname_this = (
                f"ERA5_{variable}_{region}_{unique_dec_year}_{season}_{time_freq}.npy"
            )

            # if the file does not exist, then raise an error
            if not os.path.exists(os.path.join(arrs_dir, fname_this)):
                raise FileNotFoundError(f"File {fname_this} does not exist.")

            # load the data
            data_this = np.load(os.path.join(arrs_dir, fname_this))

            # Take the mean over the first dimension and append to the data arr
            data_arr_detrended[i, :, :] = np.mean(data_this, axis=0)
    else:
        data_arr_detrended = np.zeros_like(data_arr)

    # Print the shape of data arr detredned
    print("--------------------------------")
    print(f"Shape of data arr detrended: {data_arr_detrended.shape}")
    print("--------------------------------")

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

        # If year to extract this is greater than 2020
        if year_to_extract_this >= 2019:
            # Set up the fname for the times
            times_fname = f"ERA5_{variable}_{region}_{year_to_extract_this}_{season}_{time_freq}_times_*_*.npy"
        else:
            # Set up the fname for the times
            times_fname = f"ERA5_{variable}_{region}_{year_to_extract_this}_{season}_{time_freq}_times.npy"

        # Glob the times files
        times_files = glob.glob(os.path.join(metadata_dir, times_fname))

        # If there are no times files then raise an error
        if len(times_files) == 0:
            raise FileNotFoundError(
                f"No times files found for {os.path.join(metadata_dir, times_fname)}."
            )
        elif len(times_files) > 1:
            raise ValueError(
                f"Multiple times files found for {os.path.join(metadata_dir, times_fname)}."
            )

        # load the data for this
        data_this = np.load(os.path.join(arrs_dir, fname_this))

        # if the variable is tas
        if variable == "tas":
            # print("--------------------------------")
            # print(f"Data shape: {data_this.shape}")
            # print("Detrending the data this")
            # print("--------------------------------")

            # Set up a new array for detrending the data
            data_arr_detrended_full = np.zeros(
                (data_this.shape[0], data_this.shape[1], data_this.shape[2])
            )

            # Loop over the lats and lons
            for j in range(len(lats)):
                for k in range(len(lons)):
                    # Calculate the slope and intersect
                    slope_T_this, intercept_T_this, _, _, _ = linregress(
                        unique_dec_years,
                        data_arr_detrended[:, j, k],
                    )

                    # Calculate the trend line this
                    trend_line_this = slope_T_this * unique_dec_years + intercept_T_this

                    # Find the final point on the trend line
                    final_point_this = trend_line_this[-1]

                    # # print the unique dec years
                    # print(f"Unique dec years: {unique_dec_years}")

                    # # print the year to extract this
                    # print(f"Year to extract this: {year_to_extract_this}")

                    # # print trhe type of the first unique dec year
                    # print(f"Type of first unique dec year: {type(unique_dec_years[0])}")

                    # # print the type of the year to extract this
                    # print(f"Type of year to extract this: {type(year_to_extract_this)}")

                    # find the index of year to extract this in unique_dec_years
                    try:
                        y_index = np.where(unique_dec_years == year_to_extract_this)[0][
                            0
                        ]
                    except IndexError as e:
                        print(
                            f"Error: year_to_extract_this {year_to_extract_this} not found in unique_dec_years"
                        )
                        print(f"Min unique_dec_years: {np.min(unique_dec_years)}")
                        print(f"Max unique_dec_years: {np.max(unique_dec_years)}")
                        print(f"year_to_extract_this: {year_to_extract_this}")
                        print(f"unique_dec_years: {unique_dec_years}")
                        raise e

                    # Remove the trend for the current year
                    # from the data subset
                    # Loop over the first dimension to do this
                    for l in range(data_this.shape[0]):
                        # Detrend the data
                        # data_this[l, j, k] = (
                        #     final_point_this - trend_line_this[y_index] + data_this[l, j, k]
                        # )
                        data_arr_detrended_full[l, j, k] = (
                            final_point_this
                            - trend_line_this[y_index]
                            + data_this[l, j, k]
                        )

        # load the times for this
        times_this = np.load(times_files[0])

        # convert the times to cftime
        times_this_cf = cftime.num2date(
            times_this,
            units=units,
            calendar=calendar,
        )

        # find the index of this time in the tyimes_this_cf
        # Match only by date (year, month, day), ignoring hours/minutes
        try:
            # Create a list of date matches (ignoring time)
            date_matches = []
            target_date = (
                time_to_extract_this.year,
                time_to_extract_this.month,
                time_to_extract_this.day,
            )

            for idx, time_cf in enumerate(times_this_cf):
                cf_date = (time_cf.year, time_cf.month, time_cf.day)
                if cf_date == target_date:
                    date_matches.append(idx)

            if len(date_matches) == 0:
                raise IndexError(f"No date match found for {target_date}")
            elif len(date_matches) > 1:
                print(
                    f"Warning: Multiple time matches found for date {target_date}, using first match"
                )

            time_index = date_matches[0]
        except IndexError as e:
            print(
                f"Error: Date {time_to_extract_this.year}-{time_to_extract_this.month:02d}-{time_to_extract_this.day:02d} not found in times array"
            )
            print(f"Data file: {fname_this}")
            print(f"Times file: {times_files[0]}")
            print(f"Available times range: {times_this_cf[0]} to {times_this_cf[-1]}")
            print(f"Total times available: {len(times_this_cf)}")
            print(f"type of time_to_extract_this: {type(time_to_extract_this)}")
            print(f"type of times_this_cf: {type(times_this_cf[0])}")
            print(f"Available dates in times array:")
            for t in times_this_cf:
                print(
                    f" - {t.year}-{t.month:02d}-{t.day:02d} {t.hour:02d}:{t.minute:02d}:{t.second:02d}"
                )
            raise e

        # if the variable is tas then set the data this to the detrended data
        if variable == "tas":
            data_this = data_arr_detrended_full

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
                -5,
                -4,
                -3,
                -2,
                -1,
                1,
                2,
                3,
                4,
                5,
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


# Define a function to plot the composites for the model data
def plot_composites_model(
    subset_df: pd.DataFrame,
    subset_arrs: List[np.ndarray],
    clim_arrs: List[np.ndarray],
    index_dicts: List[Dict[str, Any]],
    variables: List[str],
    lats_paths: List[str],
    lons_paths: List[str],
    suptitle: str = None,
    figsize: Tuple[int, int] = (20, 12),
):
    """
    Plots the composites for the model data.

    Args:
    =====
        subset_df (pd.DataFrame): The subset dataframe.
        subset_arrs (List[np.ndarray]): The list of subset arrays.
        clim_arrs (List[np.ndarray]): The list of climatology arrays.
        index_dicts (List[Dict[str, Any]]): The list of index dictionaries.
        variables (List[str]): The list of variables to plot.
        lats_paths (List[str]): The list of latitude paths.
        lons_paths (List[str]): The list of longitude paths.
        suptitle (str): The suptitle for the plot.
        figsize (Tuple[int, int]): The figure size.

    Returns:
    ========
        None

    """

    # Print the len of the subset df
    print(f"Plotting composites over: {len(subset_df)} days")

    # Set up a figure with 1 nrows and 3 ncols
    # Set up a figure with a custom gridspec layout
    fig = plt.figure(figsize=figsize, layout="constrained")

    # set up the gridspec object
    gs = fig.add_gridspec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # Flatten the axes
    axes_flat = [ax1, [ax2, ax3], [ax4, ax5]]

    # Loop over the axes
    for i, ax in enumerate(axes_flat):
        if i == 0:
            ax = ax1
            ax_scatter = None
        elif i == 1:
            ax = ax2
            ax_scatter = ax4
        elif i == 2:
            ax = ax3
            ax_scatter = ax5

        subset_arr_this = subset_arrs[i]

        # Set up the subset arr this
        subset_arr_this_full = np.zeros(
            (len(subset_df), subset_arr_this.shape[1], subset_arr_this.shape[2])
        )

        # extract the lons
        lons = np.load(lons_paths[i])
        # extract the lats
        lats = np.load(lats_paths[i])

        # if the variable is tas then apply detrend
        if variables[i] == "tas":
            print("Applying detrend to tas data")

            # Extract the unique effective dec years
            # from the index dict this
            index_dict_this = index_dicts[i]

            effective_dec_years_arr = np.array(index_dict_this["effective_dec_year"])

            unique_effective_dec_years = np.unique(effective_dec_years_arr)

            # Set up a new array to append to
            subset_arr_this_detrended = np.zeros(
                (
                    len(unique_effective_dec_years),
                    subset_arr_this.shape[1],
                    subset_arr_this.shape[2],
                )
            )

            # Loop over the unique effective dec years
            for j, effective_dec_year in enumerate(unique_effective_dec_years):
                # Find the index of this effective dec year in the index dict this
                index_this = np.where(effective_dec_years_arr == effective_dec_year)[0]

                # print the index this
                # print(f"Index this: {index_this}")

                # # Extract the subset arr this for this index
                subset_arr_this_detrended[j, :, :] = np.mean(
                    subset_arr_this[index_this, :, :], axis=0
                )

            # loop over the lats and lons
            for j in range(subset_arr_this_detrended.shape[1]):
                for k in range(subset_arr_this_detrended.shape[2]):
                    # Calculate the mean trend line
                    slope_T_this, intercept_T_this, _, _, _ = linregress(
                        unique_effective_dec_years,
                        subset_arr_this_detrended[:, j, k],
                    )

                    # Calculate the trend line this
                    trend_line_this = (
                        slope_T_this * unique_effective_dec_years + intercept_T_this
                    )

                    # Find the final point on the trend line
                    final_point_this = trend_line_this[-1]

                    # Loop over the unique effective dec years
                    for l in range(len(unique_effective_dec_years)):
                        # Find the indcides of the effective dec years
                        index_this_l = np.where(
                            effective_dec_years_arr == unique_effective_dec_years[l]
                        )[0]

                        # Loop over the indices
                        for m in index_this_l:
                            # Detrend the data
                            subset_arr_this[m, j, k] = (
                                final_point_this
                                - trend_line_this[l]
                                + subset_arr_this[m, j, k]
                            )

        # Loop over the rows in the subset df
        for j, (_, row) in tqdm(
            enumerate(subset_df.iterrows()),
            desc="Processing dataframe",
            total=len(subset_df),
        ):
            # Extract the init_year from the df
            init_year_df = int(row["init_year"])
            member_df = int(row["member"])
            lead_df = int(row["lead"])

            # # print the init_year, member and lead
            # print(f"Init year: {init_year_df}")
            # print(f"Member: {member_df}")
            # print(f"Lead: {lead_df}")

            # Find the index of this combination of init_year, member and lead
            # in the index
            index_dict_this = index_dicts[i]

            # print(f"Unique init years in this index dict: {sorted(set(index_dict_this['init_year']))}")
            # print(f"Unique members in this index dict: {sorted(set(index_dict_this['member']))}")
            # print(f"Unique leads in this index dict: {sorted(set(index_dict_this['lead']))}")

            # Convert lists to NumPy arrays for element-wise comparison
            init_year_array = np.array(index_dict_this["init_year"])
            member_array = np.array(index_dict_this["member"])
            lead_array = np.array(index_dict_this["lead"])

            # Construct the condition using element-wise comparison
            condition = (
                (init_year_array == init_year_df)
                & (member_array == member_df)
                & (lead_array == lead_df)
            )

            # Use np.where to find the index
            iyear_member_lead_index = np.where(condition)[0][0]

            # apply this to the subset arra this
            subset_arr_this_j = subset_arr_this[iyear_member_lead_index, :, :]

            # append this to the subset arr this full
            subset_arr_this_full[j, :, :] = subset_arr_this_j

        # if variable is psl then set the cmap to bwr
        if variables[i] == "psl":
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
        elif variables[i] == "tas":
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
            for l in range(len(lats)):
                for j in range(len(lons)):
                    point = shapely.geometry.Point(lons[j], lats[l])
                    for country in country_shapely:
                        if country.contains(point):
                            MASK_MATRIX_TMP[l, j] = 1.0

            # Reshape the mask to match the shape of the data
            MASK_MATRIX_RESHAPED = MASK_MATRIX_TMP

            # print the shape of the mask
            print(f"Shape of the mask: {MASK_MATRIX_RESHAPED.shape}")

            # print teh sum of the mask
            print(f"Sum of the mask: {np.sum(MASK_MATRIX_RESHAPED)}")
        elif variables[i] == "sfcWind":
            cmap = "PRGn"
            levels = np.array(
                [
                    -5,
                    -4,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            )
        else:
            raise ValueError(
                f"Variable {variables[i]} not recognised. Must be psl, tas or sfcWind."
            )

        # set up the countries list
        countries = [
            "Ireland",
            "Germany",
            "France",
            "Netherlands",
            "Belgium",
            "Denmark",
        ]

        uk = ["United Kingdom"]

        # Set up the countries shapefile
        countries_shp = shpreader.natural_earth(
            resolution="10m",
            category="cultural",
            name="admin_0_countries",
        )

        # set up a list of subset countries
        subset_countries = []
        for country in shpreader.Reader(countries_shp).records():
            if country.attributes["NAME"] in countries:
                subset_countries.append(country.geometry)

        uk_country = []
        for country in shpreader.Reader(countries_shp).records():
            if country.attributes["NAME"] in uk:
                uk_country.append(country.geometry)

        # set up mask matrix EU
        MASK_MATRIX_EU = np.zeros((len(lats), len(lons)))
        MASK_MATRIX_UK = np.zeros((len(lats), len(lons)))
        # Loop over the latitude and longitude points
        for l in range(len(lats)):
            for j in range(len(lons)):
                point = shapely.geometry.Point(lons[j], lats[l])
                for country in subset_countries:
                    if country.contains(point):
                        MASK_MATRIX_EU[l, j] = 1.0
                for country in uk_country:
                    if country.contains(point):
                        MASK_MATRIX_UK[l, j] = 1.0

        # Take the mean over the first dimensions of the arr this
        subset_arr_this_mean = np.mean(subset_arr_this_full, axis=0)

        # Calculate the anoms this
        anoms_this = subset_arr_this_mean - clim_arrs[i]

        # if variable is tas then apply the mask
        if variables[i] == "tas":
            # Apply the mask to the temperature data
            anoms_this = np.ma.masked_where(MASK_MATRIX_RESHAPED == 0, anoms_this)
        elif variables[i] == "psl":
            anoms_this = anoms_this / 100

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

        # add the coastlines
        ax.coastlines()

        # add the gridlines
        ax.gridlines()

        # set up the colorbar beneath the plot
        cbar = fig.colorbar(
            im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8, location="bottom"
        )

        # Set the colorbar ticks
        cbar.set_ticks(levels)

        # if the variable is tas or sfcwind
        if variables[i] == "tas" or variables[i] == "sfcWind":
            # Set up the anoms for scatter this
            anoms_scatter_this = subset_arr_this_full - clim_arrs[i]

            # expand the mask to match the shape of the data
            MASK_MATRIX_EU_EXP = np.broadcast_to(
                MASK_MATRIX_EU, anoms_scatter_this.shape
            )
            MASK_MATRIX_UK_EXP = np.broadcast_to(
                MASK_MATRIX_UK, anoms_scatter_this.shape
            )

            # Apply the mask
            anoms_scatter_this_eu = np.ma.masked_where(
                MASK_MATRIX_EU_EXP == 0, anoms_scatter_this
            )
            anoms_scatter_this_uk = np.ma.masked_where(
                MASK_MATRIX_UK_EXP == 0, anoms_scatter_this
            )

            # rpint the shape of the anoms scatter this
            print(f"Shape of the anoms scatter this: {anoms_scatter_this.shape}")
            # print the shape of the anoms scatter this eu
            print(f"Shape of the anoms scatter this eu: {anoms_scatter_this_eu.shape}")

            # print the shape of the anoms scatter this uk
            print(f"Shape of the anoms scatter this uk: {anoms_scatter_this_uk.shape}")

            # take the spatial means of the anoms for the EU and UK
            anoms_scatter_this_mean_eu = np.mean(anoms_scatter_this_eu, axis=(1, 2))
            anoms_scatter_this_mean_uk = np.mean(anoms_scatter_this_uk, axis=(1, 2))

            # calculate the pearson correlation
            corr, _ = pearsonr(
                anoms_scatter_this_mean_eu,
                anoms_scatter_this_mean_uk,
            )

            # Fit a straight line to the data
            m, b = np.polyfit(
                anoms_scatter_this_mean_uk,
                anoms_scatter_this_mean_eu,
                1,
            )

            # plot the scatter
            ax_scatter.scatter(
                anoms_scatter_this_mean_uk,
                anoms_scatter_this_mean_eu,
                color="red",
                label=f"r={corr:.2f}\n slope={m:.2f}",
                s=1,
                marker="o",
            )

            # plot the line
            ax_scatter.plot(
                anoms_scatter_this_mean_uk,
                m * anoms_scatter_this_mean_uk + b,
                color="blue",
                linestyle="--",
            )

            # set up a title
            ax_scatter.set_title(f"UK vs nearest EU neighbours")

            # set the x and y labels
            ax_scatter.set_xlabel("UK Anomalies")

            ax_scatter.set_ylabel("EU Anomalies")

            # include a legend in the top left
            ax_scatter.legend(loc="upper left")

    # set up the suptitle
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=16)

    return None


# Plot the composites
# for a subset of the df
def plot_composites(
    subset_df: pd.DataFrame,
    subset_arrs: List[np.ndarray],
    clim_arrs: List[np.ndarray],
    dates_lists: List[List[cftime.DatetimeProlepticGregorian]],
    variables: List[str],
    lats_paths: List[str],
    lons_paths: List[str],
    winter_years: np.ndarray = np.arange(1960, 2018 + 1, 1),
    suptitle: str = None,
    figsize: Tuple[int, int] = (20, 12),
):
    """
    Plots the composites for the given variables.

    Args:
    ======
        subset_df (pd.DataFrame): The subset dataframe.
        subset_arrs (List[np.ndarray]): The list of subset arrays.
        clim_arrs (List[np.ndarray]): The list of climatology arrays.
        dates_lists (List[List[cftime.DatetimeProlepticGregorian]]): The list of dates lists.
        variables (List[str]): The list of variables to plot.
        lats_paths (List[str]): The list of latitude paths.
        lons_paths (List[str]): The list of longitude paths.
        winter_years (np.ndarray): The years to load.
        suptitle (str): The suptitle for the plot.
        figsize (Tuple[int, int]): The figure size.

    Returns:
    ========
        None
    """

    # Print the len of the subset df
    print(f"Plotting composites over: {len(subset_df)} days")

    # extract the dates from the subset df
    dates = subset_df["time"].values

    # format these as datetimes
    subset_dates = [datetime.strptime(date, "%Y-%m-%d") for date in dates]

    # print the dates
    print("Dates to composite over: ", dates)

    # Set up a figure with 1 nrows and 3 ncols
    # Set up a figure with a custom gridspec layout
    fig = plt.figure(figsize=figsize, layout="constrained")

    # set up the gridspec object
    gs = fig.add_gridspec(3, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, :], projection=ccrs.PlateCarree())
    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    ax3 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])

    # Flatten the axes
    axes_flat = [ax1, [ax2, ax3], [ax4, ax5]]

    # Loop over the axes
    for i, ax in enumerate(axes_flat):
        # if i == 0
        if i == 0:
            ax = ax1
            ax_scatter = None
        elif i == 1:
            ax = ax2
            ax_scatter = ax4
        elif i == 2:
            ax = ax3
            ax_scatter = ax5

        # Print the dates list this
        print(f"Dates list this: {dates_lists[i]}")

        # print the typeof the first dates values
        print(f"Type of first dates value: {type(dates_lists[i][0])}")

        # o the same for the dates to composite over
        print(f"Type of first dates value: {type(dates[0])}")

        # load the lats and lons
        lats = np.load(lats_paths[i])
        lons = np.load(lons_paths[i])

        # if the variable is psl then set the cmap to bwr
        if variables[i] == "psl":
            subset_dates_cf = []
            # format the subset dates to extract
            for date in subset_dates:
                date_this_cf = cftime.DatetimeGregorian(
                    date.year, date.month, date.day, hour=11, calendar="gregorian"
                )
                subset_dates_cf.append(date_this_cf)

            # Set up the cmap
            cmap = "bwr"

            # Sert up the levels
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
        elif variables[i] == "tas":
            # format the subset dates
            subset_dates_cf = []
            # format the subset dates to extract
            for date in subset_dates:
                date_this_cf = cftime.DatetimeProlepticGregorian(
                    date.year,
                    date.month,
                    date.day,
                    hour=0,
                    calendar="proleptic_gregorian",
                )
                subset_dates_cf.append(date_this_cf)

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
            for l in range(len(lats)):
                for j in range(len(lons)):
                    point = shapely.geometry.Point(lons[j], lats[l])
                    for country in country_shapely:
                        if country.contains(point):
                            MASK_MATRIX_TMP[l, j] = 1.0

            # Reshape the mask to match the shape of the data
            MASK_MATRIX_RESHAPED = MASK_MATRIX_TMP

            # print the shape of the mask
            print(f"Shape of the mask: {MASK_MATRIX_RESHAPED.shape}")

            # print teh sum of the mask
            print(f"Sum of the mask: {np.sum(MASK_MATRIX_RESHAPED)}")
        elif variables[i] == "sfcWind":
            # format the subset dates
            subset_dates_cf = []
            # format the subset dates to extract
            for date in subset_dates:
                date_this_cf = cftime.DatetimeProlepticGregorian(
                    date.year,
                    date.month,
                    date.day,
                    hour=0,
                    calendar="proleptic_gregorian",
                )
                subset_dates_cf.append(date_this_cf)

            cmap = "PRGn"
            levels = np.array(
                [
                    -5,
                    -4,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            )
        else:
            raise ValueError(
                f"Variable {variables[i]} not recognised. Must be psl, tas or sfcWind."
            )

        # set up the countries list
        countries = [
            "Ireland",
            "Germany",
            "France",
            "Netherlands",
            "Belgium",
            "Denmark",
        ]

        uk = ["United Kingdom"]

        # Set up the countries shapefile
        countries_shp = shpreader.natural_earth(
            resolution="10m",
            category="cultural",
            name="admin_0_countries",
        )

        # set up a list of subset countries
        subset_countries = []
        for country in shpreader.Reader(countries_shp).records():
            if country.attributes["NAME"] in countries:
                subset_countries.append(country.geometry)

        uk_country = []
        for country in shpreader.Reader(countries_shp).records():
            if country.attributes["NAME"] in uk:
                uk_country.append(country.geometry)

        # print the subset countries
        print(f"Subset countries: {subset_countries}")

        # print the uk country
        print(f"UK country: {uk_country}")

        # print the len of subset countries
        print(f"Len of subset countries: {len(subset_countries)}")

        # print the UK country
        print(f"Len of UK country: {len(uk_country)}")

        # set up mask matrix EU
        MASK_MATRIX_EU = np.zeros((len(lats), len(lons)))
        MASK_MATRIX_UK = np.zeros((len(lats), len(lons)))
        # Loop over the latitude and longitude points
        for l in range(len(lats)):
            for j in range(len(lons)):
                point = shapely.geometry.Point(lons[j], lats[l])
                for country in subset_countries:
                    if country.contains(point):
                        MASK_MATRIX_EU[l, j] = 1.0
                for country in uk_country:
                    if country.contains(point):
                        MASK_MATRIX_UK[l, j] = 1.0

        # print the shape of the mask
        print(f"Shape of the mask EU: {MASK_MATRIX_EU.shape}")
        print(f"Shape of the mask UK: {MASK_MATRIX_UK.shape}")

        # print the sum of the mask
        print(f"Sum of the mask EU: {np.sum(MASK_MATRIX_EU)}")
        print(f"Sum of the mask UK: {np.sum(MASK_MATRIX_UK)}")

        # # now print the subset dates cf
        # print(f"Subset dates cf: {subset_dates_cf}")

        # # print the i
        # print(f"iteration i: {i}")

        # # and now print the dates list i
        # print(f"Dates list i: {dates_lists[i]}")

        # find the indexes of the subset dates cf in the dates list i
        indexes = []

        for date in subset_dates_cf:
            # find the index of the date in the dates list
            index = np.where(dates_lists[i] == date)[0][0]
            indexes.append(index)

        # # print the indexes
        # print(f"Indexes: {indexes}")
        subset_arr_this_i = subset_arrs[i]

        # # print the shape of subset arr this
        # print(f"Shape of subset arr this: {subset_arr_this.shape}")

        # # if the variable is tas then apply some detrending
        # if variables[i] == "tas":
        #     print("-----------------------------------------------")
        #     print("Applying a detrend to the temperature data")
        #     print("-----------------------------------------------")
        #     # Loop over the lats and lons
        #     for l in range(len(lats)):
        #         for j in range(len(lons)):
        #             # detrend the data
        #             slope_T, intercept_T, _, _, _ = linregress(
        #                 winter_years,
        #                 subset_arr_this_i[:, l, j],
        #             )

        #             # calculate the trend line this
        #             trend_line_this = slope_T * winter_years + intercept_T

        #             # Find the final point on the trend line
        #             final_point_this = trend_line_this[-1]

        #             # Subtract the trend line from the data
        #             subset_arr_this_i[:, l, j] = final_point_this - trend_line_this + subset_arr_this_i[:, l, j]

        # Apply these index to the subset data
        subset_arr_this = subset_arr_this_i[indexes, :, :]

        # take the mean over this
        subset_arr_this_mean = np.mean(subset_arr_this, axis=0)

        # calculate the anoms
        anoms_this = subset_arr_this_mean - clim_arrs[i]

        # if the variable is tas then apply the mask
        if variables[i] == "tas":
            # Apply the mask to the temperature data
            anoms_this = np.ma.masked_where(MASK_MATRIX_RESHAPED == 0, anoms_this)
        elif variables[i] == "psl":
            anoms_this = anoms_this / 100

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

        # add coastlines
        ax.coastlines()

        # add gridlines
        ax.gridlines()

        # Set up the colorbar beneath the plot
        cbar = fig.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            aspect=50,
            shrink=0.8,
            location="bottom",
        )

        # set the ticks for the colorbar as the levels
        cbar.set_ticks(levels)

        # if the variable is tas or sfcWind
        if variables[i] == "tas" or variables[i] == "sfcWind":
            anoms_scatter_this = subset_arr_this - clim_arrs[i]
            # print the shape of anoms this scatter
            print(f"Shape of anoms this scatter: {anoms_scatter_this.shape}")

            # Expand the mask to match the shape of anoms_scatter_this
            MASK_MATRIX_EU_expanded = np.broadcast_to(
                MASK_MATRIX_EU, anoms_scatter_this.shape
            )
            MASK_MATRIX_UK_expanded = np.broadcast_to(
                MASK_MATRIX_UK, anoms_scatter_this.shape
            )

            # Apply the mask
            anoms_this_eu = np.ma.masked_where(
                MASK_MATRIX_EU_expanded == 0, anoms_scatter_this
            )
            anoms_this_uk = np.ma.masked_where(
                MASK_MATRIX_UK_expanded == 0, anoms_scatter_this
            )

            # Print the shapes to verify
            print(f"Shape of anoms_this_eu: {anoms_this_eu.shape}")
            print(f"Shape of anoms_this_uk: {anoms_this_uk.shape}")

            # take the spatial mean of the anoms for the EU and UK
            anoms_this_eu_mean = np.mean(anoms_this_eu, axis=(1, 2))
            anoms_this_uk_mean = np.mean(anoms_this_uk, axis=(1, 2))

            # calcyulate the correlation betwen the two sets of points
            corr, _ = pearsonr(anoms_this_eu_mean, anoms_this_uk_mean)

            # fit a straight line to the data
            m, b = np.polyfit(anoms_this_uk_mean, anoms_this_eu_mean, 1)

            # plot these scatter points
            ax_scatter.scatter(
                anoms_this_uk_mean,
                anoms_this_eu_mean,
                color="blue",
                label=f"r = {corr:.2f}\n slope = {m:.2f}",
                s=1,
                marker="x",
            )

            # plot the line
            ax_scatter.plot(
                anoms_this_uk_mean,
                m * anoms_this_uk_mean + b,
                color="blue",
                linestyle="--",
            )

            # set up a title
            ax_scatter.set_title(f"UK vs nearest EU neighbours")

            # set the x and y labels
            ax_scatter.set_xlabel("UK Anomalies")

            ax_scatter.set_ylabel("EU Anomalies")

            # include a legend in the top left
            ax_scatter.legend(loc="upper left")

    # include the sup title for the plot
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=14)

    return None


# Define a function to plot MSLP composites
def plot_mslp_composites(
    subset_dfs_obs: List[pd.DataFrame],
    subset_dfs_model: List[pd.DataFrame],
    subset_arrs_obs: List[np.ndarray],
    subset_arrs_model: List[np.ndarray],
    clim_arrs_obs: List[np.ndarray],
    clim_arrs_model: List[np.ndarray],
    dates_lists_obs: List[List[cftime.DatetimeProlepticGregorian]],
    model_index_dicts: List[Dict[str, np.ndarray]],
    lats_paths: List[str],
    lons_paths: List[str],
    suptitle: str = None,
    figsize: Tuple[int, int] = (20, 12),
):
    """
    Plots composites of MSLP for different thresholds.

    E.g.,

        First row -> grey points (<80th percentile of obs)

        Second row -> yellow points (>=80th percentile of obs, <obs max)

        Third row -> red points (>=obs max)

    Args:
    =====

        subset_df_obs (pd.DataFrame): The subset dataframe for observations.
        subset_df_model (pd.DataFrame): The subset dataframe for the model.
        subset_arrs_obs (List[np.ndarray]): The list of subset arrays for observations.
        subset_arrs_model (List[np.ndarray]): The list of subset arrays for the model.
        clim_arrs_obs (List[np.ndarray]): The list of climatology arrays for observations.
        clim_arrs_model (List[np.ndarray]): The list of climatology arrays for the model.
        dates_lists_obs (List[List[cftime.DatetimeProlepticGregorian]]): The list of dates lists for observations.
        dates_lists_model (List[List[cftime.DatetimeProlepticGregorian]]): The list of dates lists for the model.
        lats_paths (List[str]): The list of latitude paths.
        lons_paths (List[str]): The list of longitude paths.
        suptitle (str): The suptitle for the plot.
        figsize (Tuple[int, int]): The figure size.

    Returns:
    ========

        None

    """

    # print the mion and max of the first obs psl subset arr
    print(f"Min of first obs psl subset arr: {np.min(subset_arrs_obs[0])}")
    print(f"Max of first obs psl subset arr: {np.max(subset_arrs_obs[0])}")
    print(f"Mean of first obs psl subset arr: {np.mean(subset_arrs_obs[0])}")

    # print the mion and max of the first model psl subset arr
    print(f"Min of first model psl subset arr: {np.min(subset_arrs_model[0])}")
    print(f"Max of first model psl subset arr: {np.max(subset_arrs_model[0])}")
    print(f"Mean of first model psl subset arr: {np.mean(subset_arrs_model[0])}")

    # hardcode the cmap and levels for psl
    cmap = "coolwarm"

    # Sert up the levels
    levels = np.array(
        [
            1004,
            1006,
            1008,
            1010,
            1012,
            1014,
            1016,
            1018,
            1020,
            1022,
            1024,
            1026,
        ]
    )
    ticks = levels

    # Load the lats and lons
    lats = np.load(lats_paths[0])
    lons = np.load(lons_paths[0])

    # Set up the figure
    fig = plt.figure(figsize=figsize, layout="constrained")

    # Set up the gridspec
    gs = fig.add_gridspec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())  # Row 0, Col 0
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())  # Row 0, Col 1
    ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())  # Row 1, Col 0
    ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())  # Row 1, Col 1
    ax5 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())  # Row 2, Col 0
    ax6 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())  # Row 2, Col 1

    full_axes = [ax1, ax2, ax3, ax4, ax5, ax6]

    # Set up the pairs
    axes_pairs = [
        (ax1, ax2),
        (ax3, ax4),
        (ax5, ax6),
    ]

    # Loop over the axes pairs
    for i, (ax_obs, ax_model) in enumerate(axes_pairs):
        # Set up the subset dates to select the obs for
        subset_dates_obs_cf = []

        # Loop over the subset dates
        dates_obs_this = subset_dfs_obs[i]["time"].values

        # Format these ad datetimes
        subset_dates_obs_dt = [
            datetime.strptime(date, "%Y-%m-%d") for date in dates_obs_this
        ]

        # Format the subset dates to extract
        for date in subset_dates_obs_dt:
            date_this_cf = cftime.DatetimeGregorian(
                date.year, date.month, date.day, hour=11, calendar="gregorian"
            )
            subset_dates_obs_cf.append(date_this_cf)

        # Set up an empty list for the indices this
        indices_dates_obs_this = []

        # Print dates list obs i
        print(f"Dates list obs i: {dates_lists_obs[i]}")

        # Looop over the subset dates obs cf
        for date in subset_dates_obs_cf:
            print(f"Date: {date}")

            index_this = np.where(dates_lists_obs[i] == date)[0][0]
            indices_dates_obs_this.append(index_this)

        # Set up the subset arr this for the obs
        subset_arr_this_obs = subset_arrs_obs[i]

        # Apply these indices to the subset data
        subset_arr_this_obs = subset_arr_this_obs[indices_dates_obs_this, :, :]

        # get the N for obs this
        N_obs_this = np.shape(subset_arr_this_obs)[0]

        # Take the mean over this
        subset_arr_this_obs_mean = np.mean(subset_arr_this_obs, axis=0)

        # Calculate the obs anoms
        anoms_this_obs = subset_arr_this_obs_mean - clim_arrs_obs[i]

        # Set up the subset array this for the obs
        subset_arr_this_model = subset_arrs_model[i]

        # Set up the subset arr this model full
        subset_arr_this_model_full = np.zeros(
            (len(subset_dfs_model[i]), len(lats), len(lons))
        )

        # Set up the N for model this
        N_model_this = np.shape(subset_arr_this_model_full)[0]

        # Extract the index dict for the model this
        model_index_dict_this = model_index_dicts[i]

        # Extract the init years as arrays
        init_year_array_this = np.array(model_index_dict_this["init_year"])
        member_array_this = np.array(model_index_dict_this["member"])
        lead_array_this = np.array(model_index_dict_this["lead"])

        # zero the missing daya here
        missing_days = 0

        # Loop over the rows in this subset df for the model
        for j, (_, row) in tqdm(enumerate(subset_dfs_model[i].iterrows())):
            # Extract the init_year from the df
            init_year_df = int(row["init_year"])
            member_df = int(row["member"])
            lead_df = int(row["lead"])

            # Construct the condition for element wise comparison
            condition = (
                (init_year_array_this == init_year_df)
                & (member_array_this == member_df)
                & (lead_array_this == lead_df)
            )

            try:
                # Find the index where this condition is met
                index_this = np.where(condition)[0][0]
            except IndexError:
                print(
                    f"init year {init_year_df}, member {member_df}, lead {lead_df} not found"
                )
                missing_days += 1

            # Extract the corresponding value from the subset_arr_this_model
            subset_arr_this_model_index_this = subset_arr_this_model[index_this, :, :]

            # Store the value in the subset_arr_this_model_full
            subset_arr_this_model_full[j, :, :] = subset_arr_this_model_index_this

        # print the number of missing days
        print(f"row index: {i}")
        print(f"Number of missing days: {missing_days}")
        print(f"Model overall N: {N_model_this}")

        # Take the mean over this
        subset_arr_this_model_mean = np.mean(subset_arr_this_model_full, axis=0)

        # Calculate the model anoms
        anoms_this_model = subset_arr_this_model_mean - clim_arrs_model[i]

        # plot the obs data on the left
        im_obs = ax_obs.contourf(
            lons,
            lats,
            anoms_this_obs / 100,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            levels=levels,
            extend="both",
        )

        # plot the model data on the right
        im_model = ax_model.contourf(
            lons,
            lats,
            anoms_this_model / 100,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            levels=levels,
            extend="both",
        )

        # Inlcude a textbox in the top right for the N
        ax_obs.text(
            0.95,
            0.95,
            f"N = {N_obs_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_model.text(
            0.95,
            0.95,
            f"N = {N_model_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # add coastlines
        ax_obs.coastlines()
        ax_model.coastlines()

        # # add gridlines
        # ax_obs.gridlines()
        # ax_model.gridlines()

        # if i ==2
        if i == 2:
            # Set up the a shared cbar
            cbar = fig.colorbar(
                im_obs,
                ax=full_axes,
                orientation="horizontal",
                pad=0.01,
                shrink=0.8,
            )

            # set the ticks as levels
            cbar.set_ticks(ticks)

    # set the title for ax1 and ax2 in bold
    ax1.set_title("Obs (ERA5)", fontsize=12, fontweight="bold")

    # set the title for ax2
    ax2.set_title("Model (DePreSys)", fontsize=12, fontweight="bold")

    # include ylabels for ax1 and ax3 and ax5
    ax1.set_ylabel("< obs 80th percentile", fontsize=12, fontweight="bold")
    ax3.set_ylabel(">= obs 80th percentile", fontsize=12, fontweight="bold")
    ax5.set_ylabel(">= obs max", fontsize=12, fontweight="bold")

    # do these with textboxes instead
    ax1.text(
        0.95,
        0.05,
        "Block max days",
        ha="right",
        va="bottom",
        transform=ax1.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    ax2.text(
        0.95,
        0.05,
        "Block max days",
        ha="right",
        va="bottom",
        transform=ax2.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # set the text for ax3 and ax4
    ax3.text(
        0.95,
        0.05,
        "Extreme days",
        ha="right",
        va="bottom",
        transform=ax3.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    ax4.text(
        0.95,
        0.05,
        "Extreme days",
        ha="right",
        va="bottom",
        transform=ax4.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # set the text for ax5 and ax6
    ax5.text(
        0.95,
        0.05,
        "21-12-2010",
        ha="right",
        va="bottom",
        transform=ax5.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    ax6.text(
        0.95,
        0.05,
        "Unseen days",
        ha="right",
        va="bottom",
        transform=ax6.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # If the suptitle is not none then set it
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=12, fontweight="bold")

    # adjust the whitespace
    plt.subplots_adjust(wspace=0.05, hspace=0.05)

    return None


# Define a functoin to plot the tas wind composites
def plot_tas_wind_composites(
    subset_dfs_obs: List[pd.DataFrame],
    subset_dfs_model: List[pd.DataFrame],
    subset_arrs_obs_tas: List[np.ndarray],
    subset_arrs_model_tas: List[np.ndarray],
    subset_arrs_obs_wind: List[np.ndarray],
    subset_arrs_model_wind: List[np.ndarray],
    clim_arrs_obs_tas: List[np.ndarray],
    clim_arrs_model_tas: List[np.ndarray],
    clim_arrs_obs_wind: List[np.ndarray],
    clim_arrs_model_wind: List[np.ndarray],
    dates_lists_obs_tas: List[List[cftime.DatetimeProlepticGregorian]],
    dates_lists_obs_wind: List[List[cftime.DatetimeProlepticGregorian]],
    model_index_dicts_tas: List[Dict[str, np.ndarray]],
    model_index_dicts_wind: List[Dict[str, np.ndarray]],
    lats_path: str,
    lons_path: str,
    suptitle: str = None,
    figsize: Tuple[int, int] = (8, 9),
):
    """
    Plots the tas and wind composites for the given variables.

    Args:
    =====

        subset_dfs_obs (List[pd.DataFrame]): The list of subset dataframes for observations.
        subset_dfs_model (List[pd.DataFrame]): The list of subset dataframes for the model.
        subset_arrs_obs_tas (List[np.ndarray]): The list of tas subset arrays for observations.
        subset_arrs_model_tas (List[np.ndarray]): The list of tas subset arrays for the model.
        subset_arrs_obs_wind (List[np.ndarray]): The list of wind subset arrays for observations.
        subset_arrs_model_wind (List[np.ndarray]): The list of wind subset arrays for the model.
        clim_arrs_obs_tas (List[np.ndarray]): The list of climatology arrays for observations.
        clim_arrs_model_tas (List[np.ndarray]): The list of climatology arrays for the model.
        clim_arrs_obs_wind (List[np.ndarray]): The list of climatology arrays for observations.
        dates_lists_obs (List[List[cftime.DatetimeProlepticGregorian]]): The list of dates lists for observations.
        model_index_dicts (List[Dict[str, np.ndarray]]): The list of model index dictionaries.
        lats_path (str): The path to the latitude file.
        lons_path (str): The path to the longitude file.
        suptitle (str): The suptitle for the plot.
        figsize (Tuple[int, int]): The figure size.

    Returns:
    ========

        None

    """

    # Hardcoe the cmap and levels for tas
    cmap_tas = "bwr"
    levels_tas = np.array(
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

    # Hardcode the cmap and levels for wind
    cmap_wind = "PRGn"
    levels_wind = np.array(
        [
            -5,
            -4,
            -3,
            -2,
            -1,
            1,
            2,
            3,
            4,
            5,
        ]
    )

    # hard code the nearest neighbour countries
    nearest_neighbour_countries = [
        "Ireland",
        "Germany",
        "France",
        "Netherlands",
        "Belgium",
        "Denmark",
    ]
    uk_names = ["United Kingdom"]

    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # Set up the countries shapefile
    countries_shp = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )

    # Set up the x and y
    x, y = lons, lats

    # Set up a landmask for the temperature data
    MASK_MATRIX_TMP = np.zeros((len(lats), len(lons)))
    country_shapely = []
    for country in shpreader.Reader(countries_shp).records():
        country_shapely.append(country.geometry)

    # Loop over the latitude and longitude points
    for l in range(len(lats)):
        for j in range(len(lons)):
            point = shapely.geometry.Point(lons[j], lats[l])
            for country in country_shapely:
                if country.contains(point):
                    MASK_MATRIX_TMP[l, j] = 1.0

    # Reshape the mask to match the shape of the data
    MASK_MATRIX_RESHAPED_LAND = MASK_MATRIX_TMP

    # Set up a mask for the countries
    nn_countries_geom = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME"] in nearest_neighbour_countries:
            nn_countries_geom.append(country.geometry)

    # Set up a mask for the UK
    uk_country_geom = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME"] in uk_names:
            uk_country_geom.append(country.geometry)

    # Set up the mask matrix for the nearest neighbour countries
    MASK_MATRIX_NN = np.zeros((len(lats), len(lons)))
    MASK_MATRIX_UK = np.zeros((len(lats), len(lons)))

    # Loop over the latitude and longitude points
    for l in range(len(lats)):
        for j in range(len(lons)):
            point = shapely.geometry.Point(lons[j], lats[l])
            for country in nn_countries_geom:
                if country.contains(point):
                    MASK_MATRIX_NN[l, j] = 1.0
            for country in uk_country_geom:
                if country.contains(point):
                    MASK_MATRIX_UK[l, j] = 1.0

    plt.rcParams["figure.constrained_layout.use"] = False
    # Set up the figure
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(6, 3, figure=fig)
    # Set up the gridspec
    gs.update(wspace=0.01, hspace=0.01)

    # Set up the axes

    # Grey dots subplots
    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())  # Row 0, Col 0
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())  # Row 0, Col 1
    ax2 = fig.add_subplot(gs[0, 2])  # Row 0, Col 2
    ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())  # Row 1, Col 0
    ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())  # Row 1, Col 1
    ax5 = fig.add_subplot(gs[1, 2])  # Row 1, Col 2

    # Yellow dots sublots
    ax6 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())  # Row 2, Col 0
    ax7 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())  # Row 2, Col 1
    ax8 = fig.add_subplot(gs[2, 2])  # Row 2, Col 2
    ax9 = fig.add_subplot(gs[3, 0], projection=ccrs.PlateCarree())  # Row 3, Col 0
    ax10 = fig.add_subplot(gs[3, 1], projection=ccrs.PlateCarree())  # Row 3, Col 1
    ax11 = fig.add_subplot(gs[3, 2])  # Row 3, Col 2

    # Red dots subplots
    ax12 = fig.add_subplot(gs[4, 0], projection=ccrs.PlateCarree())  # Row 4, Col 0
    ax13 = fig.add_subplot(gs[4, 1], projection=ccrs.PlateCarree())  # Row 4, Col 1
    ax14 = fig.add_subplot(gs[4, 2])  # Row 4, Col 2
    ax15 = fig.add_subplot(gs[5, 0], projection=ccrs.PlateCarree())  # Row 5, Col 0
    ax16 = fig.add_subplot(gs[5, 1], projection=ccrs.PlateCarree())  # Row 5, Col 1
    ax17 = fig.add_subplot(gs[5, 2])  # Row 5, Col 2

    # # Set aspect ratio to square for all scatter plot axes
    scatter_axes = [
        ax2,
        ax5,
        ax8,
        ax11,
        ax14,
        ax17,
    ]  # Replace with the axes where scatter plots are drawn
    for ax in scatter_axes:
        ax.set_aspect("equal", adjustable="box")

    all_axes = [
        ax0,
        ax1,
        ax3,
        ax4,
        ax6,
        ax7,
        ax9,
        ax10,
        ax12,
        ax13,
        ax15,
        ax16,
    ]

    for i in range(len(all_axes)):
        ax_this = all_axes[i]
        plt.axis("on")
        ax_this.set_xticklabels([])
        ax_this.set_yticklabels([])
        ax_this.set_aspect("equal")

    # Set up the cmap_axes
    # cmap_axes = [
    #     ax0,
    #     ax1,

    # # set a tight layout for the gridspec
    # fig.tight_layout()

    # Set up the axes groups
    axes_groups = [
        (ax0, ax1, ax2, ax3, ax4, ax5),
        (ax6, ax7, ax8, ax9, ax10, ax11),
        (ax12, ax13, ax14, ax15, ax16, ax17),
    ]

    # Set up the names
    names_list = [
        ("Block max days", "Block max days"),
        ("Extreme days", "Extreme days"),
        ("21-12-2010", "Unseen days"),
    ]

    for i, (axes_group) in enumerate(axes_groups):
        # Set the axes up for this
        ax_temp_obs = axes_group[0]
        ax_temp_model = axes_group[1]
        ax_temp_scatter = axes_group[2]
        ax_wind_obs = axes_group[3]
        ax_wind_model = axes_group[4]
        ax_wind_scatter = axes_group[5]

        # ----------------------------
        # Detrend the model tas data
        # ----------------------------

        # extract the subset array this for tas
        subset_arr_this_tas = subset_arrs_model_tas[i]

        # Extract the unique effective dec years
        # from the index dict this
        index_dict_this_tas = model_index_dicts_tas[i]

        effective_dec_years_arr_tas = np.array(
            index_dict_this_tas["effective_dec_year"]
        )
        unique_effective_dec_years_tas = np.unique(effective_dec_years_arr_tas)

        # Set up a new array to append to
        subset_arr_this_detrended = np.zeros(
            (len(unique_effective_dec_years_tas), len(lats), len(lons))
        )

        # Loop over the unique effective dec years
        for j, effective_dec_year in enumerate(unique_effective_dec_years_tas):
            # Find the index of this effective dec year in the index dict this
            index_this = np.where(effective_dec_years_arr_tas == effective_dec_year)[0]

            # print the index this
            # print(f"Index this: {index_this}")

            # # Extract the subset arr this for this index
            subset_arr_this_detrended[j, :, :] = np.mean(
                subset_arr_this_tas[index_this, :, :], axis=0
            )

        # loop over the lats and lons
        for j in range(subset_arr_this_detrended.shape[1]):
            for k in range(subset_arr_this_detrended.shape[2]):
                # Calculate the mean trend line
                slope_T_this, intercept_T_this, _, _, _ = linregress(
                    unique_effective_dec_years_tas,
                    subset_arr_this_detrended[:, j, k],
                )

                # Calculate the trend line this
                trend_line_this = (
                    slope_T_this * unique_effective_dec_years_tas + intercept_T_this
                )

                # Find the final point on the trend line
                final_point_this = trend_line_this[-1]

                # Loop over the unique effective dec years
                for l in range(len(unique_effective_dec_years_tas)):
                    # Find the indcides of the effective dec years
                    index_this_l = np.where(
                        effective_dec_years_arr_tas == unique_effective_dec_years_tas[l]
                    )[0]

                    # Loop over the indices
                    for m in index_this_l:
                        # Detrend the data
                        subset_arr_this_tas[m, j, k] = (
                            final_point_this
                            - trend_line_this[l]
                            + subset_arr_this_tas[m, j, k]
                        )

        # Set up the subset dates to select the obs for
        subset_dates_obs_cf_this = []

        # Extract the subset dates this
        subset_dates_obs_this = subset_dfs_obs[i]["time"].values

        # Format these as datetimes
        subset_dates_obs_dt_this = [
            datetime.strptime(date, "%Y-%m-%d") for date in subset_dates_obs_this
        ]

        # Format the subset dates to extract
        for date in subset_dates_obs_dt_this:
            date_this_cf = cftime.DatetimeProlepticGregorian(
                date.year,
                date.month,
                date.day,
                hour=0,
                calendar="proleptic_gregorian",
            )

            subset_dates_obs_cf_this.append(date_this_cf)

        # Set up an empty list for the indices dates obs this
        indices_dates_obs_this_tas = []
        indices_dates_obs_this_wind = []

        # Loop over the dates in subset dtes obs cf
        for date in subset_dates_obs_cf_this:
            try:
                # Find the index of the date in the dates list
                index_this_tas = np.where(dates_lists_obs_tas[i] == date)[0][0]
            except IndexError:
                print(f"Date {date} not found in dates list obs tas for index {i}")
                print(f"Dates list obs tas: {dates_lists_obs_tas[i]}")
            indices_dates_obs_this_tas.append(index_this_tas)

            # Find the index of the date in the dates list
            index_this_wind = np.where(dates_lists_obs_wind[i] == date)[0][0]
            indices_dates_obs_this_wind.append(index_this_wind)

        # Set up the subset arr this for the obs
        subset_arr_this_obs_tas = subset_arrs_obs_tas[i]
        subset_arr_this_obs_wind = subset_arrs_obs_wind[i]

        # assert that the indices have the same length
        assert len(indices_dates_obs_this_tas) == len(
            indices_dates_obs_this_wind
        ), "Indices for tas and wind do not match in length"

        # Apply these indices to the subset data
        subset_arr_this_obs_tas = subset_arr_this_obs_tas[
            indices_dates_obs_this_tas, :, :
        ]
        subset_arr_this_obs_wind = subset_arr_this_obs_wind[
            indices_dates_obs_this_wind, :, :
        ]

        # Get the N for the obs this
        N_obs_this = np.shape(subset_arr_this_obs_tas)[0]

        # Take the mean over this
        subset_arr_this_obs_tas_mean = np.mean(subset_arr_this_obs_tas, axis=0)
        subset_arr_this_obs_wind_mean = np.mean(subset_arr_this_obs_wind, axis=0)

        # Calculate the obs anoms
        anoms_this_obs_tas = subset_arr_this_obs_tas_mean - clim_arrs_obs_tas[i]
        anoms_this_obs_wind = subset_arr_this_obs_wind_mean - clim_arrs_obs_wind[i]

        # Set up the subset arr this for the model
        subset_arr_this_model_tas = subset_arr_this_tas
        subset_arr_this_model_wind = subset_arrs_model_wind[i]

        # Set up the subset arr this model full
        subset_arr_this_model_tas_full = np.zeros(
            (len(subset_dfs_model[i]), len(lats), len(lons))
        )
        subset_arr_this_model_wind_full = np.zeros(
            (len(subset_dfs_model[i]), len(lats), len(lons))
        )

        # Set up the N for model this
        N_model_this = np.shape(subset_arr_this_model_tas_full)[0]

        # Extract the index dict for the model this
        model_index_dict_tas_this = model_index_dicts_tas[i]
        model_index_dict_wind_this = model_index_dicts_wind[i]

        # Extract the init years as arrays
        init_year_array_tas_this = np.array(model_index_dict_tas_this["init_year"])
        member_array_tas_this = np.array(model_index_dict_tas_this["member"])
        lead_array_tas_this = np.array(model_index_dict_tas_this["lead"])

        # do the same for wind speed
        init_year_array_wind_this = np.array(model_index_dict_wind_this["init_year"])
        member_array_wind_this = np.array(model_index_dict_wind_this["member"])
        lead_array_wind_this = np.array(model_index_dict_wind_this["lead"])

        # Zero the missing days her
        missing_days_tas = 0
        missing_days_wind = 0

        # Loop over the rows in this subset df for the model
        for j, (_, row) in tqdm(enumerate(subset_dfs_model[i].iterrows())):
            # Extract the init_year from the df
            init_year_df = int(row["init_year"])
            member_df = int(row["member"])
            lead_df = int(row["lead"])

            # Construct the condition for element wise comparison
            condition_tas = (
                (init_year_array_tas_this == init_year_df)
                & (member_array_tas_this == member_df)
                & (lead_array_tas_this == lead_df)
            )
            condition_wind = (
                (init_year_array_wind_this == init_year_df)
                & (member_array_wind_this == member_df)
                & (lead_array_wind_this == lead_df)
            )

            try:
                # Find the index where this condition is met
                index_this_tas = np.where(condition_tas)[0][0]
            except IndexError:
                print(
                    f"init year {init_year_df}, member {member_df}, lead {lead_df} not found for tas"
                )
                missing_days_tas += 1

            try:
                # Find the index where this condition is met
                index_this_wind = np.where(condition_wind)[0][0]
            except IndexError:
                print(
                    f"init year {init_year_df}, member {member_df}, lead {lead_df} not found for wind"
                )
                missing_days_wind += 1

            # Extract the corresponding value from the subset_arr_this_model
            subset_arr_this_model_tas_index_this = subset_arr_this_model_tas[
                index_this_tas, :, :
            ]
            subset_arr_this_model_wind_index_this = subset_arr_this_model_wind[
                index_this_wind, :, :
            ]

            # Store the value in the subset_arr_this_model_full
            subset_arr_this_model_tas_full[j, :, :] = (
                subset_arr_this_model_tas_index_this
            )
            subset_arr_this_model_wind_full[j, :, :] = (
                subset_arr_this_model_wind_index_this
            )

        # Print the row index
        print(f"Row index: {i}")
        print(f"Number of missing days for tas: {missing_days_tas}")
        print(f"Number of missing days for wind: {missing_days_wind}")
        print(f"Model overall N: {N_model_this}")

        # Take the mean over this
        subset_arr_this_model_tas_mean = np.mean(subset_arr_this_model_tas_full, axis=0)
        subset_arr_this_model_wind_mean = np.mean(
            subset_arr_this_model_wind_full, axis=0
        )

        # Calculate the model anoms
        anoms_this_model_tas = subset_arr_this_model_tas_mean - clim_arrs_model_tas[i]
        anoms_this_model_wind = (
            subset_arr_this_model_wind_mean - clim_arrs_model_wind[i]
        )

        # Apply the europe land mask to the temperature data
        anoms_this_obs_tas = np.ma.masked_where(
            MASK_MATRIX_RESHAPED_LAND == 0, anoms_this_obs_tas
        )
        anoms_this_model_tas = np.ma.masked_where(
            MASK_MATRIX_RESHAPED_LAND == 0, anoms_this_model_tas
        )

        # Plot the obs data on the left
        im_obs_tas = ax_temp_obs.contourf(
            lons,
            lats,
            anoms_this_obs_tas,
            cmap=cmap_tas,
            transform=ccrs.PlateCarree(),
            levels=levels_tas,
            extend="both",
        )

        # Plot the model data on the right
        im_model_tas = ax_temp_model.contourf(
            lons,
            lats,
            anoms_this_model_tas,
            cmap=cmap_tas,
            transform=ccrs.PlateCarree(),
            levels=levels_tas,
            extend="both",
        )

        # Plot the obs data on the left
        im_obs_wind = ax_wind_obs.contourf(
            lons,
            lats,
            anoms_this_obs_wind,
            cmap=cmap_wind,
            transform=ccrs.PlateCarree(),
            levels=levels_wind,
            extend="both",
        )

        # Plot the model data on the right
        im_model_wind = ax_wind_model.contourf(
            lons,
            lats,
            anoms_this_model_wind,
            cmap=cmap_wind,
            transform=ccrs.PlateCarree(),
            levels=levels_wind,
            extend="both",
        )

        # add coastlines to all of these
        ax_temp_obs.coastlines()
        ax_temp_model.coastlines()
        ax_wind_obs.coastlines()
        ax_wind_model.coastlines()

        # Set up the min and max lats
        min_lat = np.min(lats)
        max_lat = np.max(lats)
        min_lon = np.min(lons)
        max_lon = np.max(lons)

        # restrict the domain of the plots
        ax_temp_obs.set_extent([min_lon, max_lon, min_lat, max_lat])
        ax_temp_model.set_extent([min_lon, max_lon, min_lat, max_lat])
        ax_wind_obs.set_extent([min_lon, max_lon, min_lat, max_lat])
        ax_wind_model.set_extent([min_lon, max_lon, min_lat, max_lat])

        # Include a textbox in the rop right for N
        ax_temp_obs.text(
            0.95,
            0.95,
            f"N = {N_obs_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_temp_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_temp_model.text(
            0.95,
            0.95,
            f"N = {N_model_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_temp_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_wind_obs.text(
            0.95,
            0.95,
            f"N = {N_obs_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_wind_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_wind_model.text(
            0.95,
            0.95,
            f"N = {N_model_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_wind_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Add the colorbar
        cbar_temp = fig.colorbar(
            im_obs_tas,
            ax=(ax_temp_obs, ax_temp_model),
            orientation="horizontal",
            pad=0.0,
            shrink=0.8,
        )
        cbar_temp.set_ticks(levels_tas)

        # add the colorbar for wind
        cbar_wind = fig.colorbar(
            im_obs_wind,
            ax=(ax_wind_obs, ax_wind_model),
            orientation="horizontal",
            pad=0.0,
            shrink=0.8,
        )
        cbar_wind.set_ticks(levels_wind)

        # if i is 0
        if i == 0:
            # Set the title for ax0 and ax1 in bold
            ax_temp_obs.set_title("Obs (ERA5)", fontsize=12, fontweight="bold")
            ax_temp_model.set_title("Model (DePreSys)", fontsize=12, fontweight="bold")

        # Set up a textbox in the bottom right
        ax_temp_obs.text(
            0.95,
            0.05,
            names_list[i][0],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_temp_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_temp_model.text(
            0.95,
            0.05,
            names_list[i][1],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_temp_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_wind_obs.text(
            0.95,
            0.05,
            names_list[i][0],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_wind_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_wind_model.text(
            0.95,
            0.05,
            names_list[i][1],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_wind_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Now process the data for the scatter functions
        anoms_scatter_tas_model_this = (
            subset_arr_this_model_tas_full - clim_arrs_model_tas[i]
        )
        anoms_scatter_wind_model_this = (
            subset_arr_this_model_wind_full - clim_arrs_model_wind[i]
        )
        anoms_scatter_tas_obs_this = subset_arr_this_obs_tas - clim_arrs_obs_tas[i]
        anoms_scatter_wind_obs_this = subset_arr_this_obs_wind - clim_arrs_obs_wind[i]

        # assert that the shape of the anoms scatter is the same as the shape of the mask
        assert np.shape(anoms_scatter_tas_model_this) == np.shape(
            anoms_scatter_wind_model_this
        ), "Anoms scatter tas and wind do not match in shape"

        # assert that the obs and model are the same shape
        assert np.shape(anoms_scatter_tas_obs_this) == np.shape(
            anoms_scatter_wind_obs_this
        ), "Anoms scatter tas and wind do not match in shape"

        # Ensure the second and third dimensions match
        assert (
            np.shape(anoms_scatter_tas_model_this)[1:]
            == np.shape(anoms_scatter_tas_obs_this)[1:]
        ), "The second and third dimensions of the arrays do not match"

        # Expand the mask to match the shape of the anoms scatter this
        MASK_MATRIX_NN_RESHAPED_model = np.broadcast_to(
            MASK_MATRIX_NN, anoms_scatter_tas_model_this.shape
        )
        MASK_MATRIX_UK_RESHAPED_model = np.broadcast_to(
            MASK_MATRIX_UK, anoms_scatter_tas_model_this.shape
        )
        MASK_MATRIX_NN_RESHAPED_obs = np.broadcast_to(
            MASK_MATRIX_NN, anoms_scatter_tas_obs_this.shape
        )
        MASK_MATRIX_UK_RESHAPED_obs = np.broadcast_to(
            MASK_MATRIX_UK, anoms_scatter_tas_obs_this.shape
        )

        # Apply the mask to the anoms scatter this
        anoms_scatter_tas_model_this_NN = np.ma.masked_where(
            MASK_MATRIX_NN_RESHAPED_model == 0, anoms_scatter_tas_model_this
        )
        anoms_scatter_tas_model_this_UK = np.ma.masked_where(
            MASK_MATRIX_UK_RESHAPED_model == 0, anoms_scatter_tas_model_this
        )
        anoms_scatter_wind_model_this_NN = np.ma.masked_where(
            MASK_MATRIX_NN_RESHAPED_model == 0, anoms_scatter_wind_model_this
        )
        anoms_scatter_wind_model_this_UK = np.ma.masked_where(
            MASK_MATRIX_UK_RESHAPED_model == 0, anoms_scatter_wind_model_this
        )

        # do the same for the obs
        anoms_scatter_tas_obs_this_NN = np.ma.masked_where(
            MASK_MATRIX_NN_RESHAPED_obs == 0, anoms_scatter_tas_obs_this
        )
        anoms_scatter_tas_obs_this_UK = np.ma.masked_where(
            MASK_MATRIX_UK_RESHAPED_obs == 0, anoms_scatter_tas_obs_this
        )
        anoms_scatter_wind_obs_this_NN = np.ma.masked_where(
            MASK_MATRIX_NN_RESHAPED_obs == 0, anoms_scatter_wind_obs_this
        )
        anoms_scatter_wind_obs_this_UK = np.ma.masked_where(
            MASK_MATRIX_UK_RESHAPED_obs == 0, anoms_scatter_wind_obs_this
        )

        # Set up a list of all of these
        list_anoms_scatter = [
            anoms_scatter_tas_model_this_NN,
            anoms_scatter_tas_model_this_UK,
            anoms_scatter_wind_model_this_NN,
            anoms_scatter_wind_model_this_UK,
            anoms_scatter_tas_obs_this_NN,
            anoms_scatter_tas_obs_this_UK,
            anoms_scatter_wind_obs_this_NN,
            anoms_scatter_wind_obs_this_UK,
        ]

        # loop over and assert that each of them have three dimensions
        for arr in list_anoms_scatter:
            assert (
                len(np.shape(arr)) == 3
            ), "Anoms scatter tas and wind do not match in shape"

        # take the spatial mean of these
        anoms_this_model_nn_tas_mean = np.mean(
            anoms_scatter_tas_model_this_NN, axis=(1, 2)
        )
        anoms_this_model_uk_tas_mean = np.mean(
            anoms_scatter_tas_model_this_UK, axis=(1, 2)
        )
        anoms_this_model_nn_wind_mean = np.mean(
            anoms_scatter_wind_model_this_NN, axis=(1, 2)
        )
        anoms_this_model_uk_wind_mean = np.mean(
            anoms_scatter_wind_model_this_UK, axis=(1, 2)
        )
        anoms_this_obs_nn_tas_mean = np.mean(anoms_scatter_tas_obs_this_NN, axis=(1, 2))
        anoms_this_obs_uk_tas_mean = np.mean(anoms_scatter_tas_obs_this_UK, axis=(1, 2))
        anoms_this_obs_nn_wind_mean = np.mean(
            anoms_scatter_wind_obs_this_NN, axis=(1, 2)
        )
        anoms_this_obs_uk_wind_mean = np.mean(
            anoms_scatter_wind_obs_this_UK, axis=(1, 2)
        )

        # calculate the correlation between the two sets of points
        corr_tas_model, _ = pearsonr(
            anoms_this_model_nn_tas_mean, anoms_this_model_uk_tas_mean
        )
        if i == 2:
            corr_tas_obs = 0.0
        else:
            corr_tas_obs, _ = pearsonr(
                anoms_this_obs_nn_tas_mean, anoms_this_obs_uk_tas_mean
            )
        corr_wind_model, _ = pearsonr(
            anoms_this_model_nn_wind_mean, anoms_this_model_uk_wind_mean
        )
        if i == 2:
            corr_wind_obs = 0.0
        else:
            corr_wind_obs, _ = pearsonr(
                anoms_this_obs_nn_wind_mean, anoms_this_obs_uk_wind_mean
            )

        # Plot the scatter plot for tas model as grey circles
        ax_temp_scatter.scatter(
            anoms_this_model_nn_tas_mean,
            anoms_this_model_uk_tas_mean,
            color="grey",
            s=10,
            alpha=0.5,
            label=f"Model (r={corr_tas_model:.2f})",
        )

        # Plot the scatter plot for tas obs as black crosses
        ax_temp_scatter.scatter(
            anoms_this_obs_nn_tas_mean,
            anoms_this_obs_uk_tas_mean,
            color="black",
            s=10,
            marker="x",
            label=f"Obs (r={corr_tas_obs:.2f})",
        )

        # Plot the scatter plot for wind model as grey circles
        ax_wind_scatter.scatter(
            anoms_this_model_nn_wind_mean,
            anoms_this_model_uk_wind_mean,
            color="grey",
            s=10,
            alpha=0.5,
            label=f"Model (r={corr_wind_model:.2f})",
        )

        # Plot the scatter plot for wind obs as black crosses
        ax_wind_scatter.scatter(
            anoms_this_obs_nn_wind_mean,
            anoms_this_obs_uk_wind_mean,
            color="black",
            s=10,
            marker="x",
            label=f"Obs (r={corr_wind_obs:.2f})",
        )

        # set the x and y limits for the scatter plots
        xlims_temp = (-17, 7)
        ylims_temp = (-17, 7)

        # Set the xlims and ylims for the scatter plots
        xlims_wind = (-4, 4)
        ylims_wind = (-4, 4)

        # Set the x and y limits for the scatter plots
        ax_temp_scatter.set_xlim(xlims_temp)
        ax_temp_scatter.set_ylim(ylims_temp)
        ax_wind_scatter.set_xlim(xlims_wind)
        ax_wind_scatter.set_ylim(ylims_wind)

        # Set the x and y labels for the scatter plots
        ax_temp_scatter.set_xlabel("NN temperature anomaly (°C)", fontsize=12)
        ax_temp_scatter.set_ylabel("UK temperature anomaly (°C)", fontsize=12)
        ax_wind_scatter.set_xlabel("NN wind anomaly (m/s)", fontsize=12)
        ax_wind_scatter.set_ylabel("UK wind anomaly (m/s)", fontsize=12)

        # Set up titles for the scatter plots
        ax_temp_scatter.set_title("UK vs. NN.", fontsize=12)
        ax_wind_scatter.set_title("UK vs. NN.", fontsize=12)

        # include a legend in the top left
        ax_temp_scatter.legend(
            loc="upper left",
            fontsize=12,
            markerscale=2,
            frameon=False,
        )
        ax_wind_scatter.legend(
            loc="upper left",
            fontsize=12,
            markerscale=2,
            frameon=False,
        )

    # minimise the vertical spacing between subplots
    fig.subplots_adjust(hspace=0.001)

    # # make sure the aspect ratios are all equal
    # for ax in all_axes:
    #     ax.set_aspect("equal", adjustable="box")

    return None


# Define a functoin to plot the tas wind composites
def plot_tas_composites(
    subset_dfs_obs: List[pd.DataFrame],
    subset_dfs_model: List[pd.DataFrame],
    subset_arrs_obs_tas: List[np.ndarray],
    subset_arrs_model_tas: List[np.ndarray],
    clim_arrs_obs_tas: List[np.ndarray],
    clim_arrs_model_tas: List[np.ndarray],
    dates_lists_obs_tas: List[List[cftime.DatetimeProlepticGregorian]],
    model_index_dicts_tas: List[Dict[str, np.ndarray]],
    lats_path: str,
    lons_path: str,
    suptitle: str = None,
    figsize: Tuple[int, int] = (8, 9),
):
    """
    Plots the tas and wind composites for the given variables.

    Args:
    =====

        subset_dfs_obs (List[pd.DataFrame]): The list of subset dataframes for observations.
        subset_dfs_model (List[pd.DataFrame]): The list of subset dataframes for the model.
        subset_arrs_obs_tas (List[np.ndarray]): The list of tas subset arrays for observations.
        subset_arrs_model_tas (List[np.ndarray]): The list of tas subset arrays for the model.
        subset_arrs_obs_wind (List[np.ndarray]): The list of wind subset arrays for observations.
        subset_arrs_model_wind (List[np.ndarray]): The list of wind subset arrays for the model.
        clim_arrs_obs_tas (List[np.ndarray]): The list of climatology arrays for observations.
        clim_arrs_model_tas (List[np.ndarray]): The list of climatology arrays for the model.
        clim_arrs_obs_wind (List[np.ndarray]): The list of climatology arrays for observations.
        dates_lists_obs (List[List[cftime.DatetimeProlepticGregorian]]): The list of dates lists for observations.
        model_index_dicts (List[Dict[str, np.ndarray]]): The list of model index dictionaries.
        lats_path (str): The path to the latitude file.
        lons_path (str): The path to the longitude file.
        suptitle (str): The suptitle for the plot.
        figsize (Tuple[int, int]): The figure size.

    Returns:
    ========

        None

    """

    # Hardcoe the cmap and levels for tas
    cmap_tas = "bwr"
    levels_tas = np.array(
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

    # Hardcode the cmap and levels for wind
    cmap_wind = "PRGn"
    levels_wind = np.array(
        [
            -5,
            -4,
            -3,
            -2,
            -1,
            1,
            2,
            3,
            4,
            5,
        ]
    )

    # hard code the nearest neighbour countries
    nearest_neighbour_countries = [
        "Ireland",
        "Germany",
        "France",
        "Netherlands",
        "Belgium",
        "Denmark",
    ]
    uk_names = ["United Kingdom"]

    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # Set up the countries shapefile
    countries_shp = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )

    # Set up the x and y
    x, y = lons, lats

    # Set up a landmask for the temperature data
    MASK_MATRIX_TMP = np.zeros((len(lats), len(lons)))
    country_shapely = []
    for country in shpreader.Reader(countries_shp).records():
        country_shapely.append(country.geometry)

    # Loop over the latitude and longitude points
    for l in range(len(lats)):
        for j in range(len(lons)):
            point = shapely.geometry.Point(lons[j], lats[l])
            for country in country_shapely:
                if country.contains(point):
                    MASK_MATRIX_TMP[l, j] = 1.0

    # Reshape the mask to match the shape of the data
    MASK_MATRIX_RESHAPED_LAND = MASK_MATRIX_TMP

    # Set up a mask for the countries
    nn_countries_geom = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME"] in nearest_neighbour_countries:
            nn_countries_geom.append(country.geometry)

    # Set up a mask for the UK
    uk_country_geom = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME"] in uk_names:
            uk_country_geom.append(country.geometry)

    # Set up the mask matrix for the nearest neighbour countries
    MASK_MATRIX_NN = np.zeros((len(lats), len(lons)))
    MASK_MATRIX_UK = np.zeros((len(lats), len(lons)))

    # Loop over the latitude and longitude points
    for l in range(len(lats)):
        for j in range(len(lons)):
            point = shapely.geometry.Point(lons[j], lats[l])
            for country in nn_countries_geom:
                if country.contains(point):
                    MASK_MATRIX_NN[l, j] = 1.0
            for country in uk_country_geom:
                if country.contains(point):
                    MASK_MATRIX_UK[l, j] = 1.0

    # plt.rcParams['figure.constrained_layout.use'] = False
    # Set up the figure
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(
        3, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1]
    )
    # # Set up the gridspec
    # gs.update(wspace=0.001, hspace=0.001)

    # Grey dots subplots
    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())  # Row 0, Col 0
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())  # Row 0, Col 1
    ax2 = fig.add_subplot(gs[0, 2])  # Row 0, Col 2

    # Yellow dots sublots
    ax6 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())  # Row 2, Col 0
    ax7 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())  # Row 2, Col 1
    ax8 = fig.add_subplot(gs[1, 2], sharex=ax2)  # Row 2, Col 2

    # Red dots subplots
    ax12 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())  # Row 4, Col 0
    ax13 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())  # Row 4, Col 1
    ax14 = fig.add_subplot(gs[2, 2], sharex=ax2)  # Row 4, Col 2

    # # # Set aspect ratio to square for all scatter plot axes
    # scatter_axes = [ax2, ax8, ax14]  # Replace with the axes where scatter plots are drawn
    # for ax in scatter_axes:
    #     ax.set_aspect('equal')

    all_axes = [
        ax0,
        ax1,
        ax2,
        ax6,
        ax7,
        ax8,
        ax12,
        ax13,
        ax14,
    ]

    for i in range(len(all_axes)):
        ax_this = all_axes[i]
        plt.axis("on")
        if i not in [2, 5, 8]:
            ax_this.set_xticks([])
            ax_this.set_yticks([])
        ax_this.set_aspect("equal", adjustable="box")

    # Set up the cmap_axes
    # cmap_axes = [
    #     ax0,
    #     ax1,

    # # set a tight layout for the gridspec
    # fig.tight_layout()

    # Set up the axes groups
    axes_groups = [
        (ax0, ax1, ax2),
        (ax6, ax7, ax8),
        (ax12, ax13, ax14),
    ]

    # Set up the names
    names_list = [
        ("Block max days", "Block max days"),
        ("Extreme days", "Extreme days"),
        ("21-12-2010", "Unseen days"),
    ]

    # set the aspect ratio for the scatter plots
    for ax in [ax2, ax8, ax14]:
        ax.set_aspect("equal", adjustable="box")

    for i, (axes_group) in enumerate(axes_groups):
        # Set the axes up for this
        ax_temp_obs = axes_group[0]
        ax_temp_model = axes_group[1]
        ax_temp_scatter = axes_group[2]

        # ----------------------------
        # Detrend the model tas data
        # ----------------------------

        # extract the subset array this for tas
        subset_arr_this_tas = subset_arrs_model_tas[i]

        # Extract the unique effective dec years
        # from the index dict this
        index_dict_this_tas = model_index_dicts_tas[i]

        effective_dec_years_arr_tas = np.array(
            index_dict_this_tas["effective_dec_year"]
        )
        unique_effective_dec_years_tas = np.unique(effective_dec_years_arr_tas)

        # Set up a new array to append to
        subset_arr_this_detrended = np.zeros(
            (len(unique_effective_dec_years_tas), len(lats), len(lons))
        )

        # Loop over the unique effective dec years
        for j, effective_dec_year in enumerate(unique_effective_dec_years_tas):
            # Find the index of this effective dec year in the index dict this
            index_this = np.where(effective_dec_years_arr_tas == effective_dec_year)[0]

            # print the index this
            # print(f"Index this: {index_this}")

            # # Extract the subset arr this for this index
            subset_arr_this_detrended[j, :, :] = np.mean(
                subset_arr_this_tas[index_this, :, :], axis=0
            )

        # loop over the lats and lons
        for j in range(subset_arr_this_detrended.shape[1]):
            for k in range(subset_arr_this_detrended.shape[2]):
                # Calculate the mean trend line
                slope_T_this, intercept_T_this, _, _, _ = linregress(
                    unique_effective_dec_years_tas,
                    subset_arr_this_detrended[:, j, k],
                )

                # Calculate the trend line this
                trend_line_this = (
                    slope_T_this * unique_effective_dec_years_tas + intercept_T_this
                )

                # Find the final point on the trend line
                final_point_this = trend_line_this[-1]

                # Loop over the unique effective dec years
                for l in range(len(unique_effective_dec_years_tas)):
                    # Find the indcides of the effective dec years
                    index_this_l = np.where(
                        effective_dec_years_arr_tas == unique_effective_dec_years_tas[l]
                    )[0]

                    # Loop over the indices
                    for m in index_this_l:
                        # Detrend the data
                        subset_arr_this_tas[m, j, k] = (
                            final_point_this
                            - trend_line_this[l]
                            + subset_arr_this_tas[m, j, k]
                        )

        # Set up the subset dates to select the obs for
        subset_dates_obs_cf_this = []

        # Extract the subset dates this
        subset_dates_obs_this = subset_dfs_obs[i]["time"].values

        # Format these as datetimes
        subset_dates_obs_dt_this = [
            datetime.strptime(date, "%Y-%m-%d") for date in subset_dates_obs_this
        ]

        # Format the subset dates to extract
        for date in subset_dates_obs_dt_this:
            date_this_cf = cftime.DatetimeProlepticGregorian(
                date.year,
                date.month,
                date.day,
                hour=0,
                calendar="proleptic_gregorian",
            )

            subset_dates_obs_cf_this.append(date_this_cf)

        # Set up an empty list for the indices dates obs this
        indices_dates_obs_this_tas = []

        # Loop over the dates in subset dtes obs cf
        for date in subset_dates_obs_cf_this:
            try:
                # Find the index of the date in the dates list
                index_this_tas = np.where(dates_lists_obs_tas[i] == date)[0][0]
            except IndexError:
                print(f"Date {date} not found in dates list obs tas for index {i}")
                print(f"Dates list obs tas: {dates_lists_obs_tas[i]}")
            indices_dates_obs_this_tas.append(index_this_tas)

        # Set up the subset arr this for the obs
        subset_arr_this_obs_tas = subset_arrs_obs_tas[i]

        # Apply these indices to the subset data
        subset_arr_this_obs_tas = subset_arr_this_obs_tas[
            indices_dates_obs_this_tas, :, :
        ]

        # Get the N for the obs this
        N_obs_this = np.shape(subset_arr_this_obs_tas)[0]

        # Take the mean over this
        subset_arr_this_obs_tas_mean = np.mean(subset_arr_this_obs_tas, axis=0)

        # Calculate the obs anoms
        anoms_this_obs_tas = subset_arr_this_obs_tas_mean - clim_arrs_obs_tas[i]

        # Set up the subset arr this for the model
        subset_arr_this_model_tas = subset_arr_this_tas

        # Set up the subset arr this model full
        subset_arr_this_model_tas_full = np.zeros(
            (len(subset_dfs_model[i]), len(lats), len(lons))
        )
        subset_arr_this_model_wind_full = np.zeros(
            (len(subset_dfs_model[i]), len(lats), len(lons))
        )

        # Set up the N for model this
        N_model_this = np.shape(subset_arr_this_model_tas_full)[0]

        # Extract the index dict for the model this
        model_index_dict_tas_this = model_index_dicts_tas[i]

        # Extract the init years as arrays
        init_year_array_tas_this = np.array(model_index_dict_tas_this["init_year"])
        member_array_tas_this = np.array(model_index_dict_tas_this["member"])
        lead_array_tas_this = np.array(model_index_dict_tas_this["lead"])

        # Zero the missing days her
        missing_days_tas = 0

        # Loop over the rows in this subset df for the model
        for j, (_, row) in tqdm(enumerate(subset_dfs_model[i].iterrows())):
            # Extract the init_year from the df
            init_year_df = int(row["init_year"])
            member_df = int(row["member"])
            lead_df = int(row["lead"])

            # Construct the condition for element wise comparison
            condition_tas = (
                (init_year_array_tas_this == init_year_df)
                & (member_array_tas_this == member_df)
                & (lead_array_tas_this == lead_df)
            )

            try:
                # Find the index where this condition is met
                index_this_tas = np.where(condition_tas)[0][0]
            except IndexError:
                print(
                    f"init year {init_year_df}, member {member_df}, lead {lead_df} not found for tas"
                )
                missing_days_tas += 1

            # Extract the corresponding value from the subset_arr_this_model
            subset_arr_this_model_tas_index_this = subset_arr_this_model_tas[
                index_this_tas, :, :
            ]

            # Store the value in the subset_arr_this_model_full
            subset_arr_this_model_tas_full[j, :, :] = (
                subset_arr_this_model_tas_index_this
            )

        # Print the row index
        print(f"Row index: {i}")
        print(f"Number of missing days for tas: {missing_days_tas}")
        print(f"Model overall N: {N_model_this}")

        # Take the mean over this
        subset_arr_this_model_tas_mean = np.mean(subset_arr_this_model_tas_full, axis=0)
        subset_arr_this_model_wind_mean = np.mean(
            subset_arr_this_model_wind_full, axis=0
        )

        # Calculate the model anoms
        anoms_this_model_tas = subset_arr_this_model_tas_mean - clim_arrs_model_tas[i]

        # Apply the europe land mask to the temperature data
        anoms_this_obs_tas = np.ma.masked_where(
            MASK_MATRIX_RESHAPED_LAND == 0, anoms_this_obs_tas
        )
        anoms_this_model_tas = np.ma.masked_where(
            MASK_MATRIX_RESHAPED_LAND == 0, anoms_this_model_tas
        )

        # Plot the obs data on the left
        im_obs_tas = ax_temp_obs.contourf(
            lons,
            lats,
            anoms_this_obs_tas,
            cmap=cmap_tas,
            transform=ccrs.PlateCarree(),
            levels=levels_tas,
            extend="both",
        )

        # Plot the model data on the right
        im_model_tas = ax_temp_model.contourf(
            lons,
            lats,
            anoms_this_model_tas,
            cmap=cmap_tas,
            transform=ccrs.PlateCarree(),
            levels=levels_tas,
            extend="both",
        )

        # add coastlines to all of these
        ax_temp_obs.coastlines()
        ax_temp_model.coastlines()

        # Set up the min and max lats
        min_lat = np.min(lats)
        max_lat = np.max(lats)
        min_lon = np.min(lons)
        max_lon = np.max(lons)

        # restrict the domain of the plots
        ax_temp_obs.set_extent([min_lon, max_lon, min_lat, max_lat])
        ax_temp_model.set_extent([min_lon, max_lon, min_lat, max_lat])

        # Include a textbox in the rop right for N
        ax_temp_obs.text(
            0.95,
            0.95,
            f"N = {N_obs_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_temp_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_temp_model.text(
            0.95,
            0.95,
            f"N = {N_model_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_temp_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # if i is 2
        if i == 2:
            # Add the colorbar
            cbar_temp = fig.colorbar(
                im_obs_tas,
                ax=(ax_temp_obs, ax_temp_model),
                orientation="horizontal",
                pad=0.0,
                shrink=0.8,
            )
            cbar_temp.set_ticks(levels_tas)

        # if i is 0
        if i == 0:
            # Set the title for ax0 and ax1 in bold
            ax_temp_obs.set_title("Obs (ERA5)", fontsize=12, fontweight="bold")
            ax_temp_model.set_title("Model (DePreSys)", fontsize=12, fontweight="bold")

        # Set up a textbox in the bottom right
        ax_temp_obs.text(
            0.95,
            0.05,
            names_list[i][0],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_temp_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_temp_model.text(
            0.95,
            0.05,
            names_list[i][1],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_temp_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Now process the data for the scatter functions
        anoms_scatter_tas_model_this = (
            subset_arr_this_model_tas_full - clim_arrs_model_tas[i]
        )
        anoms_scatter_tas_obs_this = subset_arr_this_obs_tas - clim_arrs_obs_tas[i]

        # Ensure the second and third dimensions match
        assert (
            np.shape(anoms_scatter_tas_model_this)[1:]
            == np.shape(anoms_scatter_tas_obs_this)[1:]
        ), "The second and third dimensions of the arrays do not match"

        # Expand the mask to match the shape of the anoms scatter this
        MASK_MATRIX_NN_RESHAPED_model = np.broadcast_to(
            MASK_MATRIX_NN, anoms_scatter_tas_model_this.shape
        )
        MASK_MATRIX_UK_RESHAPED_model = np.broadcast_to(
            MASK_MATRIX_UK, anoms_scatter_tas_model_this.shape
        )
        MASK_MATRIX_NN_RESHAPED_obs = np.broadcast_to(
            MASK_MATRIX_NN, anoms_scatter_tas_obs_this.shape
        )
        MASK_MATRIX_UK_RESHAPED_obs = np.broadcast_to(
            MASK_MATRIX_UK, anoms_scatter_tas_obs_this.shape
        )

        # Apply the mask to the anoms scatter this
        anoms_scatter_tas_model_this_NN = np.ma.masked_where(
            MASK_MATRIX_NN_RESHAPED_model == 0, anoms_scatter_tas_model_this
        )
        anoms_scatter_tas_model_this_UK = np.ma.masked_where(
            MASK_MATRIX_UK_RESHAPED_model == 0, anoms_scatter_tas_model_this
        )

        # do the same for the obs
        anoms_scatter_tas_obs_this_NN = np.ma.masked_where(
            MASK_MATRIX_NN_RESHAPED_obs == 0, anoms_scatter_tas_obs_this
        )
        anoms_scatter_tas_obs_this_UK = np.ma.masked_where(
            MASK_MATRIX_UK_RESHAPED_obs == 0, anoms_scatter_tas_obs_this
        )

        # Set up a list of all of these
        list_anoms_scatter = [
            anoms_scatter_tas_model_this_NN,
            anoms_scatter_tas_model_this_UK,
            anoms_scatter_tas_obs_this_NN,
            anoms_scatter_tas_obs_this_UK,
        ]

        # loop over and assert that each of them have three dimensions
        for arr in list_anoms_scatter:
            assert (
                len(np.shape(arr)) == 3
            ), "Anoms scatter tas and wind do not match in shape"

        # take the spatial mean of these
        anoms_this_model_nn_tas_mean = np.mean(
            anoms_scatter_tas_model_this_NN, axis=(1, 2)
        )
        anoms_this_model_uk_tas_mean = np.mean(
            anoms_scatter_tas_model_this_UK, axis=(1, 2)
        )

        anoms_this_obs_nn_tas_mean = np.mean(anoms_scatter_tas_obs_this_NN, axis=(1, 2))
        anoms_this_obs_uk_tas_mean = np.mean(anoms_scatter_tas_obs_this_UK, axis=(1, 2))

        # calculate the correlation between the two sets of points
        corr_tas_model, _ = pearsonr(
            anoms_this_model_nn_tas_mean, anoms_this_model_uk_tas_mean
        )
        if i == 2:
            corr_tas_obs = 0.0
        else:
            corr_tas_obs, _ = pearsonr(
                anoms_this_obs_nn_tas_mean, anoms_this_obs_uk_tas_mean
            )

        # Plot the scatter plot for tas model as grey circles
        ax_temp_scatter.scatter(
            anoms_this_model_nn_tas_mean,
            anoms_this_model_uk_tas_mean,
            color="grey",
            s=10,
            alpha=0.5,
            label=f"Model (r={corr_tas_model:.2f})",
        )

        # Plot the scatter plot for tas obs as black crosses
        ax_temp_scatter.scatter(
            anoms_this_obs_nn_tas_mean,
            anoms_this_obs_uk_tas_mean,
            color="black",
            s=10,
            marker="x",
            label=f"Obs (r={corr_tas_obs:.2f})",
        )

        # set the x and y limits for the scatter plots
        xlims_temp = (-17, 7)
        ylims_temp = (-17, 7)

        # Set the x and y limits for the scatter plots
        ax_temp_scatter.set_xlim(xlims_temp)
        ax_temp_scatter.set_ylim(ylims_temp)

        if i == 2:
            # Set the x and y labels for the scatter plots
            ax_temp_scatter.set_xlabel("NN temperature anomaly (°C)", fontsize=12)

        ax_temp_scatter.set_ylabel("UK temperature anomaly (°C)", fontsize=12)

        if i == 0:
            # Set up titles for the scatter plots
            ax_temp_scatter.set_title("UK vs. NN.", fontsize=12)

        # include a legend in the top left
        ax_temp_scatter.legend(
            loc="upper left",
            fontsize=12,
            markerscale=2,
            frameon=False,
        )

    # fix the aspect ratio for all of the plots
    for ax in all_axes:
        ax.set_aspect("equal", adjustable="box")

    # # remove the xtick lables from specific axes
    # for ax in [ax0, ax6, ax12]:
    #     ax.set_xticklabels([])

    # ax2.set_xticklabels([])
    # ax8.set_xticklabels([])

    # # minimise the vertical spacing between subplots
    # fig.subplots_adjust(hspace=0.001)

    # # make sure the aspect ratios are all equal
    # for ax in all_axes:
    #     ax.set_aspect("equal", adjustable="box")

    return None


# Define a functoin to plot the tas wind composites
def plot_wind_composites(
    subset_dfs_obs: List[pd.DataFrame],
    subset_dfs_model: List[pd.DataFrame],
    subset_arrs_obs_wind: List[np.ndarray],
    subset_arrs_model_wind: List[np.ndarray],
    clim_arrs_obs_wind: List[np.ndarray],
    clim_arrs_model_wind: List[np.ndarray],
    dates_lists_obs_wind: List[List[cftime.DatetimeProlepticGregorian]],
    model_index_dicts_wind: List[Dict[str, np.ndarray]],
    lats_path: str,
    lons_path: str,
    suptitle: str = None,
    figsize: Tuple[int, int] = (8, 9),
):
    """
    Plots the tas and wind composites for the given variables.

    Args:
    =====

        subset_dfs_obs (List[pd.DataFrame]): The list of subset dataframes for observations.
        subset_dfs_model (List[pd.DataFrame]): The list of subset dataframes for the model.
        subset_arrs_obs_tas (List[np.ndarray]): The list of tas subset arrays for observations.
        subset_arrs_model_tas (List[np.ndarray]): The list of tas subset arrays for the model.
        subset_arrs_obs_wind (List[np.ndarray]): The list of wind subset arrays for observations.
        subset_arrs_model_wind (List[np.ndarray]): The list of wind subset arrays for the model.
        clim_arrs_obs_tas (List[np.ndarray]): The list of climatology arrays for observations.
        clim_arrs_model_tas (List[np.ndarray]): The list of climatology arrays for the model.
        clim_arrs_obs_wind (List[np.ndarray]): The list of climatology arrays for observations.
        dates_lists_obs (List[List[cftime.DatetimeProlepticGregorian]]): The list of dates lists for observations.
        model_index_dicts (List[Dict[str, np.ndarray]]): The list of model index dictionaries.
        lats_path (str): The path to the latitude file.
        lons_path (str): The path to the longitude file.
        suptitle (str): The suptitle for the plot.
        figsize (Tuple[int, int]): The figure size.

    Returns:
    ========

        None

    """

    # Hardcoe the cmap and levels for tas
    cmap_tas = "bwr"
    levels_tas = np.array(
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

    # Hardcode the cmap and levels for wind
    cmap_wind = "PRGn"
    levels_wind = np.array(
        [
            -5,
            -4,
            -3,
            -2,
            -1,
            1,
            2,
            3,
            4,
            5,
        ]
    )

    # hard code the nearest neighbour countries
    nearest_neighbour_countries = [
        "Ireland",
        "Germany",
        "France",
        "Netherlands",
        "Belgium",
        "Denmark",
    ]
    uk_names = ["United Kingdom"]

    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # Set up the countries shapefile
    countries_shp = shpreader.natural_earth(
        resolution="10m",
        category="cultural",
        name="admin_0_countries",
    )

    # Set up the x and y
    x, y = lons, lats

    # Set up a mask for the countries
    nn_countries_geom = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME"] in nearest_neighbour_countries:
            nn_countries_geom.append(country.geometry)

    # Set up a mask for the UK
    uk_country_geom = []
    for country in shpreader.Reader(countries_shp).records():
        if country.attributes["NAME"] in uk_names:
            uk_country_geom.append(country.geometry)

    # Set up the mask matrix for the nearest neighbour countries
    MASK_MATRIX_NN = np.zeros((len(lats), len(lons)))
    MASK_MATRIX_UK = np.zeros((len(lats), len(lons)))

    # Loop over the latitude and longitude points
    for l in range(len(lats)):
        for j in range(len(lons)):
            point = shapely.geometry.Point(lons[j], lats[l])
            for country in nn_countries_geom:
                if country.contains(point):
                    MASK_MATRIX_NN[l, j] = 1.0
            for country in uk_country_geom:
                if country.contains(point):
                    MASK_MATRIX_UK[l, j] = 1.0

    # plt.rcParams['figure.constrained_layout.use'] = False
    # Set up the figure
    fig = plt.figure(figsize=figsize, layout="constrained")
    gs = fig.add_gridspec(
        3, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1]
    )
    # # Set up the gridspec
    # gs.update(wspace=0.001, hspace=0.001)

    # Grey dots subplots
    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())  # Row 0, Col 0
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())  # Row 0, Col 1
    ax2 = fig.add_subplot(gs[0, 2])  # Row 0, Col 2

    # Yellow dots sublots
    ax6 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())  # Row 2, Col 0
    ax7 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())  # Row 2, Col 1
    ax8 = fig.add_subplot(gs[1, 2], sharex=ax2)  # Row 2, Col 2

    # Red dots subplots
    ax12 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())  # Row 4, Col 0
    ax13 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())  # Row 4, Col 1
    ax14 = fig.add_subplot(gs[2, 2], sharex=ax2)  # Row 4, Col 2

    # # # Set aspect ratio to square for all scatter plot axes
    # scatter_axes = [ax2, ax8, ax14]  # Replace with the axes where scatter plots are drawn
    # for ax in scatter_axes:
    #     ax.set_aspect('equal')

    all_axes = [
        ax0,
        ax1,
        ax2,
        ax6,
        ax7,
        ax8,
        ax12,
        ax13,
        ax14,
    ]

    for i in range(len(all_axes)):
        ax_this = all_axes[i]
        plt.axis("on")
        if i not in [2, 5, 8]:
            ax_this.set_xticks([])
            ax_this.set_yticks([])
        ax_this.set_aspect("equal", adjustable="box")

    # Set up the cmap_axes
    # cmap_axes = [
    #     ax0,
    #     ax1,

    # # set a tight layout for the gridspec
    # fig.tight_layout()

    # Set up the axes groups
    axes_groups = [
        (ax0, ax1, ax2),
        (ax6, ax7, ax8),
        (ax12, ax13, ax14),
    ]

    # Set up the names
    names_list = [
        ("Block max days", "Block max days"),
        ("Extreme days", "Extreme days"),
        ("21-12-2010", "Unseen days"),
    ]

    # set the aspect ratio for the scatter plots
    for ax in [ax2, ax8, ax14]:
        ax.set_aspect("equal", adjustable="box")

    for i, (axes_group) in enumerate(axes_groups):
        # Set the axes up for this
        ax_wind_obs = axes_group[0]
        ax_wind_model = axes_group[1]
        ax_wind_scatter = axes_group[2]

        # Set up the subset dates to select the obs for
        subset_dates_obs_cf_this = []

        # Extract the subset dates this
        subset_dates_obs_this = subset_dfs_obs[i]["time"].values

        # Format these as datetimes
        subset_dates_obs_dt_this = [
            datetime.strptime(date, "%Y-%m-%d") for date in subset_dates_obs_this
        ]

        # Format the subset dates to extract
        for date in subset_dates_obs_dt_this:
            date_this_cf = cftime.DatetimeProlepticGregorian(
                date.year,
                date.month,
                date.day,
                hour=0,
                calendar="proleptic_gregorian",
            )

            subset_dates_obs_cf_this.append(date_this_cf)

        # Set up an empty list for the indices dates obs this
        indices_dates_obs_this_wind = []

        # Loop over the dates in subset dtes obs cf
        for date in subset_dates_obs_cf_this:

            # Find the index of the date in the dates list
            index_this_wind = np.where(dates_lists_obs_wind[i] == date)[0][0]
            indices_dates_obs_this_wind.append(index_this_wind)

        # Set up the subset arr this for the obs
        subset_arr_this_obs_wind = subset_arrs_obs_wind[i]

        # Apply these indices to the subset data
        subset_arr_this_obs_wind = subset_arr_this_obs_wind[
            indices_dates_obs_this_wind, :, :
        ]

        # Get the N for the obs this
        N_obs_this = np.shape(subset_arr_this_obs_wind)[0]

        # Take the mean over this
        subset_arr_this_obs_wind_mean = np.mean(subset_arr_this_obs_wind, axis=0)

        # Calculate the obs anoms
        anoms_this_obs_wind = subset_arr_this_obs_wind_mean - clim_arrs_obs_wind[i]

        # Set up the subset arr this for the model
        subset_arr_this_model_wind = subset_arrs_model_wind[i]

        # Set up the subset arr this model full
        subset_arr_this_model_wind_full = np.zeros(
            (len(subset_dfs_model[i]), len(lats), len(lons))
        )

        # Set up the N for model this
        N_model_this = np.shape(subset_arr_this_model_wind_full)[0]

        # Extract the index dict for the model this
        model_index_dict_wind_this = model_index_dicts_wind[i]

        # do the same for wind speed
        init_year_array_wind_this = np.array(model_index_dict_wind_this["init_year"])
        member_array_wind_this = np.array(model_index_dict_wind_this["member"])
        lead_array_wind_this = np.array(model_index_dict_wind_this["lead"])

        # Zero the missing days her
        missing_days_wind = 0

        # Loop over the rows in this subset df for the model
        for j, (_, row) in tqdm(enumerate(subset_dfs_model[i].iterrows())):
            # Extract the init_year from the df
            init_year_df = int(row["init_year"])
            member_df = int(row["member"])
            lead_df = int(row["lead"])

            # Construct the condition for element wise comparison
            condition_wind = (
                (init_year_array_wind_this == init_year_df)
                & (member_array_wind_this == member_df)
                & (lead_array_wind_this == lead_df)
            )

            try:
                # Find the index where this condition is met
                index_this_wind = np.where(condition_wind)[0][0]
            except IndexError:
                print(
                    f"init year {init_year_df}, member {member_df}, lead {lead_df} not found for wind"
                )
                missing_days_wind += 1

            # Extract the corresponding value from the subset_arr_this_model
            subset_arr_this_model_wind_index_this = subset_arr_this_model_wind[
                index_this_wind, :, :
            ]

            # Store the value in the subset_arr_this_model_full
            subset_arr_this_model_wind_full[j, :, :] = (
                subset_arr_this_model_wind_index_this
            )

        # Print the row index
        print(f"Row index: {i}")
        print(f"Number of missing days for wind: {missing_days_wind}")
        print(f"Model overall N: {N_model_this}")

        # Take the mean over this
        subset_arr_this_model_wind_mean = np.mean(
            subset_arr_this_model_wind_full, axis=0
        )

        # Calculate the model anoms
        anoms_this_model_wind = (
            subset_arr_this_model_wind_mean - clim_arrs_model_wind[i]
        )

        # Plot the obs data on the left
        im_obs_wind = ax_wind_obs.contourf(
            lons,
            lats,
            anoms_this_obs_wind,
            cmap=cmap_wind,
            transform=ccrs.PlateCarree(),
            levels=levels_wind,
            extend="both",
        )

        # Plot the model data on the right
        im_model_wind = ax_wind_model.contourf(
            lons,
            lats,
            anoms_this_model_wind,
            cmap=cmap_wind,
            transform=ccrs.PlateCarree(),
            levels=levels_wind,
            extend="both",
        )

        # add coastlines to all of these
        ax_wind_obs.coastlines()
        ax_wind_model.coastlines()

        # Set up the min and max lats
        min_lat = np.min(lats)
        max_lat = np.max(lats)
        min_lon = np.min(lons)
        max_lon = np.max(lons)

        # restrict the domain of the plots
        ax_wind_obs.set_extent([min_lon, max_lon, min_lat, max_lat])
        ax_wind_model.set_extent([min_lon, max_lon, min_lat, max_lat])

        # Include a textbox in the rop right for N
        ax_wind_obs.text(
            0.95,
            0.95,
            f"N = {N_obs_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_wind_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_wind_model.text(
            0.95,
            0.95,
            f"N = {N_model_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_wind_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # if i is 2
        if i == 2:
            # add the colorbar for wind
            cbar_wind = fig.colorbar(
                im_obs_wind,
                ax=(ax_wind_obs, ax_wind_model),
                orientation="horizontal",
                pad=0.0,
                shrink=0.8,
            )
            cbar_wind.set_ticks(levels_wind)

        # if i is 0
        if i == 0:
            # Set the title for ax0 and ax1 in bold
            ax_wind_obs.set_title("Obs (ERA5)", fontsize=12, fontweight="bold")
            ax_wind_model.set_title("Model (DePreSys)", fontsize=12, fontweight="bold")

        # Set up a textbox in the bottom right
        ax_wind_obs.text(
            0.95,
            0.05,
            names_list[i][0],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_wind_obs.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )
        ax_wind_model.text(
            0.95,
            0.05,
            names_list[i][1],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_wind_model.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Now process the data for the scatter functions
        anoms_scatter_wind_model_this = (
            subset_arr_this_model_wind_full - clim_arrs_model_wind[i]
        )
        anoms_scatter_wind_obs_this = subset_arr_this_obs_wind - clim_arrs_obs_wind[i]

        # assert that the shape of the anoms scatter is the same as the shape of the mask
        # Expand the mask to match the shape of the anoms scatter this
        MASK_MATRIX_NN_RESHAPED_model = np.broadcast_to(
            MASK_MATRIX_NN, anoms_scatter_wind_model_this.shape
        )
        MASK_MATRIX_UK_RESHAPED_model = np.broadcast_to(
            MASK_MATRIX_UK, anoms_scatter_wind_model_this.shape
        )
        MASK_MATRIX_NN_RESHAPED_obs = np.broadcast_to(
            MASK_MATRIX_NN, anoms_scatter_wind_obs_this.shape
        )
        MASK_MATRIX_UK_RESHAPED_obs = np.broadcast_to(
            MASK_MATRIX_UK, anoms_scatter_wind_obs_this.shape
        )

        # Apply the mask to the anoms scatter this
        anoms_scatter_wind_model_this_NN = np.ma.masked_where(
            MASK_MATRIX_NN_RESHAPED_model == 0, anoms_scatter_wind_model_this
        )
        anoms_scatter_wind_model_this_UK = np.ma.masked_where(
            MASK_MATRIX_UK_RESHAPED_model == 0, anoms_scatter_wind_model_this
        )

        # do the same for the obs
        anoms_scatter_wind_obs_this_NN = np.ma.masked_where(
            MASK_MATRIX_NN_RESHAPED_obs == 0, anoms_scatter_wind_obs_this
        )
        anoms_scatter_wind_obs_this_UK = np.ma.masked_where(
            MASK_MATRIX_UK_RESHAPED_obs == 0, anoms_scatter_wind_obs_this
        )

        # Set up a list of all of these
        list_anoms_scatter = [
            anoms_scatter_wind_model_this_NN,
            anoms_scatter_wind_model_this_UK,
            anoms_scatter_wind_obs_this_NN,
            anoms_scatter_wind_obs_this_UK,
        ]

        # loop over and assert that each of them have three dimensions
        for arr in list_anoms_scatter:
            assert (
                len(np.shape(arr)) == 3
            ), "Anoms scatter tas and wind do not match in shape"

        # take the spatial mean of these
        anoms_this_model_nn_wind_mean = np.mean(
            anoms_scatter_wind_model_this_NN, axis=(1, 2)
        )
        anoms_this_model_uk_wind_mean = np.mean(
            anoms_scatter_wind_model_this_UK, axis=(1, 2)
        )
        anoms_this_obs_nn_wind_mean = np.mean(
            anoms_scatter_wind_obs_this_NN, axis=(1, 2)
        )
        anoms_this_obs_uk_wind_mean = np.mean(
            anoms_scatter_wind_obs_this_UK, axis=(1, 2)
        )

        corr_wind_model, _ = pearsonr(
            anoms_this_model_nn_wind_mean, anoms_this_model_uk_wind_mean
        )
        if i == 2:
            corr_wind_obs = 0.0
        else:
            corr_wind_obs, _ = pearsonr(
                anoms_this_obs_nn_wind_mean, anoms_this_obs_uk_wind_mean
            )

        # Plot the scatter plot for wind model as grey circles
        ax_wind_scatter.scatter(
            anoms_this_model_nn_wind_mean,
            anoms_this_model_uk_wind_mean,
            color="grey",
            s=10,
            alpha=0.5,
            label=f"Model (r={corr_wind_model:.2f})",
        )

        # Plot the scatter plot for wind obs as black crosses
        ax_wind_scatter.scatter(
            anoms_this_obs_nn_wind_mean,
            anoms_this_obs_uk_wind_mean,
            color="black",
            s=10,
            marker="x",
            label=f"Obs (r={corr_wind_obs:.2f})",
        )

        # set the x and y limits for the scatter plots
        xlims_temp = (-17, 7)
        ylims_temp = (-17, 7)

        # Set the xlims and ylims for the scatter plots
        xlims_wind = (-4, 4)
        ylims_wind = (-4, 4)

        # Set the x and y limits for the scatter plots
        ax_wind_scatter.set_xlim(xlims_wind)
        ax_wind_scatter.set_ylim(ylims_wind)

        if i == 2:
            # Set the x and y labels for the scatter plots
            ax_wind_scatter.set_xlabel("NN wind anomaly (m/s)", fontsize=12)
        ax_wind_scatter.set_ylabel("UK wind anomaly (m/s)", fontsize=12)

        # Set up titles for the scatter plots
        if i == 0:
            ax_wind_scatter.set_title("UK vs. NN.", fontsize=12)

        # include a legend in the top left
        ax_wind_scatter.legend(
            loc="upper left",
            fontsize=12,
            markerscale=2,
            frameon=False,
        )

    # minimise the vertical spacing between subplots
    # fig.subplots_adjust(hspace=0.001)

    # # make sure the aspect ratios are all equal
    for ax in all_axes:
        ax.set_aspect("equal", adjustable="box")

    return None


# Define a function to plot the temp quartiles
def plot_temp_quartiles(
    subset_df_model: pd.DataFrame,
    tas_var_name: str,
    subset_arr_model: np.ndarray,
    model_index_dict: Dict[str, np.ndarray],
    lats_path: str,
    lons_path: str,
    var_name: str,
    figsize: Tuple[int, int] = (8, 9),
    anoms_flag: bool = False,
    clim_filepath: str = None,
    second_subset_df: pd.DataFrame = None,
    second_subset_arr: np.ndarray = None,
    second_model_index_dict: Dict[str, np.ndarray] = None,
    gridbox: dict = None,
    quartiles: List[Tuple[float, float]] = [
        (0.0, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 1.0),
    ],
):
    """
    Plots subplots with 4 rows and 2 columns. The left column shows the full
    field MSLP and the right column shows the differences between that
    field and the one above. The 4 rows are for the 4 quartiles of the
    temperature. E.g., the top row is the warmest quartile and the bottom row
    is the coldest quartile.

    Args:
    =====

        subset_df_model (pd.DataFrame): The subset dataframe for the model.
        tas_var_name (str): The name of the temperature variable.
        subset_arr_model (np.ndarray): The subset array for the model.
        model_index_dict (Dict[str, np.ndarray]): The model index dictionary.
        lats_path (str): The path to the latitude file.
        lons_path (str): The path to the longitude file.
        var_name (str): The name of the variable to plot.
        figsize (Tuple[int, int]): The figure size.
        anoms_flag (bool): Whether to plot anomalies or not.
        clim_filepath (str): The path to the climatology file.
        second_subset_df (pd.DataFrame): The second subset dataframe for the model.
        second_subset_arr (np.ndarray): The second subset array for the model.
        second_model_index_dict (Dict[str, np.ndarray]): The second model index dictionary.
        gridbox (dict): The gridbox to plot.

    Returns:
    ========

        None

    """

    # print the min and max of the subset arr model
    print(
        f"Min and max of the subset arr model: {np.min(subset_arr_model)}, {np.max(subset_arr_model)}"
    )
    print(f"Mean of the subset arr model: {np.mean(subset_arr_model)}")

    # # assert that var_name is tas
    # assert var_name == "tas", "Variable name must be tas"

    # Assert that tas_var_name is one of the columns in the subset_df_model
    assert (
        tas_var_name in subset_df_model.columns
    ), f"Variable name {tas_var_name} not found in the subset dataframe"

    # if the anoms flag is true
    if anoms_flag:
        # Load the anomalies
        climatology_arr = np.load(clim_filepath)
        if var_name in ["psl"]:
            # Set up the levels
            cmap = "coolwarm"
            levels = np.array(
                [
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
                ]
            )
        elif var_name in ["tas"]:
            # Set up the cmap
            cmap = "coolwarm"
            # Set up the levels
            levels = np.array(
                [
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
                ]
            )
        elif var_name in ["uas", "vas", "sfcWind"]:
            if var_name in ["uas", "sfcWind"]:
                cmap = "PRGn"
            else:
                cmap = "BrBG"
            levels = np.array(
                [
                    -4,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    4,
                ]
            )
        else:
            raise ValueError(
                f"Variable name {var_name} not recognised. Must be tas, uas or vas."
            )

    else:
        if var_name in ["tas", "psl"]:
            # Set up the cmap
            cmap = "bwr"

            # Sert up the levels
            levels = np.array(
                [
                    1004,
                    1006,
                    1008,
                    1010,
                    1012,
                    1014,
                    1016,
                    1018,
                    1020,
                    1022,
                    1024,
                    1026,
                ]
            )
        elif var_name in ["uas", "vas", "sfcWind"]:
            if var_name in ["uas", "sfcWind"]:
                cmap = "BuGn"
            else:
                cmap = "BrBG"
            levels = np.array(
                [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                ]
            )
        else:
            raise ValueError(
                f"Variable name {var_name} not recognised. Must be tas, uas or vas."
            )

    # every other tick
    levels_ticks = np.arange(
        np.min(levels),
        np.max(levels) + 1,
        2,
    )

    if var_name in ["psl", "tas"]:
        levels_diff = np.array(
            [
                -4,
                -3,
                -2,
                -1,
                1,
                2,
                3,
                4,
            ]
        )
    elif var_name in ["uas", "vas", "sfcWind"]:
        levels_diff = np.array(
            [
                -0.5,
                -0.45,
                -0.4,
                -0.35,
                -0.3,
                -0.25,
                -0.2,
                -0.15,
                -0.1,
                -0.05,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
            ]
        )

        # set up the ticks
        levels_diff_ticks = np.arange(
            np.min(levels_diff),
            np.max(levels_diff) + 1,
            0.1,
        )
    else:
        raise ValueError(
            f"Variable name {var_name} not recognised. Must be tas, uas or vas."
        )

    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # Set up the figure
    fig, axs = plt.subplots(
        ncols=2,
        nrows=4,
        figsize=figsize,
        layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Set up the quartiles
    # temp_quartiles = [
    #     [0.75, 1.00],
    #     [0.50, 0.75],
    #     [0.25, 0.50],
    #     [0.00, 0.25],
    # ]
    temp_quartiles = quartiles

    # set up the axes in rows
    axes_rows = [
        [axs[0, 0], axs[0, 1]],
        [axs[1, 0], axs[1, 1]],
        [axs[2, 0], axs[2, 1]],
        [axs[3, 0], axs[3, 1]],
    ]

    # Set up the arrays of init year
    init_year_array = np.array(model_index_dict["init_year"])
    member_array = np.array(model_index_dict["member"])
    lead_array = np.array(model_index_dict["lead"])

    # Loop over the axes rows
    for i, axes_row in enumerate(axes_rows):
        # Set up the subset arr this model
        subset_arr_this_model = subset_arr_model.copy()

        # Set up the cols
        left_col_full = axes_row[0]
        right_col_diff = axes_row[1]

        # find the lower quartiles
        lower_quartile_this = np.min(
            temp_quartiles[i],
        )
        # find the upper quartile
        upper_quartile_this = np.max(
            temp_quartiles[i],
        )

        # Quantify the lower and upper bounds of the quartile
        lower_bound = np.quantile(
            subset_df_model[tas_var_name].values, lower_quartile_this
        )
        upper_bound = np.quantile(
            subset_df_model[tas_var_name].values, upper_quartile_this
        )

        # Subset the dataframe for the quartile
        subset_df_model_this = subset_df_model[
            (subset_df_model[tas_var_name] >= lower_bound)
            & (subset_df_model[tas_var_name] < upper_bound)
        ]

        # Zero the missing days
        missing_days = 0

        # Set ip the array to store the values
        subset_arr_this_model_full = np.zeros(
            (
                len(subset_df_model_this),
                subset_arr_model.shape[1],
                subset_arr_model.shape[2],
            )
        )

        # Loop over the rows in this subset df for the model
        for j, (_, row) in tqdm(enumerate(subset_df_model_this.iterrows())):
            # Extract the init year from the df
            init_year_df = int(row["init_year"])
            member_df = int(row["member"])
            lead_df = int(row["lead"])

            # Construct the condition for element wise comparison
            condition_this = (
                (init_year_array == init_year_df)
                & (member_array == member_df)
                & (lead_array == lead_df)
            )

            try:
                # Find the index where this condition is met
                index_this = np.where(condition_this)[0][0]
            except IndexError:
                print(
                    f"init year {init_year_df}, member {member_df}, lead {lead_df} not found"
                )
                missing_days += 1

            # # prit the shape of the subset_arr_model
            # print(f"Shape of subset_arr_model: {subset_arr_model.shape}")

            # Extract the corresponding value from the subset_arr_this_model
            subset_arr_this_model_index_this = subset_arr_this_model[index_this, :, :]

            # Store the value in the subset_arr_this_model_full
            subset_arr_this_model_full[j, :, :] = subset_arr_this_model_index_this

        # if the second subset df is not None
        if second_subset_df is not None:
            # Set up the subset arr this model
            subset_arr_this_model_second = second_subset_arr.copy()

            # quantify the lower and upper bounds for the quantile
            lower_bound = np.quantile(
                second_subset_df[tas_var_name].values, temp_quartiles[i][0]
            )
            upper_bound = np.quantile(
                second_subset_df[tas_var_name].values, temp_quartiles[i][1]
            )

            # # Set up the cols
            # left_col_full = axes_row[0]
            # right_col_diff = axes_row[1]

            # Subset the dataframe for the quartile
            subset_df_model_this_second = second_subset_df[
                (second_subset_df[tas_var_name] >= lower_bound)
                & (second_subset_df[tas_var_name] < upper_bound)
            ]

            # Zero the missing days
            missing_days = 0

            # Set ip the array to store the values
            subset_arr_this_model_full_second = np.zeros(
                (
                    len(subset_df_model_this_second),
                    subset_arr_model.shape[1],
                    subset_arr_model.shape[2],
                )
            )

            # extract the init year array second
            init_year_array_second = np.array(second_model_index_dict["init_year"])
            member_array_second = np.array(second_model_index_dict["member"])
            lead_array_second = np.array(second_model_index_dict["lead"])

            # Loop over the rows in this subset df for the model
            for j, (_, row) in tqdm(enumerate(subset_df_model_this_second.iterrows())):
                # Extract the init year from the df
                init_year_df = int(row["init_year"])
                member_df = int(row["member"])
                lead_df = int(row["lead"])

                # Construct the condition for element wise comparison
                condition_this = (
                    (init_year_array_second == init_year_df)
                    & (member_array_second == member_df)
                    & (lead_array_second == lead_df)
                )

                try:
                    # Find the index where this condition is met
                    index_this = np.where(condition_this)[0][0]
                except IndexError:
                    print(
                        f"init year {init_year_df}, member {member_df}, lead {lead_df} not found"
                    )
                    missing_days += 1

                # Extract the corresponding value from the subset_arr_this_model
                subset_arr_this_model_index_this = subset_arr_this_model_second[
                    index_this, :, :
                ]

                # Store the value in the subset_arr_this_model_full
                subset_arr_this_model_full_second[j, :, :] = (
                    subset_arr_this_model_index_this
                )

        # # Print the row index
        # print(f"Row index: {i}")
        # print(f"Number of missing days: {missing_days}")
        # print(f"Model overall N: {len(subset_df_model_this)}")

        # Take the mean over this
        subset_arr_this_model_mean = np.mean(subset_arr_this_model_full, axis=0)

        if second_subset_df is not None:
            subset_arr_this_model_mean_second = np.mean(
                subset_arr_this_model_full_second, axis=0
            )

        # If the anoms flag is true
        if anoms_flag:
            # Print the min max and mean of the clim arr
            print(
                f"Min, max and mean of the climatology array: {np.min(climatology_arr)}, {np.max(climatology_arr)}, {np.mean(climatology_arr)}"
            )

            # Calculate the model anoms
            subset_arr_this_model_mean = subset_arr_this_model_mean - climatology_arr

        # If subset_arr_this_model_full_second is not None
        if second_subset_arr is not None:
            print("Quantifying differences between first and second")

            if anoms_flag:
                subset_arr_this_model_mean_second = (
                    subset_arr_this_model_mean_second - climatology_arr
                )

            subset_arr_this_model_mean = (
                subset_arr_this_model_mean - subset_arr_this_model_mean_second
            )

            levels = levels_diff
            levels_ticks = np.array(
                [
                    -4,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    4,
                ]
            )

            levels_diffs = levels_diff_ticks

        # if the var name is tas /psl
        # divide by 100
        if var_name in ["psl"]:
            subset_arr_this_model_mean = subset_arr_this_model_mean / 100
            if second_subset_arr is not None:
                subset_arr_this_model_mean_second = (
                    subset_arr_this_model_mean_second / 100
                )

        if i == 0:
            warmest_composite = subset_arr_this_model_mean

        # Plot the full field on the left
        im_full = left_col_full.contourf(
            lons,
            lats,
            (subset_arr_this_model_mean),
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            levels=levels,
            extend="both",
        )

        # if gridbox is not none
        if gridbox is not None:
            # if there is more than one gridbox
            if isinstance(gridbox, list):
                print("Calculating difference in gridbox fields")

                # Hard code the n_box and south box for delta P
                n_box = gridbox[0]
                s_box = gridbox[1]

                # Extract the n_box lats and lons
                lat1_box_n, lat2_box_n = n_box["lat1"], n_box["lat2"]
                lon1_box_n, lon2_box_n = n_box["lon1"], n_box["lon2"]

                # Extract the s_box lats and lons
                lat1_box_s, lat2_box_s = s_box["lat1"], s_box["lat2"]
                lon1_box_s, lon2_box_s = s_box["lon1"], s_box["lon2"]

                # Find the indices of the lats which correspond to the gridbox
                lat1_idx_n = np.argmin(np.abs(lats - lat1_box_n))
                lat2_idx_n = np.argmin(np.abs(lats - lat2_box_n))

                # Find the indices of the lons which correspond to the gridbox
                lon1_idx_n = np.argmin(np.abs(lons - lon1_box_n))
                lon2_idx_n = np.argmin(np.abs(lons - lon2_box_n))

                # Find the indices of the lats which correspond to the gridbox
                lat1_idx_s = np.argmin(np.abs(lats - lat1_box_s))
                lat2_idx_s = np.argmin(np.abs(lats - lat2_box_s))

                # Find the indices of the lons which correspond to the gridbox
                lon1_idx_s = np.argmin(np.abs(lons - lon1_box_s))
                lon2_idx_s = np.argmin(np.abs(lons - lon2_box_s))

                # Add the gridbox to the plot
                left_col_full.plot(
                    [lon1_box_n, lon2_box_n, lon2_box_n, lon1_box_n, lon1_box_n],
                    [lat1_box_n, lat1_box_n, lat2_box_n, lat2_box_n, lat1_box_n],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Add the gridbox to the plot
                left_col_full.plot(
                    [lon1_box_s, lon2_box_s, lon2_box_s, lon1_box_s, lon1_box_s],
                    [lat1_box_s, lat1_box_s, lat2_box_s, lat2_box_s, lat1_box_s],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Calculate the mean in the gridbox
                gridbox_mean_n = np.mean(
                    subset_arr_this_model_mean[
                        lat1_idx_n : lat2_idx_n + 1, lon1_idx_n : lon2_idx_n + 1
                    ]
                )

                gridbox_mean_s = np.mean(
                    subset_arr_this_model_mean[
                        lat1_idx_s : lat2_idx_s + 1, lon1_idx_s : lon2_idx_s + 1
                    ]
                )

                # Include a textbox in the top left for the gridbox mean
                # to two S.F.
                left_col_full.text(
                    0.05,
                    0.95,
                    f"delta P = {gridbox_mean_n - gridbox_mean_s:.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=left_col_full.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )
            else:
                print("Calculating absolute value in gridbox")

                # Extract the lons and lats from the gridbox
                lat1_box, lat2_box = gridbox["lat1"], gridbox["lat2"]
                lon1_box, lon2_box = gridbox["lon1"], gridbox["lon2"]

                # Find the indices of the lats which correspond to the gridbox
                lat1_idx = np.argmin(np.abs(lats - lat1_box))
                lat2_idx = np.argmin(np.abs(lats - lat2_box))

                # Find the indices of the lons which correspond to the gridbox
                lon1_idx = np.argmin(np.abs(lons - lon1_box))
                lon2_idx = np.argmin(np.abs(lons - lon2_box))

                # Add the gridbox to the plot
                left_col_full.plot(
                    [lon1_box, lon2_box, lon2_box, lon1_box, lon1_box],
                    [lat1_box, lat1_box, lat2_box, lat2_box, lat1_box],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Calculate the mean in the gridbox
                gridbox_mean = np.mean(
                    subset_arr_this_model_mean[
                        lat1_idx : lat2_idx + 1, lon1_idx : lon2_idx + 1
                    ]
                )

                # Include a textbox in the top left for the gridbox mean
                # to two S.F.
                left_col_full.text(
                    0.05,
                    0.95,
                    f"Gridbox mean = {gridbox_mean:.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=left_col_full.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

        # Add coastlines
        left_col_full.coastlines()

        # Include a textbox in the top right for N
        # Include a textbox in the bottom right for N
        left_col_full.text(
            0.95,  # x-coordinate (right edge)
            0.05,  # y-coordinate (bottom edge)
            f"N = {len(subset_df_model_this)}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=left_col_full.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Include the qunatile range in a textbox in the bottom left
        left_col_full.text(
            0.05,
            0.05,
            f"{temp_quartiles[i][0]} - {temp_quartiles[i][1]}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=left_col_full.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Plot the difference on the right
        im_diff = right_col_diff.contourf(
            lons,
            lats,
            (subset_arr_this_model_mean - warmest_composite),
            cmap="PRGn",
            transform=ccrs.PlateCarree(),
            levels=levels_diff,
            extend="both",
        )

        # if the gridbox is not none
        if gridbox is not None:
            if isinstance(gridbox, list):
                # Calculate the difference in the gridbox fields
                print("Calculating difference in gridbox fields")

                # subset the gridbox
                gridbox_mean_n = np.mean(
                    (subset_arr_this_model_mean - warmest_composite)[
                        lat1_idx_n : lat2_idx_n + 1, lon1_idx_n : lon2_idx_n + 1
                    ]
                )
                gridbox_mean_s = np.mean(
                    (subset_arr_this_model_mean - warmest_composite)[
                        lat1_idx_s : lat2_idx_s + 1, lon1_idx_s : lon2_idx_s + 1
                    ]
                )

                # Include a textbox in the top left for the gridbox mean
                # to two S.F.
                right_col_diff.text(
                    0.05,
                    0.95,
                    f"delta P = {gridbox_mean_n - gridbox_mean_s:.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=right_col_diff.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

                # plot the gridboxes
                right_col_diff.plot(
                    [lon1_box_n, lon2_box_n, lon2_box_n, lon1_box_n, lon1_box_n],
                    [lat1_box_n, lat1_box_n, lat2_box_n, lat2_box_n, lat1_box_n],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                right_col_diff.plot(
                    [lon1_box_s, lon2_box_s, lon2_box_s, lon1_box_s, lon1_box_s],
                    [lat1_box_s, lat1_box_s, lat2_box_s, lat2_box_s, lat1_box_s],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )
            else:
                # Add the gridbox to the plot
                right_col_diff.plot(
                    [lon1_box, lon2_box, lon2_box, lon1_box, lon1_box],
                    [lat1_box, lat1_box, lat2_box, lat2_box, lat1_box],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Calculate the mean in the gridbox
                gridbox_mean = np.mean(
                    (subset_arr_this_model_mean - warmest_composite)[
                        lat1_idx : lat2_idx + 1, lon1_idx : lon2_idx + 1
                    ]
                )

                # Include a textbox in the top left for the gridbox mean
                # to two S.F.
                right_col_diff.text(
                    0.05,
                    0.95,
                    f"Gridbox mean = {gridbox_mean:.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=right_col_diff.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

        # Add coastlines
        right_col_diff.coastlines()

        # Include a textbox in the top right for N
        right_col_diff.text(
            0.95,
            0.05,
            f"N = {len(subset_df_model_this)}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=right_col_diff.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Include the quantile range in a textbox in the bottom left
        right_col_diff.text(
            0.05,
            0.05,
            f"{temp_quartiles[i][0]} - {temp_quartiles[i][1]}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=right_col_diff.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # if i == 3, then set up the cbars
        if i == 3:
            cbar_full = fig.colorbar(
                im_full,
                ax=left_col_full,
                orientation="horizontal",
                pad=0.05,
                shrink=0.8,
            )

            cbar_full.set_ticks(levels_ticks)

            cbar_diff = fig.colorbar(
                im_diff,
                ax=right_col_diff,
                orientation="horizontal",
                pad=0.05,
                shrink=0.8,
            )

            cbar_diff.set_ticks(levels_diff)

        if i == 0:
            # Set up the titles for the left and right columns
            left_col_full.set_title("Full field", fontsize=12, fontweight="bold")
            right_col_diff.set_title(
                "Difference from warmest quartile", fontsize=12, fontweight="bold"
            )

    return None


# Define a function to plot a single column variable
def plot_var_composites_model(
    subset_dfs_model: List[pd.DataFrame],
    subset_arrs_model: List[np.ndarray],
    clim_arrs_model: List[np.ndarray],
    model_index_dicts: List[Dict[str, np.ndarray]],
    lats_path: str,
    lons_path: str,
    var_name: str,
    figsize: Tuple[int, int] = (8, 9),
):
    """
    Plots the subset composites for a single model variable.

    Args:
    =====

        subset_dfs_model (List[pd.DataFrame]): The list of subset dataframes for the model.
        subset_arrs_model (List[np.ndarray]): The list of subset arrays for the model.
        clim_arrs_model (List[np.ndarray]): The list of climatology arrays for the model.
        model_index_dicts (List[Dict[str, np.ndarray]]): The list of model index dictionaries.
        lats_path (str): The path to the latitude file.
        lons_path (str): The path to the longitude file.
        var_name (str): The name of the variable to plot.
        figsize (Tuple[int, int]): The figure size.

    Returns:
    ========

        None

    """

    # if the variable is tas
    if var_name == "tas":
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
    elif var_name == "sfcWind":
        cmap = "PRGn"
        levels = np.array(
            [
                -5,
                -4,
                -3,
                -2,
                -1,
                1,
                2,
                3,
                4,
                5,
            ]
        )
    elif var_name in ["uas", "vas"]:
        cmap = "PRGn"
        levels = np.array(
            [
                -4,
                -3.5,
                -3,
                -2.5,
                -2,
                -1.5,
                -1,
                -0.5,
                0.5,
                1,
                1.5,
                2,
                2.5,
                3,
                3.5,
                4,
            ]
        )
    elif var_name == "psl":
        # Set up the levels for plotting absolute values
        # Set up the cmap
        cmap = "bwr"

        # Sert up the levels
        levels = np.array(
            [
                1004,
                1006,
                1008,
                1010,
                1012,
                1014,
                1016,
                1018,
                1020,
                1022,
                1024,
                1026,
            ]
        )
    else:
        raise ValueError(f"Variable {var_name} not supported.")

    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

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
    for l in range(len(lats)):
        for j in range(len(lons)):
            point = shapely.geometry.Point(lons[j], lats[l])
            for country in country_shapely:
                if country.contains(point):
                    MASK_MATRIX_TMP[l, j] = 1.0

    # Reshape the mask to match the shape of the data
    MASK_MATRIX_RESHAPED = MASK_MATRIX_TMP

    # Set up the figure
    fig, axs = plt.subplots(
        ncols=1,
        nrows=3,
        figsize=figsize,
        layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Set up the names
    names_list = [
        "Low DnW days",
        "High DnW days",
        "Peak DnW days",
    ]

    # Flatten the axes
    axs = axs.flatten()

    # Loop over the axes
    for i, ax_this in enumerate(axs):
        # Set up the subset arr this model
        subset_arr_this_model = subset_arrs_model[i]

        # Set up the subset arr this model full
        subset_arr_this_model_full = np.zeros(
            (len(subset_dfs_model[i]), len(lats), len(lons))
        )

        # if the var name is tas
        if var_name == "tas":
            print("Applying detrend to the tas data")

            # Extract the index dicts this
            index_dict_this = model_index_dicts[i]

            # Extract the effective dec years array
            effective_dec_years_arr = np.array(index_dict_this["effective_dec_year"])

            # Extract the unique effective dec years
            unique_effective_dec_years = np.unique(effective_dec_years_arr)

            # Set up a new array to append to
            subset_arr_this_detrended = np.zeros(
                (
                    len(unique_effective_dec_years),
                    subset_arr_this_model.shape[1],
                    subset_arr_this_model.shape[2],
                )
            )

            # Loop over the unique effective dec years
            for j, effective_dec_year in enumerate(unique_effective_dec_years):
                # Find the index of this effective dec year in the index dict this
                index_this = np.where(effective_dec_years_arr == effective_dec_year)[0]

                # Extract the subset arr this for this index
                subset_arr_this_model_this = np.mean(
                    subset_arr_this_model[index_this, :, :], axis=0
                )

                # Store the value in the subset arr this detrended
                subset_arr_this_detrended[j, :, :] = subset_arr_this_model_this

            # Loop over the lats and lons
            for j in range(len(lats)):
                for k in range(len(lons)):
                    # Detrend the data
                    slope_this, intercept_this, _, _, _ = linregress(
                        unique_effective_dec_years,
                        subset_arr_this_detrended[:, j, k],
                    )

                    # Calculate the trend line this
                    trend_line_this = (
                        slope_this * unique_effective_dec_years + intercept_this
                    )

                    # Find the final point on the trend line
                    final_point_this = trend_line_this[-1]

                    # Loop over the unique effective dec years
                    for l, eff_dec_year_this in enumerate(unique_effective_dec_years):
                        # Find the index of this effective dec year in the index dict this
                        index_this = np.where(
                            effective_dec_years_arr == eff_dec_year_this
                        )[0]

                        # Extract the subset arr this for this index
                        subset_arr_this_model[index_this, j, k] = (
                            final_point_this
                            - trend_line_this[l]
                            + subset_arr_this_model[index_this, j, k]
                        )

        # Set up the N for model this
        N_model_this = np.shape(subset_arr_this_model_full)[0]

        # Extract the index dict for the model this
        model_index_dict_this = model_index_dicts[i]

        # do the same for wind speed
        init_year_array_this = np.array(model_index_dict_this["init_year"])
        member_array_this = np.array(model_index_dict_this["member"])
        lead_array_this = np.array(model_index_dict_this["lead"])

        # Zero the missing days her
        missing_days = 0

        # Loop over the rows in this subset df for the model
        for j, (_, row) in tqdm(enumerate(subset_dfs_model[i].iterrows())):
            # Extract the init_year from the df
            init_year_df = int(row["init_year"])
            member_df = int(row["member"])
            lead_df = int(row["lead"])

            # Construct the condition for element wise comparison
            condition_this = (
                (init_year_array_this == init_year_df)
                & (member_array_this == member_df)
                & (lead_array_this == lead_df)
            )

            try:
                # Find the index where this condition is met
                index_this = np.where(condition_this)[0][0]
            except IndexError:
                print(
                    f"init year {init_year_df}, member {member_df}, lead {lead_df} not found"
                )
                missing_days += 1

            # Extract the corresponding value from the subset_arr_this_model
            subset_arr_this_model_index_this = subset_arr_this_model[index_this, :, :]

            # Store the value in the subset_arr_this_model_full
            subset_arr_this_model_full[j, :, :] = subset_arr_this_model_index_this

        # if there are any zeros in the subset_arr_this_model_full
        if np.any(subset_arr_this_model_full == 0):
            # format these as Nans
            subset_arr_this_model_full[subset_arr_this_model_full == 0] = np.nan

        # Print the row index
        print(f"Row index: {i}")
        print(f"Number of missing days: {missing_days}")
        print(f"Model overall N: {N_model_this}")

        # Take the mean over this
        subset_arr_this_model_mean = np.mean(subset_arr_this_model_full, axis=0)

        # if the variable is not psl
        if var_name not in ["psl", "uas", "vas"]:
            # Calculate the model anoms
            anoms_this_model = subset_arr_this_model_mean - clim_arrs_model[i]
        else:
            # If the variable is psl, then do not calculate anomalies
            anoms_this_model = subset_arr_this_model_mean

        # # # if i == 0
        # if i == 0:
        #     anoms_this_model_first = anoms_this_model
        # else:
        #     anoms_this_model = anoms_this_model - anoms_this_model_first

        # # Set up a new
        # levels = np.array(
        #     [
        #         -1,
        #         -0.75,
        #         -0.5,
        #         -0.25,
        #         0.25,
        #         0.5,
        #         0.75,
        #         1,
        #     ]
        # )

        # Set up wider levels for temp
        if var_name == "tas":
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

        # if the var name is psl
        # do anoms this / 100 to get in hPa
        if var_name == "psl":
            anoms_this_model = anoms_this_model / 100.0
        elif var_name == "tas":
            anoms_this_model = np.ma.masked_where(
                MASK_MATRIX_RESHAPED == 0, anoms_this_model
            )

        # Plot the model data on the right
        im_model = ax_this.contourf(
            lons,
            lats,
            anoms_this_model,
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            levels=levels,
            extend="both",
        )

        # add coastlines to all of these
        ax_this.coastlines()

        # Include a textbox in the rop right for N
        ax_this.text(
            0.95,
            0.95,
            f"N = {N_model_this}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=ax_this.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # if i == 2
        if i == 2:
            # add the colorbar for wind
            cbar = fig.colorbar(
                im_model,
                ax=ax_this,
                orientation="horizontal",
                pad=0.05,
                shrink=0.8,
            )
            cbar.set_ticks(levels)

        # if i == 0
        if i == 0:
            # Set the title for ax0 and ax1 in bold
            ax_this.set_title("Model (DePreSys)", fontsize=12, fontweight="bold")
        # Set up a textbox in the bottom right
        ax_this.text(
            0.95,
            0.05,
            names_list[i],
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=ax_this.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    return None


# Define a function to plot the multiple column variables
def plot_multi_var_composites_model(
    multi_subset_dfs_model: List[List[pd.DataFrame]],
    multi_subset_arrs_model: List[List[np.ndarray]],
    multi_clim_arrs_model: List[List[np.ndarray]],
    multi_model_index_dicts: List[List[Dict[str, np.ndarray]]],
    multi_lats_path: List[str],
    multi_lons_path: List[str],
    multi_var_names: List[str],
    figsize: Tuple[int, int] = (8, 9),
):
    """
    Plots the subset composites for multiple model variables.

    Args:
    =====

        multi_subset_dfs_model (List[List[pd.DataFrame]]): The list of list of subset dataframes for the model.
        multi_subset_arrs_model (List[List[np.ndarray]]): The list of list of subset arrays for the model.
        multi_clim_arrs_model (List[List[np.ndarray]]): The list of list of climatology arrays for the model.
        multi_model_index_dicts (List[List[Dict[str, np.ndarray]]]): The list of list of model index dictionaries.
        multi_lats_path (List[str]): The list of paths to the latitude files.
        multi_lons_path (List[str]): The list of paths to the longitude files.
        multi_var_names (List[str]): The list of variable names to plot.
        figsize (Tuple[int, int]): The figure size.

    Returns:
    ========

        None

    """

    # Set up the ncols
    ncols = 3
    nrows = len(multi_var_names)

    # Set up the figure
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"width_ratios": [1.5, 1, 1]},  # First column is 1.5x wider
    )

    # Set up the names
    names_list = [
        "Low DnW days",
        "High DnW days",
        "Peak DnW days",
    ]

    # Set up the axes
    ax1 = axs[:, 0]
    ax2 = axs[:, 1]
    ax3 = axs[:, 2]

    # Set up the list of axes
    axes_list = [ax1, ax2, ax3]

    # Set up the labels
    plot_labels = [
        ["a", "b", "c"],
        ["d", "e", "f"],
        ["g", "h", "i"],
    ]

    # Loop over the mutli var names enumerate
    for i, var_name in enumerate(multi_var_names):
        # if the variable is tas
        if var_name == "tas":
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
        elif var_name == "sfcWind":
            cmap = "PRGn"
            levels = np.array(
                [
                    -5,
                    -4,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    4,
                    5,
                ]
            )
        elif var_name in ["uas", "vas"]:
            cmap = "PRGn"
            levels = np.array(
                [
                    -4,
                    -3.5,
                    -3,
                    -2.5,
                    -2,
                    -1.5,
                    -1,
                    -0.5,
                    0.5,
                    1,
                    1.5,
                    2,
                    2.5,
                    3,
                    3.5,
                    4,
                ]
            )
        elif var_name == "psl":
            # Set up the levels for plotting absolute values
            # Set up the cmap
            cmap = "coolwarm"

            # Sert up the levels
            levels = np.array(
                [
                    1004,
                    1006,
                    1008,
                    1010,
                    1012,
                    1014,
                    1016,
                    1018,
                    1020,
                    1022,
                    1024,
                    1026,
                ]
            )
        else:
            raise ValueError(f"Variable {var_name} not supported.")

        # Extract the lats
        lats_this = np.load(multi_lats_path[i])
        lons_this = np.load(multi_lons_path[i])

        # Set up the countries shapefile
        countries_shp = shpreader.natural_earth(
            resolution="10m",
            category="cultural",
            name="admin_0_countries",
        )

        # Set up the land shapereader
        # Initialize the mask with the correct shape
        MASK_MATRIX_TMP = np.zeros((len(lats_this), len(lons_this)))
        country_shapely = []
        for country in shpreader.Reader(countries_shp).records():
            country_shapely.append(country.geometry)

        # Loop over the latitude and longitude points
        for l in range(len(lats_this)):
            for j in range(len(lons_this)):
                point = shapely.geometry.Point(lons_this[j], lats_this[l])
                for country in country_shapely:
                    if country.contains(point):
                        MASK_MATRIX_TMP[l, j] = 1.0

        # Reshape the mask to match the shape of the data
        MASK_MATRIX_RESHAPED = MASK_MATRIX_TMP

        # Extract the col this
        col_this = axes_list[i]

        # Loop over the rows this col
        for j, row_this in enumerate(col_this):
            # Set up the subset arr this model
            subset_arr_this_model = multi_subset_arrs_model[i][j]

            # Set up the subset arr this model full
            subset_arr_this_model_full = np.zeros(
                (len(multi_subset_dfs_model[i][j]), len(lats_this), len(lons_this))
            )

            # if the var name is tas
            if var_name == "tas":
                print("Applying detrend to the tas data")

                # Extract the index dicts this
                index_dict_this = multi_model_index_dicts[i][j]

                # Extract the effective dec years array
                effective_dec_years_arr = np.array(
                    index_dict_this["effective_dec_year"]
                )

                # Extract the unique effective dec years
                unique_effective_dec_years = np.unique(effective_dec_years_arr)

                # Set up a new array to append to
                subset_arr_this_detrended = np.zeros(
                    (
                        len(unique_effective_dec_years),
                        subset_arr_this_model.shape[1],
                        subset_arr_this_model.shape[2],
                    )
                )

                # Loop over the unique effective dec years
                for dec_year_i, effective_dec_year in enumerate(
                    unique_effective_dec_years
                ):
                    # Find the index of this effective dec year in the index dict this
                    index_this = np.where(
                        effective_dec_years_arr == effective_dec_year
                    )[0]

                    # Extract the subset arr this for this index
                    subset_arr_this_model_this = np.mean(
                        subset_arr_this_model[index_this, :, :], axis=0
                    )

                    # Store the value in the subset arr this detrended
                    subset_arr_this_detrended[dec_year_i, :, :] = (
                        subset_arr_this_model_this
                    )

                # Loop over the lats and lons
                for ilat in range(len(lats_this)):
                    for ilon in range(len(lons_this)):
                        # Detrend the data
                        slope_this, intercept_this, _, _, _ = linregress(
                            unique_effective_dec_years,
                            subset_arr_this_detrended[:, ilat, ilon],
                        )

                        # Calculate the trend line this
                        trend_line_this = (
                            slope_this * unique_effective_dec_years + intercept_this
                        )

                        # Find the final point on the trend line
                        final_point_this = trend_line_this[-1]

                        # Loop over the unique effective dec years
                        for l, eff_dec_year_this in enumerate(
                            unique_effective_dec_years
                        ):
                            # Find the index of this effective dec year in the index dict this
                            index_this = np.where(
                                effective_dec_years_arr == eff_dec_year_this
                            )[0]

                            # Extract the subset arr this for this index
                            subset_arr_this_model[index_this, ilat, ilon] = (
                                final_point_this
                                - trend_line_this[l]
                                + subset_arr_this_model[index_this, ilat, ilon]
                            )

            # Set up the N for model this
            N_model_this = np.shape(subset_arr_this_model_full)[0]

            # # print the i and j
            # print(f"value of index i: {i}")
            # print(f"value of index j: {j}")

            # # print the len of multi_model_index_dicts
            # print(f"Length of multi_model_index_dicts: {len(multi_model_index_dicts)}")

            # Extract the index dict for the model this
            model_index_dict_this = multi_model_index_dicts[i][j]

            # do the same for wind speed
            init_year_array_this = np.array(model_index_dict_this["init_year"])
            member_array_this = np.array(model_index_dict_this["member"])
            lead_array_this = np.array(model_index_dict_this["lead"])

            # Zero the missing days here
            missing_days_this = 0

            # Loop over the rows in this subset df for the model
            for k, (_, row) in tqdm(enumerate(multi_subset_dfs_model[i][j].iterrows())):
                # Extract the init_year from the df
                init_year_df = int(row["init_year"])
                member_df = int(row["member"])
                lead_df = int(row["lead"])

                # Construct the condition for element wise comparison
                condition_this = (
                    (init_year_array_this == init_year_df)
                    & (member_array_this == member_df)
                    & (lead_array_this == lead_df)
                )

                try:
                    # Find the index where this condition is met
                    index_this = np.where(condition_this)[0][0]
                except IndexError:
                    print(
                        f"init year {init_year_df}, member {member_df}, lead {lead_df} not found"
                    )
                    missing_days += 1

                # Extract the corresponding value from the subset_arr_this_model
                subset_arr_this_model_index_this = subset_arr_this_model[
                    index_this, :, :
                ]

                # Store the value in the subset_arr_this_model_full
                subset_arr_this_model_full[k, :, :] = subset_arr_this_model_index_this

            # anything which is a zero format as a NaN
            subset_arr_this_model_full[subset_arr_this_model_full == 0] = np.nan

            # print the min and max of the subset_arr_this_model_full
            print(
                f"Min of subset_arr_this_model_full: {np.min(subset_arr_this_model_full)}"
            )
            print(
                f"Max of subset_arr_this_model_full: {np.max(subset_arr_this_model_full)}"
            )
            print(
                f"Mean of subset_arr_this_model_full: {np.nanmean(subset_arr_this_model_full)}"
            )

            # Take the mean over this
            subset_arr_this_model_mean = np.nanmean(subset_arr_this_model_full, axis=0)

            # if the variable is not psl
            if var_name != "psl":
                # Calculate the model anoms
                anoms_this_model = (
                    subset_arr_this_model_mean - multi_clim_arrs_model[i][j]
                )
            else:
                # If the variable is psl, then do not calculate anomalies
                anoms_this_model = subset_arr_this_model_mean

            # if the var name is psl
            # do anoms this / 100 to get in hPa
            if var_name == "psl":
                anoms_this_model = anoms_this_model / 100.0
            elif var_name == "tas":
                print("NOT Masking the tas data with the land mask")
                # anoms_this_model = np.ma.masked_where(
                #     MASK_MATRIX_RESHAPED == 0, anoms_this_model
                # )

            # Print the col index
            print(f"Column index: {i}")
            print(f"Row index: {j}")
            print(f"Number of missing days: {missing_days_this}")
            print(f"Model overall N: {N_model_this}")

            # print the min and max and mean of anoms this model
            print(f"min anoms this model: {np.min(anoms_this_model)}")
            print(f"max anoms this model: {np.max(anoms_this_model)}")
            print(f"mean anoms this model: {np.mean(anoms_this_model)}")

            # Plot the model data on the right
            im_model = row_this.contourf(
                lons_this,
                lats_this,
                anoms_this_model,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels,
                extend="both",
            )

            # if the var_name is psl, then plot absolute contours
            if var_name == "psl":
                # Plot the absolute contours
                contours = row_this.contour(
                    lons_this,
                    lats_this,
                    anoms_this_model,
                    levels=levels,
                    colors="black",
                    linewidths=0.5,
                    transform=ccrs.PlateCarree(),
                )

                row_this.clabel(
                    contours,
                    levels,
                    fmt="%.0f",
                    fontsize=6,
                    inline=True,
                    inline_spacing=0.0,
                )

            # add coastlines to all of these
            row_this.coastlines()

            # Include a textbox in the top right for N
            row_this.text(
                0.95,
                0.95,
                f"N = {N_model_this}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=row_this.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # if i == 2
            if j == 2:
                # add colorbar
                # cbar = plt.colorbar(mymap, orientation='horizontal', shrink=0.7, pad=0.1)
                # cbar.set_label('SST [C]', rotation=0, fontsize=10)
                # cbar.ax.tick_params(labelsize=7, length=0)

                # add the colorbar for wind
                cbar = fig.colorbar(
                    im_model,
                    ax=row_this,
                    orientation="horizontal",
                    pad=0.05,
                    shrink=0.8,
                )

                # depending on the i set the labels
                if i == 0:

                    # Set up the ticks
                    levels = np.array(
                        [
                            1004,
                            1008,
                            1012,
                            1015,
                            1018,
                            1022,
                            1026,
                        ]
                    )

                    cbar.set_label("hPa", rotation=0, fontsize=12)
                elif i == 1:
                    cbar.set_label("°C", rotation=0, fontsize=12)
                elif i == 2:

                    cbar.set_label("m/s", rotation=0, fontsize=12)

                cbar.set_ticks(levels)

            # Set the title for each subplot based on `i`
            if i == 0 and j == 0:
                row_this.set_title("Model daily MSLP", fontsize=12, fontweight="bold")
            elif i == 1 and j == 0:
                row_this.set_title(
                    "Model daily airT anoms", fontsize=12, fontweight="bold"
                )
            elif i == 2 and j == 0:
                row_this.set_title(
                    "Model daily sfcWind anoms", fontsize=12, fontweight="bold"
                )

            # Set up a textbox in the bottom right
            row_this.text(
                0.95,
                0.05,
                names_list[j],
                horizontalalignment="right",
                verticalalignment="bottom",
                transform=row_this.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # Include the plot labels in the bottom left
            row_this.text(
                0.05,
                0.05,
                f"{plot_labels[i][j]}",
                horizontalalignment="left",
                verticalalignment="bottom",
                transform=row_this.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )

    return None


# Define a function for plotting the temp/demand quartiles for the obs
def plot_temp_demand_quartiles_obs(
    subset_df_obs: pd.DataFrame,
    quartiles_var_name: str,
    quartiles: List[Tuple[float, float]],
    subset_arr_obs: np.ndarray,
    dates_list_obs: List[str],
    var_name: str,
    lats_path: str,
    lons_path: str,
    figsize: Tuple[int, int] = (10, 6),
    anoms_flag: bool = False,
    clim_arr_obs: np.ndarray = None,
    gridbox: Optional[Dict[str, float]] = None,
    second_subset_df_obs: Optional[pd.DataFrame] = None,
    second_quartiles_var_name: Optional[str] = None,
    second_subset_arr_obs: Optional[np.ndarray] = None,
    second_dates_list_obs: Optional[List[str]] = None,
    second_quartiles: Optional[List[Tuple[float, float]]] = None,
):
    """
    Plots subplots for the composites of different quartiles of a variable.
    E.g., could be decreasing temperature or increasing electricity demand.

    Args:
    =====

        subset_df_obs (pd.DataFrame): The subset dataframe for the observations.
        quartiles_var_name (str): The variable name for the quartiles.
        quartiles (List[Tuple[float, float]]): The list of quartiles to plot.
        subset_arr_obs (np.ndarray): The subset array for the observations.
        dates_list_obs (List[str]): The list of dates for the observations.
        var_name (str): The name of the variable to plot.
        lats_path (str): The path to the latitude file.
        lons_path (str): The path to the longitude file.
        figsize (Tuple[int, int]): The figure size.
        anoms_flag (bool): Whether to calculate anomalies or not.
        clim_arr_obs (np.ndarray): The climatology array for the observations.
        gridbox (Optional[Dict[str, float]]): The gridbox to plot.
        second_subset_df_obs (Optional[pd.DataFrame]): The second subset dataframe for the observations.
        second_quartiles_var_name (Optional[str]): The variable name for the second quartiles.
        second_subset_arr_obs (Optional[np.ndarray]): The second subset array for the observations.
        second_dates_list_obs (Optional[List[str]]): The second list of dates for the observations.
        second_quartiles (Optional[List[Tuple[float, float]]]): The second list of quartiles to plot.


    Returns:
    ========

        None

    """

    # Print the columns of the subset df obs
    print(f"Columns of subset df obs: {subset_df_obs.columns.tolist()}")

    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # if the anoms flag is true
    if anoms_flag:
        # Load the anomalies
        climatology_arr = clim_arr_obs
        if var_name in ["tas", "psl"]:
            # Set up the levels
            cmap = "coolwarm"
            levels = np.array(
                [
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
                ]
            )
        elif var_name in ["uas", "vas", "sfcWind"]:
            if var_name in ["uas", "sfcWind"]:
                cmap = "PRGn"
            else:
                cmap = "BrBG"
            levels = np.array(
                [
                    -4,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    4,
                ]
            )
        else:
            raise ValueError(
                f"Variable name {var_name} not recognised. Must be tas, uas or vas."
            )

    else:
        if var_name in ["tas", "psl"]:
            # Set up the cmap
            cmap = "bwr"

            # Sert up the levels
            levels = np.array(
                [
                    1004,
                    1006,
                    1008,
                    1010,
                    1012,
                    1014,
                    1016,
                    1018,
                    1020,
                    1022,
                    1024,
                    1026,
                ]
            )
        elif var_name in ["uas", "vas", "sfcWind"]:
            if var_name in ["uas", "sfcWind"]:
                cmap = "PRGn"
            else:
                cmap = "PrGn"
            levels = np.array(
                [
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8,
                    9,
                    10,
                ]
            )
        else:
            raise ValueError(
                f"Variable name {var_name} not recognised. Must be tas, uas or vas."
            )

    # every other tick
    levels_ticks = np.arange(
        np.min(levels),
        np.max(levels) + 1,
        2,
    )

    if var_name in ["psl", "tas"]:
        levels_diff = np.array(
            [
                -4,
                -3,
                -2,
                -1,
                1,
                2,
                3,
                4,
            ]
        )
    elif var_name in ["uas", "vas", "sfcWind"]:
        levels_diff = np.array(
            [
                -0.5,
                -0.45,
                -0.4,
                -0.35,
                -0.3,
                -0.25,
                -0.2,
                -0.15,
                -0.1,
                -0.05,
                0.05,
                0.10,
                0.15,
                0.20,
                0.25,
                0.30,
                0.35,
                0.40,
                0.45,
                0.50,
            ]
        )
    else:
        raise ValueError(
            f"Variable name {var_name} not recognised. Must be tas, uas or vas."
        )

    # set up the ticks
    levels_diff_ticks = np.arange(
        np.min(levels_diff),
        np.max(levels_diff) + 1,
        0.1,
    )

    nrows = len(quartiles)

    # Set up the figure
    fig, axs = plt.subplots(
        ncols=2,
        nrows=nrows,
        figsize=figsize,
        layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Dynamically set up the axes in rows
    axes_rows = [[axs[i, 0], axs[i, 1]] for i in range(nrows)]

    # Loop over the axes rows
    for i, (left_col_full, right_col_diff) in enumerate(axes_rows):
        # Set up a copy of the obs arr here
        subset_arr_this_obs = subset_arr_obs.copy()

        # extract the highest value of the quartiles as the upper bound
        quartiles_this = quartiles[i]

        # Set up the upper and lower quartile
        # Dynamically set up the upper and lower quartile
        upper_quartile = max(quartiles_this)  # Largest value in the tuple
        lower_quartile = min(quartiles_this)

        # Quantify the lower and upper bouns of the quartile
        lower_bound_this = np.quantile(
            subset_df_obs[quartiles_var_name].values,
            lower_quartile,
        )
        upper_bound_this = np.quantile(
            subset_df_obs[quartiles_var_name].values,
            upper_quartile,
        )

        # Set up the subset df for this quartile
        subset_df_obs_this_quartile = subset_df_obs[
            (subset_df_obs[quartiles_var_name] >= lower_bound_this)
            & (subset_df_obs[quartiles_var_name] < upper_bound_this)
        ].copy()

        # Set up the array to store the values
        subset_arr_this_obs_full = np.zeros(
            (
                len(subset_df_obs_this_quartile),
                subset_arr_this_obs.shape[1],
                subset_arr_this_obs.shape[2],
            )
        )

        # Extract the dates for this subset
        dates_obs_this_quartile = subset_df_obs_this_quartile["time"].values

        # Formate these as datetimes
        dates_obs_this_quartile_dt = [
            datetime.strptime(date, "%Y-%m-%d") for date in dates_obs_this_quartile
        ]

        # If var_name is psl
        if var_name == "psl":
            subset_dates_cf = []
            # format the subset dates to extract
            for date in dates_obs_this_quartile_dt:
                date_this_cf = cftime.DatetimeGregorian(
                    date.year, date.month, date.day, hour=11, calendar="gregorian"
                )
                subset_dates_cf.append(date_this_cf)
        elif var_name == "tas":
            # format the subset dates
            subset_dates_cf = []
            # format the subset dates to extract
            for date in dates_obs_this_quartile_dt:
                date_this_cf = cftime.DatetimeProlepticGregorian(
                    date.year,
                    date.month,
                    date.day,
                    hour=0,
                    calendar="proleptic_gregorian",
                )
                subset_dates_cf.append(date_this_cf)
        elif var_name in ["uas", "vas", "sfcWind"]:
            # format the subset dates
            subset_dates_cf = []
            # format the subset dates to extract
            for date in dates_obs_this_quartile_dt:
                date_this_cf = cftime.DatetimeProlepticGregorian(
                    date.year,
                    date.month,
                    date.day,
                    hour=0,
                    calendar="proleptic_gregorian",
                )
                subset_dates_cf.append(date_this_cf)
        else:
            raise ValueError(
                f"Variable name {var_name} not recognised. Must be tas, uas or vas."
            )

        # Set up an empty list for the indices of these dates
        indices_dates_obs_this_quartile = []

        # Loop over the dates in the subset dates
        for date_this in subset_dates_cf:
            index_this = np.where(np.array(dates_list_obs) == date_this)[0][0]
            indices_dates_obs_this_quartile.append(index_this)

        # Apply these indices to the subset_arr_this_obs
        subset_arr_this_obs_quartile = subset_arr_this_obs[
            indices_dates_obs_this_quartile, :, :
        ]

        # Get the N obs this
        N_obs_this = subset_arr_this_obs_quartile.shape[0]

        # Take the mean of this
        subset_arr_this_obs_mean = np.mean(subset_arr_this_obs_quartile, axis=0)

        # if the second subset df obs is not none
        if second_subset_df_obs is not None:
            print("Calculating second subset df obs")
            # Set up a copy of the obs arr here
            subset_arr_this_obs_second = second_subset_arr_obs.copy()

            # Set up the upper and lower quartile
            quartiles_this_second = second_quartiles[i]

            upper_quartile_second = max(
                quartiles_this_second
            )  # Largest value in the tuple
            lower_quartile_second = min(quartiles_this_second)

            # Quantify the lower and upper bouns of the quartile
            lower_bound_this_second = np.quantile(
                second_subset_df_obs[second_quartiles_var_name].values,
                lower_quartile_second,
            )
            upper_bound_this_second = np.quantile(
                second_subset_df_obs[second_quartiles_var_name].values,
                upper_quartile_second,
            )

            # Set up the subset df for this quartile
            subset_df_obs_this_quartile_second = second_subset_df_obs[
                (
                    second_subset_df_obs[second_quartiles_var_name]
                    >= lower_bound_this_second
                )
                & (
                    second_subset_df_obs[second_quartiles_var_name]
                    < upper_bound_this_second
                )
            ].copy()

            # Extract the dates for this subset
            dates_obs_this_quartile_second = subset_df_obs_this_quartile_second[
                "time"
            ].values

            # Formate these as datetimes
            dates_obs_this_quartile_second_dt = [
                datetime.strptime(date, "%Y-%m-%d")
                for date in dates_obs_this_quartile_second
            ]

            # If var_name is psl
            if var_name == "psl":
                subset_dates_cf = []
                # format the subset dates to extract
                for date in dates_obs_this_quartile_second_dt:
                    date_this_cf = cftime.DatetimeGregorian(
                        date.year, date.month, date.day, hour=11, calendar="gregorian"
                    )
                    subset_dates_cf.append(date_this_cf)
            elif var_name == "tas":
                # format the subset dates
                subset_dates_cf = []
                # format the subset dates to extract
                for date in dates_obs_this_quartile_second_dt:
                    date_this_cf = cftime.DatetimeProlepticGregorian(
                        date.year,
                        date.month,
                        date.day,
                        hour=0,
                        calendar="proleptic_gregorian",
                    )
                    subset_dates_cf.append(date_this_cf)
            elif var_name in ["uas", "vas", "sfcWind"]:
                # format the subset dates
                subset_dates_cf = []
                # format the subset dates to extract
                for date in dates_obs_this_quartile_second_dt:
                    date_this_cf = cftime.DatetimeProlepticGregorian(
                        date.year,
                        date.month,
                        date.day,
                        hour=0,
                        calendar="proleptic_gregorian",
                    )
                    subset_dates_cf.append(date_this_cf)
            else:
                raise ValueError(
                    f"Variable name {var_name} not recognised. Must be tas, uas or vas."
                )

            # Set up an empty list for the indices of these dates
            indices_dates_obs_this_quartile_second = []

            # Loop over the dates in the subset dates
            for date_this in subset_dates_cf:
                index_this = np.where(np.array(second_dates_list_obs) == date_this)[0][
                    0
                ]
                indices_dates_obs_this_quartile_second.append(index_this)

            # Apply these indices to the subset_arr_this_obs
            subset_arr_this_obs_quartile_second = subset_arr_this_obs_second[
                indices_dates_obs_this_quartile_second, :, :
            ]

            # Take the mean of this
            subset_arr_this_obs_mean_second = np.mean(
                subset_arr_this_obs_quartile_second, axis=0
            )

        # if the anoms flag is true
        if anoms_flag:
            # Calculate the anomalies
            subset_arr_this_obs_mean = subset_arr_this_obs_mean - clim_arr_obs

        # If subset_arr_this_model_full_second is not None
        if second_subset_df_obs is not None:
            print("Quantifying differences between first and second")

            if anoms_flag:
                subset_arr_this_obs_mean_second = (
                    subset_arr_this_obs_mean_second - clim_arr_obs
                )
            if var_name == "psl":
                subset_arr_this_obs_mean_diff = (
                    subset_arr_this_obs_mean / 100
                    - subset_arr_this_obs_mean_second / 100
                )
            else:
                subset_arr_this_obs_mean_diff = (
                    subset_arr_this_obs_mean - subset_arr_this_obs_mean_second
                )

            # levels = levels_diff
            levels_ticks = np.array(
                [
                    -4,
                    -3,
                    -2,
                    -1,
                    1,
                    2,
                    3,
                    4,
                ]
            )

            levels_diffs = np.arange(
                np.min(levels_diff),
                np.max(levels_diff) + 1,
                0.1,
            )

        # if the var names is psl
        if var_name == "psl":
            # Convert the psl to hPa
            subset_arr_this_obs_mean = subset_arr_this_obs_mean / 100.0

        # if i == 0, then set up the warmest composite
        if i == 0:
            diff_composite = subset_arr_this_obs_mean

        # print the min, max and mean of subset_arr_this_obs_mean
        print(
            f"Min: {np.min(subset_arr_this_obs_mean):.2f}, "
            f"Max: {np.max(subset_arr_this_obs_mean):.2f}, "
            f"Mean: {np.mean(subset_arr_this_obs_mean):.2f}"
        )

        if second_subset_df_obs is not None:
            print("plotting the difference between the first and second quartile")

            print(
                f"Min: {np.min(subset_arr_this_obs_mean_diff):.2f}, "
                f"Max: {np.max(subset_arr_this_obs_mean_diff):.2f}, "
                f"Mean: {np.mean(subset_arr_this_obs_mean_diff):.2f}"
            )

            # Plot the full field on the left
            im_full = left_col_full.contourf(
                lons,
                lats,
                subset_arr_this_obs_mean_diff,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels_diff,
                extend="both",
            )
        else:
            # Plot the full field on the left
            im_full = left_col_full.contourf(
                lons,
                lats,
                subset_arr_this_obs_mean,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels,
                extend="both",
            )

        # if gridbox is not none
        if gridbox is not None:
            # if there is more than one gridbox
            if isinstance(gridbox, list):
                print("Calculating difference in gridbox fields")

                # Hard code the n_box and south box for delta P
                # n_box = dicts.uk_n_box_corrected
                # s_box = dicts.uk_s_box_corrected
                # extract the lat 1 from the two gridboxes
                n_box = gridbox[0]
                s_box = gridbox[1]

                # Extract the n_box lats and lons
                lat1_box_n, lat2_box_n = n_box["lat1"], n_box["lat2"]
                lon1_box_n, lon2_box_n = n_box["lon1"], n_box["lon2"]

                # Extract the s_box lats and lons
                lat1_box_s, lat2_box_s = s_box["lat1"], s_box["lat2"]
                lon1_box_s, lon2_box_s = s_box["lon1"], s_box["lon2"]

                # Find the indices of the lats which correspond to the gridbox
                lat1_idx_n = np.argmin(np.abs(lats - lat1_box_n))
                lat2_idx_n = np.argmin(np.abs(lats - lat2_box_n))

                # Find the indices of the lons which correspond to the gridbox
                lon1_idx_n = np.argmin(np.abs(lons - lon1_box_n))
                lon2_idx_n = np.argmin(np.abs(lons - lon2_box_n))

                # Find the indices of the lats which correspond to the gridbox
                lat1_idx_s = np.argmin(np.abs(lats - lat1_box_s))
                lat2_idx_s = np.argmin(np.abs(lats - lat2_box_s))

                # Find the indices of the lons which correspond to the gridbox
                lon1_idx_s = np.argmin(np.abs(lons - lon1_box_s))
                lon2_idx_s = np.argmin(np.abs(lons - lon2_box_s))

                # Add the gridbox to the plot
                left_col_full.plot(
                    [lon1_box_n, lon2_box_n, lon2_box_n, lon1_box_n, lon1_box_n],
                    [lat1_box_n, lat1_box_n, lat2_box_n, lat2_box_n, lat1_box_n],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Add the gridbox to the plot
                left_col_full.plot(
                    [lon1_box_s, lon2_box_s, lon2_box_s, lon1_box_s, lon1_box_s],
                    [lat1_box_s, lat1_box_s, lat2_box_s, lat2_box_s, lat1_box_s],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Calculate the mean in the gridbox
                gridbox_mean_n = np.mean(
                    subset_arr_this_obs_mean[
                        lat1_idx_n : lat2_idx_n + 1, lon1_idx_n : lon2_idx_n + 1
                    ]
                )

                gridbox_mean_s = np.mean(
                    subset_arr_this_obs_mean[
                        lat1_idx_s : lat2_idx_s + 1, lon1_idx_s : lon2_idx_s + 1
                    ]
                )

                # Include a textbox in the top left for the gridbox mean
                # to two S.F.
                left_col_full.text(
                    0.05,
                    0.95,
                    f"delta P = {gridbox_mean_n - gridbox_mean_s:.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=left_col_full.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )
            else:
                print("Calculating absolute value in gridbox")

                # Extract the lons and lats from the gridbox
                lat1_box, lat2_box = gridbox["lat1"], gridbox["lat2"]
                lon1_box, lon2_box = gridbox["lon1"], gridbox["lon2"]

                # Find the indices of the lats which correspond to the gridbox
                lat1_idx = np.argmin(np.abs(lats - lat1_box))
                lat2_idx = np.argmin(np.abs(lats - lat2_box))

                # Find the indices of the lons which correspond to the gridbox
                lon1_idx = np.argmin(np.abs(lons - lon1_box))
                lon2_idx = np.argmin(np.abs(lons - lon2_box))

                # Add the gridbox to the plot
                left_col_full.plot(
                    [lon1_box, lon2_box, lon2_box, lon1_box, lon1_box],
                    [lat1_box, lat1_box, lat2_box, lat2_box, lat1_box],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Calculate the mean in the gridbox
                gridbox_mean = np.mean(
                    subset_arr_this_obs_mean[
                        lat1_idx : lat2_idx + 1, lon1_idx : lon2_idx + 1
                    ]
                )

                # Include a textbox in the top left for the gridbox mean
                # to two S.F.
                left_col_full.text(
                    0.05,
                    0.95,
                    f"Gridbox mean = {gridbox_mean:.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=left_col_full.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

        # add coastlines to the left plot
        left_col_full.coastlines()

        # Include a textbox in the bottom right for N
        left_col_full.text(
            0.95,  # x-coordinate (right edge)
            0.05,  # y-coordinate (bottom edge)
            f"N = {N_obs_this}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=left_col_full.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Include the qunatile range in a textbox in the bottom left
        left_col_full.text(
            0.05,
            0.05,
            f"{lower_quartile:.2f} - {upper_quartile:.2f}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=left_col_full.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        if second_subset_df_obs is not None:
            # print the min, max and mean of subset_arr_this_obs_mean_second
            print(
                f"Min: {np.min(subset_arr_this_obs_mean_second):.2f}, "
                f"Max: {np.max(subset_arr_this_obs_mean_second):.2f}, "
                f"Mean: {np.mean(subset_arr_this_obs_mean_second):.2f}"
            )

            # Plot the difference on the right
            im_diff = right_col_diff.contourf(
                lons,
                lats,
                (subset_arr_this_obs_mean_second / 100),
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels,
                extend="both",
            )
        else:
            # Plot the difference on the right
            im_diff = right_col_diff.contourf(
                lons,
                lats,
                (subset_arr_this_obs_mean - diff_composite),
                cmap="PRGn",
                transform=ccrs.PlateCarree(),
                levels=levels_diff,
                extend="both",
            )

        # if the gridbox is not none
        if gridbox is not None:
            if isinstance(gridbox, list):
                # Calculate the difference in the gridbox fields
                print("Calculating difference in gridbox fields")

                # subset the gridbox
                gridbox_mean_n = np.mean(
                    (subset_arr_this_obs_mean - diff_composite)[
                        lat1_idx_n : lat2_idx_n + 1, lon1_idx_n : lon2_idx_n + 1
                    ]
                )
                gridbox_mean_s = np.mean(
                    (subset_arr_this_obs_mean - diff_composite)[
                        lat1_idx_s : lat2_idx_s + 1, lon1_idx_s : lon2_idx_s + 1
                    ]
                )

                # Include a textbox in the top left for the gridbox mean
                # to two S.F.
                right_col_diff.text(
                    0.05,
                    0.95,
                    f"delta P = {gridbox_mean_n - gridbox_mean_s:.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=right_col_diff.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

                # plot the gridboxes
                right_col_diff.plot(
                    [lon1_box_n, lon2_box_n, lon2_box_n, lon1_box_n, lon1_box_n],
                    [lat1_box_n, lat1_box_n, lat2_box_n, lat2_box_n, lat1_box_n],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                right_col_diff.plot(
                    [lon1_box_s, lon2_box_s, lon2_box_s, lon1_box_s, lon1_box_s],
                    [lat1_box_s, lat1_box_s, lat2_box_s, lat2_box_s, lat1_box_s],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )
            else:
                # Add the gridbox to the plot
                right_col_diff.plot(
                    [lon1_box, lon2_box, lon2_box, lon1_box, lon1_box],
                    [lat1_box, lat1_box, lat2_box, lat2_box, lat1_box],
                    color="green",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Calculate the mean in the gridbox
                gridbox_mean = np.mean(
                    (subset_arr_this_obs_mean - diff_composite)[
                        lat1_idx : lat2_idx + 1, lon1_idx : lon2_idx + 1
                    ]
                )

                # Include a textbox in the top left for the gridbox mean
                # to two S.F.
                right_col_diff.text(
                    0.05,
                    0.95,
                    f"Gridbox mean = {gridbox_mean:.2f}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=right_col_diff.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

        # add coastlines to the right plot
        right_col_diff.coastlines()

        # Include a textbox in the bottom right for N
        right_col_diff.text(
            0.95,  # x-coordinate (right edge)
            0.05,  # y-coordinate (bottom edge)
            f"N = {N_obs_this}",
            horizontalalignment="right",
            verticalalignment="bottom",
            transform=right_col_diff.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Include the quantile range in a textbox in the bottom left
        right_col_diff.text(
            0.05,
            0.05,
            f"{lower_quartile:.2f} - {upper_quartile:.2f}",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=right_col_diff.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # if i is the final index of nrows, then set up the cbars
        if i == nrows - 1:
            if second_subset_df_obs is not None:
                cbar_full = fig.colorbar(
                    im_full,
                    ax=left_col_full,
                    orientation="horizontal",
                    pad=0.05,
                    shrink=0.8,
                )

                cbar_full.set_ticks(levels_diff_ticks)
            else:
                cbar_full = fig.colorbar(
                    im_full,
                    ax=left_col_full,
                    orientation="horizontal",
                    pad=0.05,
                    shrink=0.8,
                )

                cbar_full.set_ticks(levels_ticks)

            cbar_diff = fig.colorbar(
                im_diff,
                ax=right_col_diff,
                orientation="horizontal",
                pad=0.05,
                shrink=0.8,
            )

            cbar_diff.set_ticks(levels_diff)

        if i == 0:
            # Set up the titles for the left and right columns
            left_col_full.set_title("Full field", fontsize=12, fontweight="bold")
            right_col_diff.set_title(
                "Difference from warmest quartile", fontsize=12, fontweight="bold"
            )

    return None


# Define a function to plot the multi var composites (psl, tas, wind)
# For the observations
def plot_multi_var_composites_obs(
    subset_df_obs: pd.DataFrame,
    subset_arrs_list_obs: List[np.ndarray],
    dates_list_obs: List[str],
    var_names: List[str],
    lats_paths: List[str],
    lons_paths: List[str],
    effective_dec_years: List[int],
    figsize: Tuple[int, int] = (10, 10),
    anoms_flag: bool = False,
    clim_arrs_list_obs: Optional[List[np.ndarray]] = None,
    cluster_assign_name: Optional[str] = None,
):
    """
    Plots multi-variable composites for observations.

    Parameters:
    ===========

        subset_df_obs (pd.DataFrame): The subset dataframe for observations.
        subset_arrs_list_obs (List[np.ndarray]): List of subset arrays for observations.
        dates_list_obs (List[str]): List of dates for observations.
        var_names (List[str]): List of variable names to plot.
        lats_paths (List[str]): List of paths to latitude files.
        lons_paths (List[str]): List of paths to longitude files.
        effective_dec_years (List[int]): List of effective decade years.
        figsize (Tuple[int, int]): Size of the figure.
        anoms_flag (bool): Flag to indicate if anomalies should be plotted.
        clim_arrs_list_obs (Optional[List[np.ndarray]]): List of climatology arrays for observations.
        cluster_assign_name (Optional[str]): Name of the cluster assignment, if applicable.

    Returns:
    ========

        None
    """

    # Set up the ncols
    ncols = 3
    nrows = len(effective_dec_years)

    # Set up the figure
    fig, axs = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=figsize,
        layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
        gridspec_kw={"width_ratios": [1.5, 1, 1]},  # First column is 1.5x wider
    )

    # print the head of the subset df obs
    print("Subset DataFrame Head:")
    print(subset_df_obs.head())

    # Print the tail of the subset df obs
    print("Subset DataFrame Tail:")
    print(subset_df_obs.tail())

    # Strip efefctive dec year column into just YYYY and format as int
    subset_df_obs["effective_dec_year"] = (
        subset_df_obs["effective_dec_year"].str[:4].astype(int)
    )

    # Print the unique effective decade years
    unique_effective_dec_years = subset_df_obs["effective_dec_year"].unique()

    # Print the firsta dn last 5 unique effective decade years
    print("Unique Effective Decade Years (First 5):", unique_effective_dec_years[:5])
    print("Unique Effective Decade Years (Last 5):", unique_effective_dec_years[-5:])

    # Print the type of these
    print("Type of Effective Decade Years:", type(unique_effective_dec_years))

    # Find the indices of the effective dec years provided in the first subset df
    effective_dec_years_indices = [
        np.where(subset_df_obs["effective_dec_year"] == year)[0][0]
        for year in effective_dec_years
    ]

    # Print the effective dec year indices
    print("Effective Decade Year Indices:", effective_dec_years_indices)

    # Loop over the effective dec year indices
    for i, year_index in enumerate(effective_dec_years_indices):
        # Print the effective dec year for this index
        print(
            f"Effective Decade Year for Index {i}: {unique_effective_dec_years[year_index]}"
        )

        # Loop over the variable names
        for j, var_name in enumerate(var_names):
            # if the variable is tas
            if var_name == "tas":
                cmap = "coolwarm"
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
                # If anoms flasg is tru and clim_arrs_list_obs is not None
                if anoms_flag and clim_arrs_list_obs is not None:
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
            elif var_name == "sfcWind":
                cmap = "YlGnBu"
                levels = np.array(
                    [
                        0.5,
                        1,
                        1.5,
                        2,
                        2.5,
                        3,
                        3.5,
                        4,
                        4.5,
                        5,
                    ]
                )
                if anoms_flag and clim_arrs_list_obs is not None:
                    cmap = "PRGn"
                    levels = np.array(
                        [
                            -5,
                            -4,
                            -3,
                            -2,
                            -1,
                            1,
                            2,
                            3,
                            4,
                            5,
                        ]
                    )
            elif var_name in ["uas", "vas"]:
                cmap = "PRGn"
                levels = np.array(
                    [
                        -4,
                        -3.5,
                        -3,
                        -2.5,
                        -2,
                        -1.5,
                        -1,
                        -0.5,
                        0.5,
                        1,
                        1.5,
                        2,
                        2.5,
                        3,
                        3.5,
                        4,
                    ]
                )
            elif var_name == "psl":
                # Set up the levels for plotting absolute values
                # Set up the cmap
                cmap = "coolwarm"

                # Sert up the levels
                levels = np.array(
                    [
                        1004,
                        1006,
                        1008,
                        1010,
                        1012,
                        1014,
                        1016,
                        1018,
                        1020,
                        1022,
                        1024,
                        1026,
                    ]
                )
            else:
                raise ValueError(f"Variable {var_name} not supported.")

            # Set up the axes for this variable
            ax_this = axs[i, j]

            # Extract the lats and lons
            lats_this = np.load(lats_paths[j])
            lons_this = np.load(lons_paths[j])

            # Set up the date model this
            date_model_this = subset_df_obs.iloc[year_index]["time"]

            # Set up the dnw this
            dnw_val_this = subset_df_obs.iloc[year_index]["demand_net_wind_max"]
            demand_this = subset_df_obs.iloc[year_index]["data_c_dt_UK_demand"]
            wind_gen_this = subset_df_obs.iloc[year_index]["total_gen"]

            # Set up the cluster this
            if cluster_assign_name is not None:
                cluster_this = subset_df_obs.iloc[year_index][cluster_assign_name]

            # Extract the data this
            subset_arr_this_obs = subset_arrs_list_obs[j][year_index, :, :]

            # If the var name is psl, divide by 100 to get hPa
            if var_name == "psl":
                subset_arr_this_obs = subset_arr_this_obs / 100.0
            elif var_name == "tas" and not anoms_flag:
                # Convert the tas to Celsius
                subset_arr_this_obs = subset_arr_this_obs - 273.15
            elif var_name in ["tas", "vas", "sfcWind"] and anoms_flag:
                assert (
                    clim_arrs_list_obs is not None
                ), "Climatology arrays must be provided for anomaly calculation."

                subset_arr_this_obs = subset_arr_this_obs - clim_arrs_list_obs[j]
            else:
                raise ValueError(
                    f"Variable {var_name} not supported for anomaly calculation."
                )

            # Plot the model data on the right
            im_model = ax_this.contourf(
                lons_this,
                lats_this,
                subset_arr_this_obs,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels,
                extend="both",
            )

            # if the var_name is psl, then plot absolute contours
            if var_name == "psl":
                # Plot the absolute contours
                contours = ax_this.contour(
                    lons_this,
                    lats_this,
                    subset_arr_this_obs,
                    levels=levels,
                    colors="black",
                    linewidths=0.5,
                    transform=ccrs.PlateCarree(),
                )

                ax_this.clabel(
                    contours,
                    levels,
                    fmt="%.0f",
                    fontsize=6,
                    inline=True,
                    inline_spacing=0.0,
                )

                # include a textbox in the bottom right with the demand and wind gen
                ax_this.text(
                    0.95,
                    0.05,
                    f"Demand = {demand_this:.2f} GW\nWind Gen = {wind_gen_this:.2f} GW",
                    horizontalalignment="right",
                    verticalalignment="bottom",
                    transform=ax_this.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

            # add coastlines to all of these
            ax_this.coastlines()

            # Include a textbox in the top right for N
            ax_this.text(
                0.95,
                0.95,
                f"{date_model_this}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=ax_this.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )

            # if the variable is psl
            if var_name == "psl":
                # Include the value in the bottom left
                ax_this.text(
                    0.05,
                    0.05,
                    f"DnW = {dnw_val_this:.2f}",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    transform=ax_this.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )
            elif var_name == "tas":
                # Include the value in the bottom left
                ax_this.text(
                    0.05,
                    0.05,
                    f"Demand = {demand_this:.2f} GW",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    transform=ax_this.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )
            elif var_name in ["uas", "vas", "sfcWind"]:
                # Include the value in the bottom left
                ax_this.text(
                    0.05,
                    0.05,
                    f"Wind Gen = {wind_gen_this:.2f} GW",
                    horizontalalignment="left",
                    verticalalignment="bottom",
                    transform=ax_this.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

            if cluster_assign_name is not None:
                # Include the cluster assignment in the top left
                ax_this.text(
                    0.05,
                    0.95,
                    f"Cluster = {cluster_this}",
                    horizontalalignment="left",
                    verticalalignment="top",
                    transform=ax_this.transAxes,
                    fontsize=12,
                    bbox=dict(facecolor="white", alpha=0.5),
                )

            # if i == 2
            if i == len(effective_dec_years_indices) - 1:
                # add colorbar
                # cbar = plt.colorbar(mymap, orientation='horizontal', shrink=0.7, pad=0.1)
                # cbar.set_label('SST [C]', rotation=0, fontsize=10)
                # cbar.ax.tick_params(labelsize=7, length=0)

                # add the colorbar for wind
                cbar = fig.colorbar(
                    im_model,
                    ax=ax_this,
                    orientation="horizontal",
                    pad=0.05,
                    shrink=0.8,
                )

                # depending on the i set the labels
                if j == 0:

                    # Set up the ticks
                    levels = np.array(
                        [
                            1004,
                            1008,
                            1012,
                            1015,
                            1018,
                            1022,
                            1026,
                        ]
                    )

                    cbar.set_label("hPa", rotation=0, fontsize=12)
                elif j == 1:
                    cbar.set_label("°C", rotation=0, fontsize=12)
                elif j == 2:

                    cbar.set_label("m/s", rotation=0, fontsize=12)

                cbar.set_ticks(levels)

            # Set the title for each subplot based on `i`
            if i == 0 and j == 0:
                ax_this.set_title("Obs daily MSLP", fontsize=12, fontweight="bold")
            elif i == 0 and j == 1:
                ax_this.set_title("Obs daily airT", fontsize=12, fontweight="bold")
                if anoms_flag:
                    ax_this.set_title(
                        "Obs daily airT anomalies", fontsize=12, fontweight="bold"
                    )
            elif i == 0 and j == 2:
                ax_this.set_title("Obs daily sfcWind", fontsize=12, fontweight="bold")
                if anoms_flag:
                    ax_this.set_title(
                        "Obs daily sfcWind anomalies", fontsize=12, fontweight="bold"
                    )

    return None


# Function to get the cluster fraction
def get_cluster_fraction(m, label):
    return (m.labels_ == label).sum() / (m.labels_.size * 1.0)


# Set up a function to do the kmeans clustering and plotting
def kmeans_clustering_and_plotting(
    subset_arr: np.ndarray,
    lats_path: str,
    lons_path: str,
    n_clusters: int = 5,
    cmap: str = "RdBu_r",
    figsize: Tuple[int, int] = (10, 10),
    silhouette_threshold: float = 0.3,
) -> Tuple[KMeans, np.ndarray, Dict]:
    """
    Perform k-means clustering on the provided subset array and plot the results.

    Parameters:
    ===========
        subset_arr (np.ndarray): The subset array to cluster.
        lats_path (str): Path to the latitude data.
        lons_path (str): Path to the longitude data.
        n_clusters (int): Number of clusters for k-means.
        cmap (str): Colormap for the plot (default is "RdBu_r").
        figsize (Tuple[int, int]): Size of the figure (default is (10, 10)).
        silhouette_threshold (float): Threshold for silhouette score to determine cluster quality.

    Returns:
    ========
        Tuple containing:
        - KMeans model object
        - Cluster assignments for each time step (with -1 for "no-type")
        - Dictionary with cluster statistics
    """
    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # Extract the nt, nlats, and nlons from the subset_arr
    nt, nx, ny = subset_arr.shape

    # Reshape the subset_arr to 2D for clustering
    subset_arr_reshaped = subset_arr.reshape(nt, ny * nx)

    # Perform k-means clustering
    m = KMeans(n_clusters=n_clusters, random_state=0).fit(subset_arr_reshaped)

    # Get cluster assignments for each time step
    cluster_labels = m.labels_

    # Calculate silhouette scores for each sample to identify weak assignments
    silhouette_scores = silhouette_score(
        subset_arr_reshaped, cluster_labels, sample_size=min(1000, nt)
    )

    # Create individual silhouette scores (approximation)
    distances_to_centers = np.zeros(nt)
    for i in range(nt):
        assigned_cluster = cluster_labels[i]
        distances_to_centers[i] = np.linalg.norm(
            subset_arr_reshaped[i] - m.cluster_centers_[assigned_cluster]
        )

    # Assign "no-type" cluster (-1) to weak assignments
    # You can adjust this threshold based on your data
    distance_threshold = np.percentile(distances_to_centers, 90)  # Top 10% of distances
    cluster_assignments = cluster_labels.copy()
    cluster_assignments[distances_to_centers > distance_threshold] = -1

    # Create statistics dictionary
    cluster_stats = {
        "total_time_steps": nt,
        "n_clusters": n_clusters,
        "cluster_counts": {},
        "cluster_percentages": {},
        "silhouette_score": silhouette_scores,
        "distance_threshold": distance_threshold,
    }

    # Calculate cluster statistics
    unique_clusters = np.unique(cluster_assignments)
    for cluster_id in unique_clusters:
        count = np.sum(cluster_assignments == cluster_id)
        cluster_stats["cluster_counts"][cluster_id] = count
        cluster_stats["cluster_percentages"][cluster_id] = count / nt * 100

    # Set up the ncols and nrow
    # if n_clusters is 4
    ncols = 2 if n_clusters == 4 else 3
    nrows = int(np.ceil(n_clusters / ncols))

    # Set up the figure
    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        layout="constrained",
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    # Set up the tags e.g. a - x depending on n_clusters
    tags = [chr(97 + i) for i in range(n_clusters)]

    # Set up the cluster plots
    cluster_plots = []

    # Loop over the clusters and plot the centroid
    for i in range(m.n_clusters):
        # Get the centroid for this cluster
        centroid_field = m.cluster_centers_[i, :].reshape(nx, ny)

        # Set up the axes for this cluster
        ax_this = axs[i // ncols, i % ncols] if nrows > 1 else axs[i]

        # Plot the centroid field
        im = ax_this.contourf(
            lons,
            lats,
            centroid_field / 100.0,  # Convert to hPa if units are hPa
            cmap=cmap,
            transform=ccrs.PlateCarree(),
            extend="both",
        )

        cluster_plots.append(im)
        ax_this.coastlines()

        # Get count for this cluster (excluding no-type assignments)
        cluster_count = np.sum(cluster_assignments == i)
        cluster_percentage = cluster_count / nt * 100

        title_this = (
            f"Cluster {i + 1} ({cluster_count} days, {cluster_percentage:.1f}%)"
        )
        ax_this.set_title(title_this, fontsize=12, fontweight="bold")

        # Add a tag to the plot
        ax_this.text(
            0.05,
            0.95,
            f"({tags[i]})",
            transform=ax_this.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="none"),
        )

        # add a colorbar for the each plot
        cbar_this = fig.colorbar(
            im,
            ax=ax_this,
            orientation="horizontal",
            pad=0.05,
            shrink=0.8,
        )
        cbar_this.set_label("hPa", rotation=0, fontsize=10)

    plt.show()

    # Print cluster assignment summary
    print(f"\nCluster Assignment Summary:")
    print(f"Total time steps: {nt}")
    print(f"Silhouette score: {silhouette_scores:.3f}")
    print(f"Distance threshold for no-type: {distance_threshold:.2f}")
    print(f"\nCluster assignments:")
    for cluster_id in sorted(unique_clusters):
        count = cluster_stats["cluster_counts"][cluster_id]
        percentage = cluster_stats["cluster_percentages"][cluster_id]
        if cluster_id == -1:
            print(f"  No-type cluster: {count} days ({percentage:.1f}%)")
        else:
            print(f"  Cluster {cluster_id + 1}: {count} days ({percentage:.1f}%)")

    return m, cluster_assignments, cluster_stats


# Define a function to create and plot cluster composites
def create_and_plot_cluster_composites(
    subset_arr: np.ndarray,
    cluster_assignments: np.ndarray,
    var_name: str,
    lats_path: str,
    lons_path: str,
    cmap: str = "RdBu_r",
    arr_clim: Optional[np.ndarray] = None,
    exclude_no_type: bool = True,
    figsize: Tuple[int, int] = (10, 10),
    levels: Optional[np.ndarray] = None,
) -> None:
    """
    Create and plot cluster composites for a given variable.

    Parameters:
    ===========
        subset_arr (np.ndarray): The subset array to analyze.
        cluster_assignments (np.ndarray): Cluster assignments for each time step.
        var_name (str): Name of the variable being analyzed.
        lats_path (str): Path to the latitude data.
        lons_path (str): Path to the longitude data.
        arr_clim (Optional[np.ndarray]): Climatology array for the variable (if available
        exclude_no_type (bool): Whether to exclude the "no-type" cluster (-1).
        figsize (Tuple[int, int]): Size of the figure.

    Returns:
    ========
        None

    """

    # Set up a dictionary for the composites
    composites = {}
    unique_clusters = np.unique(cluster_assignments)

    if exclude_no_type:
        unique_clusters = unique_clusters[unique_clusters != -1]

    for cluster_id in unique_clusters:
        # Get the indices for this cluster
        cluster_indices = np.where(cluster_assignments == cluster_id)[0]

        if len(cluster_indices) == 0:
            print(f"No data for cluster {cluster_id + 1}, skipping.")
            continue

        arr_this_cluster = subset_arr[cluster_indices, :, :]

        if arr_clim is not None:
            print(
                f"Calculating composite for cluster {cluster_id + 1} with climatology."
            )
            # Calculate the composite by averaging over the cluster indices
            composite = np.mean(arr_this_cluster, axis=0) - arr_clim
        else:
            composite = np.mean(arr_this_cluster, axis=0)

        composites[cluster_id] = {
            "composite": composite,
            "n_samples": len(cluster_indices),
            "time_indices": cluster_indices,
        }

    # Load the lats and lons
    lats = np.load(lats_path)
    lons = np.load(lons_path)

    # Set up the n_clusters
    n_clusters = len(composites)

    # Set up the ncols and nrow
    # if n_clusters is 4
    ncols = 2 if n_clusters == 4 else 3
    nrows = int(np.ceil(n_clusters / ncols))

    fig, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )

    cluster_ids = sorted(composites.keys())

    mins, maxs, mins = [], [], []

    # Loop over the cluster ids and find the min and max
    for i, cluster_id in enumerate(cluster_ids):
        # Get the composite for this cluster
        composite = composites[cluster_id]["composite"]

        # Find the min amd max values
        min_this = np.min(composite)
        max_this = np.max(composite)

        mins.append(min_this)
        maxs.append(max_this)

    # Find the min max value and max min value
    min_val = np.min(mins)
    max_val = np.max(maxs)

    for i, cluster_id in enumerate(cluster_ids):
        # Get the composite for this cluster
        composite = composites[cluster_id]["composite"]

        ax_this = axs[i // ncols, i % ncols] if nrows > 1 else axs[i]

        if levels is not None:
            # Use the provided levels for contouring
            im_this = ax_this.contourf(
                lons,
                lats,
                composite,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                levels=levels,
                extend="both",
            )
        else:
            im_this = ax_this.contourf(
                lons,
                lats,
                composite,
                vmin=min_val,
                vmax=max_val,
                cmap=cmap,
                transform=ccrs.PlateCarree(),
                extend="both",
            )

        ax_this.coastlines()
        # Set the title for this subplot
        title_this = (
            f"Cluster {cluster_id + 1} ({composites[cluster_id]['n_samples']} days)"
        )
        ax_this.set_title(title_this, fontsize=12, fontweight="bold")

    if levels is not None:
        # Set up the colorbar with the provided levels
        cbar = fig.colorbar(
            im_this,
            ax=axs,
            orientation="horizontal",
            pad=0.05,
            shrink=0.8,
        )
        cbar.set_ticks(levels)
        cbar.set_label(f"{var_name} units", rotation=0, fontsize=12)
    else:
        # Add a colorbar for the all subplots
        cbar = fig.colorbar(
            im_this,
            ax=axs,
            orientation="horizontal",
            pad=0.05,
            shrink=0.8,
        )
        cbar.set_label(f"{var_name} units", rotation=0, fontsize=12)

    plt.show()

    return composites


# Define the main function
def main():
    start_time = time.time()

    # Set up the hard coded variables
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"
    obs_df_fname = "block_maxima_obs_demand_net_wind_30-06-2025_2020-2024.csv"
    model_df_fname = "block_maxima_model_demand_net_wind_30-06-2025_2020-2024.csv"
    low_wind_path = "/home/users/benhutch/unseen_multi_year/dfs/model_all_DJF_days_lowest_0-10_percentile_wind_speed.csv"
    higher_wind_path = "/home/users/benhutch/unseen_multi_year/dfs/model_all_DJF_days_40-60_percentile_wind_speed.csv"
    winter_arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/obs/"
    metadata_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/"
    arrs_persist_dir = "/home/users/benhutch/unseen_multi_year/data"
    subset_model_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/subset/"
    model_clim_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_clim/"
    obs_df_high_demand_path = (
        "/home/users/benhutch/unseen_multi_year/dfs/df_obs_high_demand_2025-05-28.csv"
    )
    obs_df_low_temp_path = (
        "/home/users/benhutch/unseen_multi_year/dfs/df_obs_low_temp_2025-05-28.csv"
    )
    delta_p_fpath = (
        "/home/users/benhutch/unseen_multi_year/dfs/ERA5_delta_p_1961_2024_DJF_day.csv"
    )

    season = "DJF"
    time_freq = "day"
    len_winter_days = 5866

    # If the path esists, load in the obs df
    if os.path.exists(os.path.join(dfs_dir, obs_df_fname)):
        obs_df = pd.read_csv(os.path.join(dfs_dir, obs_df_fname), index_col=0)
    else:
        raise FileNotFoundError(f"File {obs_df_fname} does not exist in {dfs_dir}")

    # If the path esists, load in the model df
    if os.path.exists(os.path.join(dfs_dir, model_df_fname)):
        model_df = pd.read_csv(os.path.join(dfs_dir, model_df_fname))
    else:
        raise FileNotFoundError(f"File {model_df_fname} does not exist in {dfs_dir}")

    # Print the head of the model df
    print("Model DataFrame Head:")
    print(model_df.head())

    # Print the tail of the model df
    print("Model DataFrame Tail:")
    print(model_df.tail())

    # Load in the low wind data
    if os.path.exists(low_wind_path):
        low_wind_df = pd.read_csv(low_wind_path)
    else:
        raise FileNotFoundError(f"File {low_wind_path} does not exist")

    # Load in the higher wind data
    if os.path.exists(higher_wind_path):
        higher_wind_df = pd.read_csv(higher_wind_path)
    else:
        raise FileNotFoundError(f"File {higher_wind_path} does not exist")

    # if the high demand path exists then load the data
    if os.path.exists(obs_df_high_demand_path):
        obs_df_high_demand = pd.read_csv(obs_df_high_demand_path)

        # reset the index
        obs_df_high_demand.reset_index(drop=True, inplace=True)

        # rename "Unnamed: 0" as "time"
        if "Unnamed: 0" in obs_df_high_demand.columns:
            obs_df_high_demand.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    else:
        raise FileNotFoundError(f"File {obs_df_high_demand_path} does not exist")

    # if the low temp path exists then load the data
    if os.path.exists(obs_df_low_temp_path):
        obs_df_low_temp = pd.read_csv(obs_df_low_temp_path)

        # reset the index
        obs_df_low_temp.reset_index(drop=True, inplace=True)

        # rename "Unnamed: 0" as "time"
        if "Unnamed: 0" in obs_df_low_temp.columns:
            obs_df_low_temp.rename(columns={"Unnamed: 0": "time"}, inplace=True)
    else:
        raise FileNotFoundError(f"File {obs_df_low_temp_path} does not exist")

    # Set up the directory in which to store the dfs
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"

    # Set up the years test
    test_years = np.arange(1960, 2018 + 1, 1)
    members = np.arange(1, 10 + 1, 1)

    # Set up a list to store the missing fnames
    missing_fnames = []
    missing_fname_years = []

    # Set up an empty dataframe
    df_delta_p_full = pd.DataFrame()
    df_uas_full = pd.DataFrame()
    df_vas_full = pd.DataFrame()

    # Loop over the years
    for year in test_years:
        for member in members:
            # Set up the test fname
            test_fname = (
                f"HadGEM3-GC31-MM_dcppA-hindcast_psl_delta_p_{year}_{member}_day.csv"
            )

            # Set up the test fname for the uas and vas
            test_fname_uas = f"HadGEM3-GC31-MM_dcppA-hindcast_uas_UK_wind_box_{year}_{member}_day.csv"
            test_fname_vas = f"HadGEM3-GC31-MM_dcppA-hindcast_vas_UK_wind_box_{year}_{member}_day.csv"

            # Set up thge output dir
            # Set up the new base dir
            base_dir_new = "/home/users/benhutch/unseen_data/saved_dfs"

            # Set up the new output directory
            new_output_dir = os.path.join(
                base_dir_new,
                "delta_p",
                str(year),
            )

            # Cehck if the file exists
            if os.path.exists(os.path.join(new_output_dir, test_fname)):
                # Load the df
                df_delta_p_this = pd.read_csv(os.path.join(new_output_dir, test_fname))

                # concat the df to the full df
                df_delta_p_full = pd.concat([df_delta_p_full, df_delta_p_this])
            else:
                missing_fnames.append(test_fname)
                missing_fname_years.append(year)

            # Check if the uas file exists
            if os.path.exists(os.path.join(new_output_dir, test_fname_uas)):
                # Load the df
                df_uas_this = pd.read_csv(os.path.join(new_output_dir, test_fname_uas))

                # concat the df to the full df
                df_uas_full = pd.concat([df_uas_full, df_uas_this])
            else:
                missing_fnames.append(test_fname_uas)
                missing_fname_years.append(year)

            # Check if the vas file exists
            if os.path.exists(os.path.join(new_output_dir, test_fname_vas)):
                # Load the df
                df_vas_this = pd.read_csv(os.path.join(new_output_dir, test_fname_vas))

                # concat the df to the full df
                df_vas_full = pd.concat([df_vas_full, df_vas_this])
            else:
                missing_fnames.append(test_fname_vas)
                missing_fname_years.append(year)

    # create a new column for delta_p_hpa as the difference between
    # "data_n" and "data_s"
    df_delta_p_full["delta_p_hpa"] = (
        df_delta_p_full["data_n"] - df_delta_p_full["data_s"]
    ) / 100

    # # merge the delta p with the model df
    # model_df = model_df.merge(
    #     df_delta_p_full,
    #     on=["init_year", "member", "lead"],
    #     suffixes=("", ""),
    # )

    # # print the columns of the low wind df
    # print(f"Columns in low wind df: {low_wind_df.columns}")

    # # print the columns of the high wind df
    # print(f"Columns in higher wind df: {higher_wind_df.columns}")

    # # print the columns of the model df
    # print(f"Columns in model df: {model_df.columns}")

    # sys.exit()

    # # Check tyhe relationships of the dataframes
    # pdg_funcs.plot_multi_var_perc(
    #     obs_df=low_wind_df,
    #     model_df=low_wind_df,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="100 - temperature percentile",
    #     ylabel="10m wind speed",
    #     title="Inverted percentiles of temp. vs 10m wind speed, low wind DJF days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # # do the same for the higher wind days
    # pdg_funcs.plot_multi_var_perc(
    #     obs_df=higher_wind_df,
    #     model_df=higher_wind_df,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="100 - temperature percentile",
    #     ylabel="10m wind speed",
    #     title="Inverted percentiles of temp. vs 10m wind speed, higher wind DJF days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # calculate the 10th percentile of the data_tas_c in df low wind and df higher wind
    low_wind_10th_percentile = low_wind_df["data_tas_c"].quantile(0.1)
    higher_wind_10th_percentile = higher_wind_df["data_tas_c"].quantile(0.1)

    # Subset the dataframe to only include the rows where the data_tas_c is less than the 10th percentile
    low_wind_df = low_wind_df[low_wind_df["data_tas_c"] < low_wind_10th_percentile]
    higher_wind_df = higher_wind_df[
        higher_wind_df["data_tas_c"] < higher_wind_10th_percentile
    ]

    # print the columns in obs df
    print(f"Columns in obs df: {obs_df.columns}")

    # Print the head and tail of the obs df
    print("Head of the obs df:")
    print(obs_df.head())

    print("Tail of the obs df:")
    print(obs_df.tail())

    # print the columns in model df
    print(f"Columns in model df: {model_df.columns}")

    # # Load the psl data for the north atlantic region
    obs_psl_arr = load_obs_data(
        variable="psl",
        region="NA",
        season=season,
        time_freq=time_freq,
        winter_years=(1960, 2024),
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
        winter_years=(1960, 2024),
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
        winter_years=(1960, 2024),
        winter_dim_shape=len_winter_days,
        lat_shape=63,  # Europe region
        lon_shape=49,  # Europe region
        arrs_dir=winter_arrs_dir,
    )

    # Print the types of the arrays
    print(f"Type of obs psl array: {type(obs_psl_arr)}")
    print(f"Type of obs temperature array: {type(obs_temp_arr)}")
    print(f"Type of obs wind array: {type(obs_wind_arr)}")

    obs_psl_full, obs_psl_wmeans = obs_psl_arr
    obs_temp_full, obs_temp_wmeans = obs_temp_arr
    obs_wind_full, obs_wind_wmeans = obs_wind_arr

    # Print the shapes of the arrays
    print(f"Shape of obs psl array: {obs_psl_full.shape}")
    print(f"Shape of obs temperature array: {obs_temp_full.shape}")
    print(f"Shape of obs wind array: {obs_wind_full.shape}")
    # Print the shapes of the wmeans arrays
    print(f"Shape of obs psl wmeans: {obs_psl_wmeans.shape}")
    print(f"Shape of obs temperature wmeans: {obs_temp_wmeans.shape}")
    print(f"Shape of obs wind wmeans: {obs_wind_wmeans.shape}")

    # # Calculate the psl climatology
    obs_psl_clim = np.mean(obs_psl_full, axis=0)
    obs_tas_clim = np.mean(obs_temp_full, axis=0)
    obs_wind_clim = np.mean(obs_wind_full, axis=0)

    # print the head of the dfs
    print("Head of the obs df:")
    print(obs_df.head())

    # print the tail of the dfs
    print("Tail of the obs df:")
    print(obs_df.tail())

    # sys.exit()

    # extract the current date
    # NOTE: Hardcode the current date for now
    # current_date = "2025-05-08"
    current_date = datetime.now().strftime("%Y-%m-%d")
    # current_date = f"{current_date}_cold_temps"
    # current_date = "2025-05-28_cold_temps"

    # Set up fnames for the psl data
    psl_fname = f"ERA5_psl_NA_1960-2018_{season}_{time_freq}_{current_date}.npy"
    psl_times_fname = (
        f"ERA5_psl_NA_1960-2018_{season}_{time_freq}_times_{current_date}.npy"
    )

    # set up fnames for the temperature data
    # NOTE: Detrended temperature here
    temp_fname = (
        f"ERA5_tas_Europe_1960-2018_{season}_{time_freq}_dtr_{current_date}.npy"
    )
    temp_times_fname = (
        f"ERA5_tas_Europe_1960-2018_{season}_{time_freq}_times_dtr_{current_date}.npy"
    )

    # set up fnames for the wind data
    wind_fname = (
        f"ERA5_sfcWind_Europe_1960-2018_{season}_{time_freq}_{current_date}.npy"
    )
    wind_times_fname = (
        f"ERA5_sfcWind_Europe_1960-2018_{season}_{time_freq}_times_{current_date}.npy"
    )

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
            lats_path=os.path.join(
                metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"
            ),
            lons_path=os.path.join(
                metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"
            ),
        )

        # Save the data to the arrs_persist_dir
        np.save(os.path.join(arrs_persist_dir, psl_fname), psl_subset)
        np.save(os.path.join(arrs_persist_dir, psl_times_fname), psl_dates_list)
    else:
        print(f"PSL data already exists in {os.path.join(arrs_persist_dir, psl_fname)}")
        # Load the existing psl data
        psl_subset = np.load(os.path.join(arrs_persist_dir, psl_fname))
        psl_dates_list = np.load(
            os.path.join(arrs_persist_dir, psl_times_fname), allow_pickle=True
        )

        # print the shape of psl subset
        print(f"Shape of psl subset: {psl_subset.shape}")
        # print the shape of psl dates list
        print(f"Shape of psl dates list: {psl_dates_list.shape}")

        # Print the first and last values of the psl dates list
        print(f"First value of psl dates list: {psl_dates_list[0]}")
        print(f"Last value of psl dates list: {psl_dates_list[-1]}")

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
            lats_path=os.path.join(
                metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
            ),
            lons_path=os.path.join(
                metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
            ),
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
            lats_path=os.path.join(
                metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
            ),
            lons_path=os.path.join(
                metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
            ),
        )

        # Save the data to the arrs_persist_dir
        np.save(os.path.join(arrs_persist_dir, wind_fname), wind_subset)
        np.save(os.path.join(arrs_persist_dir, wind_times_fname), wind_dates_list)

    # load the psl data
    obs_psl_subset = np.load(os.path.join(arrs_persist_dir, psl_fname))
    obs_psl_dates_list = np.load(
        os.path.join(arrs_persist_dir, psl_times_fname), allow_pickle=True
    )

    # print the shape of psl sset
    print(f"Shape of psl subset: {obs_psl_subset.shape}")

    # print the shape of psl dates list
    print(f"Shape of psl dates list: {obs_psl_dates_list.shape}")

    # print the values of psl dates list
    print(f"PSL dates list: {obs_psl_dates_list}")

    # Now we simply want to plot all of these
    # plot 13 x matrices of 5 rows and 3 columns
    # Showing the full field psl, temperature and wind speed
    # Along with the date when each of these events occurred
    # Temperature is already detrended so don't need to do that again

    # load the temperature data
    obs_temp_subset = np.load(os.path.join(arrs_persist_dir, temp_fname))
    obs_temp_dates_list = np.load(
        os.path.join(arrs_persist_dir, temp_times_fname), allow_pickle=True
    )

    # load the wind data
    obs_wind_subset = np.load(os.path.join(arrs_persist_dir, wind_fname))
    obs_wind_dates_list = np.load(
        os.path.join(arrs_persist_dir, wind_times_fname), allow_pickle=True
    )

    # Print the first and last values of the obs psl dates list
    print(f"First value of obs psl dates list: {obs_psl_dates_list[0]}")
    print(f"Last value of obs psl dates list: {obs_psl_dates_list[-1]}")

    # Print the first and last values of the obs temp dates list
    print(f"First value of obs temp dates list: {obs_temp_dates_list[0]}")
    print(f"Last value of obs temp dates list: {obs_temp_dates_list[-1]}")

    # Print the first and last values of the obs wind dates list
    print(f"First value of obs wind dates list: {obs_wind_dates_list[0]}")
    print(f"Last value of obs wind dates list: {obs_wind_dates_list[-1]}")

    # # assert that the dates list arrays are equal
    # assert np.array_equal(
    #     obs_psl_dates_list, obs_temp_dates_list
    # ), "Dates list arrays are not equal"
    # assert np.array_equal(
    #     obs_psl_dates_list, obs_wind_dates_list
    # ), "Dates list arrays are not equal"

    print("--" * 20)
    print("Testing next function...")
    print("--" * 20)

    # Sort the obs df in terms of demand net wind max, in descending order
    obs_df_sorted = obs_df.sort_values("demand_net_wind_max", ascending=False)

    # Extract the top 5 effective dec years from this
    effective_dec_years_top = obs_df_sorted["effective_dec_year"].unique()[:5]
    effective_dec_years_second = obs_df_sorted["effective_dec_year"].unique()[5:10]

    # Find the second worst 5
    least_worst_5 = obs_df_sorted["effective_dec_year"].unique()[-5:]
    second_least_worst_5 = obs_df_sorted["effective_dec_year"].unique()[-10:-5]

    # # Extract the first 4 characters of each value and convert them to int
    # effective_dec_years_top = np.char.slice(effective_dec_years_top.astype(str), 0, 4).astype(int)

    # Convert into a list of ints
    effective_dec_years_worst = [int(year[:4]) for year in effective_dec_years_top]
    effective_dec_years_least = [int(year[:4]) for year in least_worst_5]
    effective_dec_years_second_least = [int(year[:4]) for year in second_least_worst_5]
    effective_dec_years_second_most = [
        int(year[:4]) for year in effective_dec_years_second
    ]

    # Print the shape of obs psl subset, obs temp subset and obs wind subset
    print(f"Shape of obs psl subset: {obs_psl_subset.shape}")
    print(f"Shape of obs temp subset: {obs_temp_subset.shape}")
    print(f"Shape of obs wind subset: {obs_wind_subset.shape}")

    # Set up a figure with len (obs_psl_subset) rows and 3 columns
    n_rows = obs_psl_subset.shape[0]
    n_cols = 3

    # Import teh psl lats for the obs
    obs_psl_lats = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy")
    )
    obs_psl_lons = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy")
    )

    # Unoirt the tas lats for the obs
    obs_temp_lats = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy")
    )
    obs_temp_lons = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy")
    )

    # Import the wind lats for the obs
    obs_wind_lats = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy")
    )
    obs_wind_lons = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy")
    )

    # print the head and tail of the obs df
    print("Head of the obs df:")
    print(obs_df.head())

    print("Tail of the obs df:")
    print(obs_df.tail())

    # Print the columns of the obs df
    print(f"Columns in obs df: {obs_df.columns}")

    fig, axs = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(10, 2 * n_rows),
        subplot_kw={"projection": ccrs.PlateCarree()},
        layout="constrained",
    )

    # Loop over the rows and plot the psl, temperature and wind speed
    for i in range(n_rows):
        # Set up the ax
        ax_psl = axs[i, 0] if n_cols > 1 else axs[i]
        ax_temp = axs[i, 1] if n_cols > 1 else axs[i]
        ax_wind = axs[i, 2] if n_cols > 1 else axs[i]

        # Extract the time this
        time_this = obs_df

    # model, assign, stats = kmeans_clustering_and_plotting(
    #     subset_arr=obs_psl_subset,
    #     lats_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"),
    #     lons_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"),
    #     n_clusters=5,
    #     figsize=(15, 10),
    #     cmap="RdBu_r",
    # )

    # composites_5 = create_and_plot_cluster_composites(
    #     subset_arr=obs_temp_subset,
    #     cluster_assignments=assign,
    #     var_name="tas",
    #     lats_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"),
    #     lons_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"),
    #     cmap="RdBu_r",
    #     exclude_no_type=True,
    #     figsize=(15, 10),
    # )

    # # Do the same but for anoms
    # composites_5_anoms = create_and_plot_cluster_composites(
    #     subset_arr=obs_temp_subset,
    #     cluster_assignments=assign,
    #     var_name="tas",
    #     lats_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"),
    #     lons_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"),
    #     cmap="RdBu_r",
    #     exclude_no_type=True,
    #     figsize=(15, 10),
    #     arr_clim=obs_tas_clim,
    # )

    # # Do the same but for wind anoms
    # composites_5_wind = create_and_plot_cluster_composites(
    #     subset_arr=obs_wind_subset,
    #     cluster_assignments=assign,
    #     var_name="sfcWind",
    #     lats_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"),
    #     lons_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"),
    #     cmap="Greens",
    #     exclude_no_type=True,
    #     figsize=(15, 10),
    # )

    # # Do the same but for wind anoms
    # composites_5_wind_anoms = create_and_plot_cluster_composites(
    #     subset_arr=obs_wind_subset,
    #     cluster_assignments=assign,
    #     var_name="sfcWind",
    #     lats_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"),
    #     lons_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"),
    #     cmap="PRGn",
    #     exclude_no_type=True,
    #     figsize=(15, 10),
    #     arr_clim=obs_wind_clim,
    # )

    # # Print the composites 5
    # print("Composites for 5 clusters:")
    # for cluster_id, data in composites_5.items():
    #     print(f"Cluster {cluster_id + 1}: {data['n_samples']} samples")

    # # Do the same but for 4 clusters
    # model_4, assign_4, stats_4 = kmeans_clustering_and_plotting(
    #     subset_arr=obs_psl_subset,
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"
    #     ),
    #     n_clusters=3,
    #     figsize=(10, 10),
    #     cmap="RdBu_r",
    # )

    # # Plot the tas anoms composites for 4 clusters
    # composites_4 = create_and_plot_cluster_composites(
    #     subset_arr=obs_temp_subset,
    #     cluster_assignments=assign_4,
    #     var_name="tas",
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     cmap="RdBu_r",
    #     exclude_no_type=False,
    #     figsize=(10, 10),
    #     arr_clim=obs_tas_clim,
    # )

    # # Plot the wind anoms composites for 4 clusters
    # composites_4_wind = create_and_plot_cluster_composites(
    #     subset_arr=obs_wind_subset,
    #     cluster_assignments=assign_4,
    #     var_name="sfcWind",
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     cmap="PRGn",
    #     exclude_no_type=False,
    #     figsize=(10, 10),
    #     arr_clim=obs_wind_clim,
    # )

    # # print the time taken
    # print("Composites for 4 clusters:")
    # for cluster_id, data in composites_4.items():
    #     print(f"Cluster {cluster_id + 1}: {data['n_samples']} samples")

    # # PRINT THW shape of assign
    # print(f"Shape of assign_4: {assign_4.shape}")

    # # Print the values of assign_4
    # print(f"Values of assign_4: {np.unique(assign_4)}")

    # # Add the values of assign_4 to the obs_df
    # obs_df["cluster_assign"] = assign_4

    # # Print the head of the obs_df
    # print("Head of the obs_df with cluster assignments:")
    # print(obs_df.head())

    # # Print the tail of the obs_df
    # print("Tail of the obs_df with cluster assignments:")
    # print(obs_df.tail())

    # model_3, assign_3, stats_3 = kmeans_clustering_and_plotting(
    #     subset_arr=obs_psl_subset,
    #     lats_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"),
    #     lons_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"),
    #     n_clusters=3,
    #     figsize=(10, 8),
    #     cmap="RdBu_r",
    # )

    # # Do the same but with 4 clusters for temperature
    # kmeans_clustering_and_plotting(
    #     subset_arr=obs_temp_subset,
    #     lats_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"),
    #     lons_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"),
    #     n_clusters=4,
    #     figsize=(10, 10),
    #     cmap="RdBu_r",
    # )

    # # Do the same but with 4 clusters for wind speed
    # kmeans_clustering_and_plotting(
    #     subset_arr=obs_wind_subset,
    #     lats_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"),
    #     lons_path=os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"),
    #     n_clusters=4,
    #     figsize=(10, 10),
    #     cmap="PRGn",
    # )

    # # Test the new function
    # plot_multi_var_composites_obs(
    #     subset_df_obs=obs_df.copy(),
    #     subset_arrs_list_obs=[obs_psl_subset, obs_temp_subset, obs_wind_subset],
    #     dates_list_obs=[obs_psl_dates_list, obs_temp_dates_list, obs_wind_dates_list],
    #     var_names=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #         ),
    #     ],
    #     effective_dec_years=effective_dec_years_worst,
    #     figsize=(10, 13),
    #     anoms_flag=True,
    #     clim_arrs_list_obs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     cluster_assign_name="cluster_assign",
    # )

    # # Test the new function
    # plot_multi_var_composites_obs(
    #     subset_df_obs=obs_df.copy(),
    #     subset_arrs_list_obs=[obs_psl_subset, obs_temp_subset, obs_wind_subset],
    #     dates_list_obs=[obs_psl_dates_list, obs_temp_dates_list, obs_wind_dates_list],
    #     var_names=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #         ),
    #     ],
    #     effective_dec_years=effective_dec_years_second_most,
    #     figsize=(10, 13),
    #     anoms_flag=True,
    #     clim_arrs_list_obs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     cluster_assign_name="cluster_assign",
    # )

    # # Test the new function
    # plot_multi_var_composites_obs(
    #     subset_df_obs=obs_df.copy(),
    #     subset_arrs_list_obs=[obs_psl_subset, obs_temp_subset, obs_wind_subset],
    #     dates_list_obs=[obs_psl_dates_list, obs_temp_dates_list, obs_wind_dates_list],
    #     var_names=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #         ),
    #     ],
    #     effective_dec_years=effective_dec_years_second_least,
    #     figsize=(10, 13),
    #     anoms_flag=True,
    #     clim_arrs_list_obs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     cluster_assign_name="cluster_assign",
    # )

    # # Test the new function
    # plot_multi_var_composites_obs(
    #     subset_df_obs=obs_df.copy(),
    #     subset_arrs_list_obs=[obs_psl_subset, obs_temp_subset, obs_wind_subset],
    #     dates_list_obs=[obs_psl_dates_list, obs_temp_dates_list, obs_wind_dates_list],
    #     var_names=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    #         ),
    #         os.path.join(
    #             metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #         ),
    #     ],
    #     effective_dec_years=effective_dec_years_least,
    #     figsize=(10, 13),
    #     anoms_flag=True,
    #     clim_arrs_list_obs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     cluster_assign_name="cluster_assign",
    # )

    # sys.exit()

    # load in the model subset files
    # NOTE: Updated for longer period
    model_psl_subset_fname = (
        f"HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_DnW_subset_2025-06-30.npy"
    )
    model_psl_subset_json_fname = f"HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_DnW_subset_index_list_2025-06-30.json"

    # if the file does not exist then raise an error
    if not os.path.exists(os.path.join(subset_model_dir, model_psl_subset_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_psl_subset_fname)} does not exist."
        )

    # load the model psl subset
    model_psl_subset = np.load(os.path.join(subset_model_dir, model_psl_subset_fname))

    # print the shape of the model psl subset
    print(f"Shape of model psl subset: {model_psl_subset.shape}")

    # print the values of the model psl subset
    print(f"Model psl subset: {model_psl_subset}")

    # if the json
    if not os.path.exists(os.path.join(subset_model_dir, model_psl_subset_json_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_psl_subset_json_fname)} does not exist."
        )

    # load the json file
    with open(os.path.join(subset_model_dir, model_psl_subset_json_fname), "r") as f:
        model_psl_subset_index_list = json.load(f)

    # set up the fnames for sfcWind
    model_wind_subset_fname = (
        f"HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_DnW_subset_2025-06-30.npy"
    )
    model_wind_subset_json_fname = f"HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_DnW_subset_index_list_2025-06-30.json"

    # if the file does not exist then raise an error
    if not os.path.exists(os.path.join(subset_model_dir, model_wind_subset_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_wind_subset_fname)} does not exist."
        )

    # load the model wind subset
    model_wind_subset = np.load(os.path.join(subset_model_dir, model_wind_subset_fname))

    # if the json
    if not os.path.exists(os.path.join(subset_model_dir, model_wind_subset_json_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_wind_subset_json_fname)} does not exist."
        )

    # load the json file
    with open(os.path.join(subset_model_dir, model_wind_subset_json_fname), "r") as f:
        model_wind_subset_index_list = json.load(f)

    # # print the length of the model wind subset index list
    # # Print the length of the model wind subset index list
    # print(f"Length of model wind subset index list: {np.shape(model_wind_subset_index_list['init_year'])}")

    # # print the values of the model wind subset index list
    # print(f"Model wind subset index list: {model_wind_subset_index_list}")

    # # print the keys in the index list
    # print(f"model_wind_subset index list keys: {model_wind_subset_index_list.keys()}")

    # set up the fnames for tas
    # NOTE: Updated for longer period
    model_temp_subset_fname = (
        f"HadGEM3-GC31-MM_tas_Europe_1960-2018_DJF_day_DnW_subset_2025-06-30.npy"
    )
    model_temp_subset_json_fname = f"HadGEM3-GC31-MM_tas_Europe_1960-2018_DJF_day_DnW_subset_index_list_2025-06-30.json"

    # if the file does not exist then raise an error
    if not os.path.exists(os.path.join(subset_model_dir, model_temp_subset_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_temp_subset_fname)} does not exist."
        )

    # load the model temperature subset
    model_temp_subset = np.load(os.path.join(subset_model_dir, model_temp_subset_fname))

    # if the json
    if not os.path.exists(os.path.join(subset_model_dir, model_temp_subset_json_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_temp_subset_json_fname)} does not exist."
        )

    # load the json file
    with open(os.path.join(subset_model_dir, model_temp_subset_json_fname), "r") as f:
        model_temp_subset_index_list = json.load(f)

    # Set up the fnames for the vas data
    model_vas_subset_fname = (
        f"HadGEM3-GC31-MM_vas_Europe_1960-2018_DJF_day_DnW_subset_2025-06-30.npy"
    )
    model_vas_subset_json_fname = f"HadGEM3-GC31-MM_vas_Europe_1960-2018_DJF_day_DnW_subset_index_list_2025-06-30.json"

    # if the model subset file does not exist
    if not os.path.exists(os.path.join(subset_model_dir, model_vas_subset_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_vas_subset_fname)} does not exist."
        )

    # load the model vas subset
    model_vas_subset = np.load(os.path.join(subset_model_dir, model_vas_subset_fname))

    # If the json does not exist
    if not os.path.exists(os.path.join(subset_model_dir, model_vas_subset_json_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_vas_subset_json_fname)} does not exist."
        )

    # load the json file
    with open(os.path.join(subset_model_dir, model_vas_subset_json_fname), "r") as f:
        model_vas_subset_index_list = json.load(f)

    # Set up the fnames for the uas data
    model_uas_subset_fname = (
        f"HadGEM3-GC31-MM_uas_Europe_1960-2018_DJF_day_DnW_subset_2025-06-30.npy"
    )
    model_uas_subset_json_fname = f"HadGEM3-GC31-MM_uas_Europe_1960-2018_DJF_day_DnW_subset_index_list_2025-06-30.json"

    # if the model subset file does not exist
    if not os.path.exists(os.path.join(subset_model_dir, model_uas_subset_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_uas_subset_fname)} does not exist."
        )

    # load the model uas subset
    model_uas_subset = np.load(os.path.join(subset_model_dir, model_uas_subset_fname))

    # If the json does not exist
    if not os.path.exists(os.path.join(subset_model_dir, model_uas_subset_json_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_uas_subset_json_fname)} does not exist."
        )

    # load the json file
    with open(os.path.join(subset_model_dir, model_uas_subset_json_fname), "r") as f:
        model_uas_subset_index_list = json.load(f)

    # # Set up the fnames for the psl low wind subset
    # model_low_wind_psl_subset_fname = "HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_DnW_subset_low_wind_0-10_2025-05-22.npy"
    # model_low_wind_psl_subset_json_fname = "HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_DnW_subset_low_wind_0-10_index_list_2025-05-22.json"

    # # if the model subset file does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_low_wind_psl_subset_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_low_wind_psl_subset_fname)} does not exist."
    #     )

    # # load the model low wind psl subset
    # model_low_wind_psl_subset = np.load(
    #     os.path.join(subset_model_dir, model_low_wind_psl_subset_fname)
    # )

    # # if the json does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_low_wind_psl_subset_json_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_low_wind_psl_subset_json_fname)} does not exist."
    #     )

    # # load the json file
    # with open(
    #     os.path.join(subset_model_dir, model_low_wind_psl_subset_json_fname), "r"
    # ) as f:
    #     model_low_wind_psl_subset_index_list = json.load(f)

    # # Set up the fnames for the psl higher wind subset
    # model_higher_wind_psl_subset_fname = "HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_DnW_subset_higher_wind_40-60_2025-05-22.npy"
    # model_higher_wind_psl_subset_json_fname = "HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_DnW_subset_higher_wind_40-60_index_list_2025-05-22.json"

    # # if the model subset file does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_higher_wind_psl_subset_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_higher_wind_psl_subset_fname)} does not exist."
    #     )

    # # load the model higher wind psl subset
    # model_higher_wind_psl_subset = np.load(
    #     os.path.join(subset_model_dir, model_higher_wind_psl_subset_fname)
    # )
    # # print the shape of the model higher wind psl subset
    # print(
    #     f"Shape of model higher wind psl subset: {model_higher_wind_psl_subset.shape}"
    # )

    # # if the json does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_higher_wind_psl_subset_json_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_higher_wind_psl_subset_json_fname)} does not exist."
    #     )

    # # load the json file
    # with open(
    #     os.path.join(subset_model_dir, model_higher_wind_psl_subset_json_fname), "r"
    # ) as f:
    #     model_higher_wind_psl_subset_index_list = json.load(f)

    # # Set up the fnames for uas/vas high wind/low wind subsets
    # # First for low wind uas and vas
    # model_low_wind_uas_subset_fname = "HadGEM3-GC31-MM_uas_Europe_1960-2018_DJF_day_DnW_subset_low_wind_0-10_2025-05-23.npy"
    # model_low_wind_uas_subset_json_fname = "HadGEM3-GC31-MM_uas_Europe_1960-2018_DJF_day_DnW_subset_low_wind_0-10_index_list_2025-05-23.json"

    # # Fnames for low wind vas
    # model_low_wind_vas_subset_fname = (
    #     "HadGEM3-GC31-MM_vas_Europe_1960-2018_DJF_day_DnW_subset_2025-05-23.npy"
    # )
    # model_low_wind_vas_subset_json_fname = "HadGEM3-GC31-MM_vas_Europe_1960-2018_DJF_day_DnW_subset_index_list_2025-05-23.json"

    # # Fnames for high wind uas
    # model_higher_wind_uas_subset_fname = "HadGEM3-GC31-MM_uas_Europe_1960-2018_DJF_day_DnW_subset_higher_wind_40-60_2025-05-23.npy"
    # model_higher_wind_uas_subset_json_fname = "HadGEM3-GC31-MM_uas_Europe_1960-2018_DJF_day_DnW_subset_higher_wind_40-60_index_list_2025-05-23.json"

    # # Fnames for high wind vas
    # model_higher_wind_vas_subset_fname = "HadGEM3-GC31-MM_vas_Europe_1960-2018_DJF_day_DnW_subset_higher_wind_40-60_2025-05-23.npy"
    # model_higher_wind_vas_subset_json_fname = "HadGEM3-GC31-MM_vas_Europe_1960-2018_DJF_day_DnW_subset_higher_wind_40-60_index_list_2025-05-23.json"

    # model_lower_wind_sfcWind_subset_fname = "HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_DnW_subset_low_wind_0-10_2025-05-23.npy"
    # model_lower_wind_sfcWind_subset_json_fname = "HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_DnW_subset_low_wind_0-10_index_list_2025-05-23.json"

    # model_higher_wind_sfcWind_subset_fname = "HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_DnW_subset_higher_wind_40-60_2025-05-23.npy"
    # model_higher_wind_sfcWind_subset_json_fname = "HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_DnW_subset_higher_wind_40-60_index_list_2025-05-23.json"

    # # if the model subset file does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_low_wind_uas_subset_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_low_wind_uas_subset_fname)} does not exist."
    #     )

    # # load the model low wind uas subset
    # model_low_wind_uas_subset = np.load(
    #     os.path.join(subset_model_dir, model_low_wind_uas_subset_fname)
    # )

    # # if the json does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_low_wind_uas_subset_json_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_low_wind_uas_subset_json_fname)} does not exist."
    #     )

    # # load the json file
    # with open(
    #     os.path.join(subset_model_dir, model_low_wind_uas_subset_json_fname), "r"
    # ) as f:
    #     model_low_wind_uas_subset_index_list = json.load(f)

    # # if the model subset file does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_low_wind_vas_subset_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_low_wind_vas_subset_fname)} does not exist."
    #     )

    # # load the model low wind vas subset
    # model_low_wind_vas_subset = np.load(
    #     os.path.join(subset_model_dir, model_low_wind_vas_subset_fname)
    # )

    # # if the json does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_low_wind_vas_subset_json_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_low_wind_vas_subset_json_fname)} does not exist."
    #     )

    # # load the json file
    # with open(
    #     os.path.join(subset_model_dir, model_low_wind_vas_subset_json_fname), "r"
    # ) as f:
    #     model_low_wind_vas_subset_index_list = json.load(f)

    # # if the model subset file does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_higher_wind_uas_subset_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_higher_wind_uas_subset_fname)} does not exist."
    #     )

    # # load the model higher wind uas subset
    # model_higher_wind_uas_subset = np.load(
    #     os.path.join(subset_model_dir, model_higher_wind_uas_subset_fname)
    # )

    # # if the json does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_higher_wind_uas_subset_json_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_higher_wind_uas_subset_json_fname)} does not exist."
    #     )

    # # load the json file
    # with open(
    #     os.path.join(subset_model_dir, model_higher_wind_uas_subset_json_fname), "r"
    # ) as f:
    #     model_higher_wind_uas_subset_index_list = json.load(f)

    # # if the model subset file does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_higher_wind_vas_subset_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_higher_wind_vas_subset_fname)} does not exist."
    #     )

    # # load the model higher wind vas subset
    # model_higher_wind_vas_subset = np.load(
    #     os.path.join(subset_model_dir, model_higher_wind_vas_subset_fname)
    # )

    # # if the json does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_higher_wind_vas_subset_json_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_higher_wind_vas_subset_json_fname)} does not exist."
    #     )

    # # load the json file
    # with open(
    #     os.path.join(subset_model_dir, model_higher_wind_vas_subset_json_fname), "r"
    # ) as f:
    #     model_higher_wind_vas_subset_index_list = json.load(f)

    # # if the model subset file does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_lower_wind_sfcWind_subset_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_lower_wind_sfcWind_subset_fname)} does not exist."
    #     )

    # # load the model lower wind sfcWind subset
    # model_lower_wind_sfcWind_subset = np.load(
    #     os.path.join(subset_model_dir, model_lower_wind_sfcWind_subset_fname)
    # )

    # # if the json does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_lower_wind_sfcWind_subset_json_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_lower_wind_sfcWind_subset_json_fname)} does not exist."
    #     )

    # # load the json file
    # with open(
    #     os.path.join(subset_model_dir, model_lower_wind_sfcWind_subset_json_fname), "r"
    # ) as f:
    #     model_lower_wind_sfcWind_subset_index_list = json.load(f)

    # # if the model subset file does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_higher_wind_sfcWind_subset_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_higher_wind_sfcWind_subset_fname)} does not exist."
    #     )

    # # load the model higher wind sfcWind subset
    # model_higher_wind_sfcWind_subset = np.load(
    #     os.path.join(subset_model_dir, model_higher_wind_sfcWind_subset_fname)
    # )

    # # if the json does not exist
    # if not os.path.exists(
    #     os.path.join(subset_model_dir, model_higher_wind_sfcWind_subset_json_fname)
    # ):
    #     raise FileNotFoundError(
    #         f"File {os.path.join(subset_model_dir, model_higher_wind_sfcWind_subset_json_fname)} does not exist."
    #     )

    # # load the json file
    # with open(
    #     os.path.join(subset_model_dir, model_higher_wind_sfcWind_subset_json_fname), "r"
    # ) as f:
    #     model_higher_wind_sfcWind_subset_index_list = json.load(f)

    # # print the length of the model temperature subset index list
    # # Print the length of the model temperature subset index list
    # print(f"Length of model temperature subset index list: {np.shape(model_temp_subset_index_list['init_year'])}")

    # # print the values of the model temperature subset index list
    # print(f"Model temperature subset index list: {model_temp_subset_index_list}")

    # # print the keys in the index list
    # print(f"model_temp_subset index list keys: {model_temp_subset_index_list.keys()}")

    # load in the climatology file for psl
    psl_clim_fname = f"climatology_HadGEM3-GC31-MM_psl_DJF_NA_1960_2018_day.npy"
    sfcWind_clim_fname = (
        f"climatology_HadGEM3-GC31-MM_sfcWind_DJF_Europe_1960_2018_day.npy"
    )
    tas_clim_fname = f"climatology_HadGEM3-GC31-MM_tas_DJF_Europe_1960_2018_day.npy"
    vas_clim_fname = "climatology_HadGEM3-GC31-MM_vas_DJF_Europe_1960_2018_day.npy"
    uas_clim_fname = "climatology_HadGEM3-GC31-MM_uas_DJF_Europe_1960_2018_day.npy"

    # if the file does not exist then raise an error
    if not os.path.exists(os.path.join(model_clim_dir, psl_clim_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(model_clim_dir, psl_clim_fname)} does not exist."
        )

    if not os.path.exists(os.path.join(model_clim_dir, sfcWind_clim_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(model_clim_dir, sfcWind_clim_fname)} does not exist."
        )

    if not os.path.exists(os.path.join(model_clim_dir, tas_clim_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(model_clim_dir, tas_clim_fname)} does not exist."
        )

    if not os.path.exists(os.path.join(model_clim_dir, vas_clim_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(model_clim_dir, vas_clim_fname)} does not exist."
        )

    if not os.path.exists(os.path.join(model_clim_dir, uas_clim_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(model_clim_dir, uas_clim_fname)} does not exist."
        )

    # load the climatology data
    model_psl_clim = np.load(os.path.join(model_clim_dir, psl_clim_fname))
    model_wind_clim = np.load(os.path.join(model_clim_dir, sfcWind_clim_fname))
    model_tas_clim = np.load(os.path.join(model_clim_dir, tas_clim_fname))
    model_vas_clim = np.load(os.path.join(model_clim_dir, vas_clim_fname))
    model_uas_clim = np.load(os.path.join(model_clim_dir, uas_clim_fname))

    # print the shape of the climatology data
    print(f"Shape of obs psl climatology data: {model_psl_clim.shape}")
    print(f"Shape of obs wind climatology data: {model_wind_clim.shape}")
    print(f"Shape of obs tas climatology data: {model_tas_clim.shape}")
    # print(f"Shape of obs vas climatology data: {model_vas_clim.shape}")
    # print(f"Shape of obs uas climatology data: {model_uas_clim.shape}")

    # # print the values of the climatology data
    # print(f"Obs psl climatology data: {obs_psl_clim}")
    # print(f"Obs wind climatology data: {obs_wind_clim}")
    # print(f"Obs tas climatology data: {obs_tas_clim}")

    # print the head of the mdoel df
    print("Head of the model df:")
    print(model_df.head())

    # prnt the tail of the model df
    print("Tail of the model df:")
    print(model_df.tail())

    # find the 80th percentile of the demand net wind max
    # for the obs
    obs_dnw_5th = obs_df["demand_net_wind_max"].quantile(0.01)

    # do the same for the model
    model_dnw_5th = model_df["demand_net_wind_bc_max"].quantile(0.01)

    # Find the maximum of the demand net wind max for the obs
    obs_dnw_90th = obs_df["demand_net_wind_max"].quantile(0.90)

    # Find the maximum of the demand net wind max for the model
    model_dnw_90th = model_df["demand_net_wind_bc_max"].quantile(0.90)

    # find the 99th percentile of the demand net wind max
    obs_dnw_99th = obs_df["demand_net_wind_max"].quantile(0.99)

    # do the same for the model
    model_dnw_99th = model_df["demand_net_wind_bc_max"].quantile(0.99)

    # subset the model df to grey points
    model_df_subset_grey = model_df[model_df["demand_net_wind_bc_max"] < model_dnw_5th]

    # do the same for the obs
    obs_df_subset_grey = obs_df[obs_df["demand_net_wind_max"] < obs_dnw_5th]

    # subset the model df to yellow points
    model_df_subset_yellow = model_df[
        (model_df["demand_net_wind_bc_max"] >= model_dnw_90th)
        & (model_df["demand_net_wind_bc_max"] < model_dnw_99th)
    ]

    # do the same for the obs
    obs_df_subset_yellow = obs_df[
        (obs_df["demand_net_wind_max"] >= obs_dnw_90th)
        & (obs_df["demand_net_wind_max"] < obs_dnw_99th)
    ]

    # subset the model df to red points
    model_df_subset_red = model_df[model_df["demand_net_wind_bc_max"] >= model_dnw_99th]

    # do the same for the obs
    obs_df_subset_red = obs_df[obs_df["demand_net_wind_max"] >= obs_dnw_99th]

    # print the shape of moel df subset red
    print(f"Shape of model df subset red: {model_df_subset_red.shape}")

    # print the shape of obs df subset red
    print(f"Shape of obs df subset red: {obs_df_subset_red.shape}")

    # print the values of obs df subset red
    print(f"Obs df subset red: {obs_df_subset_red}")

    # # plot the composites
    # # plot the composites f`or all of the winter days
    # plot_composites(
    #     subset_df=obs_df_low_temp,
    #     subset_arrs=[obs_psl_subset, obs_temp_subset, obs_wind_subset],
    #     clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     dates_lists=[
    #         obs_psl_dates_list,
    #         obs_temp_dates_list,
    #         obs_wind_dates_list,
    #     ],
    #     variables=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
    #         ),
    #     ],
    #     suptitle="All obs low temp",
    #     figsize=(12, 6),
    # )

    # Set up the subset dfs model
    subset_dfs_model = [
        model_df_subset_grey,
        model_df_subset_yellow,
        model_df_subset_red,
    ]

    # Set up the subset arrs model
    subset_arrs_model = [
        model_psl_subset,
        model_psl_subset,
        model_psl_subset,
    ]

    # Set up the clim arrs model
    clim_arrs_model = [
        model_psl_clim,
        model_psl_clim,
        model_psl_clim,
    ]

    # set up zeros like model psl clim
    model_psl_clim_zeros = np.zeros_like(model_psl_clim)

    clim_arrs_model_zeros = [
        model_psl_clim_zeros,
        model_psl_clim_zeros,
        model_psl_clim_zeros,
    ]

    # # Set up zeros like obs psl clim
    # obs_psl_clim_zeros = np.zeros_like(obs_psl_clim)

    # clim_arrs_obs_zeros = [
    #     obs_psl_clim_zeros,
    #     obs_psl_clim_zeros,
    #     obs_psl_clim_zeros,
    # ]

    # # Set up the clim arrs obs
    # clim_arrs_obs = [
    #     obs_psl_clim,
    #     obs_psl_clim,
    #     obs_psl_clim,
    # ]

    # # Set up the dates lists obs
    # dates_lists_obs = [
    #     obs_psl_dates_list,
    #     obs_psl_dates_list,
    #     obs_psl_dates_list,
    # ]

    # Set up the model index dicts
    model_index_dicts = [
        model_psl_subset_index_list,
        model_psl_subset_index_list,
        model_psl_subset_index_list,
    ]

    # Set up the lats path
    lats_paths = [
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"),
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"),
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"),
    ]

    # Set up the lons path
    lons_paths = [
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"),
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"),
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"),
    ]

    # Set up the suptitle
    suptitle = (
        "ERA5 and HadGEM3-GC31-MM DnW max MSLP anoms, relative to 1960-2018 climatology"
    )

    # # # Now test the new function
    # plot_mslp_composites(
    #     subset_dfs_obs=[obs_df_low_temp, obs_df_low_temp, obs_df_low_temp],
    #     subset_dfs_model=subset_dfs_model,
    #     subset_arrs_obs=[obs_psl_subset, obs_psl_subset, obs_psl_subset],
    #     subset_arrs_model=subset_arrs_model,
    #     clim_arrs_obs=clim_arrs_obs_zeros,
    #     clim_arrs_model=clim_arrs_model_zeros,
    #     dates_lists_obs=dates_lists_obs,
    #     model_index_dicts=model_index_dicts,
    #     lats_paths=lats_paths,
    #     lons_paths=lons_paths,
    #     suptitle=suptitle,
    #     figsize=(8, 9),
    # )

    # # print the columsn in obs_df_high_demand
    # print(f"Columns in obs_df_high_demand: {obs_df_low_temp.columns}")

    # # Load in the data for high demand
    # obs_psl_subset_high_demand = np.load(
    #     "/home/users/benhutch/unseen_multi_year/data/ERA5_psl_NA_1960-2018_DJF_day_2025-05-28.npy"
    # )
    # obs_psl_dates_list_high_demand = np.load(
    #     "/home/users/benhutch/unseen_multi_year/data/ERA5_psl_NA_1960-2018_DJF_day_times_2025-05-28.npy",
    #     allow_pickle=True,
    # )

    # # Now test the new function
    # plot_temp_demand_quartiles_obs(
    #     subset_df_obs=obs_df_low_temp,
    #     quartiles_var_name="data_tas_c",
    #     quartiles=[
    #         (0.75, 1.0),
    #         (0.5, 0.75),
    #         (0.25, 0.5),
    #         (0, 0.25),
    #     ],
    #     subset_arr_obs=obs_psl_subset,
    #     dates_list_obs=obs_psl_dates_list,
    #     var_name="psl",
    #     lats_path=lats_paths[0],
    #     lons_path=lons_paths[0],
    #     figsize=(8, 10),
    #     anoms_flag=False,
    #     clim_arr_obs=None,
    #     gridbox=[
    #         dicts.uk_n_box_tight,
    #         dicts.uk_s_box_tight,
    #     ],
    # )

    # # plot_temp_demand_quartiles_obs(
    # #     subset_df_obs=obs_df_low_temp,
    # #     quartiles_var_name="data_tas_c",
    # #     quartiles=[
    # #         (0.75, 1.0),
    # #         (0.5, 0.75),
    # #         (0.25, 0.5),
    # #         (0, 0.25),
    # #     ],
    # #     subset_arr_obs=obs_psl_subset,
    # #     dates_list_obs=obs_psl_dates_list,
    # #     var_name="psl",
    # #     lats_path=lats_paths[0],
    # #     lons_path=lons_paths[0],
    # #     figsize=(8, 10),
    # #     anoms_flag=False,
    # #     clim_arr_obs=None,
    # #     gridbox=dicts.uk_s_box_corrected,
    # # )

    # plot_temp_demand_quartiles_obs(
    #     subset_df_obs=obs_df_high_demand,
    #     quartiles_var_name="elec_demand_5yrRmean_nohols",
    #     quartiles=[
    #         (0, 0.25),
    #         (0.25, 0.5),
    #         (0.5, 0.75),
    #         (0.75, 1.0),
    #     ],
    #     subset_arr_obs=obs_psl_subset_high_demand,
    #     dates_list_obs=obs_psl_dates_list_high_demand,
    #     var_name="psl",
    #     lats_path=lats_paths[0],
    #     lons_path=lons_paths[0],
    #     figsize=(8, 10),
    #     anoms_flag=False,
    #     clim_arr_obs=None,
    #     gridbox=[
    #         dicts.uk_n_box_tight,
    #         dicts.uk_s_box_tight,
    #     ],
    # )

    # # plot_temp_demand_quartiles_obs(
    # #     subset_df_obs=obs_df_high_demand,
    # #     quartiles_var_name="elec_demand_5yrRmean_nohols",
    # #     quartiles=[
    # #         (0, 0.25),
    # #         (0.25, 0.5),
    # #         (0.5, 0.75),
    # #         (0.75, 1.0),
    # #     ],
    # #     subset_arr_obs=obs_psl_subset_high_demand,
    # #     dates_list_obs=obs_psl_dates_list_high_demand,
    # #     var_name="psl",
    # #     lats_path=lats_paths[0],
    # #     lons_path=lons_paths[0],
    # #     figsize=(8, 10),
    # #     anoms_flag=False,
    # #     clim_arr_obs=None,
    # #     gridbox=dicts.uk_s_box_corrected,
    # # )

    # # sys.exit()

    # lats_europe_tas = os.path.join(
    #     metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    # )
    # lons_europe_tas = os.path.join(
    #     metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    # )

    # # Do the same for temperature anoms
    # plot_temp_demand_quartiles_obs(
    #     subset_df_obs=obs_df_low_temp,
    #     quartiles_var_name="data_tas_c",
    #     quartiles=[
    #         (0.75, 1.0),
    #         (0.5, 0.75),
    #         (0.25, 0.5),
    #         (0, 0.25),
    #     ],
    #     subset_arr_obs=obs_temp_subset,
    #     dates_list_obs=obs_temp_dates_list,
    #     var_name="tas",
    #     lats_path=lats_europe_tas,
    #     lons_path=lons_europe_tas,
    #     figsize=(6, 10),
    #     anoms_flag=True,
    #     clim_arr_obs=obs_tas_clim,
    #     gridbox=dicts.wind_gridbox_south,
    # )

    # # DO the same for wind anoms
    # plot_temp_demand_quartiles_obs(
    #     subset_df_obs=obs_df_low_temp,
    #     quartiles_var_name="data_tas_c",
    #     quartiles=[
    #         (0.75, 1.0),
    #         (0.5, 0.75),
    #         (0.25, 0.5),
    #         (0, 0.25),
    #     ],
    #     subset_arr_obs=obs_wind_subset,
    #     dates_list_obs=obs_wind_dates_list,
    #     var_name="sfcWind",
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     figsize=(6, 10),
    #     anoms_flag=True,
    #     clim_arr_obs=obs_wind_clim,
    #     gridbox=dicts.wind_gridbox_south,
    # )

    # # # Plot the differences for psl
    # # plot_temp_demand_quartiles_obs(
    # #     subset_df_obs=obs_df_high_demand,
    # #     quartiles_var_name="elec_demand_5yrRmean_nohols",
    # #     quartiles=[
    # #         (0.0, 0.25),
    # #         (0.25, 0.5),
    # #         (0.5, 0.75),
    # #         (0.75, 1.0),
    # #     ],
    # #     subset_arr_obs=obs_psl_subset_high_demand,
    # #     dates_list_obs=obs_psl_dates_list_high_demand,
    # #     var_name="psl",
    # #     lats_path=lats_paths[0],
    # #     lons_path=lons_paths[0],
    # #     figsize=(10, 10),
    # #     anoms_flag=False,
    # #     clim_arr_obs=None,
    # #     gridbox=[
    # #         dicts.uk_n_box_corrected,
    # #         dicts.uk_s_box_corrected,
    # #     ],
    # #     second_subset_df_obs=obs_df_low_temp,
    # #     second_quartiles_var_name="data_tas_c",
    # #     second_subset_arr_obs=obs_psl_subset,
    # #     second_dates_list_obs=obs_psl_dates_list,
    # #     second_quartiles=[
    # #         (0.75, 1.0),
    # #         (0.5, 0.75),
    # #         (0.25, 0.5),
    # #         (0, 0.25),
    # #     ],
    # # )

    # # Load in the data for high demand for tas
    # obs_temp_subset_high_demand = np.load(
    #     "/home/users/benhutch/unseen_multi_year/data/ERA5_tas_Europe_1960-2018_DJF_day_dtr_2025-05-28.npy"
    # )
    # obs_temp_dates_list_high_demand = np.load(
    #     "/home/users/benhutch/unseen_multi_year/data/ERA5_tas_Europe_1960-2018_DJF_day_times_dtr_2025-05-28.npy",
    #     allow_pickle=True,
    # )

    # # Plot the differences for tas
    # plot_temp_demand_quartiles_obs(
    #     subset_df_obs=obs_df_high_demand,
    #     quartiles_var_name="elec_demand_5yrRmean_nohols",
    #     quartiles=[
    #         (0.0, 0.25),
    #         (0.25, 0.5),
    #         (0.5, 0.75),
    #         (0.75, 1.0),
    #     ],
    #     subset_arr_obs=obs_temp_subset_high_demand,
    #     dates_list_obs=obs_temp_dates_list_high_demand,
    #     var_name="tas",
    #     lats_path=lats_europe_tas,
    #     lons_path=lons_europe_tas,
    #     figsize=(6, 10),
    #     anoms_flag=True,
    #     clim_arr_obs=obs_tas_clim,
    #     gridbox=dicts.wind_gridbox_south,
    # )

    # # Load in the data for high demand for wind
    # obs_wind_subset_high_demand = np.load(
    #     "/home/users/benhutch/unseen_multi_year/data/ERA5_sfcWind_Europe_1960-2018_DJF_day_2025-05-28.npy"
    # )
    # obs_wind_dates_list_high_demand = np.load(
    #     "/home/users/benhutch/unseen_multi_year/data/ERA5_sfcWind_Europe_1960-2018_DJF_day_times_2025-05-28.npy",
    #     allow_pickle=True,
    # )

    # # Plot the differences for wind
    # plot_temp_demand_quartiles_obs(
    #     subset_df_obs=obs_df_high_demand,
    #     quartiles_var_name="elec_demand_5yrRmean_nohols",
    #     quartiles=[
    #         (0.0, 0.25),
    #         (0.25, 0.5),
    #         (0.5, 0.75),
    #         (0.75, 1.0),
    #     ],
    #     subset_arr_obs=obs_wind_subset_high_demand,
    #     dates_list_obs=obs_wind_dates_list_high_demand,
    #     var_name="sfcWind",
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     figsize=(6, 10),
    #     anoms_flag=True,
    #     clim_arr_obs=obs_wind_clim,
    #     gridbox=dicts.wind_gridbox_south,
    # )

    # sys.exit()

    # # plot the composite
    # # plot the composites for all of the winter days
    # plot_composites(
    #     subset_df=obs_df_subset_red,
    #     subset_arrs=[obs_psl_subset, obs_temp_subset, obs_wind_subset],
    #     clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     dates_lists=[
    #         obs_psl_dates_list,
    #         obs_temp_dates_list,
    #         obs_wind_dates_list,
    #     ],
    #     variables=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
    #         ),
    #     ],
    #     suptitle="All obs DnW max",
    #     figsize=(12, 6),
    # )

    # sys.exit()

    # Set up the subset dfs obs
    subset_dfs_obs = [
        obs_df_subset_grey,
        obs_df_subset_yellow,
        obs_df_subset_red,
    ]

    # Set up the subset dfs model
    subset_dfs_model = [
        model_df_subset_grey,
        model_df_subset_yellow,
        model_df_subset_red,
    ]

    # # Set up the subset arrs obs
    # subset_arrs_obs = [
    #     obs_psl_subset,
    #     obs_psl_subset,
    #     obs_psl_subset,
    # ]

    # # Set up the subset arrs obs tas
    # subset_arrs_obs_tas = [
    #     obs_temp_subset,
    #     obs_temp_subset,
    #     obs_temp_subset,
    # ]

    # # Set up the subset arrs obs wind
    # subset_arrs_obs_wind = [
    #     obs_wind_subset,
    #     obs_wind_subset,
    #     obs_wind_subset,
    # ]

    # Set up the subset arrs model
    subset_arrs_model = [
        model_psl_subset,
        model_psl_subset,
        model_psl_subset,
    ]

    # Set up the subset arrs model tas
    subset_arrs_model_tas = [
        model_temp_subset,
        model_temp_subset,
        model_temp_subset,
    ]

    # Set up the subset arrs model wind
    subset_arrs_model_wind = [
        model_wind_subset,
        model_wind_subset,
        model_wind_subset,
    ]

    # Set up the subset arrs model vas
    subset_arrs_model_vas = [
        model_vas_subset,
        model_vas_subset,
        model_vas_subset,
    ]

    # Set up the subset arrs model uas
    subset_arrs_model_uas = [
        model_uas_subset,
        model_uas_subset,
        model_uas_subset,
    ]

    # print the shape of the model psl subset
    print(f"Shape of model psl subset: {model_psl_subset.shape}")

    # # perform the clustering on the model psl subset
    # model_3, model_assign_3, model_stats_3 = kmeans_clustering_and_plotting(
    #     subset_arr=model_psl_subset,
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"
    #     ),
    #     n_clusters=3,
    #     figsize=(10, 10),
    #     cmap="RdBu_r",
    # )

    # levels = np.array(
    #     [
    #         -6,
    #         -5,
    #         -4,
    #         -3,
    #         -2,
    #         -1,
    #         1,
    #         2,
    #         3,
    #         4,
    #         5,
    #         6,
    #     ]
    # )

    # # Plot the tas anoms composites for the clusters
    # model_composites_3_tas = create_and_plot_cluster_composites(
    #     subset_arr=model_temp_subset,
    #     cluster_assignments=model_assign_3,
    #     var_name="tas",
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     cmap="RdBu_r",
    #     exclude_no_type=False,
    #     figsize=(10, 10),
    #     arr_clim=model_tas_clim,
    #     levels=levels,
    # )

    # levels = np.array(
    #     [
    #         -2,
    #         -1.5,
    #         -1,
    #         -0.5,
    #         0.5,
    #         1,
    #         1.5,
    #         2,
    #     ]
    # )

    # # Do the same for the wind anoms
    # model_composites_3_wind = create_and_plot_cluster_composites(
    #     subset_arr=model_wind_subset,
    #     cluster_assignments=model_assign_3,
    #     var_name="sfcWind",
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     cmap="PRGn",
    #     exclude_no_type=False,
    #     figsize=(10, 10),
    #     arr_clim=model_wind_clim,
    #     levels=levels,
    # )

    # levels = np.array(
    #     [
    #         -4,
    #         -3.5,
    #         -3,
    #         -2.5,
    #         -2,
    #         -1.5,
    #         -1,
    #         -0.5,
    #         0.5,
    #         1,
    #         1.5,
    #         2,
    #         2.5,
    #         3,
    #         3.5,
    #         4,
    #     ]
    # )

    # # Do the same for the uas anoms
    # model_composites_3_uas = create_and_plot_cluster_composites(
    #     subset_arr=model_uas_subset,
    #     cluster_assignments=model_assign_3,
    #     var_name="uas",
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_uas_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_uas_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     cmap="PRGn",
    #     exclude_no_type=False,
    #     figsize=(10, 10),
    #     levels=levels,
    # )

    # # Do the same for the vas anoms
    # model_composites_3_vas = create_and_plot_cluster_composites(
    #     subset_arr=model_vas_subset,
    #     cluster_assignments=model_assign_3,
    #     var_name="vas",
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     cmap="PRGn",
    #     exclude_no_type=False,
    #     figsize=(10, 10),
    #     levels=levels,
    # )

    # Set up the save dir for the climatology data
    save_dir_clim = "/home/users/benhutch/unseen_multi_year/model_clim"

    # bSet up fnames for the model clims
    model_psl_clim_NA_fname = "HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_clim.npy"
    model_tas_clim_Europe_fname = "HadGEM3-GC31-MM_tas_Europe_1960-2018_DJF_day_clim.npy"
    model_wind_clim_Europe_fname = "HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_clim.npy"
    model_uas_clim_Europe_fname = "HadGEM3-GC31-MM_uas_Europe_1960-2018_DJF_day_clim.npy"
    model_vas_clim_Europe_fname = "HadGEM3-GC31-MM_vas_Europe_1960-2018_DJF_day_clim.npy"

    # Set up a list of fnames
    model_clim_fnames = [
        model_psl_clim_NA_fname,
        model_tas_clim_Europe_fname,
        model_wind_clim_Europe_fname,
        model_uas_clim_Europe_fname,
        model_vas_clim_Europe_fname,
    ]

    # Set up a list of teh clim arrs
    clim_arrs = [
        model_psl_clim,
        model_tas_clim,
        model_wind_clim,
        model_uas_clim,
        model_vas_clim,
    ]

    # Loop over this list
    for i, clim_arr in enumerate(clim_arrs):
        # Set up the save path
        save_path = os.path.join(save_dir_clim, model_clim_fnames[i])

        if not os.path.exists(save_dir_clim):
            os.makedirs(save_dir_clim)

        if not os.path.exists(save_path):
            print(f"Saving {model_clim_fnames[i]} to {save_path}")
            np.save(save_path, clim_arr)

    sys.exit()


    # Load the psl lats
    psl_lats = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy")
    )
    psl_lons = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy")
    )

    # Load the tas lats
    tas_lats = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy")
    )
    tas_lons = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy")
    )

    # Load the wind lats
    wind_lats = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy")
    )
    wind_lons = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy")
    )

    # Load the uas lats
    uas_lats = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_uas_Europe_1960_DJF_day_lats.npy")
    )
    uas_lons = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_uas_Europe_1960_DJF_day_lons.npy")
    )

    # Load the vas lats
    vas_lats = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_lats.npy")
    )
    vas_lons = np.load(
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_lons.npy")
    )

    subset_dfs_test = ["/home/users/benhutch/unseen_multi_year/dfs/model_df_subset_cluster_1_zonal_block_max_dnw.csv"]

    # Check if the first file exists in subset_dfs_test
    if os.path.exists(subset_dfs_test[0]):
        print(f"File {subset_dfs_test[0]} exists.")

        subset_dfs_test = [
            pd.read_csv(subset_dfs_test[0]),
        ]
    else:
        raise FileNotFoundError(f"File {subset_dfs_test[0]} does not exist.")

    # Loop over the subset dfs model
    for i, subset_df in enumerate(subset_dfs_test):
        # Print the shape of the subset df
        print(f"Shape of subset df {i}: {subset_df.shape}")

        n_rows = subset_df.shape[0]

        # Set up the subset arr this for model
        subset_arr_this_model_psl = np.zeros(
            (n_rows,len(psl_lats), len(psl_lons))
        )
        subset_arr_this_model_tas = np.zeros(
            (n_rows, len(tas_lats), len(tas_lons))
        )
        subset_arr_this_model_wind = np.zeros(
            (n_rows, len(wind_lats), len(wind_lons))
        )
        subset_arr_this_model_uas = np.zeros(
            (n_rows,len(uas_lats), len(uas_lons))
        )
        subset_arr_this_model_vas = np.zeros(
            (n_rows,len(vas_lats), len(vas_lons))
        )

        # extract the model index dict this
        model_index_dict_this = model_index_dicts[i]

        # Extract the init years as arrays
        # Extract the init years as arrays
        init_year_array_this = np.array(model_index_dict_this["init_year"])
        member_array_this = np.array(model_index_dict_this["member"])
        lead_array_this = np.array(model_index_dict_this["lead"])

        # zero the missing daya here
        missing_days = 0

        # Create figure with GridSpec to control layout
        fig = plt.figure(figsize=(15, 16))  # Slightly taller to accommodate colorbar
        
        # Create GridSpec with space for colorbar at bottom
        gs = fig.add_gridspec(6, 5,  # 6 rows (5 for plots, 1 for colorbar)
                            height_ratios=[1, 1, 1, 1, 1, 0.1],  # Last row smaller for colorbar
                            hspace=0.3, wspace=0.1)  # Reduced spacing between plots
        
        # Create axes array
        axs = np.empty((5, 5), dtype=object)
        for i in range(5):
            for j in range(5):
                axs[i, j] = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
        
        # Create axis for colorbar
        cax = fig.add_subplot(gs[5, :])
        
        # Set up the levels
        levels_psl_abs = np.array(
            [
                996,
                998,
                1000,
                1002,
                1004,
                1006,
                1008,
                1010,
                1012,
                1014,
                1016,
                1018,
                1020,
                1022,
                1024,
                1026,
            ]
        )

        # Loop over the rows in this subset df for the model
        for j, (_, row) in tqdm(enumerate(subset_df.iterrows())):
            # Extract the init_year from the df
            init_year_df = int(row["init_year"])
            member_df = int(row["member"])
            lead_df = int(row["lead"])
            winter_year_df = int(row["winter_year"])

            # Construct the condition for element wise comparison
            condition = (
                (init_year_array_this == init_year_df)
                & (member_array_this == member_df)
                & (lead_array_this == lead_df)
            )

            try:
                # Find the index where this condition is met
                index_this = np.where(condition)[0][0]
            except IndexError:
                print(
                    f"init year {init_year_df}, member {member_df}, lead {lead_df} not found"
                )
                missing_days += 1

            # Extract the corresponding value from the subset_arr_this_model
            subset_arr_this_model_psl_index_this = model_psl_subset[index_this, :, :]
            subset_arr_this_model_tas_index_this = model_temp_subset[index_this, :, :]
            subset_arr_this_model_wind_index_this = model_wind_subset[index_this, :, :]
            subset_arr_this_model_uas_index_this = model_uas_subset[index_this, :, :]
            subset_arr_this_model_vas_index_this = model_vas_subset[index_this, :, :]

            # Set up the ax this
            ax_this = axs[j // 5, j % 5]

            # Plot the psl this
            psl_plot_this = ax_this.contourf(
                psl_lons,
                psl_lats,
                subset_arr_this_model_psl_index_this / 100, # Convert to hPa
                levels=levels_psl_abs,
                cmap="RdBu_r",
                transform=ccrs.PlateCarree(),
                extend="both",
            )

            # add coastlines
            ax_this.coastlines()

            # Set up the title
            ax_this.set_title(
                f"i_y: {init_year_df}, m: {member_df}, w_y: {winter_year_df}, l: {lead_df}",
                fontsize=8,
            )

            # # Store the value in the subset_arr_this_model_full
            # subset_arr_this_model_psl[j, :, :] = subset_arr_this_model_psl_index_this
            # subset_arr_this_model_tas[j, :, :] = subset_arr_this_model_tas_index_this
            # subset_arr_this_model_wind[j, :, :] = subset_arr_this_model_wind_index_this
            # subset_arr_this_model_uas[j, :, :] = subset_arr_this_model_uas_index_this
            # subset_arr_this_model_vas[j, :, :] = subset_arr_this_model_vas_index_this

        # Create colorbar using the dedicated axis
        cbar = fig.colorbar(
            psl_plot_this,
            cax=cax,
            orientation="horizontal",
            label="Mean sea level pressure (hPa)"
        )
        
        # Adjust colorbar label size and position
        cbar.ax.tick_params(labelsize=10)
        cbar.ax.set_title("Mean sea level pressure (hPa)", 
                         fontsize=10, pad=10)

        # Show the figure
        plt.show()

        sys.exit()

        # Print the number of missing days
        print(f"Number of missing days for model {i}: {missing_days}")

        # Take the mean over this
        model_psl_subset_this = subset_arr_this_model_psl
        model_temp_subset_this = subset_arr_this_model_tas
        model_wind_subset_this = subset_arr_this_model_wind
        model_uas_subset_this = subset_arr_this_model_uas
        model_vas_subset_this = subset_arr_this_model_vas

        # print the shape of the model psl subset this
        print(f"Shape of model psl subset this: {model_psl_subset_this.shape}")



        # perform the clustering on the model psl subset
        model_3, model_assign_3, model_stats_3 = kmeans_clustering_and_plotting(
            subset_arr=model_psl_subset_this,
            lats_path=os.path.join(
                metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"
            ),
            lons_path=os.path.join(
                metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"
            ),
            n_clusters=3,
            figsize=(10, 10),
            cmap="RdBu_r",
        )

        levels = np.array(
            [
                -6,
                -5,
                -4,
                -3,
                -2,
                -1,
                1,
                2,
                3,
                4,
                5,
                6,
            ]
        )

        # # Plot the tas anoms composites for the clusters
        # model_composites_3_tas = create_and_plot_cluster_composites(
        #     subset_arr=model_temp_subset_this,
        #     cluster_assignments=model_assign_3,
        #     var_name="tas",
        #     lats_path=os.path.join(
        #         metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
        #     ),
        #     lons_path=os.path.join(
        #         metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
        #     ),
        #     cmap="RdBu_r",
        #     exclude_no_type=False,
        #     figsize=(10, 10),
        #     arr_clim=model_tas_clim,
        #     levels=levels,
        # )

        levels = np.array(
            [
                -2,
                -1.5,
                -1,
                -0.5,
                0.5,
                1,
                1.5,
                2,
            ]
        )

        # # Do the same for the wind anoms
        # model_composites_3_wind = create_and_plot_cluster_composites(
        #     subset_arr=model_wind_subset_this,
        #     cluster_assignments=model_assign_3,
        #     var_name="sfcWind",
        #     lats_path=os.path.join(
        #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
        #     ),
        #     lons_path=os.path.join(
        #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
        #     ),
        #     cmap="PRGn",
        #     exclude_no_type=False,
        #     figsize=(10, 10),
        #     arr_clim=model_wind_clim,
        #     levels=levels,
        # )

        # levels = np.array(
        #     [
        #         -4,
        #         -3.5,
        #         -3,
        #         -2.5,
        #         -2,
        #         -1.5,
        #         -1,
        #         -0.5,
        #         0.5,
        #         1,
        #         1.5,
        #         2,
        #         2.5,
        #         3,
        #         3.5,
        #         4,
        #     ]
        # )

        # # Do the same for the uas anoms
        # model_composites_3_uas = create_and_plot_cluster_composites(
        #     subset_arr=model_uas_subset_this,
        #     cluster_assignments=model_assign_3,
        #     var_name="uas",
        #     lats_path=os.path.join(
        #         metadata_dir, "HadGEM3-GC31-MM_uas_Europe_1960_DJF_day_lats.npy"
        #     ),
        #     lons_path=os.path.join(
        #         metadata_dir, "HadGEM3-GC31-MM_uas_Europe_1960_DJF_day_lons.npy"
        #     ),
        #     cmap="PRGn",
        #     exclude_no_type=False,
        #     figsize=(10, 10),
        #     levels=levels,
        # )

        # # Do the same for the vas anoms
        # model_composites_3_vas = create_and_plot_cluster_composites(
        #     subset_arr=model_vas_subset_this,
        #     cluster_assignments=model_assign_3,
        #     var_name="vas",
        #     lats_path=os.path.join(
        #         metadata_dir, "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_lats.npy"
        #     ),
        #     lons_path=os.path.join(
        #         metadata_dir, "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_lons.npy"
        #     ),
        #     cmap="PRGn",
        #     exclude_no_type=False,
        #     figsize=(10, 10),
        #     levels=levels,
        # )

        # Add the model assign 3 to the model df subset grey
        subset_dfs_model[i]["cluster_assign"] = model_assign_3

        # Print the head and tail of the model df subset
        print("Model df subset head:")
        print(subset_dfs_model[i].head())
        print("Model df subset tail:")
        print(subset_dfs_model[i].tail())

        # Subset the model df subset to only the cluster 1 (index of cluster 2)
        model_df_subset_cluster_1 = subset_dfs_model[i][
            subset_dfs_model[i]["cluster_assign"] == 1
        ]

        # Print the shape of the model df subset cluster 1
        print(f"Shape of model df subset cluster 1: {model_df_subset_cluster_1.shape}")

        # Pirnt the head and tail of the model df subset cluster 1
        print("Model df subset cluster 1 head:")
        print(model_df_subset_cluster_1.head())

        print("Model df subset cluster 1 tail")
        print(model_df_subset_cluster_1.tail())

        # Set up a fname for this df
        fname = f"model_df_subset_cluster_1_zonal_block_max_dnw.csv"

        save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

        fpath = os.path.join(save_dir, fname)

        if not os.path.exists(fpath):
            # Save the model df subset cluster 1 to a csv
            model_df_subset_cluster_1.to_csv(fpath, index=False)
            print(f"Saved model df subset cluster 1 to {fpath}")

        sys.exit()

    # Print the time taken
    print(f"Time taken: {time.time() - start_time:.2f} seconds")
    print("------------------------------")
    print("Exiting script.")
    print("------------------------------")

    sys.exit()

    # # Set up the clim arrs obs
    # clim_arrs_obs = [
    #     obs_psl_clim,
    #     obs_psl_clim,
    #     obs_psl_clim,
    # ]

    # # set up the clim arrs obs tas
    # clim_arrs_obs_tas = [
    #     obs_tas_clim,
    #     obs_tas_clim,
    #     obs_tas_clim,
    # ]

    # # Set up the clim arrs obs wind
    # clim_arrs_obs_wind = [
    #     obs_wind_clim,
    #     obs_wind_clim,
    #     obs_wind_clim,
    # ]

    # Set up the clim arrs model
    clim_arrs_model = [
        model_psl_clim,
        model_psl_clim,
        model_psl_clim,
    ]

    # set up the clim arr for model tas
    clim_arrs_model_tas = [
        model_tas_clim,
        model_tas_clim,
        model_tas_clim,
    ]

    # Set up the clim arrs model wind
    clim_arrs_model_wind = [
        model_wind_clim,
        model_wind_clim,
        model_wind_clim,
    ]

    # Set up the clim arrs model vas
    clim_arrs_model_vas = [
        model_vas_clim,
        model_vas_clim,
        model_vas_clim,
    ]

    # Set up the clim arrs model uas
    clim_arrs_model_uas = [
        model_uas_clim,
        model_uas_clim,
        model_uas_clim,
    ]

    # # Set up the dates lists obs
    # dates_lists_obs = [
    #     obs_psl_dates_list,
    #     obs_psl_dates_list,
    #     obs_psl_dates_list,
    # ]

    # # set up the dates list obs tas
    # dates_lists_obs_tas = [
    #     obs_temp_dates_list,
    #     obs_temp_dates_list,
    #     obs_temp_dates_list,
    # ]

    # # Set up the dates lists obs wind
    # dates_lists_obs_wind = [
    #     obs_wind_dates_list,
    #     obs_wind_dates_list,
    #     obs_wind_dates_list,
    # ]

    # Set up the model index dicts
    model_index_dicts = [
        model_psl_subset_index_list,
        model_psl_subset_index_list,
        model_psl_subset_index_list,
    ]

    # Set up the model index dicts tas
    model_index_dicts_tas = [
        model_temp_subset_index_list,
        model_temp_subset_index_list,
        model_temp_subset_index_list,
    ]

    # Set up the model index dicts wind
    model_index_dicts_wind = [
        model_wind_subset_index_list,
        model_wind_subset_index_list,
        model_wind_subset_index_list,
    ]

    # # Set up the model index dicts vas
    model_index_dicts_vas = [
        model_vas_subset_index_list,
        model_vas_subset_index_list,
        model_vas_subset_index_list,
    ]

    # Set up the model index dicts uas
    model_index_dicts_uas = [
        model_uas_subset_index_list,
        model_uas_subset_index_list,
        model_uas_subset_index_list,
    ]

    # Set up the lats path
    lats_paths = [
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"),
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"),
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy"),
    ]

    # Set up the lons path
    lons_paths = [
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"),
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"),
        os.path.join(metadata_dir, "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy"),
    ]

    # lats_path

    lats_europe = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    )
    lons_europe = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    )

    # set up the lons europe for wind
    lons_europe_wind = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    )
    lats_europe_wind = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    )

    # Set up the lons europe vas
    lons_europe_vas = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_lons.npy"
    )
    lats_europe_vas = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_lats.npy"
    )

    # Set up the lons europe uas
    lons_europe_uas = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_uas_Europe_1960_DJF_day_lons.npy"
    )
    lats_europe_uas = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_uas_Europe_1960_DJF_day_lats.npy"
    )

    # Set up the suptitle
    suptitle = (
        "ERA5 and HadGEM3-GC31-MM DnW max MSLP anoms, relative to 1960-2018 climatology"
    )

    # Set up the figure size
    figsize = (12, 6)

    # # # Now test the new function
    # plot_mslp_composites(
    #     subset_dfs_obs=subset_dfs_obs,
    #     subset_dfs_model=subset_dfs_model,
    #     subset_arrs_obs=subset_arrs_obs,
    #     subset_arrs_model=subset_arrs_model,
    #     clim_arrs_obs=clim_arrs_obs,
    #     clim_arrs_model=clim_arrs_model,
    #     dates_lists_obs=dates_lists_obs,
    #     model_index_dicts=model_index_dicts,
    #     lats_paths=lats_paths,
    #     lons_paths=lons_paths,
    #     suptitle=suptitle,
    #     figsize=(8, 9),
    # )

    # sys.exit()

    # # plot the wind composites
    # plot_wind_composites(
    #     subset_dfs_obs=subset_dfs_obs,
    #     subset_dfs_model=subset_dfs_model,
    #     subset_arrs_obs_wind=subset_arrs_obs_wind,
    #     subset_arrs_model_wind=subset_arrs_model_vas,
    #     clim_arrs_obs_wind=clim_arrs_obs_wind,
    #     clim_arrs_model_wind=clim_arrs_model_vas,
    #     dates_lists_obs_wind=dates_lists_obs_wind,
    #     model_index_dicts_wind=model_index_dicts_vas,
    #     lats_path=lats_europe_vas,
    #     lons_path=lons_europe_vas,
    #     suptitle=suptitle,
    #     figsize=(10, 10),
    # )

    # # Check tyhe relationships of the dataframes
    # pdg_funcs.plot_multi_var_perc(
    #     obs_df=low_wind_df,
    #     model_df=low_wind_df,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="100 - temperature percentile",
    #     ylabel="10m wind speed",
    #     title="Inverted percentiles of temp. vs 10m wind speed, low wind DJF days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # # # do the same for the higher wind days
    # pdg_funcs.plot_multi_var_perc(
    #     obs_df=obs_df,
    #     model_df=model_df,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="demand_net_wind_bc_max",
    #     y_var_name_model="data_tas_c_drift_bc_dt",
    #     xlabel="Demand net wind percentiles",
    #     ylabel="Temperature (°C)",
    #     title="Percentiles of DnW vs temperature and wind speed, block max DnW DJF days",
    #     legend_y1="Temperature (°C)",
    #     legend_y2="10m wind speed (m/s)",
    #     y2_var_name_model="data_sfcWind_drift_bc_dt",
    #     y2_label="10m wind speed (m/s)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     y1_zero_line=True,
    # )

    # sys.exit()

    # # # PLot the deamnd net wind on the y-axis and the delta P on the y2 axis
    # pdg_funcs.plot_multi_var_perc(
    #     obs_df=obs_df,
    #     model_df=model_df,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="demand_net_wind_bc_max",
    #     y_var_name_model="demand_net_wind_bc_max",
    #     xlabel="Demand net wind percentiles",
    #     ylabel="Demand net wind (GW)",
    #     title="Percentiles of DnW vs demand net wind, block max DnW DJF days",
    #     legend_y1="Demand net wind (GW)",
    #     legend_y2="10m wind speed (m/s)",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     xlims=(80.0, 105.0),
    # )

    # sys.exit()

    # set up the quartiles
    quartiles = [
        (0.80, 0.85),
        (0.85, 0.90),
        (0.90, 0.95),
        (0.95, 1.0),
    ]

    # # print the collumns in the model df
    # print(f"Columns in model df: {model_df.columns}")

    # # test the new function for plotting temp quartiles
    # plot_temp_quartiles(
    #     subset_df_model=model_df,
    #     tas_var_name="demand_net_wind_bc_max",
    #     subset_arr_model=model_psl_subset,
    #     model_index_dict=model_psl_subset_index_list,
    #     lats_path=lats_paths[0],
    #     lons_path=lons_paths[0],
    #     var_name="psl",
    #     figsize=(8, 10),
    #     anoms_flag=False,
    #     clim_filepath=None,
    #     gridbox=[
    #         dicts.uk_n_box_tight,
    #         dicts.uk_s_box_tight,
    #     ],
    #     quartiles=quartiles,
    # )

    # # sys.exit()

    # # # do the same for the higher wind
    # plot_temp_quartiles(
    #     subset_df_model=model_df,
    #     tas_var_name="demand_net_wind_bc_max",
    #     subset_arr_model=model_psl_subset,
    #     model_index_dict=model_psl_subset_index_list,
    #     lats_path=lats_paths[0],
    #     lons_path=lons_paths[0],
    #     var_name="psl",
    #     figsize=(8, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, psl_clim_fname),
    #     gridbox=[
    #         dicts.uk_n_box_tight,
    #         dicts.uk_s_box_tight,
    #     ],
    #     quartiles=quartiles,
    # )

    # # # Do the same for temperature
    # plot_temp_quartiles(
    #     subset_df_model=model_df,
    #     tas_var_name="demand_net_wind_bc_max",
    #     subset_arr_model=model_temp_subset,
    #     model_index_dict=model_temp_subset_index_list,
    #     lats_path=lats_europe,
    #     lons_path=lons_europe,
    #     var_name="tas",
    #     figsize=(6, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, tas_clim_fname),
    #     gridbox=dicts.wind_gridbox_south,
    #     quartiles=quartiles,
    # )

    # # sys.exit()

    # # # Do the same for wind speed
    # plot_temp_quartiles(
    #     subset_df_model=model_df,
    #     tas_var_name="demand_net_wind_bc_max_bc",
    #     subset_arr_model=model_wind_subset,
    #     model_index_dict=model_wind_subset_index_list,
    #     lats_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy"
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir, "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy"
    #     ),
    #     var_name="sfcWind",
    #     figsize=(6, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, sfcWind_clim_fname),
    #     gridbox=dicts.wind_gridbox_south,
    #     quartiles=quartiles,
    # )

    # sys.exit()

    # Plot the differences between lower wind and higher wind (full field)
    # low - high in this case
    # plot_temp_quartiles(
    #     subset_df_model=low_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_low_wind_psl_subset,
    #     model_index_dict=model_low_wind_psl_subset_index_list,
    #     lats_path=lats_paths[0],
    #     lons_path=lons_paths[0],
    #     var_name="tas",
    #     figsize=(10, 10),
    #     anoms_flag=False,
    #     clim_filepath=None,
    #     second_subset_df=higher_wind_df,
    #     second_subset_arr=model_higher_wind_psl_subset,
    #     second_model_index_dict=model_higher_wind_psl_subset_index_list,
    # )

    # # Plot temp quartiles, but for anoms
    # plot_temp_quartiles(
    #     subset_df_model=low_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_low_wind_psl_subset,
    #     model_index_dict=model_low_wind_psl_subset_index_list,
    #     lats_path=lats_paths[0],
    #     lons_path=lons_paths[0],
    #     var_name="tas",
    #     figsize=(10, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, psl_clim_fname),
    #     gridbox=dicts.wind_gridbox_subset,
    # )

    # # Do the same for the high wind days but for anoms
    # plot_temp_quartiles(
    #     subset_df_model=higher_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_higher_wind_psl_subset,
    #     model_index_dict=model_higher_wind_psl_subset_index_list,
    #     lats_path=lats_paths[0],
    #     lons_path=lons_paths[0],
    #     var_name="tas",
    #     figsize=(10, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, psl_clim_fname),
    #     gridbox=dicts.wind_gridbox_subset,
    # )

    # # plot_temp_quartiles(
    # #     subset_df_model=low_wind_df,
    # #     tas_var_name="data_tas_c",
    # #     subset_arr_model=model_low_wind_psl_subset,
    # #     model_index_dict=model_low_wind_psl_subset_index_list,
    # #     lats_path=lats_paths[0],
    # #     lons_path=lons_paths[0],
    # #     var_name="tas",
    # #     figsize=(10, 10),
    # #     anoms_flag=True,
    # #     clim_filepath=os.path.join(model_clim_dir, psl_clim_fname),
    # #     second_subset_df=higher_wind_df,
    # #     second_subset_arr=model_higher_wind_psl_subset,
    # #     second_model_index_dict=model_higher_wind_psl_subset_index_list,
    # # )

    # # Low wind, uas and vas composites first
    # plot_temp_quartiles(
    #     subset_df_model=low_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_low_wind_uas_subset,
    #     model_index_dict=model_low_wind_uas_subset_index_list,
    #     lats_path=lats_europe_uas,
    #     lons_path=lons_europe_uas,
    #     var_name="uas",
    #     figsize=(10, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, uas_clim_fname),
    #     gridbox=dicts.wind_gridbox_subset,
    # )

    # # Now for low wind vas
    # plot_temp_quartiles(
    #     subset_df_model=low_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_low_wind_vas_subset,
    #     model_index_dict=model_low_wind_vas_subset_index_list,
    #     lats_path=lats_europe_vas,
    #     lons_path=lons_europe_vas,
    #     var_name="vas",
    #     figsize=(10, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, vas_clim_fname),
    #     gridbox=dicts.wind_gridbox_subset,
    # )

    # # Now for higher wind, uas and vas composites first
    # plot_temp_quartiles(
    #     subset_df_model=higher_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_higher_wind_uas_subset,
    #     model_index_dict=model_higher_wind_uas_subset_index_list,
    #     lats_path=lats_europe_uas,
    #     lons_path=lons_europe_uas,
    #     var_name="uas",
    #     figsize=(10, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, uas_clim_fname),
    #     gridbox=dicts.wind_gridbox_subset,
    # )

    # # Now for higher wind vas
    # plot_temp_quartiles(
    #     subset_df_model=higher_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_higher_wind_vas_subset,
    #     model_index_dict=model_higher_wind_vas_subset_index_list,
    #     lats_path=lats_europe_vas,
    #     lons_path=lons_europe_vas,
    #     var_name="vas",
    #     figsize=(10, 10),
    #     anoms_flag=True,
    #     clim_filepath=os.path.join(model_clim_dir, vas_clim_fname),
    #     gridbox=dicts.wind_gridbox_subset,
    # )

    # # Plot temp quartiels for sfcWind
    # plot_temp_quartiles(
    #     subset_df_model=low_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_lower_wind_sfcWind_subset,
    #     model_index_dict=model_lower_wind_sfcWind_subset_index_list,
    #     lats_path=lats_europe,
    #     lons_path=lons_europe,
    #     var_name="sfcWind",
    #     figsize=(10, 10),
    #     anoms_flag=False,
    #     clim_filepath=None,
    #     gridbox=dicts.wind_gridbox_subset,
    # )

    # # Now for higher wind sfcWind
    # plot_temp_quartiles(
    #     subset_df_model=higher_wind_df,
    #     tas_var_name="data_tas_c",
    #     subset_arr_model=model_higher_wind_sfcWind_subset,
    #     model_index_dict=model_higher_wind_sfcWind_subset_index_list,
    #     lats_path=lats_europe,
    #     lons_path=lons_europe,
    #     var_name="sfcWind",
    #     figsize=(10, 10),
    #     anoms_flag=False,
    #     clim_filepath=None,
    #     gridbox=dicts.wind_gridbox_subset,
    # )

    # sys.exit()

    # # test the new function
    # plot_var_composites_model(
    #     subset_dfs_model=subset_dfs_model,
    #     subset_arrs_model=subset_arrs_model_vas,
    #     clim_arrs_model=clim_arrs_model_vas,
    #     model_index_dicts=model_index_dicts_vas,
    #     lats_path=lats_europe_vas,
    #     lons_path=lons_europe_vas,
    #     var_name="vas",
    #     figsize=(10, 10),
    # )

    # test the new new function
    multi_subset_dfs_list = [
        subset_dfs_model,
        subset_dfs_model,
        subset_dfs_model,
    ]

    multi_subset_arrs_list = [
        subset_arrs_model,
        subset_arrs_model_tas,
        subset_arrs_model_wind,
    ]

    multi_clim_arrs_list = [
        clim_arrs_model,
        clim_arrs_model_tas,
        clim_arrs_model_wind,
    ]

    multi_model_index_dicts_list = [
        model_index_dicts,
        model_index_dicts_tas,
        model_index_dicts_wind,
    ]

    lats_paths_list = [
        lats_paths[0],
        lats_europe,
        lats_europe_wind,
    ]

    lons_paths_list = [
        lons_paths[0],
        lons_europe,
        lons_europe_wind,
    ]

    plot_width = 10
    plot_height = 8

    # print the shape of subset arrs tas and subset arras wind
    print(f"Shape of subset arrs tas: {subset_arrs_model_tas[0].shape}")
    print(f"Shape of subset arrs wind: {subset_arrs_model_wind[0].shape}")

    # # Plot the var composites for psl
    # plot_var_composites_model(
    #     subset_dfs_model=subset_dfs_model,
    #     subset_arrs_model=subset_arrs_model,
    #     clim_arrs_model=clim_arrs_model,
    #     model_index_dicts=model_index_dicts,
    #     lats_path=lats_paths[0],
    #     lons_path=lons_paths[0],
    #     var_name="psl",
    #     figsize=(10, 10),
    # )

    # # Do the same for tas
    # plot_var_composites_model(
    #     subset_dfs_model=subset_dfs_model,
    #     subset_arrs_model=subset_arrs_model_tas,
    #     clim_arrs_model=clim_arrs_model_tas,
    #     model_index_dicts=model_index_dicts_tas,
    #     lats_path=lats_europe,
    #     lons_path=lons_europe,
    #     var_name="tas",
    #     figsize=(10, 10),
    # )

    # # Do the same for wind
    # plot_var_composites_model(
    #     subset_dfs_model=subset_dfs_model,
    #     subset_arrs_model=subset_arrs_model_wind,
    #     clim_arrs_model=clim_arrs_model_wind,
    #     model_index_dicts=model_index_dicts_wind,
    #     lats_path=lats_europe,
    #     lons_path=lons_europe,
    #     var_name="sfcWind",
    #     figsize=(10, 10),
    # )

    # # test the new function for plotting multiple variables
    plot_multi_var_composites_model(
        multi_subset_dfs_model=multi_subset_dfs_list,
        multi_subset_arrs_model=multi_subset_arrs_list,
        multi_clim_arrs_model=multi_clim_arrs_list,
        multi_model_index_dicts=multi_model_index_dicts_list,
        multi_lats_path=lats_paths_list,
        multi_lons_path=lons_paths_list,
        multi_var_names=["psl", "tas", "sfcWind"],
        figsize=(plot_width, plot_height),
    )

    sys.exit()

    # test the new function
    plot_var_composites_model(
        subset_dfs_model=subset_dfs_model,
        subset_arrs_model=subset_arrs_model_uas,
        clim_arrs_model=clim_arrs_model_uas,
        model_index_dicts=model_index_dicts_uas,
        lats_path=lats_europe_uas,
        lons_path=lons_europe_uas,
        var_name="uas",
        figsize=(10, 10),
    )

    # Do the same for vas
    plot_var_composites_model(
        subset_dfs_model=subset_dfs_model,
        subset_arrs_model=subset_arrs_model_vas,
        clim_arrs_model=clim_arrs_model_vas,
        model_index_dicts=model_index_dicts_vas,
        lats_path=lats_europe_vas,
        lons_path=lons_europe_vas,
        var_name="vas",
        figsize=(10, 10),
    )

    sys.exit()

    # test the function for just plotting the tas composites
    plot_tas_composites(
        subset_dfs_obs=subset_dfs_obs,
        subset_dfs_model=subset_dfs_model,
        subset_arrs_obs_tas=subset_arrs_obs_tas,
        subset_arrs_model_tas=subset_arrs_model_tas,
        clim_arrs_obs_tas=clim_arrs_obs_tas,
        clim_arrs_model_tas=clim_arrs_model_tas,
        dates_lists_obs_tas=dates_lists_obs_tas,
        model_index_dicts_tas=model_index_dicts_tas,
        lats_path=lats_europe,
        lons_path=lons_europe,
        suptitle=suptitle,
        figsize=(10, 10),
    )

    sys.exit()

    # test the tas wind composites function
    plot_tas_wind_composites(
        subset_dfs_obs=subset_dfs_obs,
        subset_dfs_model=subset_dfs_model,
        subset_arrs_obs_tas=subset_arrs_obs_tas,
        subset_arrs_model_tas=subset_arrs_model_tas,
        subset_arrs_obs_wind=subset_arrs_obs_wind,
        subset_arrs_model_wind=subset_arrs_model_wind,
        clim_arrs_obs_tas=clim_arrs_obs_tas,
        clim_arrs_model_tas=clim_arrs_model_tas,
        clim_arrs_obs_wind=clim_arrs_obs_wind,
        clim_arrs_model_wind=clim_arrs_model_wind,
        dates_lists_obs_tas=dates_lists_obs_tas,
        dates_lists_obs_wind=dates_lists_obs_wind,
        model_index_dicts_tas=model_index_dicts_tas,
        model_index_dicts_wind=model_index_dicts_wind,
        lats_path=lats_europe,
        lons_path=lons_europe,
        suptitle=suptitle,
        figsize=(12, 28),
    )

    sys.exit()

    # # Plot the composites for all the data
    # plot_composites_model(
    #     subset_df=model_df,
    #     subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
    #     clim_arrs=[model_psl_clim, model_tas_clim, model_wind_clim],
    #     index_dicts=[
    #         model_psl_subset_index_list,
    #         model_temp_subset_index_list,
    #         model_wind_subset_index_list,
    #     ],
    #     variables=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
    #         ),
    #     ],
    #     suptitle="All model DnW max",
    #     figsize=(12, 6),
    # )

    # # Quantify the 80th percentil of demand net wind max
    # # for the observations
    # obs_dnw_80th = obs_df["demand_net_wind_max"].quantile(0.8)

    # # Subset the model df to where demand_net_wind_max_bc
    # # exceeds this
    # model_subset_df = model_df[
    #     model_df["demand_net_wind_max_bc"] >= obs_dnw_80th
    # ]

    # # print the length of the model subset df
    # print(f"Length of model subset df: {len(model_subset_df)}")

    # # test the new function
    # plot_composites_model(
    #     subset_df=model_subset_df,
    #     subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
    #     clim_arrs=[model_psl_clim, model_tas_clim, model_wind_clim],
    #     index_dicts=[
    #         model_psl_subset_index_list,
    #         model_temp_subset_index_list,
    #         model_wind_subset_index_list,
    #     ],
    #     variables=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
    #         ),
    #     ],
    #     suptitle="Model DnW max > 80th percentile obs DnW max",
    #     figsize=(12, 6),
    # )

    # # quantify the 90th percentile of demand net wind max in the obs
    # obs_dnw_90th = obs_df["demand_net_wind_max"].quantile(0.9)

    # # subset the obs df to where this is exceeded
    # model_subset_df_90 = model_df[
    #     model_df["demand_net_wind_max_bc"] >= obs_dnw_90th
    # ]

    # # print the length of the model subset df
    # print(f"Length of model subset df 90: {len(model_subset_df_90)}")

    # # test the new function
    # plot_composites_model(
    #     subset_df=model_subset_df_90,
    #     subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
    #     clim_arrs=[model_psl_clim, model_tas_clim, model_wind_clim],
    #     index_dicts=[
    #         model_psl_subset_index_list,
    #         model_temp_subset_index_list,
    #         model_wind_subset_index_list,
    #     ],
    #     variables=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
    #         ),
    #     ],
    #     suptitle="Model DnW max > 90th percentile obs DnW max",
    #     figsize=(12, 6),
    # )

    # # same for the 95th percentile
    # obs_dnw_95th = obs_df["demand_net_wind_max"].quantile(0.95)

    # # subset the obs df to where this is exceeded
    # model_subset_df_95 = model_df[
    #     model_df["demand_net_wind_max_bc"] >= obs_dnw_95th
    # ]

    # # print the length of the model subset df
    # print(f"Length of model subset df 95: {len(model_subset_df_95)}")
    # # test the new function
    # plot_composites_model(
    #     subset_df=model_subset_df_95,
    #     subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
    #     clim_arrs=[model_psl_clim, model_tas_clim, model_wind_clim],
    #     index_dicts=[
    #         model_psl_subset_index_list,
    #         model_temp_subset_index_list,
    #         model_wind_subset_index_list,
    #     ],
    #     variables=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
    #         ),
    #     ],
    #     suptitle="Model DnW max > 95th percentile obs DnW max",
    #     figsize=(12, 6),
    # )

    # # find the obs max
    # obs_max = obs_df["demand_net_wind_max"].max()

    # # subset the model df to where this is exceeded
    # model_subset_df_max = model_df[
    #     model_df["demand_net_wind_max_bc"] >= obs_max
    # ]

    # # print the length of the model subset df
    # print(f"Length of model subset df max: {len(model_subset_df_max)}")

    # # test the new function
    # plot_composites_model(
    #     subset_df=model_subset_df_max,
    #     subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
    #     clim_arrs=[model_psl_clim, model_tas_clim, model_wind_clim],
    #     index_dicts=[
    #         model_psl_subset_index_list,
    #         model_temp_subset_index_list,
    #         model_wind_subset_index_list,
    #     ],
    #     variables=["psl", "tas", "sfcWind"],
    #     lats_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
    #         ),
    #     ],
    #     lons_paths=[
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
    #         ),
    #         os.path.join(
    #             metadata_dir,
    #             "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
    #         ),
    #     ],
    #     suptitle="Model DnW max > max obs DnW max",
    #     figsize=(12, 6),
    # )

    # # plot all of the data for psl
    # plot_data_postage_stamp(
    #     subset_arr=psl_subset,
    #     clim_arr=obs_psl_clim,
    #     dates_list=psl_dates_list,
    #     variable="psl",
    #     region="NA",
    #     season=season,
    #     lats_path=os.path.join(
    #         metadata_dir,
    #         "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir,
    #         "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
    #     ),
    # )

    # # Plot the postage stamps for tas
    # plot_data_postage_stamp(
    #     subset_arr=temp_subset,
    #     clim_arr=obs_tas_clim,
    #     dates_list=temp_dates_list,
    #     variable="tas",
    #     region="Europe",
    #     season=season,
    #     lats_path=os.path.join(
    #         metadata_dir,
    #         "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir,
    #         "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
    #     ),
    # )

    # # Plot the postage stamps for wind speed
    # plot_data_postage_stamp(
    #     subset_arr=wind_subset,
    #     clim_arr=obs_wind_clim,
    #     dates_list=wind_dates_list,
    #     variable="sfcWind",
    #     region="Europe",
    #     season=season,
    #     lats_path=os.path.join(
    #         metadata_dir,
    #         "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
    #     ),
    #     lons_path=os.path.join(
    #         metadata_dir,
    #         "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
    #     ),
    # )

    # Find the 80th percentile value of demand net wind max
    dnw_80th = obs_df["demand_net_wind_max"].quantile(0.8)

    # subset the obs df to where this is excedded
    subset_df = obs_df[obs_df["demand_net_wind_max"] >= dnw_80th]

    # print the shape of obs psl arr
    print(f"Shape of obs psl arr: {obs_psl_arr.shape}")
    # print the shape of obs temp arr
    print(f"Shape of obs temp arr: {obs_temp_arr.shape}")
    # print the shape of obs wind arr
    print(f"Shape of obs wind arr: {obs_wind_arr.shape}")

    # print the shape of obs psl clim
    print(f"Shape of obs psl clim: {obs_psl_clim.shape}")
    # print the shape of obs temp clim
    print(f"Shape of obs temp clim: {obs_tas_clim.shape}")
    # print the shape of obs wind clim
    print(f"Shape of obs wind clim: {obs_wind_clim.shape}")

    # plot the composites for all of the winter days
    plot_composites(
        subset_df=obs_df,
        subset_arrs=[psl_subset, temp_subset, wind_subset],
        clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
        dates_lists=[
            psl_dates_list,
            temp_dates_list,
            wind_dates_list,
        ],
        variables=["psl", "tas", "sfcWind"],
        lats_paths=[
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
            ),
        ],
        lons_paths=[
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
            ),
        ],
        suptitle="All obs DnW max",
        figsize=(12, 6),
    )

    # plot the composites for this
    plot_composites(
        subset_df=subset_df,
        subset_arrs=[psl_subset, temp_subset, wind_subset],
        clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
        dates_lists=[
            psl_dates_list,
            temp_dates_list,
            wind_dates_list,
        ],
        variables=["psl", "tas", "sfcWind"],
        lats_paths=[
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
            ),
        ],
        lons_paths=[
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
            ),
        ],
        suptitle="Obs DnW max > 80th percentile obs DnW max",
        figsize=(12, 6),
    )

    # Also do the 10th percentile for this
    dnw_10th = obs_df["demand_net_wind_max"].quantile(0.9)
    subset_df_10th = obs_df[obs_df["demand_net_wind_max"] >= dnw_10th]

    # plot this composite
    plot_composites(
        subset_df=subset_df_10th,
        subset_arrs=[psl_subset, temp_subset, wind_subset],
        clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
        dates_lists=[
            psl_dates_list,
            temp_dates_list,
            wind_dates_list,
        ],
        variables=["psl", "tas", "sfcWind"],
        lats_paths=[
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
            ),
        ],
        lons_paths=[
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
            ),
        ],
        suptitle="Obs DnW max > 90th percentile obs DnW max",
        figsize=(12, 6),
    )

    # Find the max demand net wind max
    dnw_max = obs_df["demand_net_wind_max"].max()

    # Subset the obs df to where this is exceeded
    subset_df_max = obs_df[obs_df["demand_net_wind_max"] >= dnw_max]

    # plot this composite
    plot_composites(
        subset_df=subset_df_max,
        subset_arrs=[psl_subset, temp_subset, wind_subset],
        clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
        dates_lists=[
            psl_dates_list,
            temp_dates_list,
            wind_dates_list,
        ],
        variables=["psl", "tas", "sfcWind"],
        lats_paths=[
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lats.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lats.npy",
            ),
        ],
        lons_paths=[
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_psl_NA_1960_DJF_day_lons.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy",
            ),
            os.path.join(
                metadata_dir,
                "HadGEM3-GC31-MM_sfcWind_Europe_1960_DJF_day_lons.npy",
            ),
        ],
        suptitle="Obs DnW max >= max obs DnW max",
        figsize=(12, 6),
    )

    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print("Done!")

    return None


if __name__ == "__main__":
    main()

# %%

# %%

# %%
