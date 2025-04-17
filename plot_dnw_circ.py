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
from typing import List, Tuple, Dict, Any
from datetime import datetime, timedelta
from scipy.stats import pearsonr, linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable


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
    unique_dec_years = np.arange(1960, 2018 + 1, 1)

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

        # Set up the fname for the times
        times_fname = f"ERA5_{variable}_{region}_{year_to_extract_this}_{season}_{time_freq}_times.npy"

        # If the file does not exist then raise an error
        if not os.path.exists(os.path.join(metadata_dir, times_fname)):
            raise FileNotFoundError(
                f"File {os.path.join(metadata_dir, times_fname)} does not exist."
            )

        # load the data for this
        data_this = np.load(os.path.join(arrs_dir, fname_this))

        # if the variable is tas
        if variable == "tas":
            print("--------------------------------")
            print(f"Data shape: {data_this.shape}")
            print("Detrending the data this")
            print("--------------------------------")

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
                    y_index = np.where(unique_dec_years == year_to_extract_this)[0][0]

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
        else:
            # print the shape of the data this
            print(f"Data shape: {data_this.shape}")

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

    # hardcode the cmap and levels for psl
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

    plt.rcParams['figure.constrained_layout.use'] = False
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
    scatter_axes = [ax2, ax5, ax8, ax11, ax14, ax17]  # Replace with the axes where scatter plots are drawn
    for ax in scatter_axes:
        ax.set_aspect('equal', adjustable='box')

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
                print(
                    f"Date {date} not found in dates list obs tas for index {i}"
                )
                print(
                    f"Dates list obs tas: {dates_lists_obs_tas[i]}"
                )
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
        assert np.shape(anoms_scatter_tas_model_this)[1:] == np.shape(anoms_scatter_tas_obs_this)[1:], \
            "The second and third dimensions of the arrays do not match"

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
    gs = fig.add_gridspec(3, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])
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
                print(
                    f"Date {date} not found in dates list obs tas for index {i}"
                )
                print(
                    f"Dates list obs tas: {dates_lists_obs_tas[i]}"
                )
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
        assert np.shape(anoms_scatter_tas_model_this)[1:] == np.shape(anoms_scatter_tas_obs_this)[1:], \
            "The second and third dimensions of the arrays do not match"

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
    gs = fig.add_gridspec(3, 3, figure=fig, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])
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


# Define the main function
def main():
    start_time = time.time()

    # Set up the hard coded variables
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"
    obs_df_fname = "block_maxima_obs_demand_net_wind.csv"
    model_df_fname = "block_maxima_model_demand_net_wind.csv"
    winter_arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/obs/"
    metadata_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/"
    arrs_persist_dir = "/home/users/benhutch/unseen_multi_year/data"
    subset_model_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/subset/"
    model_clim_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_clim/"
    season = "DJF"
    time_freq = "day"
    len_winter_days = 5324

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

    # print the columns in obs df
    print(f"Columns in obs df: {obs_df.columns}")

    # print the columns in model df
    print(f"Columns in model df: {model_df.columns}")

    # sys.exit()

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

    # extract the current date
    current_date = datetime.now().strftime("%Y-%m-%d")

    # Set up fnames for the psl data
    psl_fname = f"ERA5_psl_NA_1960-2018_{season}_{time_freq}_{current_date}.npy"
    psl_times_fname = f"ERA5_psl_NA_1960-2018_{season}_{time_freq}_times_{current_date}.npy"

    # set up fnames for the temperature data
    # NOTE: Detrended temperature here
    temp_fname = f"ERA5_tas_Europe_1960-2018_{season}_{time_freq}_dtr_{current_date}.npy"
    temp_times_fname = f"ERA5_tas_Europe_1960-2018_{season}_{time_freq}_times_dtr_{current_date}.npy"

    # set up fnames for the wind data
    wind_fname = f"ERA5_sfcWind_Europe_1960-2018_{season}_{time_freq}_{current_date}.npy"
    wind_times_fname = f"ERA5_sfcWind_Europe_1960-2018_{season}_{time_freq}_times_{current_date}.npy"

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

    # sys.exit()

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

    # # assert that the dates list arrays are equal
    # assert np.array_equal(
    #     obs_psl_dates_list, obs_temp_dates_list
    # ), "Dates list arrays are not equal"
    # assert np.array_equal(
    #     obs_psl_dates_list, obs_wind_dates_list
    # ), "Dates list arrays are not equal"

    # sys.exit()

    # load in the model subset files
    model_psl_subset_fname = (
        f"HadGEM3-GC31-MM_psl_NA_1960-2018_{season}_{time_freq}_DnW_subset_2025-04-16.npy"
    )
    model_psl_subset_json_fname = (
        f"HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_DnW_subset_index_list_2025-04-16.json"
    )

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

    # print the length of the model psl subset index list
    # Print the length of the model psl subset index list
    print(
        f"Length of model psl subset index list: {np.shape(model_psl_subset_index_list['init_year'])}"
    )

    # print the values of the model psl subset index list
    print(f"Model psl subset index list: {model_psl_subset_index_list}")

    # print the keys in the index list
    print(f"model_psl_subset index list keys: {model_psl_subset_index_list.keys()}")

    # set up the fnames for sfcWind
    model_wind_subset_fname = (
        f"HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_{season}_{time_freq}_DnW_subset_2025-04-16.npy"
    )
    model_wind_subset_json_fname = (
        f"HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_DnW_subset_index_list_2025-04-16.json"
    )

    # if the file does not exist then raise an error
    if not os.path.exists(os.path.join(subset_model_dir, model_wind_subset_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_wind_subset_fname)} does not exist."
        )

    # load the model wind subset
    model_wind_subset = np.load(os.path.join(subset_model_dir, model_wind_subset_fname))

    # # print the shape of the model wind subset
    # print(f"Shape of model wind subset: {model_wind_subset.shape}")

    # # print the values of the model wind subset
    # print(f"Model wind subset: {model_wind_subset}")

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
    model_temp_subset_fname = (
        f"HadGEM3-GC31-MM_tas_Europe_1960-2018_{season}_{time_freq}_DnW_subset_2025-04-16.npy"
    )
    model_temp_subset_json_fname = (
        f"HadGEM3-GC31-MM_tas_Europe_1960-2018_DJF_day_DnW_subset_index_list_2025-04-16.json"
    )

    # if the file does not exist then raise an error
    if not os.path.exists(os.path.join(subset_model_dir, model_temp_subset_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_temp_subset_fname)} does not exist."
        )

    # load the model temperature subset
    model_temp_subset = np.load(os.path.join(subset_model_dir, model_temp_subset_fname))

    # # print the shape of the model temperature subset
    # print(f"Shape of model temperature subset: {model_temp_subset.shape}")

    # # print the values of the model temperature subset
    # print(f"Model temperature subset: {model_temp_subset}")

    # if the json
    if not os.path.exists(os.path.join(subset_model_dir, model_temp_subset_json_fname)):
        raise FileNotFoundError(
            f"File {os.path.join(subset_model_dir, model_temp_subset_json_fname)} does not exist."
        )

    # load the json file
    with open(os.path.join(subset_model_dir, model_temp_subset_json_fname), "r") as f:
        model_temp_subset_index_list = json.load(f)

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

    # load the climatology data
    model_psl_clim = np.load(os.path.join(model_clim_dir, psl_clim_fname))
    model_wind_clim = np.load(os.path.join(model_clim_dir, sfcWind_clim_fname))
    model_tas_clim = np.load(os.path.join(model_clim_dir, tas_clim_fname))

    # print the shape of the climatology data
    print(f"Shape of obs psl climatology data: {model_psl_clim.shape}")
    print(f"Shape of obs wind climatology data: {model_wind_clim.shape}")
    print(f"Shape of obs tas climatology data: {model_tas_clim.shape}")

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
    obs_dnw_80th = obs_df["demand_net_wind_max"].quantile(0.80)

    # Find the maximum of the demand net wind max for the obs
    obs_dnw_max = obs_df["demand_net_wind_max"].max()

    # subset the model df to grey points
    model_df_subset_grey = model_df[
        model_df["demand_net_wind_bc_max_bc"] < obs_dnw_80th
    ]

    # do the same for the obs
    obs_df_subset_grey = obs_df[obs_df["demand_net_wind_max"] < obs_dnw_80th]

    # subset the model df to yellow points
    model_df_subset_yellow = model_df[
        (model_df["demand_net_wind_bc_max_bc"] >= obs_dnw_80th)
        & (model_df["demand_net_wind_bc_max_bc"] < obs_dnw_max)
    ]

    # do the same for the obs
    obs_df_subset_yellow = obs_df[
        (obs_df["demand_net_wind_max"] >= obs_dnw_80th)
        & (obs_df["demand_net_wind_max"] < obs_dnw_max)
    ]

    # subset the model df to red points
    model_df_subset_red = model_df[model_df["demand_net_wind_bc_max_bc"] >= obs_dnw_max]

    # do the same for the obs
    obs_df_subset_red = obs_df[obs_df["demand_net_wind_max"] >= obs_dnw_max]

    # print the shape of moel df subset red
    print(f"Shape of model df subset red: {model_df_subset_red.shape}")

    # print the shape of obs df subset red
    print(f"Shape of obs df subset red: {obs_df_subset_red.shape}")

    # print the values of obs df subset red
    print(f"Obs df subset red: {obs_df_subset_red}")

    # plot the composites
    # plot the composites f`or all of the winter days
    # plot_composites(
    #     subset_df=obs_df_subset_yellow,
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

    # Set up the subset arrs obs
    subset_arrs_obs = [
        obs_psl_subset,
        obs_psl_subset,
        obs_psl_subset,
    ]

    # Set up the subset arrs obs tas
    subset_arrs_obs_tas = [
        obs_temp_subset,
        obs_temp_subset,
        obs_temp_subset,
    ]

    # Set up the subset arrs obs wind
    subset_arrs_obs_wind = [
        obs_wind_subset,
        obs_wind_subset,
        obs_wind_subset,
    ]

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

    # Set up the clim arrs obs
    clim_arrs_obs = [
        obs_psl_clim,
        obs_psl_clim,
        obs_psl_clim,
    ]

    # set up the clim arrs obs tas
    clim_arrs_obs_tas = [
        obs_tas_clim,
        obs_tas_clim,
        obs_tas_clim,
    ]

    # Set up the clim arrs obs wind
    clim_arrs_obs_wind = [
        obs_wind_clim,
        obs_wind_clim,
        obs_wind_clim,
    ]

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

    # Set up the dates lists obs
    dates_lists_obs = [
        obs_psl_dates_list,
        obs_psl_dates_list,
        obs_psl_dates_list,
    ]

    # set up the dates list obs tas
    dates_lists_obs_tas = [
        obs_temp_dates_list,
        obs_temp_dates_list,
        obs_temp_dates_list,
    ]

    # Set up the dates lists obs wind
    dates_lists_obs_wind = [
        obs_wind_dates_list,
        obs_wind_dates_list,
        obs_wind_dates_list,
    ]

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

    lats_europe = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lats.npy"
    )
    lons_europe = os.path.join(
        metadata_dir, "HadGEM3-GC31-MM_tas_Europe_1960_DJF_day_lons.npy"
    )

    # Set up the suptitle
    suptitle = (
        "ERA5 and HadGEM3-GC31-MM DnW max MSLP anoms, relative to 1960-2018 climatology"
    )

    # Set up the figure size
    figsize = (12, 6)

    # # Now test the new function
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

    # plot the wind composites
    plot_wind_composites(
        subset_dfs_obs=subset_dfs_obs,
        subset_dfs_model=subset_dfs_model,
        subset_arrs_obs_wind=subset_arrs_obs_wind,
        subset_arrs_model_wind=subset_arrs_model_wind,
        clim_arrs_obs_wind=clim_arrs_obs_wind,
        clim_arrs_model_wind=clim_arrs_model_wind,
        dates_lists_obs_wind=dates_lists_obs_wind,
        model_index_dicts_wind=model_index_dicts_wind,
        lats_path=lats_europe,
        lons_path=lons_europe,
        suptitle=suptitle,
        figsize=(10, 10),
    )

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
