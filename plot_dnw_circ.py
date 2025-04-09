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
from scipy.stats import pearsonr


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

        # Set up the subset arr this
        subset_arr_this_full = np.zeros(
            (len(subset_df), subset_arrs[i].shape[1], subset_arrs[i].shape[2])
        )

        # extract the lons
        lons = np.load(lons_paths[i])
        # extract the lats
        lats = np.load(lats_paths[i])

        # Loop over the rows in the subset df
        for j, (_, row) in tqdm(enumerate(subset_df.iterrows()), desc="Processing dataframe", total=len(subset_df)):
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
                (init_year_array == init_year_df) &
                (member_array == member_df) &
                (lead_array == lead_df)
            )

            # Use np.where to find the index
            iyear_member_lead_index = np.where(condition)[0][0]

            # apply this to the subset arra this
            subset_arr_this = subset_arrs[i][iyear_member_lead_index, :, :]

            # append this to the subset arr this full
            subset_arr_this_full[j, :, :] = subset_arr_this

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
        cbar = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, shrink=0.8,
                            location="bottom")
        
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
            anoms_scatter_this_mean_eu = np.mean(
                anoms_scatter_this_eu, axis=(1, 2)
            )
            anoms_scatter_this_mean_uk = np.mean(
                anoms_scatter_this_uk, axis=(1, 2)
            )

            # calculate the pearson correlation
            corr, _ = pearsonr(
                anoms_scatter_this_mean_eu,
                anoms_scatter_this_mean_uk,
            )

            # plot the scatter
            ax_scatter.scatter(
                anoms_scatter_this_mean_uk,
                anoms_scatter_this_mean_eu,
                color="red",
                label=f"r={corr:.2f}",
                s=1,
                marker="o",
            )

            # Fit a straight line to the data
            m, b = np.polyfit(
                anoms_scatter_this_mean_uk,
                anoms_scatter_this_mean_eu,
                1,
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

        # Apply these index to the subset data
        subset_arr_this = subset_arrs[i][indexes, :, :]

        # print the shape of subset arr this
        print(f"Shape of subset arr this: {subset_arr_this.shape}")

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

            # plot these scatter points
            ax_scatter.scatter(
                anoms_this_uk_mean,
                anoms_this_eu_mean,
                color="blue",
                label=f"r = {corr:.2f}",
                s=1,
                marker="x",
            )

            # fit a straight line to the data
            m, b = np.polyfit(anoms_this_uk_mean, anoms_this_eu_mean, 1)

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

    # load in the model subset files
    model_psl_subset_fname = (
        f"HadGEM3-GC31-MM_psl_NA_1960-2018_{season}_{time_freq}_DnW_subset.npy"
    )
    model_psl_subset_json_fname = (
        f"HadGEM3-GC31-MM_psl_NA_1960-2018_DJF_day_DnW_subset_index_list.json"
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
        f"HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_{season}_{time_freq}_DnW_subset.npy"
    )
    model_wind_subset_json_fname = (
        f"HadGEM3-GC31-MM_sfcWind_Europe_1960-2018_DJF_day_DnW_subset_index_list.json"
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
        f"HadGEM3-GC31-MM_tas_Europe_1960-2018_{season}_{time_freq}_DnW_subset.npy"
    )
    model_temp_subset_json_fname = (
        f"HadGEM3-GC31-MM_tas_Europe_1960-2018_DJF_day_DnW_subset_index_list.json"
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
    sfcWind_clim_fname = f"climatology_HadGEM3-GC31-MM_sfcWind_DJF_Europe_1960_2018_day.npy"
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
    obs_psl_clim = np.load(os.path.join(model_clim_dir, psl_clim_fname))
    obs_wind_clim = np.load(os.path.join(model_clim_dir, sfcWind_clim_fname))
    obs_tas_clim = np.load(os.path.join(model_clim_dir, tas_clim_fname))

    # print the shape of the climatology data
    print(f"Shape of obs psl climatology data: {obs_psl_clim.shape}")
    print(f"Shape of obs wind climatology data: {obs_wind_clim.shape}")
    print(f"Shape of obs tas climatology data: {obs_tas_clim.shape}")

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

    # Quantify the 80th percentil of demand net wind max
    # for the observations
    obs_dnw_80th = obs_df["demand_net_wind_max"].quantile(0.8)

    # Subset the model df to where demand_net_wind_max_bc
    # exceeds this
    model_subset_df = model_df[
        model_df["demand_net_wind_max_bc"] >= obs_dnw_80th
    ]

    # print the length of the model subset df
    print(f"Length of model subset df: {len(model_subset_df)}")

    # test the new function
    plot_composites_model(
        subset_df=model_subset_df,
        subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
        clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
        index_dicts=[
            model_psl_subset_index_list,
            model_temp_subset_index_list,
            model_wind_subset_index_list,
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
        figsize=(12, 6),
    )

    # quantify the 90th percentile of demand net wind max in the obs
    obs_dnw_90th = obs_df["demand_net_wind_max"].quantile(0.9)

    # subset the obs df to where this is exceeded
    model_subset_df_90 = model_df[
        model_df["demand_net_wind_max_bc"] >= obs_dnw_90th
    ]

    # print the length of the model subset df
    print(f"Length of model subset df 90: {len(model_subset_df_90)}")

    # test the new function
    plot_composites_model(
        subset_df=model_subset_df_90,
        subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
        clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
        index_dicts=[
            model_psl_subset_index_list,
            model_temp_subset_index_list,
            model_wind_subset_index_list,
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
        figsize=(12, 6),
    )

    # same for the 95th percentile
    obs_dnw_95th = obs_df["demand_net_wind_max"].quantile(0.95)

    # subset the obs df to where this is exceeded
    model_subset_df_95 = model_df[
        model_df["demand_net_wind_max_bc"] >= obs_dnw_95th
    ]

    # print the length of the model subset df
    print(f"Length of model subset df 95: {len(model_subset_df_95)}")
    # test the new function
    plot_composites_model(
        subset_df=model_subset_df_95,
        subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
        clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
        index_dicts=[
            model_psl_subset_index_list,
            model_temp_subset_index_list,
            model_wind_subset_index_list,
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
        figsize=(12, 6),
    )

    # find the obs max
    obs_max = obs_df["demand_net_wind_max"].max()

    # subset the model df to where this is exceeded
    model_subset_df_max = model_df[
        model_df["demand_net_wind_max_bc"] >= obs_max
    ]

    # print the length of the model subset df
    print(f"Length of model subset df max: {len(model_subset_df_max)}")

    # test the new function
    plot_composites_model(
        subset_df=model_subset_df_max,
        subset_arrs=[model_psl_subset, model_temp_subset, model_wind_subset],
        clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
        index_dicts=[
            model_psl_subset_index_list,
            model_temp_subset_index_list,
            model_wind_subset_index_list,
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
        figsize=(12, 6),
    )

    sys.exit()

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

    # # Find the 80th percentile value of demand net wind max
    # dnw_80th = obs_df["demand_net_wind_max"].quantile(0.8)

    # # subset the obs df to where this is excedded
    # subset_df = obs_df[obs_df["demand_net_wind_max"] >= dnw_80th]

    # # plot the composites for all of the winter days
    # plot_composites(
    #     subset_df=obs_df,
    #     subset_arrs=[obs_psl_arr, obs_temp_arr, obs_wind_arr],
    #     clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     dates_lists=[
    #         psl_dates_list,
    #         temp_dates_list,
    #         wind_dates_list,
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
    #     figsize=(12, 6),
    # )

    # # plot the composites for this
    # plot_composites(
    #     subset_df=subset_df,
    #     subset_arrs=[psl_subset, temp_subset, wind_subset],
    #     clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     dates_lists=[
    #         psl_dates_list,
    #         temp_dates_list,
    #         wind_dates_list,
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
    #     figsize=(12, 6),
    # )

    # # Also do the 10th percentile for this
    # dnw_10th = obs_df["demand_net_wind_max"].quantile(0.9)
    # subset_df_10th = obs_df[obs_df["demand_net_wind_max"] >= dnw_10th]

    # # plot this composite
    # plot_composites(
    #     subset_df=subset_df_10th,
    #     subset_arrs=[psl_subset, temp_subset, wind_subset],
    #     clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     dates_lists=[
    #         psl_dates_list,
    #         temp_dates_list,
    #         wind_dates_list,
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
    #     figsize=(12, 6),
    # )

    # # Find the max demand net wind max
    # dnw_max = obs_df["demand_net_wind_max"].max()

    # # Subset the obs df to where this is exceeded
    # subset_df_max = obs_df[obs_df["demand_net_wind_max"] >= dnw_max]

    # # plot this composite
    # plot_composites(
    #     subset_df=subset_df_max,
    #     subset_arrs=[psl_subset, temp_subset, wind_subset],
    #     clim_arrs=[obs_psl_clim, obs_tas_clim, obs_wind_clim],
    #     dates_lists=[
    #         psl_dates_list,
    #         temp_dates_list,
    #         wind_dates_list,
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
    #     figsize=(12, 6),
    # )

    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")
    print("Done!")

    return None


if __name__ == "__main__":
    main()

# %%
