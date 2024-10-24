#!/usr/bin/env python

"""
plotting_functions.py
=============================

Script to store the functions for plotting the MSLP outlook for specific
periods, for both the observations (ERA5) and the hindcast (DePreSys).

Usage:
------

    $ python plotting_functions.py    

"""

# Imports
import os
import sys
import glob
import time
import re

# Third-party libraries
import numpy as np
import pandas as pd
import xarray as xr
import iris
import iris.coords
import cftime
import matplotlib.pyplot as plt
import matplotlib.ticker as mplticker
import cartopy.crs as ccrs

# Specific third-party imports
from matplotlib import colors
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm
from scipy.stats import pearsonr

# Import types
from typing import Any, Callable, Union, List, Tuple


# Formatting functions for 4 significant figures and 1 decimal point
def format_func(
    x: float,
    pos: int,
):
    """
    Formats the x-axis ticks as significant figures.

    Args:
        x (float): The tick value.
        pos (int): The position of the tick.

    Returns:
        str: The formatted tick value.
    """
    return f"{x:.4g}"


def format_func_one_decimal(
    x: float,
    pos: int,
):
    """
    Formats the x-axis ticks to one decimal point.

    Args:
        x (float): The tick value.
        pos (int): The position of the tick.

    Returns:
        str: The formatted tick value.
    """
    return f"{x:.1f}"


# write a function for plotting the full field MSLP data
# for a full period of interest e.g. November 2010 -> March 2011
def plot_mslp_anoms(
    start_date: str,
    end_date: str,
    title: str,
    variable: str = "msl",
    freq: str = "amon",
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    ERA5_regrid_path: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl.nc",
    climatology_period: list[int] = [1990, 2020],
    calc_anoms: bool = False,
):
    """
    Grabs the MSLP anomalies for a given period of interest and plots them.

    Args:
        start_date (str): The start date of the period of interest.
        end_date (str): The end date of the period of interest.
        title (str): The title of the plot.
        variable (str): The variable of interest.
        freq (str): The frequency of the data.
        lat_bounds (list): The latitude bounds for the plot.
        lon_bounds (list): The longitude bounds for the plot.
        ERA5_regrid_path (str): The path to the regridded ERA5 data.

    Returns:
        None
    """

    # Load the observed data
    ds = xr.open_mfdataset(
        ERA5_regrid_path,
        chunks={"time": 10},
        combine="by_coords",
        parallel=False,
        engine="netcdf4",
        coords="minimal",
    )

    # If expver is present in the observations
    if "expver" in ds.coords:
        # Combine the first two expver variables
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # if the variable is not in the dataset
    if variable not in ds:
        # raise an error
        raise ValueError(f"{variable} not in dataset")

    # calculate the ds climatology
    if calc_anoms:
        # Strip the month from the start and end dates
        # format is "YYYY-MM-DD"
        start_month = start_date[5:7]
        end_month = end_date[5:7]

        # form the list of months to subset the data
        if start_month == end_month:
            months = [start_month]
        elif start_month < end_month:
            months = [
                str(i).zfill(2) for i in range(int(start_month), int(end_month) + 1)
            ]
        else:
            months = [str(i).zfill(2) for i in range(int(start_month), 13)]
            months += [str(i).zfill(2) for i in range(1, int(end_month) + 1)]

        # convert all the months to integers
        months = [int(i) for i in months]

        # print the months
        print(f"months to subset to: {months}")

        # subset the data to the region
        ds_clim = ds.sel(
            lat=slice(lat_bounds[0], lat_bounds[1]),
            lon=slice(lon_bounds[0], lon_bounds[1]),
        )

        # subset the data
        ds_clim = ds_clim.sel(time=ds["time.month"].isin(months))

        # Select the years
        ds_clim = ds_clim.sel(
            time=slice(
                f"{climatology_period[0]}-01-01", f"{climatology_period[1]}-12-31"
            )
        )

        # calculate the climatology
        climatology = ds_clim[variable].mean(dim="time")

    # select the variable
    ds = ds[variable].sel(time=slice(start_date, end_date)).mean(dim="time")

    # subset to the region of interest
    ds = ds.sel(
        lat=slice(lat_bounds[0], lat_bounds[1]), lon=slice(lon_bounds[0], lon_bounds[1])
    )

    # extract the lons
    lons = ds["lon"].values
    lats = ds["lat"].values

    if calc_anoms:
        # calculate the anomalies
        field = (ds.values - climatology.values) / 100  # convert to hPa
    else:
        field = ds.values / 100  # convert to hPa

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks = clevs

        # ensure that these are floats
        clevs = clevs.astype(float)
        ticks = ticks.astype(float)
    else:
        # define the contour levels
        clevs = np.array(np.arange(988, 1024 + 1, 2))
        ticks = clevs

        # ensure that these are ints
        clevs = clevs.astype(int)
        ticks = ticks.astype(int)

    # # print the shape of the inputs
    # print(f"lons shape: {lons.shape}")
    # print(f"lats shape: {lats.shape}")
    # print(f"field shape: {field.shape}")
    # print(f"clevs shape: {clevs.shape}")

    # # print the field values
    # print(f"field values: {field}")

    # Define the custom diverging colormap
    # cs = ["purple", "blue", "lightblue", "lightgreen", "lightyellow", "orange", "red", "darkred"]
    # cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # custom colormap
    cs = [
        "#4D65AD",
        "#3E97B7",
        "#6BC4A6",
        "#A4DBA4",
        "#D8F09C",
        "#FFFEBE",
        "#FFD27F",
        "#FCA85F",
        "#F57244",
        "#DD484C",
        "#B51948",
    ]
    # cs = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # plot the data
    mymap = ax.contourf(
        lons, lats, field, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend="both"
    )
    contours = ax.contour(
        lons,
        lats,
        field,
        clevs,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )
    if calc_anoms:
        ax.clabel(
            contours, clevs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )

    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"mean sea level pressure {climatology_period[0]}-{climatology_period[1]} anomaly (hPa)",
            rotation=0,
            fontsize=10,
        )

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label("mean sea level pressure (hPa)", rotation=0, fontsize=10)

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    # save figure to file
    # plt.savefig('../images/8_python_simple_map_plot_sst_anoms_300dpi.png',
    # format='png', dpi=300)

    # plt.close()

    if calc_anoms:
        print("Returning the climatology")
        climatology_values = climatology.values
    else:
        print("Returning None")
        climatology_values = None

    return climatology_values


# define a function for plotting the temperature/wind contours underneath
# the mslp contours
def plot_mslp_anoms_temp_wind_obs(
    start_date: str,
    end_date: str,
    title: str,
    variable: str = "t2m",
    psl_variable: str = "msl",
    freq: str = "Amon",
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    ERA5_regrid_path: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl.nc",
    climatology_period: list[int] = [1990, 2020],
    calc_anoms: bool = False,
) -> None:
    """
    Grabs the MSLP data and surface variable (e.g. temperature, 10m wind speed)
    for a given period of interest and plots them. Plots the surface variable
    as contours underneath the MSLP contours.

    Args:
        start_date (str): The start date of the period of interest.
        end_date (str): The end date of the period of interest.
        title (str): The title of the plot.
        variable (str): The variable of interest.
        psl_variable (str): The MSLP variable.
        freq (str): The frequency of the data.
        lat_bounds (list): The latitude bounds for the plot.
        lon_bounds (list): The longitude bounds for the plot.
        ERA5_regrid_path (str): The path to the regridded ERA5 data.
        climatology_period (list): The climatology period.
        calc_anoms (bool): Whether to calculate anomalies.

    Returns:
        None
    """

    # Load the observed data
    ds = xr.open_mfdataset(
        ERA5_regrid_path,
        chunks={"time": 10},
        combine="by_coords",
        parallel=False,
        engine="netcdf4",
        coords="minimal",
    )

    # If expver is present in the observations
    if "expver" in ds.coords:
        # Combine the first two expver variables
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # if the variable is not in the dataset
    if variable not in ds:
        # raise an error
        raise ValueError(f"{variable} not in dataset")

    # if the psl_variable is not in the dataset
    if psl_variable not in ds:
        # raise an error
        raise ValueError(f"{psl_variable} not in dataset")

    # if calc_anoms is True
    if calc_anoms:
        # Strip the month from the start and end dates
        # format is "YYYY-MM-DD"
        start_month = start_date[5:7]
        end_month = end_date[5:7]

        # form the list of months to subset the data
        if start_month == end_month:
            months = [start_month]
        elif start_month < end_month:
            months = [
                str(i).zfill(2) for i in range(int(start_month), int(end_month) + 1)
            ]
        else:
            months = [str(i).zfill(2) for i in range(int(start_month), 13)]
            months += [str(i).zfill(2) for i in range(1, int(end_month) + 1)]

        # convert all the months to integers
        months = [int(i) for i in months]

        # print the months
        print(f"months to subset to: {months}")

        # subset the data to the region
        ds_clim = ds.sel(
            lat=slice(lat_bounds[0], lat_bounds[1]),
            lon=slice(lon_bounds[0], lon_bounds[1]),
        )

        # subset the data
        ds_clim = ds_clim.sel(time=ds["time.month"].isin(months))

        # Select the years
        ds_clim = ds_clim.sel(
            time=slice(
                f"{climatology_period[0]}-01-01", f"{climatology_period[1]}-12-31"
            )
        )

        # calculate the climatology
        psl_climatology = ds_clim[psl_variable].mean(dim="time")

        # calculate the variable climatology
        var_climatology = ds_clim[variable].mean(dim="time")

    # select the variable
    ds_var = ds[variable].sel(time=slice(start_date, end_date)).mean(dim="time")

    # select the psl variable
    ds_psl = ds[psl_variable].sel(time=slice(start_date, end_date)).mean(dim="time")

    # subset to the region of interest
    ds_var = ds_var.sel(
        lat=slice(lat_bounds[0], lat_bounds[1]), lon=slice(lon_bounds[0], lon_bounds[1])
    )

    # subset to the region of interest
    ds_psl = ds_psl.sel(
        lat=slice(lat_bounds[0], lat_bounds[1]), lon=slice(lon_bounds[0], lon_bounds[1])
    )

    # extract the lons
    lons = ds_var["lon"].values
    lats = ds_var["lat"].values

    if calc_anoms:
        # calculate the anomalies
        field_var = ds_var.values - var_climatology.values
        field_psl = (ds_psl.values - psl_climatology.values) / 100  # convert to hPa
    else:
        field_var = ds_var.values
        field_psl = ds_psl.values / 100

        # if variable in t2m, tas
        if variable in ["t2m", "tas"]:
            # convert to degrees celsius
            field_var -= 273.15

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs_psl = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks_psl = clevs_psl

        # ensure that these are floats
        clevs_psl = clevs_psl.astype(float)
        ticks_psl = ticks_psl.astype(float)

        # depending on the variable
        if variable in ["t2m", "tas"]:
            # -18 to +18 in 2 degree intervals
            clevs_var = np.array(
                [
                    -5.0,
                    -4.5,
                    -4.0,
                    -3.5,
                    -3.0,
                    -2.5,
                    -2.0,
                    -1.5,
                    -1.0,
                    -0.5,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                ]
            )
            ticks_var = clevs_var

            # set up tjhe cmap
            cmap = "bwr"

            # set the cbar label
            cbar_label = "temperature (°C)"
        elif variable in ["u10", "v10", "sfcWind", "si10"]:
            # 0 to 20 in 2 m/s intervals
            clevs_var = np.array(
                [
                    -1.4,
                    -1.2,
                    -1.0,
                    -0.8,
                    -0.6,
                    -0.4,
                    -0.2,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                    1.2,
                    1.4,
                ]
            )
            ticks_var = clevs_var

            # set up the cmap
            cmap = "PRGn"

            # set the cbar label
            cbar_label = "10m wind speed (m/s)"
        else:
            raise ValueError(f"Unknown variable {variable}")

    else:
        # define the contour levels for the variable
        # should be 19 of them
        if variable in ["t2m", "tas"]:
            # -18 to +18 in 2 degree intervals
            clevs_var = np.array(np.arange(-18, 18 + 1, 2))
            ticks_var = clevs_var

            # set up tjhe cmap
            cmap = "bwr"

            # set the cbar label
            cbar_label = "temperature (°C)"

        elif variable in ["u10", "v10", "sfcWind", "si10"]:
            # 0 to 20 in 2 m/s intervals
            clevs_var = np.array(np.arange(0, 12 + 1, 1))
            ticks_var = clevs_var

            # set up the cmap
            cmap = "RdPu"

            # set the cbar label
            cbar_label = "10m wind speed (m/s)"
        else:
            raise ValueError(f"Unknown variable {variable}")

        # define the contour levels
        clevs_psl = np.array(np.arange(988, 1024 + 1, 2))
        ticks_psl = clevs_psl

        # ensure that these are ints
        clevs_psl = clevs_psl.astype(int)
        ticks_psl = ticks_psl.astype(int)

    # print the len of clevs_psl
    print(f"len of clevs_psl: {len(clevs_psl)}")
    print(f"len of clevs_var: {len(clevs_var)}")

    # print field_var and field_psl
    print(f"field_var shape: {field_var.shape}")
    print(f"field_psl shape: {field_psl.shape}")
    print(f"field_var values: {field_var}")
    print(f"field_psl values: {field_psl}")

    # print the field var min and the field var max
    print(f"field_var min: {field_var.min()}")
    print(f"field_var max: {field_var.max()}")

    if variable in ["si10", "sfcWind"] and not calc_anoms:
        # set up the extend
        extend = "max"
    else:
        extend = "both"

    # plot the data
    mymap = ax.contourf(
        lons,
        lats,
        field_var,
        clevs_var,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend=extend,
    )

    # plot the psl contours
    contours = ax.contour(
        lons,
        lats,
        field_psl,
        clevs_psl,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )

    if calc_anoms:
        ax.clabel(
            contours, clevs_psl, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs_psl, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )
    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"{cbar_label} {climatology_period[0]}-{climatology_period[1]} anomaly",
            rotation=0,
            fontsize=10,
        )

        # # add contour lines to the colorbar
        # cbar.add_lines(mymap)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label(cbar_label, rotation=0, fontsize=10)

        # # set up invisible contour lines for field_var
        # contour_var = ax.contour(
        #     lons,
        #     lats,
        #     field_var,
        #     clevs_var,
        #     colors="k",
        #     transform=ccrs.PlateCarree(),
        #     linewidth=0.2,
        #     alpha=0.5,
        # )

        # # add contour lines to the colorbar
        # cbar.add_lines(contour_var)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks_var)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    if calc_anoms:
        print("Returning the climatology")
        var_climatology_values = var_climatology.values
        psl_climatology_values = psl_climatology.values
    else:
        print("Returning None")
        var_climatology_values = None
        psl_climatology_values = None

    return var_climatology_values, psl_climatology_values


# Write a function which does the same, but for the model data
def plot_mslp_anoms_model(
    init_year: int,
    months: list,
    lead_year: int,
    member: int,
    title: str,
    model: str = "HadGEM3-GC31-MM",
    variable: str = "psl",
    freq: str = "Amon",
    experiment: str = "dcppA-hindcast",
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    climatology_period: list[int] = [1960, 1990],
    grid_bounds: list[float] = [-180.0, 180.0, -90.0, 90.0],
    calc_anoms: bool = False,
    grid_file: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl_first_timestep_msl.nc",
    files_loc_path: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
    saved_clim_path: str = None,
) -> None:
    """
    Grabs the MSLP anomalies for a given period of interest and plots them.

    Args:
        start_date (str): The start date of the period of interest.
        end_date (str): The end date of the period of interest.
        member (str): The member of the model.
        title (str): The title of the plot.
        model (str): The model of interest.
        variable (str): The variable of interest.
        freq (str): The frequency of the data.
        lat_bounds (list): The latitude bounds for the plot.
        lon_bounds (list): The longitude bounds for the plot.
        climatology_period (list): The climatology period.
        calc_anoms (bool): Whether to calculate anomalies.
        grid_file (str): The path to the grid file.
        files_loc_path (str): The path to the file locations.
        saved_clim_path (str): The path to the saved climatology.

    Returns:
        None
    """

    # if calc_anoms is False and saved_clim_path is not none
    # raise an error
    if not calc_anoms and saved_clim_path is not None:
        raise ValueError("Cannot have saved_clim_path if calc_anoms is False")

    # Check that the csv file exists
    if not os.path.exists(files_loc_path):
        raise FileNotFoundError(f"Cannot find the file {files_loc_path}")

    # Load in the csv file
    csv_data = pd.read_csv(files_loc_path)

    # print the data we seek
    print(f"model: {model}")
    print(f"experiment: {experiment}")
    print(f"variable: {variable}")
    print(f"frequency: {freq}")

    # Extract the path for the given model, experiment and variable
    model_path = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == variable)
        & (csv_data["frequency"] == freq),
        "path",
    ].values[0]

    # Assert that theb model path exists
    assert os.path.exists(model_path), f"Cannot find the model path {model_path}"

    # Assert that the model path is not empty
    assert os.listdir(model_path), f"Model path {model_path} is empty"

    # print the model path
    print(f"Model path: {model_path}")

    # Extract the root of the model path
    model_path_root = model_path.split("/")[1]

    # print the model path root
    print(f"Model path root: {model_path_root}")

    # example file
    # /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/psl/HadGEM3-GC31-MM/psl_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s1965-r2i1_gn_196511-197603.nc

    # Extract the year from the start and end dates
    # start_year = start_date[:4]
    # end_year = end_date[:4]

    # # print the start and end years
    # print(f"start year: {start_year}")
    # print(f"end year: {end_year}")

    # depending on the model_path_root
    if model_path_root == "work":
        raise NotImplementedError("work path not implemented yet")
    elif model_path_root == "gws":
        # Create the path
        path = f"{model_path}/{variable}_{freq}_{model}_{experiment}_s{init_year}-r{member}i*_*_{init_year}??-*.nc"

        # print the path
        print(f"path: {path}")
        # print("correct path: /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/psl/HadGEM3-GC31-MM/psl_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s1965-r2i1_gn_196511-197603.nc")

        # glob this path
        files = glob.glob(path)

        # assert that files has length 1
        assert len(files) == 1, f"files has length {len(files)}"

        # print the loaded file
        print(f"Loaded file: {files[0]}")

        # extract the file
        file = files[0]
    elif model_path_root == "badc":
        raise NotImplementedError("home path not implemented yet")
    else:
        raise ValueError(f"Unknown model path root {model_path_root}")

    # load the observed data as a cube
    obs = iris.load_cube(grid_file)

    # Load the model data ad a cube
    cube = iris.load_cube(file)

    # # if expver is a coord in the obs cube
    # if "expver" in obs.dims():
    #     # combine the first two expver variables
    #     obs = obs.extract(iris.Constraint(expver=1)) + obs.extract(iris.Constraint(expver=5))

    # # print the obs cube
    # print(f"obs cube: {obs}")
    # print(f"model cube: {cube}")

    # regrid the model data to the obs grid
    regrid_cube = cube.regrid(obs, iris.analysis.Linear())

    # # print the regridded cube
    # print(f"regridded cube: {regrid_cube}")

    # if the months are ONDJFM
    if months == [10, 11, 12, 1, 2, 3]:
        # set up the start and end dates
        start_date = f"{init_year + lead_year}-10-01"
        end_date = f"{init_year + lead_year + 1}-03-30"
    else:
        raise ValueError("Only implemented for ONDJFM")

    # convert the YYYY-MM-DD to cftime objects
    start_date_cf = cftime.datetime.strptime(start_date, "%Y-%m-%d", calendar="360_day")
    end_date_cf = cftime.datetime.strptime(end_date, "%Y-%m-%d", calendar="360_day")

    # Slice between the start date and end date
    regrid_cube = regrid_cube.extract(
        iris.Constraint(time=lambda cell: start_date_cf <= cell.point <= end_date_cf)
    )

    # take the mean over the time dimension
    regrid_cube = regrid_cube.collapsed("time", iris.analysis.MEAN)

    # subset to the region of interest
    regrid_cube = regrid_cube.intersection(
        latitude=(lat_bounds[0], lat_bounds[1]),
        longitude=(lon_bounds[0], lon_bounds[1]),
    )

    if calc_anoms and saved_clim_path is None:
        print("Caculating the climatology for the model data")

        # initialise a cube list
        ds_list = []

        # Loop over the years in the climatology period
        for year in tqdm(range(climatology_period[0], climatology_period[1] + 1)):
            # Create the path
            path = f"{model_path}/{variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

            # # print the path
            # print(f"path: {path}")

            # glob this path
            files = glob.glob(path)

            # assert that files has length 1
            assert len(files) == 1, f"files has length {len(files)}"

            # # print the loaded file
            # print(f"Loaded file: {files[0]}")

            # extract the file
            file = files[0]

            # # Load the model data ad a cube
            # cube = iris.load_cube(file)

            # load the observed data using xarray
            ds = xr.open_dataset(file)

            # Set up the yyyy-mm-dd format
            start_date_this = cftime.datetime.strptime(
                f"{year}-{start_date[5:10]}", "%Y-%m-%d", calendar="360_day"
            )

            # Use all of the lead years
            end_date_this = cftime.datetime.strptime(
                f"{year + 10}-{end_date[5:10]}", "%Y-%m-%d", calendar="360_day"
            )

            # Slice between the start date and end date and take the mean
            # cube = cube.extract(iris.Constraint(time=lambda cell: start_date_this <= cell.point <= end_date_this))

            # slice between the start date and end date
            ds = ds.sel(time=slice(start_date_this, end_date_this))

            # append the ds to the cube list
            ds_list.append(ds[variable])

        # concatenate with a new time dimension using xarray
        ds_clim = xr.concat(ds_list, dim="time")

        # convert to a cube from xarray
        cube_clim = ds_clim.to_iris()

        # regrid the model data to the obs grid
        regrid_cube_clim = cube_clim.regrid(obs, iris.analysis.Linear())

        # subset to the region of interest
        regrid_cube_clim = regrid_cube_clim.intersection(
            latitude=(lat_bounds[0], lat_bounds[1]),
            longitude=(lon_bounds[0], lon_bounds[1]),
        )

        # calculate the time mean of this
        regrid_cube_clim = regrid_cube_clim.collapsed("time", iris.analysis.MEAN)
    elif calc_anoms and saved_clim_path is not None:
        print("loading saved climatology")

        # load the saved climatology .npy file
        climatology_array = np.load(saved_clim_path)
    # # print the regridded cube
    # print(f"regridded cube: {regrid_cube}")

    # extract the lats and lons values
    lats = regrid_cube.coord("latitude").points
    lons = regrid_cube.coord("longitude").points

    if calc_anoms and saved_clim_path is None:
        # calculate the anomalies
        field = (regrid_cube.data - regrid_cube_clim.data) / 100
    elif calc_anoms and saved_clim_path is not None:
        # calculate the anomalies
        field = (regrid_cube.data - climatology_array) / 100
    else:
        # extract the data values
        field = regrid_cube.data / 100  # convert to hPa

    # # print the shape of the lats and lons
    # print(f"lats shape: {lats.shape}")
    # print(f"lons shape: {lons.shape}")
    # print(f"field shape: {field.shape}")

    # # print the values of the field
    # print(f"field values: {field}")
    # print(f"lats values: {lats}")
    # print(f"lons values: {lons}")

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks = clevs

        # ensure that these are floats
        clevs = clevs.astype(float)
        ticks = ticks.astype(float)
    else:
        # define the contour levels
        clevs = np.array(np.arange(988, 1024 + 1, 2))
        ticks = clevs

        # ensure that these are ints
        clevs = clevs.astype(int)
        ticks = ticks.astype(int)

    # custom colormap
    cs = [
        "#4D65AD",
        "#3E97B7",
        "#6BC4A6",
        "#A4DBA4",
        "#D8F09C",
        "#FFFEBE",
        "#FFD27F",
        "#FCA85F",
        "#F57244",
        "#DD484C",
        "#B51948",
    ]
    # cs = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # plot the data
    mymap = ax.contourf(
        lons, lats, field, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend="both"
    )

    # plot the contours
    contours = ax.contour(
        lons,
        lats,
        field,
        clevs,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )

    # if calc_anoms is True
    if calc_anoms:
        ax.clabel(
            contours, clevs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )

    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"mean sea level pressure {climatology_period[0]}-{climatology_period[1]} anomaly (hPa)",
            rotation=0,
            fontsize=10,
        )

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label("mean sea level pressure (hPa)", rotation=0, fontsize=10)

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    return None


# define a function for plotting the temp and wind speed for the model data
def plot_mslp_var_model(
    init_year: int,
    months: list,
    lead_year: int,
    member: int,
    title: str,
    sf_variable: str,
    psl_variable: str = "psl",
    model: str = "HadGEM3-GC31-MM",
    freq: str = "Amon",
    experiment: str = "dcppA-hindcast",
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    climatology_period: list[int] = [1960, 1990],
    grid_bounds: list[float] = [-180.0, 180.0, -90.0, 90.0],
    calc_anoms: bool = False,
    grid_file: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl_first_timestep_msl.nc",
    files_loc_path: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
    saved_clim_var_path: str = None,
    saved_clim_psl_path: str = None,
    use_bc_fields: bool = False,
) -> None:
    """
    Grabs the MSLP anomalies, as well as surface variable (e.g. temp, wind)
    data for a given period of interest and plots them.

    Args:
        init_year (int): The initial year of the period of interest.
        months (list): The months of interest.
        member (str): The member of the ensemble to plot.
        lead_year (int): The lead year of the period of interest.
        title (str): The title of the plot.
        sf_variable (str): The surface variable to plot (e.g. 'temp', 'wind').
        psl_variable (str, optional): The pressure variable to use. Defaults to 'psl'.
        model (str, optional): The model to use. Defaults to 'HadGEM3-GC31-MM'.
        freq (str, optional): The frequency of the data. Defaults to 'Amon'.
        experiment (str, optional): The experiment to use. Defaults to 'dcppA-hindcast'.
        lat_bounds (list, optional): The latitude bounds for the plot. Defaults to [30, 80].
        lon_bounds (list, optional): The longitude bounds for the plot. Defaults to [-90, 30].
        climatology_period (list[int], optional): The period to use for the climatology. Defaults to [1960, 1990].
        grid_bounds (list[float], optional): The bounds for the grid. Defaults to [-180.0, 180.0, -90.0, 90.0].
        calc_anoms (bool, optional): Whether to calculate anomalies. Defaults to False.
        grid_file (str, optional): The path to the grid file. Defaults to '/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl_first_timestep_msl.nc'.
        files_loc_path (str, optional): The path to the file locations. Defaults to '/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv'.
        saved_clim_var_path (str, optional): The path to the saved climatology for the variable. Defaults to None.
        saved_clim_psl_path (str, optional): The path to the saved climatology for the pressure variable. Defaults to None.
        use_bc_fields (bool, optional): Whether to use the bias corrected fields. Defaults to False.

    Returns:
        None
    """

    if not calc_anoms and (
        saved_clim_var_path is not None or saved_clim_psl_path is not None
    ):
        raise ValueError(
            "Cannot have saved_clim_var_path or saved_clim_psl_path if calc_anoms is False"
        )

    # if use_bc_fields is True
    if use_bc_fields:
        print("Only implemented for specific time period, not climatology")

    # # assert that both saved paths are not none
    # assert (
    #     saved_clim_var_path is not None and saved_clim_psl_path is not None
    # ), "Both saved paths must be provided"

    # exract the start year
    # start_year = start_date[:4]
    # end_year = end_date[:4]

    # # print the start and end years
    # print(f"start year: {start_year}")
    # print(f"end year: {end_year}")

    # Check that the csv file exists
    if not os.path.exists(files_loc_path):
        raise FileNotFoundError(f"Cannot find the file {files_loc_path}")

    # Load in the csv file
    csv_data = pd.read_csv(files_loc_path)

    # Print the data we seek
    print(f"model: {model}")
    print(f"experiment: {experiment}")
    print(f"variable: {sf_variable}")
    print(f"frequency: {freq}")

    # Extract the path for the given model, experiment and variable
    model_path_var = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == sf_variable)
        & (csv_data["frequency"] == freq),
        "path",
    ].values[0]

    # Extract the path for the given model, experiment and variable
    # for the psl variable
    model_path_psl = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == psl_variable)
        & (csv_data["frequency"] == freq),
        "path",
    ].values[0]

    # Assert that theb model path exists
    assert os.path.exists(
        model_path_var
    ), f"Cannot find the model path {model_path_var}"

    # assert that the other model path exists
    assert os.path.exists(
        model_path_psl
    ), f"Cannot find the model path {model_path_psl}"

    # Extract the model_path_root for var
    model_path_root_var = model_path_var.split("/")[1]
    model_path_root_psl = model_path_psl.split("/")[1]

    # Create a list
    model_paths = [model_path_var, model_path_psl]
    model_path_roots = [model_path_root_var, model_path_root_psl]
    variables = [sf_variable, psl_variable]

    print("Using non-bias corrected fields directly from /badc/ archive")
    # create an empty list of files
    files_to_extract = []

    # Loop over the model path roots
    for model_path, model_path_root, variable in zip(
        model_paths, model_path_roots, variables
    ):
        # depending on the model_path_root
        if model_path_root == "work":
            raise NotImplementedError("work path not implemented yet")
        elif model_path_root == "gws":
            # Create the path
            path = f"{model_path}/{variable}_{freq}_{model}_{experiment}_s{init_year}-r{member}i*_*_{init_year}??-*.nc"

            # print the path
            print(f"path: {path}")
            # print("correct path: /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/psl/HadGEM3-GC31-MM/psl_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s1965-r2i1_gn_196511-197603.nc")

            # glob this path
            files = glob.glob(path)

            # assert that files has length 1
            assert len(files) == 1, f"files has length {len(files)}"

            # print the loaded file
            print(f"Loaded file: {files[0]}")

            # extract the file
            file = files[0]

        elif model_path_root == "badc":
            raise NotImplementedError("home path not implemented yet")
        else:
            raise ValueError(f"Unknown model path root {model_path_root}")

        # append the file to the files to extract
        files_to_extract.append(file)

    # Load the gridspec file as a cube
    obs = iris.load_cube(grid_file)

    # Load the model data ad a cube
    cube_var = iris.load_cube(files_to_extract[0])
    cube_psl = iris.load_cube(files_to_extract[1])

    # regrid the model data to the obs grid
    regrid_cube_var = cube_var.regrid(obs, iris.analysis.Linear())
    regrid_cube_psl = cube_psl.regrid(obs, iris.analysis.Linear())

    # if the months are ONDJFM
    if months == [10, 11, 12, 1, 2, 3]:
        # set up the start and end dates
        start_date = f"{init_year + lead_year}-10-01"
        end_date = f"{init_year + lead_year + 1}-03-30"
    else:
        raise ValueError("Only implemented for ONDJFM")

    # convert the YYYY-MM-DD to cftime objects
    start_date_cf = cftime.datetime.strptime(start_date, "%Y-%m-%d", calendar="360_day")
    end_date_cf = cftime.datetime.strptime(end_date, "%Y-%m-%d", calendar="360_day")

    # Slice between the start date and end date
    regrid_cube_var = regrid_cube_var.extract(
        iris.Constraint(time=lambda cell: start_date_cf <= cell.point <= end_date_cf)
    )
    regrid_cube_psl = regrid_cube_psl.extract(
        iris.Constraint(time=lambda cell: start_date_cf <= cell.point <= end_date_cf)
    )

    # take the mean over the time dimension
    regrid_cube_var = regrid_cube_var.collapsed("time", iris.analysis.MEAN)
    regrid_cube_psl = regrid_cube_psl.collapsed("time", iris.analysis.MEAN)

    # subset to the region of interest
    regrid_cube_var = regrid_cube_var.intersection(
        latitude=(lat_bounds[0], lat_bounds[1]),
        longitude=(lon_bounds[0], lon_bounds[1]),
    )
    regrid_cube_psl = regrid_cube_psl.intersection(
        latitude=(lat_bounds[0], lat_bounds[1]),
        longitude=(lon_bounds[0], lon_bounds[1]),
    )

    # # set up a list of regrid cubes
    # regrid_cubes = [regrid_cube_var, regrid_cube_psl]

    # if use the bias corrected fields
    if use_bc_fields:
        print(
            "Quantifying the anomalies as the difference between the bc fields for a given period and non-bc climatology"
        )

        # Set ip the directory in which the files are stored
        bc_files_dir = "/work/scratch-nopw2/benhutch/test_nc/"

        # set up the lead
        lead = 1
        init = "1960-2018"

        # Create the path
        # sfcWind_bias_correction_HadGEM3-GC31-MM_lead1_month12_init1960-2018.nc
        bc_path = os.path.join(bc_files_dir, f"{sf_variable}_bias_correction_{model}_*.nc")


    if calc_anoms and saved_clim_var_path is None and saved_clim_psl_path is None:
        print("Caculating the climatology for the model data")

        # initialise a cube list
        clim_cubes = []

        # Loop over the cubes
        for model_path, variable in zip(model_paths, variables):
            # initialise a cube list
            ds_list = []
            # Loop over the years in the climatology period
            for year in tqdm(range(climatology_period[0], climatology_period[1] + 1)):
                # Create the path
                path = f"{model_path}/{variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

                # glob this path
                files = glob.glob(path)

                # assert that files has length 1
                assert len(files) == 1, f"files has length {len(files)}"

                # load the observed data using xarray
                ds = xr.open_dataset(files[0])

                # Set up the yyyy-mm-dd format
                start_date_this = cftime.datetime.strptime(
                    f"{year}-{start_date[5:10]}", "%Y-%m-%d", calendar="360_day"
                )
                end_date_this = cftime.datetime.strptime(
                    f"{year + 10}-{end_date[5:10]}", "%Y-%m-%d", calendar="360_day"
                )

                # slice between the start date and end date
                ds = ds.sel(time=slice(start_date_this, end_date_this))

                # append the ds to the cube list
                ds_list.append(ds[variable])

            # Concatenate with a new time dimension using xarray
            ds_clim = xr.concat(ds_list, dim="time")

            # convert to a cube from xarray
            cube_clim = ds_clim.to_iris()

            # regrid the model data to the obs grid
            regrid_cube_clim = cube_clim.regrid(obs, iris.analysis.Linear())

            # subset to the region of interest
            regrid_cube_clim = regrid_cube_clim.intersection(
                latitude=(lat_bounds[0], lat_bounds[1]),
                longitude=(lon_bounds[0], lon_bounds[1]),
            )

            # calculate the time mean of this
            regrid_cube_clim = regrid_cube_clim.collapsed("time", iris.analysis.MEAN)

            # append the regrid_cube_clim to the clim_cubes
            clim_cubes.append(regrid_cube_clim)
    elif (
        calc_anoms
        and saved_clim_var_path is not None
        and saved_clim_psl_path is not None
    ):
        print("loading saved climatology")

        # load the saved climatology .npy file
        climatology_var_array = np.load(saved_clim_var_path)
        climatology_psl_array = np.load(saved_clim_psl_path)

    # extract the lats an dlons
    lats = regrid_cube_var.coord("latitude").points
    lons = regrid_cube_var.coord("longitude").points

    if calc_anoms and saved_clim_var_path is None and saved_clim_psl_path is None:
        # Calculate the anomalies
        field_var = regrid_cube_var.data - clim_cubes[0].data  # either temp or wind
        field_psl = (regrid_cube_psl.data - clim_cubes[1].data) / 100  # convert to hPa
    elif (
        calc_anoms
        and saved_clim_var_path is not None
        and saved_clim_psl_path is not None
    ):
        # Calculate the anomalies
        field_var = regrid_cube_var.data - climatology_var_array
        field_psl = (regrid_cube_psl.data - climatology_psl_array) / 100
    else:
        # Extract the data values
        field_var = regrid_cube_var.data
        field_psl = regrid_cube_psl.data / 100

        # if variable is temp
        if sf_variable in ["temp", "t2m", "tas"]:
            # convert to celsius
            field_var -= 273.15

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs_psl = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks_psl = clevs_psl

        # ensure that these are floats
        clevs_psl = clevs_psl.astype(float)
        ticks_psl = ticks_psl.astype(float)

        # depending on the variable
        if sf_variable in ["t2m", "tas"]:
            # -18 to +18 in 2 degree intervals
            clevs_var = np.array(
                [
                    -5.0,
                    -4.5,
                    -4.0,
                    -3.5,
                    -3.0,
                    -2.5,
                    -2.0,
                    -1.5,
                    -1.0,
                    -0.5,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                ]
            )
            ticks_var = clevs_var

            # set up tjhe cmap
            cmap = "bwr"

            # set the cbar label
            cbar_label = "temperature (°C)"
        elif sf_variable in ["u10", "v10", "sfcWind", "si10"]:
            # 0 to 20 in 2 m/s intervals
            clevs_var = np.array(
                [
                    -1.4,
                    -1.2,
                    -1.0,
                    -0.8,
                    -0.6,
                    -0.4,
                    -0.2,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                    1.2,
                    1.4,
                ]
            )
            ticks_var = clevs_var

            # set up the cmap
            cmap = "PRGn"

            # set the cbar label
            cbar_label = "10m wind speed (m/s)"
        else:
            raise ValueError(f"Unknown variable {variable}")

    else:
        # define the contour levels for the variable
        # should be 19 of them
        if sf_variable in ["t2m", "tas"]:
            # -18 to +18 in 2 degree intervals
            clevs_var = np.array(np.arange(-18, 18 + 1, 2))
            ticks_var = clevs_var

            # set up tjhe cmap
            cmap = "bwr"

            # set the cbar label
            cbar_label = "temperature (°C)"

        elif sf_variable in ["u10", "v10", "sfcWind", "si10"]:
            # 0 to 20 in 2 m/s intervals
            clevs_var = np.array(np.arange(0, 12 + 1, 1))
            ticks_var = clevs_var

            # set up the cmap
            cmap = "RdPu"

            # set the cbar label
            cbar_label = "10m wind speed (m/s)"
        else:
            raise ValueError(f"Unknown variable {variable}")

        # define the contour levels
        clevs_psl = np.array(np.arange(988, 1024 + 1, 2))
        ticks_psl = clevs_psl

        # ensure that these are ints
        clevs_psl = clevs_psl.astype(int)
        ticks_psl = ticks_psl.astype(int)

    # print the len of clevs_psl
    print(f"len of clevs_psl: {len(clevs_psl)}")
    print(f"len of clevs_var: {len(clevs_var)}")

    # print field_var and field_psl
    print(f"field_var shape: {field_var.shape}")
    print(f"field_psl shape: {field_psl.shape}")
    print(f"field_var values: {field_var}")
    print(f"field_psl values: {field_psl}")

    # print the field var min and the field var max
    print(f"field_var min: {field_var.min()}")
    print(f"field_var max: {field_var.max()}")

    if variable in ["si10", "sfcWind"] and not calc_anoms:
        # set up the extend
        extend = "max"
    else:
        extend = "both"

    # plot the data
    mymap = ax.contourf(
        lons,
        lats,
        field_var,
        clevs_var,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend=extend,
    )

    # plot the psl contours
    contours = ax.contour(
        lons,
        lats,
        field_psl,
        clevs_psl,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )

    if calc_anoms:
        ax.clabel(
            contours, clevs_psl, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs_psl, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )
    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"{cbar_label} {climatology_period[0]}-{climatology_period[1]} anomaly",
            rotation=0,
            fontsize=10,
        )

        # # add contour lines to the colorbar
        # cbar.add_lines(mymap)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label(cbar_label, rotation=0, fontsize=10)

        # # set up invisible contour lines for field_var
        # contour_var = ax.contour(
        #     lons,
        #     lats,
        #     field_var,
        #     clevs_var,
        #     colors="k",
        #     transform=ccrs.PlateCarree(),
        #     linewidth=0.2,
        #     alpha=0.5,
        # )

        # # add contour lines to the colorbar
        # cbar.add_lines(contour_var)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks_var)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    return None


# Write a function for plotting the composites for the observations
# e.g. identify the 95th %tile of events for demand, wind, or demand-net-wind
# then print how many events are in the 95th %tile
# then plot the composite psl patterns for these events
def plot_composite_obs(
    title: str,
    energy_variable: str,
    percentile: float,
    months: list[int] = [11, 12, 1, 2, 3],
    psl_variable: str = "msl",
    freq: str = "Amon",
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    energy_df_path: str = "/home/users/benhutch/unseen_multi_year/dfs/obs_df_NDJFM_wind_demand_1960-2018_dnw.csv",
    ERA5_regrid_path: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl.nc",
    climatology_period: list[int] = [1990, 2020],
    calc_anoms: bool = False,
) -> None:
    """
    Identifies the percentile threshold for demand, wind power, or demand net
    wind, and plots a psl composite of the events that exceed this threshold.

    Args:
        title (str): The title of the plot.
        energy_variable (str): The energy variable to be used for identifying the percentile threshold.
        percentile (float): The percentile to be used as the threshold.
        months (list[int], optional): The months to be used for the composite plot. Defaults to [11, 12, 1, 2, 3].
        psl_variable (str, optional): The pressure level variable to be used for the composite plot. Defaults to "msl".
        freq (str, optional): The frequency of the data. Defaults to "Amon".
        lat_bounds (list, optional): The latitude boundaries for the plot. Defaults to [30, 80].
        lon_bounds (list, optional): The longitude boundaries for the plot. Defaults to [-90, 30].
        energy_df_path (str, optional): The path to the energy dataframe. Defaults to "/home/users/benhutch/unseen_multi_year/dfs/obs_df_NDJFM_wind_demand_1960-2018_dnw.csv".
        ERA5_regrid_path (str, optional): The path to the ERA5 regridded data. Defaults to "/gws/nopw/j04/cp4cds1_vol1/data/era5/pressure_levels/6hr/native/psl/psl_era5_6hr_native_19790101-20191231.nc".
        climatology_period (list[int], optional): The period to be used for the climatology. Defaults to [1990, 2020].
        calc_anoms (bool, optional): Whether to calculate anomalies. Defaults to False.

    Returns:
        None
    """

    # set up the dictionary for the energy variables
    energy_dict = {
        "demand": "United_Kingdom_demand",
        "wind": "total_gen",
        "demand_net_wind": "demand_net_wind",
    }

    # assert that energy_variable is in ["demand", "wind", "demand_net_wind"]
    assert energy_variable in [
        "demand",
        "wind",
        "demand_net_wind",
    ], f"Unknown energy variable {energy_variable}, must be in ['demand', 'wind', 'demand_net_wind']"

    # Assert that the energy df path exists
    assert os.path.exists(
        energy_df_path
    ), f"Cannot find the energy df path {energy_df_path}"

    # Load the energy df
    energy_df = pd.read_csv(energy_df_path)

    # if "Unnamed: 0" in energy_df.columns:
    if "Unnamed: 0" in energy_df.columns:
        # Convert to datetime
        energy_df["Unnamed: 0"] = pd.to_datetime(energy_df["Unnamed: 0"], format="%Y")

        # Set as the index
        energy_df.set_index("Unnamed: 0", inplace=True)

        # strptime to just be the year
        energy_df.index = energy_df.index.strftime("%Y")

        # remove the name of the index
        energy_df.index.name = None

    # if the correct column for the specified energy variable is in the df
    if energy_dict[energy_variable] in energy_df.columns:
        # Subset the df to this column
        energy_series = energy_df[energy_dict[energy_variable]]
    else:
        raise ValueError(
            f"Cannot find the column {energy_dict[energy_variable]} in the energy df"
        )

    # Calculate the percentile threshold
    threshold = energy_series.quantile(percentile)

    if energy_variable != "wind":
        # identify the number of events above the threshold
        num_events = len(energy_series[energy_series > threshold])
        print(f"Number of events above the {percentile} percentile: {num_events}")

        # Find the years of the events above the threshold
        years = energy_series[energy_series > threshold].index
        print(f"Years of the events above the {percentile} percentile: {years}")
    else:
        # identify the number of events below the threshold
        num_events = len(energy_series[energy_series < threshold])
        print(f"Number of events below the {percentile} percentile: {num_events}")

        # Find the years of the events below the threshold
        years = energy_series[energy_series < threshold].index
        print(f"Years of the events below the {percentile} percentile: {years}")

    # Load the ERA5 regridded data
    ds = xr.open_mfdataset(
        ERA5_regrid_path,
        chunks={"time": 10},
        combine="by_coords",
        parallel=False,
        engine="netcdf4",
        coords="minimal",
    )

    # If expver is present in the observations
    if "expver" in ds.coords:
        # Combine the first two expver variables
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # if the variable is not in the ds
    if psl_variable not in ds:
        raise ValueError(f"Cannot find the variable {psl_variable} in the ds")

    # if calc anoms is true
    if calc_anoms:
        print("Calculating the climatology for the observations")

        # assert that the months are integers
        assert all(
            isinstance(month, int) for month in months
        ), "Months must be integers"

        # subset the data to the region
        ds_clim = ds.sel(
            lat=slice(lat_bounds[0], lat_bounds[1]),
            lon=slice(lon_bounds[0], lon_bounds[1]),
        )

        # subset the data to the months
        ds_clim = ds_clim.sel(time=ds_clim["time.month"].isin(months))

        # Select the years
        ds_clim = ds_clim.sel(
            time=slice(
                f"{climatology_period[0]}-01-01", f"{climatology_period[1]}-12-31"
            )
        )

        # Calculate the climatology
        ds_clim = ds_clim[psl_variable].mean(dim="time")

    # ensure that years is a list of ints
    years = [int(year) for year in years]

    # set up an empty list
    ds_list = []

    # loop through the years
    for year in tqdm(years):
        # subset the data to the region
        ds_year = ds.sel(
            lat=slice(lat_bounds[0], lat_bounds[1]),
            lon=slice(lon_bounds[0], lon_bounds[1]),
        )

        # subset the data to the months
        ds_year = ds_year.sel(time=ds_year["time.month"].isin(months))

        # Select the years
        ds_year = ds_year.sel(
            time=slice(f"{year}-{months[0]}-01", f"{year + 1}-{months[-1]}-31")
        )

        # # print the ds_year
        # print(f"ds_year: {ds_year}")

        # append the ds_year to the ds_list
        ds_list.append(ds_year[psl_variable])

    # Concatenate with a new time dimension using xarray
    ds_composite = xr.concat(ds_list, dim="time")

    # take the time mean
    ds_composite = ds_composite.mean(dim="time")

    # Etract the lat and lon points
    lats = ds_composite["lat"].values
    lons = ds_composite["lon"].values

    # if calc_anoms is True
    if calc_anoms:
        # Calculate the anomalies
        field = (ds_composite.values - ds_clim.values) / 100  # convert to hPa
    else:
        # Extract the data values
        field = ds_composite.values / 100  # convert to hPa

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks = clevs

        # ensure that these are floats
        clevs = clevs.astype(float)
        ticks = ticks.astype(float)
    else:
        # define the contour levels
        clevs = np.array(np.arange(988, 1024 + 1, 2))
        ticks = clevs

        # ensure that these are ints
        clevs = clevs.astype(int)
        ticks = ticks.astype(int)

    # # print the shape of the inputs
    # print(f"lons shape: {lons.shape}")
    # print(f"lats shape: {lats.shape}")
    # print(f"field shape: {field.shape}")
    # print(f"clevs shape: {clevs.shape}")

    # # print the field values
    # print(f"field values: {field}")

    # Define the custom diverging colormap
    # cs = ["purple", "blue", "lightblue", "lightgreen", "lightyellow", "orange", "red", "darkred"]
    # cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # custom colormap
    cs = [
        "#4D65AD",
        "#3E97B7",
        "#6BC4A6",
        "#A4DBA4",
        "#D8F09C",
        "#FFFEBE",
        "#FFD27F",
        "#FCA85F",
        "#F57244",
        "#DD484C",
        "#B51948",
    ]
    # cs = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # plot the data
    mymap = ax.contourf(
        lons, lats, field, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend="both"
    )
    contours = ax.contour(
        lons,
        lats,
        field,
        clevs,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )
    if calc_anoms:
        ax.clabel(
            contours, clevs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )

    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    # include a textbox in the top left
    ax.text(
        0.02,
        0.95,
        f"N = {num_events}",
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="black",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"mean sea level pressure {climatology_period[0]}-{climatology_period[1]} anomaly (hPa)",
            rotation=0,
            fontsize=10,
        )

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label("mean sea level pressure (hPa)", rotation=0, fontsize=10)

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    return None


# define a function to plot the composites for the observations
# with the MSLP contours and the surface variables
def plot_composite_var_obs(
    title: str,
    energy_variable: str,
    percentile: float,
    months: list[int] = [11, 12, 1, 2, 3],
    sf_variable: str = "t2m",
    psl_variable: str = "msl",
    freq: str = "Amon",
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    energy_df_path: str = "/home/users/benhutch/unseen_multi_year/dfs/obs_df_NDJFM_wind_demand_1960-2018_dnw.csv",
    ERA5_regrid_path: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl.nc",
    climatology_period: list[int] = [1990, 2020],
    calc_anoms: bool = False,
) -> None:
    """
    Identifies the percentile threshold for demand, wind power, or demand net
    wind, and plots a psl composite of the events that exceed this threshold.

    Args:
        title (str): The title of the plot.
        energy_variable (str): The energy variable to be used for identifying the percentile threshold.
        percentile (float): The percentile to be used as the threshold.
        months (list[int], optional): The months to be used for the composite plot. Defaults to [11, 12, 1, 2, 3].
        sf_variable (str, optional): The surface variable to be used for the composite plot. Defaults to "t2m".
        psl_variable (str, optional): The pressure level variable to be used for the composite plot. Defaults to "msl".
        freq (str, optional): The frequency of the data. Defaults to "Amon".
        lat_bounds (list, optional): The latitude boundaries for the plot. Defaults to [30, 80].
        lon_bounds (list, optional): The longitude boundaries for the plot. Defaults to [-90, 30].
        energy_df_path (str, optional): The path to the energy dataframe. Defaults to "/home/users/benhutch/unseen_multi_year/dfs/obs_df_NDJFM_wind_demand_1960-2018_dnw.csv".
        ERA5_regrid_path (str, optional): The path to the ERA5 regridded data. Defaults to "/gws/nopw/j04/cp4cds1_vol1/data/era5/pressure_levels/6hr/native/psl/psl_era5_6hr_native_19790101-20191231.nc".
        climatology_period (list[int], optional): The period to be used for the climatology. Defaults to [1990, 2020].
        calc_anoms (bool, optional): Whether to calculate anomalies. Defaults to False.

    Returns:
        None
    """

    # set up the dictionary for the energy variables
    energy_dict = {
        "demand": "United_Kingdom_demand",
        "wind": "total_gen",
        "demand_net_wind": "demand_net_wind",
    }

    # assert that energy_variable is in ["demand", "wind", "demand_net_wind"]
    assert energy_variable in [
        "demand",
        "wind",
        "demand_net_wind",
    ], f"Unknown energy variable {energy_variable}, must be in ['demand', 'wind', 'demand_net_wind']"

    # Assert that the energy df path exists
    assert os.path.exists(
        energy_df_path
    ), f"Cannot find the energy df path {energy_df_path}"

    # Load the energy df
    energy_df = pd.read_csv(energy_df_path)

    # if "Unnamed: 0" in energy_df.columns:
    if "Unnamed: 0" in energy_df.columns:
        # Convert to datetime
        energy_df["Unnamed: 0"] = pd.to_datetime(energy_df["Unnamed: 0"], format="%Y")

        # Set as the index
        energy_df.set_index("Unnamed: 0", inplace=True)

        # strptime to just be the year
        energy_df.index = energy_df.index.strftime("%Y")

        # remove the name of the index
        energy_df.index.name = None
    else:
        raise NotImplementedError("Unnamed: 0 not in the columns")

    # if the correct column for the specified energy variable is in the df
    if energy_dict[energy_variable] in energy_df.columns:
        # Subset the df to this column
        energy_series = energy_df[energy_dict[energy_variable]]
    else:
        raise ValueError(
            f"Cannot find the column {energy_dict[energy_variable]} in the energy df"
        )

    # Calculate the percentile threshold
    threshold = energy_series.quantile(percentile)

    if energy_variable != "wind":
        # identify the number of events above the threshold
        num_events = len(energy_series[energy_series > threshold])
        print(f"Number of events above the {percentile} percentile: {num_events}")

        # Find the years of the events above the threshold
        years = energy_series[energy_series > threshold].index
        print(f"Years of the events above the {percentile} percentile: {years}")
    else:
        # identify the number of events below the threshold
        num_events = len(energy_series[energy_series < threshold])
        print(f"Number of events below the {percentile} percentile: {num_events}")

        # Find the years of the events below the threshold
        years = energy_series[energy_series < threshold].index
        print(f"Years of the events below the {percentile} percentile: {years}")

    # Load the ERA5 regridded data
    ds = xr.open_mfdataset(
        ERA5_regrid_path,
        chunks={"time": 10},
        combine="by_coords",
        parallel=False,
        engine="netcdf4",
        coords="minimal",
    )

    # If expver is present in the observations
    if "expver" in ds.coords:
        # Combine the first two expver variables
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # if the variable is not in the ds
    if psl_variable not in ds:
        raise ValueError(f"Cannot find the variable {psl_variable} in the ds")

    if calc_anoms:
        # assert that the months are integers
        assert all(
            isinstance(month, int) for month in months
        ), "Months must be integers"

        # subset the data to the region
        ds_clim = ds.sel(
            lat=slice(lat_bounds[0], lat_bounds[1]),
            lon=slice(lon_bounds[0], lon_bounds[1]),
        )

        # subset the data to the months
        ds_clim = ds_clim.sel(time=ds_clim["time.month"].isin(months))

        # Select the years
        ds_clim = ds_clim.sel(
            time=slice(
                f"{climatology_period[0]}-01-01", f"{climatology_period[1]}-12-31"
            )
        )

        # Calculate the climatology
        ds_clim_psl = ds_clim[psl_variable].mean(dim="time")

        # Calculate the variable climatology
        ds_clim_var = ds_clim[sf_variable].mean(dim="time")

    # ensure that years is a list of ints
    years = [int(year) for year in years]

    # set up an empty list
    ds_list_var = []
    ds_list_psl = []

    # Loop over the years
    for year in tqdm(years):
        # subset the data to the region
        ds_year = ds.sel(
            lat=slice(lat_bounds[0], lat_bounds[1]),
            lon=slice(lon_bounds[0], lon_bounds[1]),
        )

        # subset the data to the months
        ds_year = ds_year.sel(time=ds_year["time.month"].isin(months))

        # Select the years
        ds_year = ds_year.sel(
            time=slice(f"{year}-{months[0]}-01", f"{year + 1}-{months[-1]}-31")
        )

        # append the ds_year to the ds_list
        ds_list_var.append(ds_year[sf_variable])
        ds_list_psl.append(ds_year[psl_variable])

    # Concatenate with a new time dimension using xarray
    ds_composite_var = xr.concat(ds_list_var, dim="time")
    ds_composite_psl = xr.concat(ds_list_psl, dim="time")

    # Take the time mean
    ds_composite_var = ds_composite_var.mean(dim="time")
    ds_composite_psl = ds_composite_psl.mean(dim="time")

    # Etract the lat and lon points
    lats = ds_composite_var["lat"].values
    lons = ds_composite_var["lon"].values

    # if calc_anoms is True
    if calc_anoms:
        # Calculate the anomalies
        field_var = ds_composite_var.values - ds_clim_var.values
        field_psl = (ds_composite_psl.values - ds_clim_psl.values) / 100
    else:
        # Extract the data values
        field_var = ds_composite_var.values
        field_psl = ds_composite_psl.values / 100

        # if variable is temp
        if sf_variable in ["temp", "t2m", "tas"]:
            # convert to celsius
            field -= 273.15

        # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs_psl = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks_psl = clevs_psl

        # ensure that these are floats
        clevs_psl = clevs_psl.astype(float)
        ticks_psl = ticks_psl.astype(float)

        # depending on the variable
        if sf_variable in ["t2m", "tas"]:
            # -18 to +18 in 2 degree intervals
            clevs_var = np.array(
                [
                    -5.0,
                    -4.5,
                    -4.0,
                    -3.5,
                    -3.0,
                    -2.5,
                    -2.0,
                    -1.5,
                    -1.0,
                    -0.5,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                ]
            )
            ticks_var = clevs_var

            # set up tjhe cmap
            cmap = "bwr"

            # set the cbar label
            cbar_label = "temperature (°C)"
        elif sf_variable in ["u10", "v10", "sfcWind", "si10"]:
            # 0 to 20 in 2 m/s intervals
            clevs_var = np.array(
                [
                    -1.4,
                    -1.2,
                    -1.0,
                    -0.8,
                    -0.6,
                    -0.4,
                    -0.2,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                    1.2,
                    1.4,
                ]
            )
            ticks_var = clevs_var

            # set up the cmap
            cmap = "PRGn_r"

            # set the cbar label
            cbar_label = "10m wind speed (m/s)"
        else:
            raise ValueError(f"Unknown variable {sf_variable}")

    else:
        # define the contour levels for the variable
        # should be 19 of them
        if sf_variable in ["t2m", "tas"]:
            # -18 to +18 in 2 degree intervals
            clevs_var = np.array(np.arange(-18, 18 + 1, 2))
            ticks_var = clevs_var

            # set up tjhe cmap
            cmap = "bwr"

            # set the cbar label
            cbar_label = "temperature (°C)"

        elif sf_variable in ["u10", "v10", "sfcWind", "si10"]:
            # 0 to 20 in 2 m/s intervals
            clevs_var = np.array(np.arange(0, 12 + 1, 1))
            ticks_var = clevs_var

            # set up the cmap
            cmap = "RdPu"

            # set the cbar label
            cbar_label = "10m wind speed (m/s)"
        else:
            raise ValueError(f"Unknown variable {sf_variable}")

        # define the contour levels
        clevs_psl = np.array(np.arange(988, 1024 + 1, 2))
        ticks_psl = clevs_psl

        # ensure that these are ints
        clevs_psl = clevs_psl.astype(int)
        ticks_psl = ticks_psl.astype(int)

    # print the len of clevs_psl
    print(f"len of clevs_psl: {len(clevs_psl)}")
    print(f"len of clevs_var: {len(clevs_var)}")

    # print field_var and field_psl
    print(f"field_var shape: {field_var.shape}")
    print(f"field_psl shape: {field_psl.shape}")
    # print(f"field_var values: {field_var}")
    # print(f"field_psl values: {field_psl}")

    # # print the field var min and the field var max
    # print(f"field_var min: {field_var.min()}")
    # print(f"field_var max: {field_var.max()}")

    if sf_variable in ["si10", "sfcWind"] and not calc_anoms:
        # set up the extend
        extend = "max"
    else:
        extend = "both"

    # plot the data
    mymap = ax.contourf(
        lons,
        lats,
        field_var,
        clevs_var,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend=extend,
    )

    # plot the psl contours
    contours = ax.contour(
        lons,
        lats,
        field_psl,
        clevs_psl,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )

    if calc_anoms:
        ax.clabel(
            contours, clevs_psl, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs_psl, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )
    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"{cbar_label} {climatology_period[0]}-{climatology_period[1]} anomaly",
            rotation=0,
            fontsize=10,
        )

        # # add contour lines to the colorbar
        # cbar.add_lines(mymap)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label(cbar_label, rotation=0, fontsize=10)

        # # set up invisible contour lines for field_var
        # contour_var = ax.contour(
        #     lons,
        #     lats,
        #     field_var,
        #     clevs_var,
        #     colors="k",
        #     transform=ccrs.PlateCarree(),
        #     linewidth=0.2,
        #     alpha=0.5,
        # )

        # # add contour lines to the colorbar
        # cbar.add_lines(contour_var)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks_var)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    return None


# Write a function which will plot the 95% composite for the model data
# just plotting the absolute and anomaly MSLP fields in this case
def plot_composite_model(
    title: str,
    energy_variable: str,
    percentile: float,
    months: list[int] = [11, 12, 1, 2, 3],
    model: str = "HadGEM3-GC31-MM",
    psl_variable: str = "psl",
    freq: str = "Amon",
    experiment: str = "dcppA-hindcast",
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    climatology_period: list[int] = [1988, 2018],
    grid_bounds: list[float] = [-180.0, 180.0, -90.0, 90.0],
    calc_anoms: bool = False,
    energy_df_path: str = "/home/users/benhutch/unseen_multi_year/dfs/model_df_NDJFM_wind_demand_1960-2018_dnw.csv",
    grid_file: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl_first_timestep_msl.nc",
    files_loc_path: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
) -> None:
    """
    Identifies the percentile threshold for demand, wind power, or demand net
    wind, and plots a psl composite of the events that exceed this threshold.

    Args:
        title (str): The title of the plot.
        energy_variable (str): The energy variable to be used for identifying the percentile threshold.
        percentile (float): The percentile to be used as the threshold.
        months (list[int], optional): The months to be used for the composite plot. Defaults to [11, 12, 1, 2, 3].
        model (str, optional): The model to be used for the composite plot. Defaults to "HadGEM3-GC31-MM".
        psl_variable (str, optional): The pressure level variable to be used for the composite plot. Defaults to "psl".
        freq (str, optional): The frequency of the data. Defaults to "Amon".
        experiment (str, optional): The experiment to be used for the composite plot. Defaults to "dcppA-hindcast".
        lat_bounds (list, optional): The latitude boundaries for the plot. Defaults to [30, 80].
        lon_bounds (list, optional): The longitude boundaries for the plot. Defaults to [-90, 30].
        climatology_period (list[int], optional): The period to be used for the climatology. Defaults to [1990, 2020].
        grid_bounds (list[float], optional): The grid boundaries for the plot. Defaults to [-180.0, 180.0, -90.0, 90.0].
        calc_anoms (bool, optional): Whether to calculate anomalies. Defaults to False.
        energy_df_path (str, optional): The path to the energy dataframe. Defaults to "/home/users/benhutch/unseen_multi_year/dfs/obs_df_NDJFM_wind_demand_1960-2018_dnw.csv".
        grid_file (str, optional): The path to the grid file. Defaults to "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl_first_timestep_msl.nc".
        files_loc_path (str, optional): The path to the files location file. Defaults to "/home/users/benhutch/unseen_multi_year/paths/paths_20240117

    Returns:
        None
    """

    # set up the dictionary for the energy variables
    energy_dict = {
        "demand": "United_Kingdom_demand",
        "wind": "total_gen",
        "demand_net_wind": "demand_net_wind",
    }

    # assert that energy_variable is in ["demand", "wind", "demand_net_wind"]
    assert energy_variable in [
        "demand",
        "wind",
        "demand_net_wind",
    ], f"Unknown energy variable {energy_variable}, must be in ['demand', 'wind', 'demand_net_wind']"

    # Assert that the energy df path exists
    assert os.path.exists(
        energy_df_path
    ), f"Cannot find the energy df path {energy_df_path}"

    # Load the energy df
    energy_df = pd.read_csv(energy_df_path)

    # if "Unnamed: 0" in energy_df.columns:
    if "Unnamed: 0" in energy_df.columns:
        # Convert to datetime
        energy_df["Unnamed: 0"] = pd.to_datetime(energy_df["Unnamed: 0"], format="%Y")

        # Set as the index
        energy_df.set_index("Unnamed: 0", inplace=True)

        # strptime to just be the year
        energy_df.index = energy_df.index.strftime("%Y")

        # remove the name of the index
        energy_df.index.name = None
    else:
        raise NotImplementedError("Unnamed: 0 not in the columns")

    # format as an int member
    energy_df["member"] = energy_df["member"].astype(int)

    # extract the unique members
    unique_members = energy_df["member"].unique()

    # set this as an index
    energy_df.set_index("member", append=True, inplace=True)

    # print the head of the energy df
    print(f"energy_df head: {energy_df.head()}")

    # if the correct column for the specified energy variable is in the df
    if energy_dict[energy_variable] in energy_df.columns:
        # Subset the df to this column
        energy_series = energy_df[energy_dict[energy_variable]]
    else:
        raise ValueError(
            f"Cannot find the column {energy_dict[energy_variable]} in the energy df"
        )

    # Calculate the percentile threshold
    threshold = energy_series.quantile(percentile)

    if energy_variable != "wind":
        # identify the number of events above the threshold
        num_events = len(energy_series[energy_series > threshold])
        print(f"Number of events above the {percentile} percentile: {num_events}")

        # Find the years of the events above the threshold
        years_members = energy_series[energy_series > threshold].index
        print(f"Years of the events above the {percentile} percentile: {years_members}")
    else:
        # identify the number of events below the threshold
        num_events = len(energy_series[energy_series < threshold])
        print(f"Number of events below the {percentile} percentile: {num_events}")

        # Find the years of the events below the threshold
        years_members = energy_series[energy_series < threshold].index
        print(f"Years of the events below the {percentile} percentile: {years_members}")

    # Check that the csv file exits
    assert os.path.exists(
        files_loc_path
    ), f"Cannot find the files location path {files_loc_path}"

    # Load the files location
    files_loc = pd.read_csv(files_loc_path)

    # print the data we seek
    print(f"model: {model}")
    print(f"experiment: {experiment}")
    print(f"freq: {freq}")
    print(f"psl_variable: {psl_variable}")

    # Extract the path for the given model, experiment, freq, and variable
    model_path = files_loc[
        (files_loc["model"] == model)
        & (files_loc["experiment"] == experiment)
        & (files_loc["frequency"] == freq)
        & (files_loc["variable"] == psl_variable)
    ]["path"].values[0]

    # assert that the model path exists
    assert os.path.exists(model_path), f"Cannot find the model path {model_path}"

    # extract the model path root
    model_path_root = model_path.split("/")[1]

    # pritn the mdoel path root
    print(f"model_path_root: {model_path_root}")

    # set up an empty list of files
    files_list = []

    # loop over the multi index
    for year, member in years_members:
        # depending on the model_path_root
        if model_path_root == "work":
            raise NotImplementedError("work path not implemented yet")
        elif model_path_root == "gws":
            # Create the path
            path = f"{model_path}/{psl_variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

            # # print the path
            # print(f"path: {path}")
            # # print("correct path: /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/psl/HadGEM3-GC31-MM/psl_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s1965-r2i1_gn_196511-197603.nc")

            # glob this path
            files = glob.glob(path)

            # assert that files has length 1
            assert len(files) == 1, f"files has length {len(files)}"

            # # print the loaded file
            # print(f"Loaded file: {files[0]}")

            # extract the file
            file = files[0]
        elif model_path_root == "badc":
            raise NotImplementedError("home path not implemented yet")
        else:
            raise ValueError(f"Unknown model path root {model_path_root}")

        # append the file to the files_list
        files_list.append(file)

    # # print the files list
    # print(f"files_list: {files_list}")

    ds_comp_list = []

    # loop over the files list
    for file, (year, member) in tqdm(zip(files_list, years_members)):
        # Load the model data
        ds = xr.open_dataset(file)

        # format the year as an int
        year = int(year)

        # if the variable is not in the ds
        if psl_variable not in ds:
            raise ValueError(f"Cannot find the variable {psl_variable} in the ds")

        # Set up the times to extract
        start_date_this = cftime.datetime.strptime(
            f"{year}-{months[0]}-01", "%Y-%m-%d", calendar="360_day"
        )
        end_date_this = cftime.datetime.strptime(
            f"{year + 1}-{months[-1]}-30", "%Y-%m-%d", calendar="360_day"
        )

        # slice between the start and end dates
        ds = ds.sel(time=slice(start_date_this, end_date_this))

        # append the ds to the list
        ds_comp_list.append(ds[psl_variable])

    # Concatenate with a new time dimension using xarray
    ds_composite = xr.concat(ds_comp_list, dim="time")

    # Convert to a cube
    cube = ds_composite.to_iris()

    # load the obs cube
    cube_obs = iris.load_cube(grid_file)

    # regrid the model data to the obs grid
    cube_regrid = cube.regrid(cube_obs, iris.analysis.Linear())

    # Subset to the region of interest
    cube_regrid = cube_regrid.intersection(
        latitude=(lat_bounds[0], lat_bounds[1]),
        longitude=(lon_bounds[0], lon_bounds[1]),
    )

    # Calculate the time mean of this
    cube_regrid = cube_regrid.collapsed("time", iris.analysis.MEAN)

    # print cube regrid
    print(f"cube_regrid: {cube_regrid}")

    if calc_anoms:
        # assert that all of the months are integers
        assert all(
            isinstance(month, int) for month in months
        ), "Months must be integers"

        init_year_list = []

        # loop over the years
        for year in tqdm(range(climatology_period[0], climatology_period[1] + 1)):
            member_list = []
            # loop over the unique members
            for member in unique_members:
                # create the path
                path = f"{model_path}/{psl_variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

                # glob this path
                files = glob.glob(path)

                # assert that files has length 1
                assert (
                    len(files) == 1
                ), f"files has length {len(files)} for year {year} and member {member} and path {path}"

                # open all of the files
                member_ds = xr.open_mfdataset(
                    files[0],
                    combine="nested",
                    concat_dim="time",
                    preprocess=lambda ds: preprocess(
                        ds=ds,
                        year=year,
                        variable=psl_variable,
                        months=months,
                    ),
                    parallel=False,
                    engine="netcdf4",
                    coords="minimal",  # expecting identical coords
                    data_vars="minimal",  # expecting identical vars
                    compat="override",  # speed up
                ).squeeze()

                # id init year == climatology_period[0]
                # and member == unique_members[0]
                if year == climatology_period[0] and member == unique_members[0]:
                    # set the new integer time
                    member_ds = set_integer_time_axis(
                        xro=member_ds,
                        frequency=freq,
                        first_month_attr=True,
                    )
                else:
                    # set the new integer time
                    member_ds = set_integer_time_axis(
                        xro=member_ds,
                        frequency=freq,
                    )

                # append the member_ds to the member_list
                member_list.append(member_ds)
            # Concatenate with a new member dimension using xarray
            member_ds = xr.concat(member_list, dim="member")
            # append the member_ds to the init_year_list
            init_year_list.append(member_ds)
        # Concatenate the init_year list along the init dimension
        # and rename as lead time
        ds = xr.concat(init_year_list, "init").rename({"time": "lead"})

        # set up the members
        ds["member"] = unique_members
        ds["init"] = np.arange(climatology_period[0], climatology_period[1] + 1)

        # print ds
        print(f"ds: {ds}")

        # calculate the climatology
        # extract the data for the variable
        ds_psl = ds[psl_variable]

        # take the mean over lead dimension
        ds_clim = ds_psl.mean(dim="lead")

        # take the mean over member dimension
        ds_clim = ds_clim.mean(dim="member")

        # take the mean over init dimension
        ds_clim = ds_clim.mean(dim="init")

        # convert to a cube
        cube_clim = ds_clim.to_iris()

        # regrid the model data to the obs grid
        cube_clim_regrid = cube_clim.regrid(cube_obs, iris.analysis.Linear())

        # subset to the region of interest
        cube_clim_regrid = cube_clim_regrid.intersection(
            latitude=(lat_bounds[0], lat_bounds[1]),
            longitude=(lon_bounds[0], lon_bounds[1]),
        )

        # print the cube_clim_regrid
        print(f"cube_clim_regrid: {cube_clim_regrid}")

    # extract the lats and lons
    lats = cube_regrid.coord("latitude").points
    lons = cube_regrid.coord("longitude").points

    # if calc_anoms is True
    if calc_anoms:
        field = (cube_regrid.data - cube_clim_regrid.data) / 100  # convert to hPa
    else:
        field = cube_regrid.data / 100

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks = clevs

        # ensure that these are floats
        clevs = clevs.astype(float)
        ticks = ticks.astype(float)
    else:
        # define the contour levels
        clevs = np.array(np.arange(988, 1024 + 1, 2))
        ticks = clevs

        # ensure that these are ints
        clevs = clevs.astype(int)
        ticks = ticks.astype(int)

    # # print the shape of the inputs
    # print(f"lons shape: {lons.shape}")
    # print(f"lats shape: {lats.shape}")
    # print(f"field shape: {field.shape}")
    # print(f"clevs shape: {clevs.shape}")

    # # print the field values
    # print(f"field values: {field}")

    # Define the custom diverging colormap
    # cs = ["purple", "blue", "lightblue", "lightgreen", "lightyellow", "orange", "red", "darkred"]
    # cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # custom colormap
    cs = [
        "#4D65AD",
        "#3E97B7",
        "#6BC4A6",
        "#A4DBA4",
        "#D8F09C",
        "#FFFEBE",
        "#FFD27F",
        "#FCA85F",
        "#F57244",
        "#DD484C",
        "#B51948",
    ]
    # cs = ["#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8", "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", cs)

    # plot the data
    mymap = ax.contourf(
        lons, lats, field, clevs, transform=ccrs.PlateCarree(), cmap=cmap, extend="both"
    )
    contours = ax.contour(
        lons,
        lats,
        field,
        clevs,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )
    if calc_anoms:
        ax.clabel(
            contours, clevs, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )

    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    # include a textbox in the top left
    ax.text(
        0.02,
        0.95,
        f"N = {num_events}",
        verticalalignment="top",
        horizontalalignment="left",
        transform=ax.transAxes,
        color="black",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.5"),
    )

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"mean sea level pressure {climatology_period[0]}-{climatology_period[1]} anomaly (hPa)",
            rotation=0,
            fontsize=10,
        )

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label("mean sea level pressure (hPa)", rotation=0, fontsize=10)

        # add contour lines to the colorbar
        cbar.add_lines(contours)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    return None


# plot composites with the surface variables
def plot_composite_var_model(
    title: str,
    energy_variable: str,
    percentile: float,
    months: list[int] = [11, 12, 1, 2, 3],
    model: str = "HadGEM3-GC31-MM",
    sf_variable: str = "t2m",
    psl_variable: str = "psl",
    freq: str = "Amon",
    experiment: str = "dcppA-hindcast",
    lat_bounds: list = [30, 80],
    lon_bounds: list = [-90, 30],
    climatology_period: list[int] = [1988, 2018],
    grid_bounds: list[float] = [-180.0, 180.0, -90.0, 90.0],
    calc_anoms: bool = False,
    energy_df_path: str = "/home/users/benhutch/unseen_multi_year/dfs/model_df_NDJFM_wind_demand_1960-2018_dnw.csv",
    grid_file: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl_first_timestep_msl.nc",
    files_loc_path: str = "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv",
) -> None:
    """
    Identifies the events above/below the percentile threshold for demand, wind power, or demand net
    wind, and plots a psl composite of the events that exceed this threshold on top of the surface variable contours.

    Args:
        title (str): The title of the plot.
        energy_variable (str): The energy variable to be used for identifying the percentile threshold.
        percentile (float): The percentile to be used as the threshold.
        months (list[int], optional): The months to be used for the composite plot. Defaults to [11, 12, 1, 2, 3].
        model (str, optional): The model to be used for the composite plot. Defaults to "HadGEM3-GC31-MM".
        sf_variable (str, optional): The surface variable to be used for the composite plot. Defaults to "t2m".
        psl_variable (str, optional): The pressure level variable to be used for the composite plot. Defaults to "psl".
        freq (str, optional): The frequency of the data. Defaults to "Amon".
        experiment (str, optional): The experiment to be used for the composite plot. Defaults to "dcppA-hindcast".
        lat_bounds (list, optional): The latitude boundaries for the plot. Defaults to [30, 80].
        lon_bounds (list, optional): The longitude boundaries for the plot. Defaults to [-90, 30].
        climatology_period (list[int], optional): The period to be used for the climatology. Defaults to [1990, 2020].
        grid_bounds (list[float], optional): The grid boundaries for the plot. Defaults to [-180.0, 180.0, -90.0, 90.0].
        calc_anoms (bool, optional): Whether to calculate anomalies. Defaults to False.
        energy_df_path (str, optional): The path to the energy dataframe. Defaults to "/home/users/benhutch/unseen_multi_year/dfs/obs_df_NDJFM_wind_demand_1960-2018_dnw.csv".
        grid_file (str, optional): The path to the grid file. Defaults to "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl_first_timestep_msl.nc".
        files_loc_path (str, optional): The path to the files location file. Defaults to "/home/users/benhutch/unseen_multi_year/paths/paths_20240117T122513.csv".

    Returns:
        None
    """

    # assert that all of the months are integers

    # set up the dictionary for the energy variables
    energy_dict = {
        "demand": "United_Kingdom_demand",
        "wind": "total_gen",
        "demand_net_wind": "demand_net_wind",
    }

    # assert that energy_variable is in ["demand", "wind", "demand_net_wind"]
    assert energy_variable in [
        "demand",
        "wind",
        "demand_net_wind",
    ], f"Unknown energy variable {energy_variable}, must be in ['demand', 'wind', 'demand_net_wind']"

    # Assert that the energy df path exists
    assert os.path.exists(
        energy_df_path
    ), f"Cannot find the energy df path {energy_df_path}"

    # Load the energy df
    energy_df = pd.read_csv(energy_df_path)

    # if "Unnamed: 0" in energy_df.columns:
    if "Unnamed: 0" in energy_df.columns:
        # Convert to datetime
        energy_df["Unnamed: 0"] = pd.to_datetime(energy_df["Unnamed: 0"], format="%Y")

        # Set as the index
        energy_df.set_index("Unnamed: 0", inplace=True)

        # strptime to just be the year
        energy_df.index = energy_df.index.strftime("%Y")

        # remove the name of the index
        energy_df.index.name = None
    else:
        raise NotImplementedError("Unnamed: 0 not in the columns")

    # format as an int member
    energy_df["member"] = energy_df["member"].astype(int)

    # extract the unique members
    unique_members = energy_df["member"].unique()

    # set this as an index
    energy_df.set_index("member", append=True, inplace=True)

    # # print the head of the energy df
    # print(f"energy_df head: {energy_df.head()}")

    # if the correct column for the specified energy variable is in the df
    if energy_dict[energy_variable] in energy_df.columns:
        # Subset the df to this column
        energy_series = energy_df[energy_dict[energy_variable]]
    else:
        raise ValueError(
            f"Cannot find the column {energy_dict[energy_variable]} in the energy df"
        )

    # Calculate the percentile threshold
    threshold = energy_series.quantile(percentile)

    if energy_variable != "wind":
        # identify the number of events above the threshold
        num_events = len(energy_series[energy_series > threshold])
        print(f"Number of events above the {percentile} percentile: {num_events}")

        # Find the years of the events above the threshold
        years_members = energy_series[energy_series > threshold].index
        print(f"Years of the events above the {percentile} percentile: {years_members}")
    else:
        # identify the number of events below the threshold
        num_events = len(energy_series[energy_series < threshold])
        print(f"Number of events below the {percentile} percentile: {num_events}")

        # Find the years of the events below the threshold
        years_members = energy_series[energy_series < threshold].index
        print(f"Years of the events below the {percentile} percentile: {years_members}")

    # Check that the csv file exits
    assert os.path.exists(
        files_loc_path
    ), f"Cannot find the files location path {files_loc_path}"

    # Load the files location
    files_loc = pd.read_csv(files_loc_path)

    # print the data we seek
    print(f"model: {model}")
    print(f"experiment: {experiment}")
    print(f"freq: {freq}")
    print(f"psl_variable: {sf_variable}")

    # extract the model_path_var
    model_path_var = files_loc.loc[
        (files_loc["model"] == model)
        & (files_loc["experiment"] == experiment)
        & (files_loc["frequency"] == freq)
        & (files_loc["variable"] == sf_variable)
    ]["path"].values[0]

    # Extract the path for the given model, experiment, freq, and variable
    model_path_psl = files_loc.loc[
        (files_loc["model"] == model)
        & (files_loc["experiment"] == experiment)
        & (files_loc["frequency"] == freq)
        & (files_loc["variable"] == psl_variable)
    ]["path"].values[0]

    # assert that the model path exists
    assert os.path.exists(
        model_path_var
    ), f"Cannot find the model path {model_path_var}"

    # extract the model path psl exists
    assert os.path.exists(
        model_path_psl
    ), f"Cannot find the model path {model_path_psl}"

    # extract the model path root
    model_path_root_var = model_path_var.split("/")[1]
    model_path_root_psl = model_path_psl.split("/")[1]

    # create a series of lists
    model_paths = [model_path_var, model_path_psl]
    model_path_roots = [model_path_root_var, model_path_root_psl]
    variables = [sf_variable, psl_variable]

    # set up an empty list of files
    list_files_lists = []

    # loop over the model paths
    for model_path, model_path_root, variable in zip(
        model_paths, model_path_roots, variables
    ):
        # set up an empty list of files
        files_list = []

        # loop over the multi index
        for year, member in years_members:
            # depending on the model_path_root
            if model_path_root == "work":
                raise NotImplementedError("work path not implemented yet")
            elif model_path_root == "gws":
                # Create the path
                path = f"{model_path}/{variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

                # glob this path
                files = glob.glob(path)

                # assert that files has length 1
                assert len(files) == 1, f"files has length {len(files)}"

                # extract the file
                file = files[0]
            elif model_path_root == "badc":
                raise NotImplementedError("home path not implemented yet")
            else:
                raise ValueError(f"Unknown model path root {model_path_root}")

            # append the file to the files_list
            files_list.append(file)

        # append the files_list to the list_files_lists
        list_files_lists.append(files_list)

    # create a list of ds_comp_lists
    regridded_cubes = []

    # load the obs cube
    cube_obs = iris.load_cube(grid_file)

    # loop over the files list
    for files_list, variable in tqdm(zip(list_files_lists, variables)):
        ds_comp_list = []
        # loop over the files
        for file, (year, member) in zip(files_list, years_members):
            # Load the model data
            ds = xr.open_dataset(file)

            # format the year as an int
            year = int(year)

            # if the variable is not in the ds
            if variable not in ds:
                raise ValueError(f"Cannot find the variable {variable} in the ds")

            # Set up the times to extract
            start_date_this = cftime.datetime.strptime(
                f"{year}-{months[0]}-01", "%Y-%m-%d", calendar="360_day"
            )
            end_date_this = cftime.datetime.strptime(
                f"{year + 1}-{months[-1]}-30", "%Y-%m-%d", calendar="360_day"
            )

            # slice between the start and end dates
            ds = ds.sel(time=slice(start_date_this, end_date_this))

            # append the ds to the list
            ds_comp_list.append(ds[variable])

        # concatenate with a new time dimension using xarray
        ds_composite = xr.concat(ds_comp_list, dim="time")

        # convert to a cube
        cube = ds_composite.to_iris()

        # regrid the model data to the obs grid
        cube_regrid = cube.regrid(cube_obs, iris.analysis.Linear())

        # subset to the region of interest
        cube_regrid = cube_regrid.intersection(
            latitude=(lat_bounds[0], lat_bounds[1]),
            longitude=(lon_bounds[0], lon_bounds[1]),
        )

        # calculate the time mean of this
        cube_regrid = cube_regrid.collapsed("time", iris.analysis.MEAN)

        # append the cube_regrid to the list_ds_comp_lists
        regridded_cubes.append(cube_regrid)

    if calc_anoms:
        # Set up an empty list for the full ds's
        climatology_ds_list = []

        # loop over the variables
        for variable, model_path in zip(variables, model_paths):
            init_year_list = []
            for year in tqdm(range(climatology_period[0], climatology_period[1] + 1)):
                member_list = []
                for member in unique_members:
                    path = f"{model_path}/{variable}_{freq}_{model}_{experiment}_s{year}-r{member}i*_*_{year}??-*.nc"

                    # glob this path
                    files = glob.glob(path)

                    # assert
                    assert (
                        len(files) == 1
                    ), f"files has length {len(files)} for year {year} and member {member} and path {path}"

                    # open all of the files
                    member_ds = xr.open_mfdataset(
                        files[0],
                        combine="nested",
                        concat_dim="time",
                        preprocess=lambda ds: preprocess(
                            ds=ds,
                            year=year,
                            variable=variable,
                            months=months,
                        ),
                        parallel=False,
                        engine="netcdf4",
                        coords="minimal",  # expecting identical coords
                        data_vars="minimal",  # expecting identical vars
                        compat="override",  # speed up
                    ).squeeze()

                    # id init year == climatology_period[0]
                    # and member == unique_members[0]
                    if year == climatology_period[0] and member == unique_members[0]:
                        # set the new integer time
                        member_ds = set_integer_time_axis(
                            xro=member_ds,
                            frequency=freq,
                            first_month_attr=True,
                        )
                    else:
                        # set the new integer time
                        member_ds = set_integer_time_axis(
                            xro=member_ds,
                            frequency=freq,
                        )

                    # append the member_ds to the member_list
                    member_list.append(member_ds)
                # Concatenate with a new member dimension using xarray
                member_ds = xr.concat(member_list, dim="member")
                # append the member_ds to the init_year_list
                init_year_list.append(member_ds)
            # Concatenate the init_year list along the init dimension
            # and rename as lead time
            ds = xr.concat(init_year_list, "init").rename({"time": "lead"})

            # set up the members
            ds["member"] = unique_members
            ds["init"] = np.arange(climatology_period[0], climatology_period[1] + 1)

            # extract the variable
            ds_var = ds[variable]

            # take the mean over lead dimension
            ds_clim = ds_var.mean(dim="lead")

            # take the mean over member dimension
            ds_clim = ds_clim.mean(dim="member")

            # take the mean over init dimension
            ds_clim = ds_clim.mean(dim="init")

            # convert to a cube
            cube_clim = ds_clim.to_iris()

            # regrid the model data to the obs grid
            cube_clim_regrid = cube_clim.regrid(cube_obs, iris.analysis.Linear())

            # subset to the region of interest
            cube_clim_regrid = cube_clim_regrid.intersection(
                latitude=(lat_bounds[0], lat_bounds[1]),
                longitude=(lon_bounds[0], lon_bounds[1]),
            )

            # append the cube_clim_regrid to the climatology_ds_list
            climatology_ds_list.append(cube_clim_regrid)

    # extract the lats and lons
    lats = regridded_cubes[0].coord("latitude").points
    lons = regridded_cubes[0].coord("longitude").points

    # if calc_anoms is True
    if calc_anoms:
        field_var = regridded_cubes[0].data - climatology_ds_list[0].data
        field_psl = (
            regridded_cubes[1].data - climatology_ds_list[1].data
        ) / 100  # convert to hPa
    else:
        field_var = regridded_cubes[0].data
        field_psl = regridded_cubes[1].data / 100

        # if the variable is temp
        if sf_variable in ["t2m", "tas"]:
            field_var -= 273.15

    # set up the figure
    fig, ax = plt.subplots(
        figsize=(10, 5), subplot_kw=dict(projection=ccrs.PlateCarree())
    )

    # if calc_anoms is True
    if calc_anoms:
        # clevs = np.linspace(-8, 8, 18)
        clevs_psl = np.array(
            [
                -8.0,
                -7.0,
                -6.0,
                -5.0,
                -4.0,
                -3.0,
                -2.0,
                -1.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ]
        )
        ticks_psl = clevs_psl

        # ensure that these are floats
        clevs_psl = clevs_psl.astype(float)
        ticks_psl = ticks_psl.astype(float)

        # depending on the variable
        if sf_variable in ["t2m", "tas"]:
            # -18 to +18 in 2 degree intervals
            clevs_var = np.array(
                [
                    -5.0,
                    -4.5,
                    -4.0,
                    -3.5,
                    -3.0,
                    -2.5,
                    -2.0,
                    -1.5,
                    -1.0,
                    -0.5,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    2.5,
                    3.0,
                    3.5,
                    4.0,
                    4.5,
                    5.0,
                ]
            )
            ticks_var = clevs_var

            # set up tjhe cmap
            cmap = "bwr"

            # set the cbar label
            cbar_label = "temperature (°C)"
        elif sf_variable in ["u10", "v10", "sfcWind", "si10"]:
            # 0 to 20 in 2 m/s intervals
            clevs_var = np.array(
                [
                    -1.4,
                    -1.2,
                    -1.0,
                    -0.8,
                    -0.6,
                    -0.4,
                    -0.2,
                    0.2,
                    0.4,
                    0.6,
                    0.8,
                    1.0,
                    1.2,
                    1.4,
                ]
            )
            ticks_var = clevs_var

            # set up the cmap
            cmap = "PRGn_r"

            # set the cbar label
            cbar_label = "10m wind speed (m/s)"
        else:
            raise ValueError(f"Unknown variable {variable}")

    else:
        # define the contour levels for the variable
        # should be 19 of them
        if sf_variable in ["t2m", "tas"]:
            # -18 to +18 in 2 degree intervals
            clevs_var = np.array(np.arange(-18, 18 + 1, 2))
            ticks_var = clevs_var

            # set up tjhe cmap
            cmap = "bwr"

            # set the cbar label
            cbar_label = "temperature (°C)"

        elif sf_variable in ["u10", "v10", "sfcWind", "si10"]:
            # 0 to 20 in 2 m/s intervals
            clevs_var = np.array(np.arange(0, 12 + 1, 1))
            ticks_var = clevs_var

            # set up the cmap
            cmap = "RdPu"

            # set the cbar label
            cbar_label = "10m wind speed (m/s)"
        else:
            raise ValueError(f"Unknown variable {variable}")

        # define the contour levels
        clevs_psl = np.array(np.arange(988, 1024 + 1, 2))
        ticks_psl = clevs_psl

        # ensure that these are ints
        clevs_psl = clevs_psl.astype(int)
        ticks_psl = ticks_psl.astype(int)

    # print the len of clevs_psl
    print(f"len of clevs_psl: {len(clevs_psl)}")
    print(f"len of clevs_var: {len(clevs_var)}")

    # print field_var and field_psl
    print(f"field_var shape: {field_var.shape}")
    print(f"field_psl shape: {field_psl.shape}")
    print(f"field_var values: {field_var}")
    print(f"field_psl values: {field_psl}")

    # print the field var min and the field var max
    print(f"field_var min: {field_var.min()}")
    print(f"field_var max: {field_var.max()}")

    if variable in ["si10", "sfcWind"] and not calc_anoms:
        # set up the extend
        extend = "max"
    else:
        extend = "both"

    # plot the data
    mymap = ax.contourf(
        lons,
        lats,
        field_var,
        clevs_var,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        extend=extend,
    )

    # plot the psl contours
    contours = ax.contour(
        lons,
        lats,
        field_psl,
        clevs_psl,
        colors="black",
        transform=ccrs.PlateCarree(),
        linewidth=0.2,
        alpha=0.5,
    )

    if calc_anoms:
        ax.clabel(
            contours, clevs_psl, fmt="%.1f", fontsize=8, inline=True, inline_spacing=0.0
        )
    else:
        ax.clabel(
            contours, clevs_psl, fmt="%.4g", fontsize=8, inline=True, inline_spacing=0.0
        )
    # add coastlines
    ax.coastlines()

    # format the gridlines and labels
    gl = ax.gridlines(
        draw_labels=True, linewidth=0.5, color="black", alpha=0.5, linestyle=":"
    )
    gl.xlabels_top = False
    gl.xlocator = mplticker.FixedLocator(np.arange(-180, 180, 30))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.xlabel_style = {"size": 7, "color": "black"}
    gl.ylabels_right = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.ylabel_style = {"size": 7, "color": "black"}

    if calc_anoms:
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func_one_decimal),
        )
        # add colorbar label
        cbar.set_label(
            f"{cbar_label} {climatology_period[0]}-{climatology_period[1]} anomaly",
            rotation=0,
            fontsize=10,
        )

        # # add contour lines to the colorbar
        # cbar.add_lines(mymap)
    else:
        # add colorbar
        cbar = plt.colorbar(
            mymap,
            orientation="horizontal",
            shrink=0.7,
            pad=0.1,
            format=FuncFormatter(format_func),
        )
        cbar.set_label(cbar_label, rotation=0, fontsize=10)

        # # set up invisible contour lines for field_var
        # contour_var = ax.contour(
        #     lons,
        #     lats,
        #     field_var,
        #     clevs_var,
        #     colors="k",
        #     transform=ccrs.PlateCarree(),
        #     linewidth=0.2,
        #     alpha=0.5,
        # )

        # # add contour lines to the colorbar
        # cbar.add_lines(contour_var)
    cbar.ax.tick_params(labelsize=7, length=0)
    # set the ticks
    cbar.set_ticks(ticks_var)

    # add title
    ax.set_title(title, fontsize=12, weight="bold")

    # make plot look nice
    plt.tight_layout()

    return None


# define a function for preprocessing
def preprocess(
    ds: xr.Dataset,
    year: int,
    variable: str,
    months: list[int] = [11, 12, 1, 2, 3],
) -> xr.Dataset:
    """
    Preprocesses the model data by subsetting to the months of interest.

    Args:
        ds (xr.Dataset): The model dataset to be preprocessed.
        year (int): The year of the data.
        variable (str): The variable to be preprocessed.
        months (list[int], optional): The months to be used for the preprocessing. Defaults to [11, 12, 1, 2, 3].

    Returns:
        xr.Dataset: The preprocessed model dataset.
    """

    # if year is not an int, format as an int
    if not isinstance(year, int):
        year = int(year)

    # if the variable is not in the ds
    if variable not in ds:
        raise ValueError(f"Cannot find the variable {variable} in the ds")

    # # Set up the times to extract
    # start_date_this = cftime.datetime.strptime(
    #     f"{year}-{months[0]}-01", "%Y-%m-%d", calendar="360_day"
    # )
    # end_date_this = cftime.datetime.strptime(
    #     f"{year + 1}-{months[-1]}-30", "%Y-%m-%d", calendar="360_day"
    # )

    # # slice between the start and end dates
    # ds = ds.sel(time=slice(start_date_this, end_date_this))

    # extract the specific months
    ds = ds.sel(time=ds["time.month"].isin(months))

    return ds


def set_integer_time_axis(
    xro: Union[xr.DataArray, xr.Dataset],
    frequency: str = "Amon",
    offset: int = 1,
    time_dim: str = "time",
    first_month_attr: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Set time axis to integers starting from `offset`.

    Used in hindcast preprocessing before the concatenation of `intake-esm` happens.

    Inputs:
    xro: xr.DataArray or xr.Dataset
        The input xarray DataArray or Dataset whose time axis is to be modified.

    frequency: str, optional
        The frequency of the data. Default is "Amon".

    offset: int, optional
        The starting point for the new integer time axis. Default is 1.

    time_dim: str, optional
        The name of the time dimension in the input xarray object. Default is "time".

    first_month_attr: bool, optional
        Whether to include the first month as an attribute in the dataset.
        Default is False.

    Returns:
    xr.DataArray or xr.Dataset
        The input xarray object with the time axis set to integers starting from `offset`.
    """

    if first_month_attr:
        # Extract the first forecast year-month pair
        first_month = xro[time_dim].values[0]

        # Add the first month as an attribute to the dataset
        xro.attrs["first_month"] = str(first_month)

        # add an attribute for the type of the time axis
        xro.attrs["time_axis_type"] = type(first_month).__name__

    xro[time_dim] = np.arange(offset, offset + xro[time_dim].size)
    return xro


# Define a function which plots the time series for one of:
# * Demand
# * Wind power
# * Demand net wind
# along with the NAO time series
# For the observations in the first instance, then move onto the model
def plot_nao_ts_obs(
    start_date: str,
    end_date: str,
    title: str,
    energy_variable: str,
    months: list[int] = [11, 12, 1, 2, 3],
    psl_variable: str = "msl",
    freq: str = "Amon",
    energy_df_path: str = "/home/users/benhutch/unseen_multi_year/dfs/obs_df_NDJFM_wind_demand_1960-2018_dnw.csv",
    ERA5_regrid_path: str = "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl.nc",
    azores_grid: dict = {"lon1": -28, "lon2": -20, "lat1": 36, "lat2": 40},
    iceland_grid: dict = {"lon1": -25, "lon2": -16, "lat1": 63, "lat2": 70},
    standardise_ts: bool = False,
    invert_predictand: bool = False,
    vertical_lines: list = None,
    percentile_years: bool = False,
    set_ylims: list = None,
) -> None:
    """
    Grabs the mslp data for the observations and the .csv containing the demand,
    wind power, or demand net wind data and plots the time series for the given
    energy variable alongside the NAO time series.

    Args:
        start_date (str): The start date of the time series. Format: "YYYY-MM-DD".
        end_date (str): The end date of the time series. Format: "YYYY-MM-DD".
        title (str): The title of the plot.
        energy_variable (str): The energy variable to be used for the time series.
        months (list[int], optional): The months to be used for the time series. Defaults to [11, 12, 1, 2, 3].
        psl_variable (str, optional): The pressure level variable to be used for the time series. Defaults to "msl".
        freq (str, optional): The frequency of the data. Defaults to "Amon".
        energy_df_path (str, optional): The path to the energy dataframe. Defaults to "/home/users/benhutch/unseen_multi_year/dfs/obs_df_NDJFM_wind_demand_1960-2018_dnw.csv".
        ERA5_regrid_path (str, optional): The path to the regridded ERA5 data. Defaults to "/gws/nopw/j04/canari/users/benhutch/ERA5/global_regrid_sel_region_psl.nc".
        azores_grid (dict, optional): The grid boundaries for the Azores region. Defaults to {"lon1": -28, "lon2": -20, "lat1": 36, "lat2": 40}.
        iceland_grid (dict, optional): The grid boundaries for the Iceland region. Defaults to {"lon1": -25, "lon2": -16, "lat1": 63, "lat2": 70}.
        standardise_ts (bool, optional): Whether to standardise the time series. Defaults to False.
        invert_predictand (bool, optional): Whether to invert the predictand. Defaults to False.
        vertical_lines (list, optional): The vertical lines to be added to the plot. Defaults to None.
        percentile_years: (bool, optional): Whether to calculate the percentile years. Defaults to False.
        set_ylims (list, optional): The y limits to be set for the plot. Defaults to None.

    Returns:
        None
    """

    # assert that the start date is formatted in the correct way
    assert re.match(
        r"\d{4}-\d{2}-\d{2}", start_date
    ), f"Start date {start_date} not formatted correctly"

    # assert that the end date is formatted in the correct way
    assert re.match(
        r"\d{4}-\d{2}-\d{2}", end_date
    ), f"End date {end_date} not formatted correctly"

    # extract the start year
    start_year = int(start_date.split("-")[0])
    end_year = int(end_date.split("-")[0])

    # Set up the dictionary for the energy variables
    energy_dict = {
        "demand": "United_Kingdom_demand",
        "wind": "total_gen",
        "demand_net_wind": "demand_net_wind",
    }

    # assert that energy_variable is in ["demand", "wind", "demand_net_wind"]
    assert energy_variable in [
        "demand",
        "wind",
        "demand_net_wind",
    ], f"Unknown energy variable {energy_variable}, must be in ['demand', 'wind', 'demand_net_wind']"

    # Assert that the energy df path exists
    assert os.path.exists(
        energy_df_path
    ), f"Cannot find the energy df path {energy_df_path}"

    # assert that the ERA5 regrid path exists
    assert os.path.exists(
        ERA5_regrid_path
    ), f"Cannot find the ERA5 regrid path {ERA5_regrid_path}"

    # assert all of the months are integers
    assert all(
        isinstance(month, int) for month in months
    ), f"months must all be integers, got {months}"

    # Load the energy df
    energy_df = pd.read_csv(energy_df_path)

    # if "Unnamed: 0" in energy_df.columns:
    if "Unnamed: 0" in energy_df.columns:
        # Convert to datetime
        energy_df["Unnamed: 0"] = pd.to_datetime(energy_df["Unnamed: 0"], format="%Y")

        # Set as the index
        energy_df.set_index("Unnamed: 0", inplace=True)

        # strptime to just be the year
        energy_df.index = energy_df.index.strftime("%Y")

        # remove the name of the index
        energy_df.index.name = None

    # if start year and end year are both part of the index
    if str(start_year) in energy_df.index and str(end_year) in energy_df.index:
        # Subset the df to the start and end years
        energy_df = energy_df.loc[str(start_year) : str(end_year)]
    else:
        raise ValueError(
            f"Cannot find the start year {start_year} and end year {end_year} in the energy df"
        )

    # if the correct column for the specified energy variable is in the df
    if energy_dict[energy_variable] in energy_df.columns:
        # Subset the df to this column
        energy_series = energy_df[energy_dict[energy_variable]]
    else:
        raise ValueError(
            f"Cannot find the column {energy_dict[energy_variable]} in the energy df"
        )

    # Load the ERA5 regridded data
    ds = xr.open_mfdataset(
        ERA5_regrid_path,
        chunks={"time": 10},
        combine="by_coords",
        parallel=False,
        engine="netcdf4",
        coords="minimal",
    )

    # If expver is present in the observations
    if "expver" in ds.coords:
        # Combine the first two expver variables
        ds = ds.sel(expver=1).combine_first(ds.sel(expver=5))

    # if the variable is not in the ds
    if psl_variable not in ds:
        raise ValueError(f"Cannot find the variable {psl_variable} in the ds")
    else:
        # extract the variable
        ds = ds[psl_variable]

    # subset the data to the months
    ds = ds.sel(time=ds["time.month"].isin(months))

    # Subset to the start and end dates
    ds = ds.sel(time=slice(start_date, end_date))

    # shift time back by int(months[-1]) and resample to take the annual mean
    ds_shifted = ds.shift(time=-int(months[-1])).resample(time="A").mean()

    # subset to the azores and iceland regions
    ds_azores = ds_shifted.sel(
        lat=slice(azores_grid["lat1"], azores_grid["lat2"]),
        lon=slice(azores_grid["lon1"], azores_grid["lon2"]),
    ).mean(dim=["lat", "lon"]) - ds_shifted.sel(
        lat=slice(azores_grid["lat1"], azores_grid["lat2"]),
        lon=slice(azores_grid["lon1"], azores_grid["lon2"]),
    ).mean(
        dim=["lat", "lon"]
    ).mean(
        dim="time"
    )
    ds_iceland = ds_shifted.sel(
        lat=slice(iceland_grid["lat1"], iceland_grid["lat2"]),
        lon=slice(iceland_grid["lon1"], iceland_grid["lon2"]),
    ).mean(dim=["lat", "lon"]) - ds_shifted.sel(
        lat=slice(iceland_grid["lat1"], iceland_grid["lat2"]),
        lon=slice(iceland_grid["lon1"], iceland_grid["lon2"]),
    ).mean(
        dim=["lat", "lon"]
    ).mean(
        dim="time"
    )

    # calculate the NAO as the difference between the azores and iceland regions
    nao = (ds_azores - ds_iceland) / 100  # convert to hPa

    # print the nao values
    print(f"nao values: {nao.values}")

    # remove any Nans
    nao = nao.dropna(dim="time")

    # extract the obs nao
    obs_nao = nao.values

    # extract the energy_var
    energy_var = energy_series.values

    # extract the index
    index = energy_series.index.values

    # # print the shape of the obs_nao
    # print(f"obs_nao shape: {obs_nao.shape}")
    # print(f"energy_var shape: {energy_var.shape}")
    # print(f"index shape: {index.shape}")

    # # print the nao values
    # print(f"nao values: {nao.values}")
    # print(f"energy_var values: {energy_series.values}")
    print(f"index values: {energy_series.index.values}")

    # print the type of index
    print(f"type of index: {type(index[0])}")

    # convert the index to a list of ints
    index = [int(str(date)[:4]) for date in index]

    # if the length of the obs_nao is not equal to the length of the energy_var
    if len(obs_nao) != len(energy_var):
        # subset energy var to [:len(obs_nao) + 1]
        energy_var = energy_var[:-1]
        index = index[:-1]

        # assert that the length of the obs_nao is equal to the length of the energy_var
        assert len(obs_nao) == len(
            energy_var
        ), "Lengths of obs_nao and energy_var not equal"
        assert len(obs_nao) == len(index), "Lengths of obs_nao and index not equal"

    # if standardise_ts is True
    if standardise_ts:
        # standardise the obs_nao
        obs_nao = (obs_nao - obs_nao.mean()) / obs_nao.std()

        # standardise the energy_var
        energy_var = (energy_var - energy_var.mean()) / energy_var.std()

    # set up the figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # set up a horizontal black line at 0
    ax.axhline(0, color="black")

    # plot the observed nao index
    ax.plot(index, obs_nao, color="black", label="NAO index")

    if invert_predictand:
        ax.plot(index, -energy_var, color="red", label=f"inverted {energy_variable}")
    else:
        # plot the energy_var
        ax.plot(index, energy_var, color="red", label=energy_variable)

    # use pearsonr to calculate the r and p values
    r, p = pearsonr(obs_nao, energy_var)

    # Include these values in a textbox
    ax.text(
        0.05,
        0.95,
        (f"r = {r:.2f} (p = {p:.2f})"),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # if percentile years is True
    if percentile_years:
        # set up the percentile
        percentile = 5

        # find the NAo years beneath the 5th percentile
        nao_percentile = np.percentile(obs_nao, percentile)

        # find the years where the NAO is beneath the 5th percentile
        nao_percentile_years = np.where(obs_nao < nao_percentile)[0]

        # apply this to the index
        index_percentile_nao = [index[year] for year in nao_percentile_years]

        # print the index_percentile_nao
        print(f"index_percentile_nao: {index_percentile_nao}")

        # find the 1 - percentile for the energy variable (if not wind)
        if energy_variable != "wind":
            print(
                f"variable is not wind, calculating the {100 - percentile} percentile"
            )

            # print energy var min and max
            print(f"energy_var min: {energy_var.min()}")
            print(f"energy_var max: {energy_var.max()}")

            # find the energy variable percentile
            energy_percentile = np.percentile(energy_var, 100 - percentile)

            # print the energy_percentile
            print(f"energy_percentile: {energy_percentile}")

            # find the years where the energy variable is beneath the 1 - percentile
            energy_percentile_years = np.where(energy_var > energy_percentile)[0]

            # apply this to the index
            index_percentile_energy = [index[year] for year in energy_percentile_years]

            # print the index_percentile_energy
            print(f"index_percentile_energy: {index_percentile_energy}")
        else:
            # find the energy variable percentile
            energy_percentile = np.percentile(energy_var, percentile)

            # print the energy_percentile
            print(f"energy_percentile: {energy_percentile}")

            # find the years where the energy variable is beneath the 1 - percentile
            energy_percentile_years = np.where(energy_var < energy_percentile)[0]

            # apply this to the index
            index_percentile_energy = [index[year] for year in energy_percentile_years]

            # print the index_percentile_energy
            print(f"index_percentile_energy: {index_percentile_energy}")

        # loop over the ints
        for nao_year, energy_year in zip(index_percentile_nao, index_percentile_energy):
            # add a vertical line
            ax.axvline(nao_year, color="blue", linestyle="--")
            ax.axvline(energy_year, color="hotpink", linestyle="-.")

            # include a label
            ax.text(
                nao_year - 0.6,
                1.5,
                f"{nao_year}",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=10,
                color="blue",
            )

            # include a label
            ax.text(
                energy_year - 0.6,
                1.5,
                f"{energy_year}",
                rotation=90,
                verticalalignment="center",
                horizontalalignment="center",
                fontsize=10,
                color="hotpink",
            )

    else:
        # if vertical lines is not none
        if vertical_lines is not None:
            # assert that verrtical lines is a list of ints
            assert all(
                isinstance(vertical_line, int) for vertical_line in vertical_lines
            ), f"vertical_lines must all be integers, got {vertical_lines}"

            # loop over the ints
            for vertical_line in vertical_lines:
                # add a vertical line
                ax.axvline(vertical_line, color="hotpink", linestyle="--")

                # include a label
                ax.text(
                    vertical_line - 0.6,
                    1.5,
                    f"{vertical_line}",
                    rotation=90,
                    verticalalignment="center",
                    horizontalalignment="center",
                    fontsize=10,
                    color="hotpink",
                )

    # if set_ylims is not None
    if set_ylims is not None:
        # assert that set_ylims is a list of ints
        assert all(
            isinstance(ylim, int) for ylim in set_ylims
        ), f"set_ylims must all be integers, got {set_ylims}"

        # set the y limits
        ax.set_ylim(set_ylims)

    # set the title
    ax.set_title(title, fontsize=12, weight="bold")

    # only plot the xticks every 5 years
    ax.set_xticks(index[::5])

    # set the ylable
    ax.set_ylabel("NAO index / energy variable normalised anomalies")

    # include a legend
    ax.legend(loc="upper right")

    return None


def main():
    """
    Main function for testing purposes.
    """

    # Define the start and end dates
    # start_date = "1965-11-01"
    # end_date = "1966-03-30"
    # member = "2"
    # title = "MSLP Anomalies for November 1965 to March 1966, member r2i1p1f2 HadGEM3-GC31-MM"

    # Set up the constsnats
    # title = "MSLP composites for the 95th percentile of demand-net-wind events, observations"
    # energy_variable = "demand_net_wind"
    # percentile = 0.95

    # set up the constants
    start_date = "1960-11-01"
    end_date = "2018-03-31"
    title = "Observed correlations between NDJFM NAO index and demand net wind, UK"
    energy_variable = "demand_net_wind"

    # call the functions
    plot_nao_ts_obs(
        start_date=start_date,
        end_date=end_date,
        title=title,
        energy_variable=energy_variable,
    )

    # Call the function
    # plot_composite_model(
    #     title=title,
    #     energy_variable=energy_variable,
    #     percentile=percentile,
    #     calc_anoms=True,
    # )

    # # Call the function
    # plot_mslp_anoms_model(
    #     start_date=start_date,
    #     end_date=end_date,
    #     member=member,
    #     title=title,
    #     calc_anoms=True,
    # )

    # # call rthe new obs function
    # plot_mslp_anoms_temp_wind_obs(
    #     start_date=start_date,
    #     end_date=end_date,
    #     title=title,
    #     calc_anoms=False,
    # )

    return None


if __name__ == "__main__":
    main()
