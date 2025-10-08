# %%
"""
process_ERA5_wind_gen.py
=======================

This script processes daily ERA5 10m wind speed data into an estimate of wind
power generation for the UK.

Usage:
------

    $ python process_ERA5_wind_gen.py

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
import argparse

# Third-party imports
import iris
import cftime
import iris.analysis
import iris.analysis.cartography
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
import shapely.geometry
import iris.coord_categorisation
import xarray as xr

# Specific imports
from tqdm import tqdm
from iris import cube
from iris.util import equalise_attributes
from iris.cube import Cube, CubeList

# Imports from Hannah's functions
sys.path.append(
    "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/"
)

# Importing the necessary functions from the module
from European_hourly_hub_height_winds_2023_model import (
    country_mask,
    load_power_curves,
    load_wind_speed_and_take_to_hubheight,
)


# Define a function to process area weighted mean for wind farm locations
def process_area_weighted_mean(
    path_to_farm_locations_ons: str,
    path_to_farm_locations_ofs: str,
    cube: iris.cube.Cube,
    lats: np.ndarray = None,
    lons: np.ndarray = None,
) -> iris.cube.Cube:
    """
    Process area weighted mean for wind farm locations.

    Args:
        path_to_farm_locations_ons (str): Path to onshore wind farm locations.
        path_to_farm_locations_ofs (str): Path to offshore wind farm locations.
        cube (iris.cube.Cube): Cube containing the wind speed data.

    Returns:
        pd.DataFrame: DataFrame containing the area weighted mean wind speeds
        for onshore and offshore wind farms
    """
    # Load the onshore and offshore wind farm locations
    onshore_farm_locations = iris.load_cube(path_to_farm_locations_ons)
    offshore_farm_locations = iris.load_cube(path_to_farm_locations_ofs)

    # Get the lats and lons from the cube
    lats_wp, lons_wp = iris.analysis.cartography.get_xy_grids(onshore_farm_locations)

    # Add the onshore and offshore wind farm locations together
    combined_farm_locations = onshore_farm_locations + offshore_farm_locations

    # Set the coordinates and the bounds
    for coordinate in ["latitude", "longitude"]:
        combined_farm_locations.coord(coordinate).units = "degrees"
        combined_farm_locations.coord(coordinate).guess_bounds()

    # Get the x and y grids from the combined farm locations
    lons_wind, lats_wind = iris.analysis.cartography.get_xy_grids(
        cube=cube,
    )

    # Print the cube
    print(f"Cube shape: {cube.shape}")
    # Print the type of the cube
    print(f"Cube type: {type(cube)}")

    # Debugging statements to verify inputs
    print(f"Type of onshore_farm_locations: {type(onshore_farm_locations)}")
    print(f"Type of offshore_farm_locations: {type(offshore_farm_locations)}")
    print(f"Type of combined_farm_locations: {type(combined_farm_locations)}")
    print(f"Type of cube: {type(cube)}")

    # Ensure the cube is valid
    if not isinstance(cube, Cube):
        raise TypeError("The 'cube' parameter is not an Iris Cube. Please check the input.")

    # Print the type of the subset cube
    # print(f"Type of cube[0, :, :]: {type(cube[0, :, :])}")
    # print(f"Shape of cube[0, :, :]: {cube[0, :, :].shape}")
    # print(f"Values of cube[0, :, :]: {cube[0, :, :]}")

    # Get the area weights of the target drig
    wind_weights = iris.analysis.cartography.area_weights(
        cube=cube,
    )

    # Get the weights of the wind farm grid
    wf_weights = iris.analysis.cartography.area_weights(
        cube=combined_farm_locations,
    )

    # Print the shapes of the weights
    print(f"Shape of wind_weights: {wind_weights.shape}")
    print(f"Shape of wf_weights: {wf_weights.shape}")

    # Print the shape of the combined farm locations
    print(f"Shape of combined_farm_locations: {combined_farm_locations.shape}")

    # Get the wind farms total over the area
    wind_farm_total_over_area = combined_farm_locations / wf_weights

    # Print the shape of the wind farm total over area
    print(f"Shape of wind_farm_total_over_area: {wind_farm_total_over_area.shape}")
    # Print the shape of the cube
    print(f"Shape of cube: {cube.shape}")

    # Regrid the wind farm locations to the wind speed cube
    regridded_farm_locations = wind_farm_total_over_area.regrid(
        cube, iris.analysis.AreaWeighted()
    )

    # Undo the divide by dA
    regridded_farm_locations_total_corrected = regridded_farm_locations * wind_weights

    return regridded_farm_locations_total_corrected


# Define a function for loading the wind speed and taking it to hub height
def load_wind_speed_and_take_to_hubheight(
    ERA5_cube: iris.cube.Cube,
    land_mask: np.ndarray,
    height_of_wind_speed: float = 10.0,
    u_cube: iris.cube.Cube = None,
    v_cube: iris.cube.Cube = None,
) -> iris.cube.Cube:
    """
    Load wind speed data and take it to hub height.

    Args:
        ERA5_cube (iris.cube.Cube): Cube containing the ERA5 wind speed data.
        land_mask (np.ndarray): Land mask for the region.
        height_of_wind_speed (float): Height of the wind speed data in meters.

    Returns:
        iris.cube.Cube: Cube containing the wind speed data at hub height.
    """

    # Perform the correction to hub height
    correction_hubheight = land_mask * (71.0 / height_of_wind_speed) ** (
        1.0 / 7.0
    ) + abs(land_mask - 1) * (92.0 / height_of_wind_speed) ** (1.0 / 7.0)

    # If u_cube and v_cube are provided, calculate the wind speed
    if u_cube is not None and v_cube is not None:
        # Calculate the wind speed from u and v components
        wind_speed = (u_cube ** 2 + v_cube ** 2) ** 0.5
        ERA5_cube = wind_speed

    # Calculate the speed at hub height
    ERA5_cube_hubheight = ERA5_cube * correction_hubheight

    # Set the units to m/s
    ERA5_cube_hubheight.units = "m/s"
    # Set the standard name to wind speed
    ERA5_cube_hubheight.standard_name = "wind_speed"

    # Set the long name to wind speed at hub height
    ERA5_cube_hubheight.long_name = "Wind speed at hub height (m/s)"

    return ERA5_cube_hubheight


# Convert the wind speed at hub height to wind power generation
# using the power curves
def convert_wind_speed_to_power_generation(
    ERA5_cube_hubheight: iris.cube.Cube,
    pc_winds: np.ndarray,
    pc_power_ons: np.ndarray,
    pc_power_ofs: np.ndarray,
    land_mask: np.ndarray = None,
    farm_locations: np.ndarray = None,
) -> np.ndarray:
    """
    Convert wind speed at hub height to wind power generation.

    Args:
        ERA5_cube_hubheight (iris.cube.Cube): Cube containing the wind speed data at hub height.
        pc_winds (np.ndarray): Array of wind speeds from the power curve.
        pc_power_ons (np.ndarray): Array of onshore power generation from the power curve.
        pc_power_ofs (np.ndarray): Array of offshore power generation from the power curve.
        land_mask (np.ndarray, optional): Land mask for the region. Defaults to None.
        farm_locations (np.ndarray, optional): Wind farm locations. Defaults to None.

    Returns:
        np.ndarray: Array of wind power generation in GW.
    """

    # If ERA5_cube_hubheight is not a cube, raise an error
    if isinstance(ERA5_cube_hubheight, iris.cube.Cube):
        print("ERA5_cube_hubheight is a cube, processing it.")
        hh_ws_values = np.array(ERA5_cube_hubheight.data)  # Convert memoryview to NumPy array
        hh_ws_values_flat = hh_ws_values.ravel()  # Flatten the data

        # Digitize the wind speed data to the power curve
        digitized_winds = np.digitize(
            hh_ws_values_flat,
            pc_winds,
            right=False,  # Use left edges for bins
        ).reshape(
            hh_ws_values.shape  # Reshape to match the original shape
        )  # Reshape back to original shape if needed
    else:
        print("ERA5_cube_hubheight is not a cube, using it directly as an array.")
        hh_ws_values = ERA5_cube_hubheight
        # hh_ws_values_flat = hh_ws_values.ravel()

        print("Using searchsorted for digitization.")
        # Digitize the wind speed data to the power curve
        # Assuming pc_winds is sorted
        digitized_winds = np.searchsorted(pc_winds, hh_ws_values, side="right")
    
    # Make sure the bins don't go out of bounds
    digitized_winds[digitized_winds == len(pc_winds)] = 500

    # Apply. the landmask and power curve to the digitized winds onshore/offshore
    p_hh_ons = (
        land_mask
        * 0.5
        * (pc_power_ons[digitized_winds - 1] + pc_power_ons[digitized_winds])
    )
    p_hh_ofs = (
        (1 - land_mask)
        * 0.5
        * (pc_power_ofs[digitized_winds - 1] + pc_power_ofs[digitized_winds])
    )

    p_hh_total = p_hh_ons + p_hh_ofs

    # Get the time series accumulated over the country
    p_hh_total_GW = (p_hh_total * farm_locations) / 1000.0  # Convert to GW

    print("Summing -2, -1 axes for total power generation.")
    # sum across latitude and longitude
    p_hh_total_GW = np.sum(p_hh_total_GW, axis=(-2, -1))

    return p_hh_total_GW


# Define the main function
def main():
    # Start the timer
    start_time = time.time()

    # Set up the hardcoded variables
    # ERA5_file_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_wind_test_1960_1961.nc"
    ERA5_file_path = (
        "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_wind_daily_1952_2020.nc"
    )
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

    # Set the country for the mask
    COUNTRY = "United Kingdom"
    
    # # Set up the constraints
    # constraint_lon = (-6.25, 4.58)  # degrees east
    # constraint_lat = (50.28, 59.72)  # degrees north

    # Set up the fname
    fname = "ERA5_UK_wind_power_generation_cfs_constrained_1961_2025_daily_test_26092025.csv"
    fpath = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/Hannah_wind"

    # if the directory does not exist, create it
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    # If the full path already exists, then exit
    full_path = os.path.join(fpath, fname)

    if os.path.exists(full_path):
        print(f"File {full_path} already exists. Exiting.")
        return None

        # Load and process the observed data to compare against
    # Set up the path to the observed data
    base_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/"

    # Set up the remaining years
    # remaining_years = [str(year) for year in range(1960, 2025 + 1)]
    remaining_years = ["2018"]

    # Set up the path to the observed data
    remaining_files_dir = os.path.join(base_path, "year_month")

    # Set up an empty cubelist
    obs_cubelist_u10 = []
    obs_cubelist_v10 = []

#  ERA5_EU_85000_zg_T_U_V2018_03.nc

    # Loop over the remaining years
    for year in tqdm(remaining_years):
        for month in ["01", "02", "03", "12"]:
            # if the year is 2025 and the month is 12, then skip
            if year == "2025" and month == "12":
                continue
            
            # Set up the fname this
            # fname_this = f"ERA5_EU_T_U10_V10_msl{year}_{month}.nc"
            fname_this = f"ERA5_EU_85000_zg_T_U_V{year}_{month}.nc"

            # Set up the path to the observed data
            obs_path_this = os.path.join(remaining_files_dir, fname_this)

            # Load the observed data
            obs_cube_u10 = iris.load_cube(obs_path_this, "u10")
            obs_cube_v10 = iris.load_cube(obs_path_this, "v10")

            # Append to the cubelist
            obs_cubelist_u10.append(obs_cube_u10)
            obs_cubelist_v10.append(obs_cube_v10)

    # convert the list to a cube list
    obs_cubelist_u10 = iris.cube.CubeList(obs_cubelist_u10)
    obs_cubelist_v10 = iris.cube.CubeList(obs_cubelist_v10)

    # removed the attributes
    removed_attrs_u10 = equalise_attributes(obs_cubelist_u10)
    removed_attrs_v10 = equalise_attributes(obs_cubelist_v10)

    obs_cube_u10 = obs_cubelist_u10.concatenate_cube()
    obs_cube_v10 = obs_cubelist_v10.concatenate_cube()

    # # SUbset the cubes to the first time step
    # obs_cube_u10_first = obs_cube_u10[0, :, :]  # Subset the first time step
    # obs_cube_v10_first = obs_cube_v10[0, :, :]  # Subset the first time step

    # # Print the shapes of the cubes
    # print(f"Shape of obs_cube_u10: {obs_cube_u10.shape}")
    # print(f"Shape of obs_cube_v10: {obs_cube_v10.shape}")
    # print(f"Shape of obs_cube_u10_first: {obs_cube_u10_first.shape}")
    # print(f"Shape of obs_cube_v10_first: {obs_cube_v10_first.shape}")

    # # Calculate the wind speed from the data
    # # Calculate wind speed
    # windspeed_10m = (obs_cube_u10 ** 2 + obs_cube_v10 ** 2) ** 0.5
    # windspeed_10m.rename("si10")

    # # rename as obs cube
    # ERA5_cube = windspeed_10m

    # # If the test file path ERA5 exists, then load it
    # if os.path.exists(ERA5_file_path):
    #     print(f"Loading test ERA5 file from {ERA5_file_path}")
    #     ERA5_cube = iris.load_cube(ERA5_file_path)
    # else:
    #     print(f"Test ERA5 file not found at {ERA5_file_path}")

    # If the test file path dps exists, then load it
    if os.path.exists(test_dps_file_path):
        print(f"Loading test DPS file from {test_dps_file_path}")
        dps_cube = iris.load_cube(test_dps_file_path)
    else:
        print(f"Test DPS file not found at {test_dps_file_path}")

    # Intersect the u10 and v10 cubes
    obs_cube_u10 = obs_cube_u10.intersection(longitude=(-180, 180), latitude=(0, 90))
    obs_cube_v10 = obs_cube_v10.intersection(longitude=(-180, 180), latitude=(0, 90))

    dps_cube = dps_cube.intersection(longitude=(-180, 180), latitude=(0, 90))

    # Limit the depresys cube to the first time step
    dps_cube = dps_cube[0, :, :]

    # Find the min and max lon and lat for the obs cube
    min_lon_obs = np.min(obs_cube_u10.coord("longitude").points)
    max_lon_obs = np.max(obs_cube_u10.coord("longitude").points)
    min_lat_obs = np.min(obs_cube_u10.coord("latitude").points)
    max_lat_obs = np.max(obs_cube_u10.coord("latitude").points)

    # # Intersect to the European grid
    # obs_cube_u10 = obs_cube_u10.intersection(
    #     longitude=(eu_grid["lon1"], eu_grid["lon2"]),
    #     latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    # )
    # obs_cube_v10 = obs_cube_v10.intersection(
    #     longitude=(eu_grid["lon1"], eu_grid["lon2"]),
    #     latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    # )

    # Same for the depresys cube
    dps_cube = dps_cube.intersection(
        longitude=(min_lon_obs, max_lon_obs),
        latitude=(min_lat_obs, max_lat_obs),
    )

#    # Set up a figure with two subplots, sharing the same cbar
#     fig, ax = plt.subplots(
#         nrows=1,
#         ncols=2,
#         figsize=(12, 6),
#         subplot_kw={"projection": ccrs.PlateCarree()},
#     )

#     # Plot the u10 data
#     mesh_u10_original = ax[0].pcolormesh(
#         dps_cube.coord("longitude").points,
#         dps_cube.coord("latitude").points,
#         dps_cube.data[:, :],
#         cmap="viridis",
#         transform=ccrs.PlateCarree(),
#     )

#     ax[0].set_title("U10 Wind Speed (m/s) - Depresys")

#     ax[0].coastlines()
#     ax[0].set_extent(
#         [
#             eu_grid["lon1"],
#             eu_grid["lon2"],
#             eu_grid["lat1"],
#             eu_grid["lat2"],
#         ],
#         crs=ccrs.PlateCarree(),
#     )

#     # Plot the regridded u10 data
#     mesh_u10_regridded = ax[1].pcolormesh(
#         obs_cube_u10.coord("longitude").points,
#         obs_cube_u10.coord("latitude").points,
#         obs_cube_u10.data[0, :, :],
#         cmap="viridis",
#         transform=ccrs.PlateCarree(),
#     )

#     ax[1].set_title("U10 Wind Speed (m/s) - ERA5 U10")
#     ax[1].coastlines()

#     ax[1].set_extent(
#         [
#             eu_grid["lon1"],
#             eu_grid["lon2"],
#             eu_grid["lat1"],
#             eu_grid["lat2"],
#         ],
#         crs=ccrs.PlateCarree(),
#     )

#     # # Insert a colorbar for both plots
#     # cbar = fig.colorbar(
#     #     mesh_u10_original,
#     #     ax=ax,
#     #     orientation="horizontal",
#     #     fraction=0.02,
#     #     pad=0.1,
#     #     label="Wind Speed (m/s)",
#     # )
#     # cbar.set_ticks(np.arange(0, 30, 5))  # Set ticks for the colorbar
#     # cbar.ax.tick_params(labelsize=10)  # Set colorbar tick label size

#     # Set a seperate colorbar for each plot
#     cbar_u10_original = fig.colorbar(
#         mesh_u10_original,
#         ax=ax[0],
#         orientation="horizontal",
#         fraction=0.02,
#         pad=0.1,
#         label="Wind Speed (m/s)",
#     )

#     # Set up the otehr colorbar
#     cbar_u10_regridded = fig.colorbar(
#         mesh_u10_regridded,
#         ax=ax[1],
#         orientation="horizontal",
#         fraction=0.02,
#         pad=0.1,
#         label="Wind Speed (m/s)",
#     )

#     plt.tight_layout()

#     plt.show()

#     # sys.exit()

    # Calculate the wind speed from the data
    # Calculate wind speed
    # windspeed_10m = (obs_cube_u10 ** 2 + obs_cube_v10 ** 2) ** 0.5
    # ERA5_cube = windspeed_10m

    # Regrid the ERA5 data to the DePreSys grid
    # ERA5_cube_rg = ERA5_cube.regrid(dps_cube, iris.analysis.Linear())

    # # Print the min, max, and mean of the ERA5 data before regridding
    # print(f"U10 data min: {np.min(obs_cube_u10.data):.2f} m/s")
    # print(f"U10 data max: {np.max(obs_cube_u10.data):.2f} m/s")
    # print(f"U10 data mean: {np.mean(obs_cube_u10.data):.2f} m/s")
    # print(f"V10 data min: {np.min(obs_cube_v10.data):.2f} m/s")
    # print(f"V10 data max: {np.max(obs_cube_v10.data):.2f} m/s")
    # print(f"V10 data mean: {np.mean(obs_cube_v10.data):.2f} m/s")

    # Regrid the u10 and v10 data
    obs_cube_u10_rg = obs_cube_u10.regrid(dps_cube, iris.analysis.Linear())
    obs_cube_v10_rg = obs_cube_v10.regrid(dps_cube, iris.analysis.Linear())

    # Test the subsettting of these datasets
    # Subset the first time step of the u10 and v10 cubes
    u10_first_test = obs_cube_u10_rg[0, :, :]  # Subset the first time step
    v10_first_test = obs_cube_v10_rg[0, :, :]  # Subset the first time step

    # Print the shapes of the cubes
    print(f"Shape of obs_cube_u10: {obs_cube_u10_rg.shape}")
    print(f"Shape of obs_cube_v10: {obs_cube_v10_rg.shape}")

    # # Print the min, max, and mean of the u10 and v10 cubes
    # print(f"U10 data min: {np.min(obs_cube_u10_rg.data):.2f} m/s")
    # print(f"U10 data max: {np.max(obs_cube_u10_rg.data):.2f} m/s")
    # print(f"U10 data mean: {np.mean(obs_cube_u10_rg.data):.2f} m/s")
    # print(f"V10 data min: {np.min(obs_cube_v10_rg.data):.2f} m/s")
    # print(f"V10 data max: {np.max(obs_cube_v10_rg.data):.2f} m/s")
    # print(f"V10 data mean: {np.mean(obs_cube_v10_rg.data):.2f} m/s")

#    # Set up a figure with two subplots, sharing the same cbar
#     fig, ax = plt.subplots(
#         nrows=1,
#         ncols=2,
#         figsize=(12, 6),
#         subplot_kw={"projection": ccrs.PlateCarree()},
#     )

#     # Plot the u10 data
#     mesh_u10_original = ax[0].pcolormesh(
#         obs_cube_u10.coord("longitude").points,
#         obs_cube_u10.coord("latitude").points,
#         obs_cube_u10.data[0, :, :],
#         cmap="viridis",
#         transform=ccrs.PlateCarree(),
#     )

#     ax[0].set_title("U10 Wind Speed (m/s) - Original")

#     ax[0].coastlines()
#     ax[0].set_extent(
#         [
#             eu_grid["lon1"],
#             eu_grid["lon2"],
#             eu_grid["lat1"],
#             eu_grid["lat2"],
#         ],
#         crs=ccrs.PlateCarree(),
#     )

#     # Plot the regridded u10 data
#     mesh_u10_regridded = ax[1].pcolormesh(
#         obs_cube_u10_rg.coord("longitude").points,
#         obs_cube_u10_rg.coord("latitude").points,
#         obs_cube_u10_rg.data[0, :, :],
#         cmap="viridis",
#         transform=ccrs.PlateCarree(),
#     )

#     ax[1].set_title("U10 Wind Speed (m/s) - Regridded")
#     ax[1].coastlines()

#     ax[1].set_extent(
#         [
#             eu_grid["lon1"],
#             eu_grid["lon2"],
#             eu_grid["lat1"],
#             eu_grid["lat2"],
#         ],
#         crs=ccrs.PlateCarree(),
#     )

#     # Insert a colorbar for both plots
#     cbar = fig.colorbar(
#         mesh_u10_original,
#         ax=ax,
#         orientation="horizontal",
#         fraction=0.02,
#         pad=0.1,
#         label="Wind Speed (m/s)",
#     )
#     cbar.set_ticks(np.arange(0, 30, 5))  # Set ticks for the colorbar
#     cbar.ax.tick_params(labelsize=10)  # Set colorbar tick label size

#     plt.tight_layout()

#     plt.show()

#     sys.exit()

    # # Limit the ERA5 cube to the constrained region
    # ERA5_cube_rg = ERA5_cube_rg.intersection(
    #     longitude=constraint_lon,
    #     latitude=constraint_lat,
    # )

    # # If any values are negative, set them to zero
    # if np.any(ERA5_cube_rg.data < 0):
    #     print("Negative values found in ERA5 data. Setting them to zero.")
    #     ERA5_cube_rg.data[ERA5_cube_rg.data < 0] = 0

    # # print the min, max, and mean of the ERA5 data
    # print(f"ERA5 data min: {np.min(ERA5_cube_rg.data):.2f} m/s")
    # print(f"ERA5 data max: {np.max(ERA5_cube_rg.data):.2f} m/s")
    # print(f"ERA5 data mean: {np.mean(ERA5_cube_rg.data):.2f} m/s")

    # Load the power curves
    pc_winds, pc_power_ons, pc_power_ofs = load_power_curves(
        path_onshore_curve=onshore_pc_path, path_offshore_curve=offshore_pc_path
    )

    # # Check whether the bins in pc_winds are in ascending order
    # if not np.all(np.diff(pc_winds) > 0):
    #     raise ValueError("Wind speed bins in power curves must be in ascending order.")
    
    # # Check whether the bin intervals are evenly spaced
    # if not np.all(np.diff(pc_winds) == np.diff(pc_winds)[0]):
    #     raise ValueError("Wind speed bins in power curves must be evenly spaced.")

    # # Make a country mask for the UK
    MASK_MATRIX_RESHAPE, LONS, LATS = country_mask(
        dataset=obs_cube_u10_rg,
        COND="si10",
        COUNTRY="United Kingdom",
    )

    # sys.exit()

    # Tets the new function to process area weighted mean for wind farm locations
    regridded_farm_locations = process_area_weighted_mean(
        path_to_farm_locations_ons=path_to_farm_locations_ons,
        path_to_farm_locations_ofs=path_to_farm_locations_ofs,
        cube=obs_cube_u10_rg[0, :, :],  # Use the first time step of the regridded u10 cube
    )

    # # print the regridded farm locations
    # print(f"Regridded farm locations shape: {regridded_farm_locations.shape}")
    # print(f"Regridded farm locations min: {np.min(regridded_farm_locations.data):.2f}")
    # print(f"Regridded farm locations max: {np.max(regridded_farm_locations.data):.2f}")
    # print(
    #     f"Regridded farm locations mean: {np.mean(regridded_farm_locations.data):.2f}"
    # )

    # Load the wind speed and take it to hub height
    ERA5_cube_hubheight = load_wind_speed_and_take_to_hubheight(
        ERA5_cube=None,  # Use None since we are using the regridded u10 cube
        land_mask=MASK_MATRIX_RESHAPE,
        height_of_wind_speed=10.0,
        u_cube=obs_cube_u10_rg,  # Use the first time step of the regridded u10 cube
        v_cube=obs_cube_v10_rg,  # Use the first time step of the regridded v10 cube
    )

    # Print the ERA5 cube hub height
    print(f"ERA5 cube hub height shape: {ERA5_cube_hubheight}")

    # Convert the iris cube to an xarry dataarray
    ERA5_cube_hubheight_xr = xr.DataArray.from_iris(
        ERA5_cube_hubheight,
    )

    print(f"Type of ERA5 cube hub height: {type(ERA5_cube_hubheight_xr)}")
    print(f"Shape of ERA5 cube hub height: {ERA5_cube_hubheight_xr.shape}")
    print(f"Values of ERA5 cube hub height: {ERA5_cube_hubheight_xr}")

    # print the min, max, and mean of the ERA5 cube hub height
    print(f"ERA5 cube hub height min: {np.min(ERA5_cube_hubheight_xr.values):.2f} m/s")
    print(f"ERA5 cube hub height max: {np.max(ERA5_cube_hubheight_xr.values):.2f} m/s")
    print(f"ERA5 cube hub height mean: {np.mean(ERA5_cube_hubheight_xr.values):.2f} m/s")

    # Print the coords
    print(f"Coords of ERA5 cube hub height: {ERA5_cube_hubheight_xr.coords}")

    # Resample into daily data
    ERA5_cube_hubheight_daily = ERA5_cube_hubheight_xr.resample(
        valid_time="1D",  # Resample to daily data
    ).mean()

    # Print the daily resampled ERA5 cube hub height
    print(f"Daily resampled ERA5 cube hub height shape: {ERA5_cube_hubheight_daily.shape}")
    print(f"Daily resampled ERA5 cube hub height: {ERA5_cube_hubheight_daily}")

    # Convert back to iris cube
    ERA5_cube_hubheight = xr.DataArray.to_iris(
        ERA5_cube_hubheight_daily,
    )

    # Print the min, max, and mean of the ERA5 cube hub height after resampling
    print(f"Daily resampled ERA5 cube hub height min: {np.min(ERA5_cube_hubheight.data):.2f} m/s")
    print(f"Daily resampled ERA5 cube hub height max: {np.max(ERA5_cube_hubheight.data):.2f} m/s")
    print(f"Daily resampled ERA5 cube hub height mean: {np.mean(ERA5_cube_hubheight.data):.2f} m/s")

    # Print the type of the ERA5 cube hub height
    print(f"Type of ERA5 cube hub height: {type(ERA5_cube_hubheight)}")
    print(f"Shape of ERA5 cube hub height: {ERA5_cube_hubheight.shape}")
    print(f"Values of ERA5 cube hub height: {ERA5_cube_hubheight}")

    sys.exit()

    # Test the new function
    p_hh_total_GW = convert_wind_speed_to_power_generation(
        ERA5_cube_hubheight=ERA5_cube_hubheight,
        pc_winds=pc_winds,
        pc_power_ons=pc_power_ons,
        pc_power_ofs=pc_power_ofs,
        land_mask=MASK_MATRIX_RESHAPE,
        farm_locations=regridded_farm_locations.data,
    )

    # print the power generation
    print(f"Power generation shape: {p_hh_total_GW.shape}")
    print(f"Power generation min: {np.min(p_hh_total_GW):.2f} GW")
    print(f"Power generation max: {np.max(p_hh_total_GW):.2f} GW")
    print(f"Power generation mean: {np.mean(p_hh_total_GW):.2f} GW")
    print(f"Type of power generation: {type(p_hh_total_GW)}")

    # Extract the time series
    wp_cfs = p_hh_total_GW / 1000.0  # Convert to MW

    cfs = wp_cfs / (np.sum(regridded_farm_locations.data) / 1000000.0)  # Convert to CFs

    # Limit to the first time step
    wp_cfs_first = wp_cfs[0]

    # print the sum of this
    print(f"Sum of wind power generation: {np.sum(wp_cfs_first):.2f} GW")
    # print the shape of the wind power generation
    print(f"Shape of wind power generation: {wp_cfs_first.shape}")

    # Get the wind power generation time series
    wp_cf_ts = pd.DataFrame(
        data=wp_cfs,
        index=ERA5_cube_hubheight.coord("time").points,
        columns=["Wind Power Generation (GW)"],
    )

    # Add the capacity factor to the dataframe
    wp_cf_ts["Capacity Factor"] = cfs


    # Print the head of the dataframe
    print(wp_cf_ts.head())

    # Print the tail of the dataframe
    print(wp_cf_ts.tail())

    # print the stats of the dataframe
    print(f"Dataframe description:\n{wp_cf_ts.describe()}")

    # Save the dataframe to a csv file
    print("Saving the dataframe to a csv file")
    wp_cf_ts.to_csv(full_path)

    end_time = time.time()

    # Print the time taken to run the script
    print(f"Time taken to run the script: {end_time - start_time:.2f} seconds")

    return None


if __name__ == "__main__":
    # Start the main function
    main()

# %%
