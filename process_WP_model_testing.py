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

# Specific imports from local modules
from process_ERA5_wind_gen import (
    process_area_weighted_mean,
    convert_wind_speed_to_power_generation,
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
    test_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/HadGEM3-GC31-MM_sfcWind_UK_2018_DJF_day.npy"
    lats_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_UK_2018_DJF_day_lats.npy"
    lons_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_UK_2018_DJF_day_lons.npy"
    members_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_UK_2018_DJF_day_members.npy"
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

    # Set up the constraints
    constraint_lon = (-6.25, 4.58)  # degrees east
    constraint_lat = (50.28, 59.72)  # degrees north

    # Hard code the country to UK
    COUNTRY = "United Kingdom"

    # Load the test data
    test_data = np.load(test_file_path)

    lats = np.load(lats_file_path)
    lons = np.load(lons_file_path)
    members = np.load(members_file_path)

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

    sys.exit()

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
    capacity_factor = (
        total_gen_MW
        / (np.sum(
            rg_farm_locations.data) / 1000000.0)
    )

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

    # End the timer and print the execution time
    end_time = time.time()

    print(f"Execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
# %%
