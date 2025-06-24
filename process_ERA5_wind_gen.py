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

# Third-party imports
import iris
import iris.analysis
import iris.analysis.cartography
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

# Specific imports
from tqdm import tqdm
from iris import cube

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

    # Get the area weights of the target drig
    wind_weights = iris.analysis.cartography.area_weights(
        cube=cube[0, :, :],
    )

    # Get the weights of the wind farm grid
    wf_weights = iris.analysis.cartography.area_weights(
        cube=combined_farm_locations,
    )

    # Get the wind farms total over the area
    wind_farm_total_over_area = combined_farm_locations / wf_weights

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

    # Set up the constraints
    constraint_lon = (-6.25, 4.58)  # degrees east
    constraint_lat = (50.28, 59.72)  # degrees north

    # Set up the fname
    fname = "ERA5_UK_wind_power_generation_cfs_constrained_1952_2020.csv"
    fpath = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/Hannah_wind"

    # if the directory does not exist, create it
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    # If the full path already exists, then exit
    full_path = os.path.join(fpath, fname)

    if os.path.exists(full_path):
        print(f"File {full_path} already exists. Exiting.")
        return None

    # If the test file path ERA5 exists, then load it
    if os.path.exists(ERA5_file_path):
        print(f"Loading test ERA5 file from {ERA5_file_path}")
        ERA5_cube = iris.load_cube(ERA5_file_path)
    else:
        print(f"Test ERA5 file not found at {ERA5_file_path}")

    # If the test file path dps exists, then load it
    if os.path.exists(test_dps_file_path):
        print(f"Loading test DPS file from {test_dps_file_path}")
        dps_cube = iris.load_cube(test_dps_file_path)
    else:
        print(f"Test DPS file not found at {test_dps_file_path}")

    # Intersect the cubes
    ERA5_cube = ERA5_cube.intersection(longitude=(-180, 180), latitude=(0, 90))
    dps_cube = dps_cube.intersection(longitude=(-180, 180), latitude=(0, 90))

    # Limit the depresys cube to the first time step
    dps_cube = dps_cube[0, :, :]

    # Intersect to the European grid
    ERA5_cube = ERA5_cube.intersection(
        longitude=(eu_grid["lon1"], eu_grid["lon2"]),
        latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    )

    # Same for the depresys cube
    dps_cube = dps_cube.intersection(
        longitude=(eu_grid["lon1"], eu_grid["lon2"]),
        latitude=(eu_grid["lat1"], eu_grid["lat2"]),
    )

    # Regrid the ERA5 data to the DePreSys grid
    ERA5_cube_rg = ERA5_cube.regrid(dps_cube, iris.analysis.Linear())

    # Limit the ERA5 cube to the constrained region
    ERA5_cube_rg = ERA5_cube_rg.intersection(
        longitude=constraint_lon,
        latitude=constraint_lat,
    )

    # If any values are negative, set them to zero
    if np.any(ERA5_cube_rg.data < 0):
        print("Negative values found in ERA5 data. Setting them to zero.")
        ERA5_cube_rg.data[ERA5_cube_rg.data < 0] = 0

    # print the min, max, and mean of the ERA5 data
    print(f"ERA5 data min: {np.min(ERA5_cube_rg.data):.2f} m/s")
    print(f"ERA5 data max: {np.max(ERA5_cube_rg.data):.2f} m/s")
    print(f"ERA5 data mean: {np.mean(ERA5_cube_rg.data):.2f} m/s")

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
    
    # sys.exit()

    # Make a country mask for the UK
    MASK_MATRIX_RESHAPE, LONS, LATS = country_mask(
        dataset=ERA5_cube_rg,
        COND="si10",
        COUNTRY="United Kingdom",
    )

    # Tets the new function to process area weighted mean for wind farm locations
    regridded_farm_locations = process_area_weighted_mean(
        path_to_farm_locations_ons=path_to_farm_locations_ons,
        path_to_farm_locations_ofs=path_to_farm_locations_ofs,
        cube=ERA5_cube_rg,
    )

    # print the regridded farm locations
    print(f"Regridded farm locations shape: {regridded_farm_locations.shape}")
    print(f"Regridded farm locations min: {np.min(regridded_farm_locations.data):.2f}")
    print(f"Regridded farm locations max: {np.max(regridded_farm_locations.data):.2f}")
    print(
        f"Regridded farm locations mean: {np.mean(regridded_farm_locations.data):.2f}"
    )

    # Load the wind speed and take it to hub height
    ERA5_cube_hubheight = load_wind_speed_and_take_to_hubheight(
        ERA5_cube=ERA5_cube_rg,
        land_mask=MASK_MATRIX_RESHAPE,
        height_of_wind_speed=10.0,
    )

    # print the ERA5 cube hub height
    print(f"ERA5 cube hub height shape: {ERA5_cube_hubheight.shape}")
    print(f"ERA5 cube hub height min: {np.min(ERA5_cube_hubheight.data):.2f} m/s")
    print(f"ERA5 cube hub height max: {np.max(ERA5_cube_hubheight.data):.2f} m/s")
    print(f"ERA5 cube hub height mean: {np.mean(ERA5_cube_hubheight.data):.2f} m/s")

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
