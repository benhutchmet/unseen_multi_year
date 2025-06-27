# %%
"""
process_WP_model_drift_dt.py
=======================

Now we have processed the model data via drift and mean bias correction, and
also detrending, we can now process the 10m wind speed data into UK wind power
generation dataframes and consider whether the values are realistic/comparable
to those from the obs.

Usage:
------

    $ python process_WP_model_drift_dt.py --init_year <init_year>

Example Usage:
-------------

    $ python process_WP_model_drift_dt.py --init_year 1961

Arguments:
----------

    init_year : int : the initialisation year of the model data to process, e.g. 1961, 1971, 1981, 1991, 2001, 2011, 2021.

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

from process_WP_model_testing import (
    load_wind_speed_and_take_to_hubheight_model,
)

# # more specific imports from other modules
# from plot_dnw_circ import (
#     load_obs_data,
# )

# Imports from Hannah's functions
sys.path.append(
    "/home/users/benhutch/for_martin/for_martin/creating_wind_power_generation/"
)

# Importing the necessary functions from the module
from European_hourly_hub_height_winds_2023_model import (
    load_power_curves,
)

# Define the main function to process the model data
def main():
    # Set up a timer
    start_time = time.time()

    # Set up the hard-coded variables
    subset_arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/subset_WP/"

    # Set up the directory in which the model data is stored
    store_dir = "/home/users/benhutch/unseen_multi_year/split_data_anomalies/"

    # Set up the output df path
    output_df_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/model/WP_gen"

    test_dps_file_path = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/s1961-r9i1p1f2/day/sfcWind/gn/files/d20200417/sfcWind_day_HadGEM3-GC31-MM_dcppA-hindcast_s1961-r9i1p1f2_gn_19720101-19720330.nc"
    test_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/HadGEM3-GC31-MM_sfcWind_Europe_2018_DJF_day.npy"
    lats_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_Europe_2018_DJF_day_lats.npy"
    lons_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_Europe_2018_DJF_day_lons.npy"
    members_file_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata/HadGEM3-GC31-MM_sfcWind_Europe_2018_DJF_day_members.npy"

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

    # If the output df path does not exist, create it
    if not os.path.exists(output_df_path):
        os.makedirs(output_df_path)
        print(f"Created output directory: {output_df_path}")

    # Format country with _
    COUNTRY_str = COUNTRY.replace(" ", "_")

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Process model data for wind power generation in the UK."
    )
    parser.add_argument(
        "--init_year",
        type=int,
        required=True,
        help="The initialisation year of the model data to process, e.g. 1961, 1971, 1981, 1991, 2001, 2011, 2021.",
    )

    # Check if running in IPython
    if "ipykernel_launcher" in sys.argv[0]:
        # Manually set arguments for IPython
        args = parser.parse_args(["--init_year", "1961"])
        print("Running in IPython, using default arguments.")
        print(f"Using initialisation year: {args.init_year}")
    else:
        # Parse arguments normally
        args = parser.parse_args()

    # Set up the years
    years = np.arange(1960, 2018 + 1, 1)

    # Loop over the years
    for year_this in tqdm(years):
        # Print the initialisation year we are processing
        print(f"Processing model data for initialisation year: {year_this}")

        # Set up the output path for the model data
        output_model_data_path = os.path.join(
            output_df_path,
            f"HadGEM3-GC31-MM_WP_gen_{COUNTRY_str}_{year_this}_drift_bc_no_dt.csv",
        )

        # If the output model data path already exists, remove it
        if os.path.exists(output_model_data_path):
            print(
                f"Output model data path {output_model_data_path} already exists."
            )
            continue

        # Set up the year_no_this
        year_no_this = (int(year_this) - 1960) + 1

        # Set up the fname for the array data to load
        fname_year = f"data_anoms_plus_obs_year_{year_no_this}_no_dt.npy"

        fpath = os.path.join(store_dir, fname_year)

        # If the file does not exist, raise an error
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"The file {fpath} does not exist. Please check the path and try again."
            )
        
        # Load the numpy array data
        print(f"Loading data from {fpath}...")
        data = np.load(fpath)

        # Load the latitudes and longitudes
        print(f"Loading latitudes from {lats_file_path}...")
        lats = np.load(lats_file_path)
        print(f"Loading longitudes from {lons_file_path}...")
        lons = np.load(lons_file_path)
        print(f"Loading members from {members_file_path}...")
        members = np.load(members_file_path)

        # Print the shape of the data
        print(f"Data shape: {data.shape}")

        # Print the min and max values of the data
        print(f"Data min: {np.min(data)}, Data max: {np.max(data)}")
        # Print the mean and std of the data
        print(f"Data mean: {np.mean(data)}, Data std: {np.std(data)}")

        # Load the power curves
        print("Loading power curves...")
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

        # Load the cibe
        print(f"Loading test cube from {test_dps_file_path}...")
        test_cube = iris.load_cube(test_dps_file_path)
        subset_cube_locs = test_cube[0, :, :]

        # Process the area weighted means for the wind farm locations
        rg_farm_locations = process_area_weighted_mean(
            path_to_farm_locations_ons=path_to_farm_locations_ons,
            path_to_farm_locations_ofs=path_to_farm_locations_ofs,
            cube=subset_cube_locs,
        )

        # Find the min and max latitudes and longitudes
        min_lat, max_lat = np.min(lats), np.max(lats)
        min_lon, max_lon = np.min(lons), np.max(lons)

        # Print the min and max latitudes and longitudes
        print(f"Min latitude: {min_lat}, Max latitude: {max_lat}")
        print(f"Min longitude: {min_lon}, Max longitude: {max_lon}")

        # Intersect the rg_farm_locations to this region
        rg_farm_locations = rg_farm_locations.intersection(
            longitude=(min_lon, max_lon),
            latitude=(min_lat, max_lat),
        )

        # Load the wind speed data and take to hubheight
        ws_hh = load_wind_speed_and_take_to_hubheight_model(
            model_data=data,
            land_mask=MASK_MATRIX_RESHAPE,
            height_of_wind_speed=10.0,  # 10m wind speed
        )

        # Count the number of values which are 0.0
        num_zero_values = np.sum(ws_hh == 0.0)
        print(f"Number of zero values in wind speed data: {num_zero_values}")

        # Print the % of values which are 0.0
        total_values = ws_hh.size
        percent_zero_values = (num_zero_values / total_values) * 100
        print(f"Percentage of zero values in wind speed data: {percent_zero_values:.2f}%")

        # Print the shape of the wind speed data
        print(f"Wind speed data shape: {ws_hh.shape}")
        # Print the min and max values of the wind speed data
        print(f"Wind speed data min: {np.min(ws_hh)}, Wind speed data max: {np.max(ws_hh)}")
        # Print the mean and std of the wind speed data
        print(f"Wind speed data mean: {np.mean(ws_hh)}, Wind speed data std: {np.std(ws_hh)}")

        # Print the shape of wind speeds at hub height
        print(f"Wind speeds at hub height shape: {ws_hh.shape}")

        # Convert the wind speed data to power generation data
        p_hh_total_GW = convert_wind_speed_to_power_generation(
            ERA5_cube_hubheight=ws_hh,
            pc_winds=pc_winds,
            pc_power_ons=pc_power_ons,
            pc_power_ofs=pc_power_ofs,
            land_mask=MASK_MATRIX_RESHAPE,
            farm_locations=rg_farm_locations.data,
        )

        # Print the shape of the power generation data
        print(f"Power generation data shape: {p_hh_total_GW.shape}")

        # Print the type of the power generation data
        print(f"Power generation data type: {type(p_hh_total_GW)}")

        # Extract the data
        p_hh_total_GW_vals = np.array(p_hh_total_GW.data)

        # Convert form GW to MW
        total_gen_MW = p_hh_total_GW_vals / 1000.0

        # Get the capacity factor
        capacity_factors = (
            total_gen_MW / (np.sum(rg_farm_locations.data) / 1000000.0)
        )

        # Print the shape of the capacity factors
        print(f"Capacity factors shape: {capacity_factors.shape}")

        # Print the min and max values of the capacity factors
        print(f"Capacity factors min: {np.min(capacity_factors)}, "
                f"Capacity factors max: {np.max(capacity_factors)}")
        # Print the mean and std of the capacity factors
        print(f"Capacity factors mean: {np.mean(capacity_factors)}, "
                f"Capacity factors std: {np.std(capacity_factors)}")
        
        # Set up a model df to store the data in
        model_df = pd.DataFrame()

        # Fix the loop dimensions based on the actual structure of capacity_factors
        for i_member, member in enumerate(members):
            for i_day in range(capacity_factors.shape[1]):  # Use the second dimension of capacity_factors
                for i_wyear in range(capacity_factors.shape[2]):  # Use the third dimension of capacity_factors
                    # Extract the model values this
                    cf_this = capacity_factors[i_member, i_day, i_wyear]

                    # Set up the df this
                    model_df_this = pd.DataFrame(
                        {
                            "member": [i_member + 1],  # Use i_member + 1 to represent the member
                            "lead": [i_day + 1],  # Use i_day + 1 to represent the day
                            "wyear": [i_wyear + 1],  # Use i_wyear + 1 to represent the winter year
                            "capacity_factor": [cf_this],
                        }
                    )

                    # Concat this to the model df
                    model_df = pd.concat([model_df, model_df_this])
        
        # Inclue an additional column for the initialisation year
        model_df["init_year"] = year_this
        
        # Print the head of the model df
        print("Model DataFrame head:")
        print(model_df.head())

        # Print the tail of the model df
        print("Model DataFrame tail:")
        print(model_df.tail())

        # describe the model df
        print("Model DataFrame description:")
        print(model_df.describe())

        # Save the model df to a csv file
        print(f"Saving model DataFrame to {output_model_data_path}...")
        model_df.to_csv(output_model_data_path, index=False)

    # Set up an end timer
    end_time = time.time()

    # Print the total time taken
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

    return None

if __name__ == "__main__":
    # Run the main function
    main()

    # Print a message indicating that the script has finished running
    print("Script finished running successfully.")
# %%
