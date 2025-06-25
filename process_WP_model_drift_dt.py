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
    args = parser.parse_args()

    # Print the initialisation year we are processing
    print(f"Processing model data for initialisation year: {args.init_year}")

    # Set up the fname for the array data to load
    fname_year = f"HadGEM3-GC31-MM_sfcWind_Europe_{args.init_year}_DJF_day_drift_bc_anoms_1960-2018_dt.npy"

    fpath = os.path.join(subset_arrs_dir, fname_year)

    # If the file does not exist, raise an error
    if not os.path.exists(fpath):
        raise FileNotFoundError(
            f"The file {fpath} does not exist. Please check the path and try again."
        )
    
    # Load the numpy array data
    print(f"Loading data from {fpath}...")
    data = np.load(fpath)

    # Print the shape of the data
    print(f"Data shape: {data.shape}")

    # Print the min and max values of the data
    print(f"Data min: {np.min(data)}, Data max: {np.max(data)}")
    # Print the mean and std of the data
    print(f"Data mean: {np.mean(data)}, Data std: {np.std(data)}")

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