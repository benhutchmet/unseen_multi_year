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

    $ python process_WP_model_drift_dt.py

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

    # Set up the hard-coded 


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