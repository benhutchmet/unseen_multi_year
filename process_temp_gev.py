#!/usr/bin/env python
"""
process_temp_gev.py
===================

This script processes the temperature extremes for fidelity testing via GEV fitting.

"""
#%%
# Local imports
import os
import sys
import glob
import time
import argparse
import warnings

# Third-party imports
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
import shapely.geometry
import cartopy.io.shapereader as shpreader
import iris
import cftime

# Specific imports
from tqdm import tqdm
from matplotlib import gridspec
from datetime import datetime, timedelta

from scipy.optimize import curve_fit
from scipy.stats import linregress, percentileofscore, gaussian_kde
from scipy.stats import genextreme as gev
from sklearn.metrics import mean_squared_error, r2_score
from iris.util import equalise_attributes

# Local imports
import gev_functions as gev_funcs
from process_dnw_gev import select_leads_wyears_DJF

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import (
    sigmoid, dot_plot
)

# Silence warnings
warnings.filterwarnings("ignore")

# Define the main function
def main():
    # Start the timer
    start_time = time.time()

    # Set up the directory in which the dfs are stored
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"

    # Load the model temperature data
    df_model_tas = pd.read_csv(
        os.path.join(dfs_dir, "HadGEM3-GC31-MM_dcppA-hindcast_tas_United_Kingdom_1960-2018_day.csv"
        )
    )

    # Process the model temperature data for DJF
    df_model_tas_djf = select_leads_wyears_DJF(
        df=df_model_tas,
        wyears=np.arange(1, 11 + 1),
    )

    # Set up the effective dec year column
    df_model_tas_djf["effective_dec_year"] = df_model_tas_djf["init_year"] + (df_model_tas_djf["winter_year"] - 1)

    # Load the observed data
    df_obs_tas = pd.read_csv(
        os.path.join(dfs_dir, "ERA5_tas_United_Kingdom_1960-2018_daily_2024-11-26.csv")
    )

    # Convert the 'time' column to datetime, assuming it represents days since "1950-01-01 00:00:00"
    df_obs_tas["time"] = pd.to_datetime(df_obs_tas["time"], origin="1950-01-01", unit="D")

    # subset the obs data to D, J, F
    df_obs_tas = df_obs_tas[df_obs_tas["time"].dt.month.isin([12, 1, 2])]

    # new column for temp in C
    df_obs_tas["data_c"] = df_obs_tas["data"] - 273.15

    # Make sure that the time column is a datetime
    df_obs_tas["time"] = pd.to_datetime(df_obs_tas["time"])

    # Apply the effective dec year to the df obs tas
    df_obs_tas["effective_dec_year"] = df_obs_tas.apply(
        lambda row: gev_funcs.determine_effective_dec_year(row), axis=1
    )

    # Set up the common winter years
    # NOTE: Exclude 1960 as only 10 members initialised in 1960
    # available for this year
    common_wyears = np.arange(1961, 2017 + 1)

    # Subset the model data to the common winter years
    df_model_tas_djf = df_model_tas_djf[
        df_model_tas_djf["effective_dec_year"].isin(common_wyears)
    ]

    # Subset the obs data to the common winter years
    df_obs_tas = df_obs_tas[
        df_obs_tas["effective_dec_year"].isin(common_wyears)
    ]

    # Create a new column for data tas c in df_model_full_djf
    df_model_tas_djf["data_tas_c"] = df_model_tas_djf["data"] - 273.15

    # Apply the block minima transform to the obs data
    block_minima_obs_tas = gev_funcs.obs_block_min_max(
        df=df_obs_tas,
        time_name="effective_dec_year",
        min_max_var_name="data_c",
        new_df_cols=[],
        process_min=True,
    )

    # Apply the block minima transform to the model data
    block_minima_model_tas = gev_funcs.model_block_min_max(
        df=df_model_tas_djf,
        time_name="init_year",
        min_max_var_name="data_tas_c",
        new_df_cols=[],
        winter_year="winter_year",
        process_min=True,
    )

    # Ensure effective dec year is in the block minima model tas
    block_minima_model_tas["effective_dec_year"] = block_minima_model_tas["init_year"] + (block_minima_model_tas["winter_year"] - 1)

    # Compare the trends
    gev_funcs.compare_trends(
        model_df_full_field=df_model_tas_djf,
        obs_df_full_field=df_obs_tas,
        model_df_block=block_minima_model_tas,
        obs_df_block=block_minima_obs_tas,
        model_var_name_full_field="data_tas_c",
        obs_var_name_full_field="data_c",
        model_var_name_block="data_tas_c_min",
        obs_var_name_block="data_c_min",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends (no bias correction or detrend)",
        figsize=(15, 5),
        window_size=10,
    )


    # Apply the lead time dependent mean bias correction for temperature
    

# If name is main
if __name__ == "__main__":
    main()
# %%
