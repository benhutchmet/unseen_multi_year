#!/usr/bin/env python

"""
process_dnw_gev.py
==================

This script processes daily obs and model data (all leads) into a dataframe containing demand net wind.

Methodology is still in development, so this script is a work in progress.

"""

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

# Define a function to set up the winter years
def select_leads_wyears_DJF(
    df: pd.DataFrame,
    wyears: list[int],
) -> pd.DataFrame:
    """
    Selects the leads for the winter years.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe containing the leads.
    wyears : list[int]
        The winter years.

    Returns
    -------
    pd.DataFrame
        The dataframe containing the leads for the winter years.
    """
    # Set up an empty dataframe to store the djf leads
    df_wyears = pd.DataFrame()

    # Loop through the winter years
    for i, wyear in enumerate(wyears):
        # Select the leads for the winter year
        leads = np.arange(31 + (i * 360), 31 + 90 + (i * 360))

        # Extract the data for these leads
        df_this = df[df["lead"].isin(leads)]

        # Include a new column for winter year
        df_this["winter_year"] = wyear

        # Concat to the new df
        df_wyears = pd.concat([df_wyears, df_this])

    return df_wyears


# Define the main function
def main():
    # Start the timer
    start = time.time()
    
    # Set up the directory in which the dfs are stored
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"

    # Load the model temperature data
    df_model_tas = pd.read_csv(
        os.path.join(dfs_dir, "HadGEM3-GC31-MM_dcppA-hindcast_tas_United_Kingdom_1960-2018_day.csv"
        )
    )

    # Load the model wind spped data
    df_model_sfcWind = pd.read_csv(
        os.path.join(dfs_dir, "HadGEM3-GC31-MM_dcppA-hindcast_sfcWind_UK_wind_box_1960-2018_day.csv"
        )
    )

    # Merge the two model dataframes on init_year, member, and lead
    df_model = df_model_tas.merge(df_model_sfcWind, on=["init_year", "member", "lead"], suffixes=("_tas", "_sfcWind"))

    # Subset the leads for the valid winter years
    winter_years = np.arange(1, 11 + 1)

    # Process the df_model_djf
    df_model_djf = select_leads_wyears_DJF(df_model, winter_years)

    # Add the column for effective dec year to the df_model_djf
    df_model_djf["effective_dec_year"] = df_model_djf["init_year"] + (df_model_djf["winter_year"] - 1)

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

    # Load the obs wind data
    df_obs_sfcWind = pd.read_csv(
        os.path.join(dfs_dir, "ERA5_sfcWind_UK_wind_box_1960-2018_daily_2025-02-26.csv")
    )

    # Convert the 'time' column to datetime, assuming it represents days since "1950-01-01 00:00:00"
    df_obs_wind["time"] = pd.to_datetime(df_obs_wind["time"], origin="1952-01-01", unit="D")

    # subset the obs data to D, J, F
    df_obs_wind = df_obs_wind[df_obs_wind["time"].dt.month.isin([12, 1, 2])]

    # Set time as the index for both dataframes
    df_obs_tas.set_index("time", inplace=True)
    df_obs_wind.set_index("time", inplace=True)

    # Join the two dataframes with suffixes
    df_obs = df_obs_tas.join(df_obs_wind, lsuffix="_tas", rsuffix="_sfcWind")


    return None

if __name__ == "__main__":
    main()