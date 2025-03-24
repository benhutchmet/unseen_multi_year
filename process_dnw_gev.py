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

# Local imports
import gev_functions as gev_funcs

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import (
    sigmoid,
)

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

# Define a function to convert 10m wind speed to UK wind power generation
def ws_to_wp_gen(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_ws_col: str,
    model_ws_col: str,
    ch_fpath: str = "/home/users/benhutch/unseen_multi_year/dfs/UK_clearheads_data_daily_1960_2018_ONDJFM.csv",
    onshore_cap: float = 15710.69,
    offshore_cap: float = 14733.02, # https://www.renewableuk.com/energypulse/ukwed/
    months: list[int] = [12, 1, 2],
    date_range: tuple[str] = ("1960-12-01", "2018-03-01"),
) -> pd.DataFrame:
    """
    Converts wind speed to wind power generation using a sigmoid fit as
    quantified from the observations.
    
    Parameters
    ==========
    
    obs_df : pd.DataFrame
        The dataframe containing the observed wind speed data.
    model_df : pd.DataFrame
        The dataframe containing the model wind speed data.
    obs_ws_col : str
        The name of the observed wind speed column.
    model_ws_col : str
        The name of the model wind speed column.
    ch_fpath : str
        The file path to the clear heads data.
    onshore_cap : float
        The onshore wind power capacity.
    offshore_cap : float
        The offshore wind power capacity.
    months : list[int]
        The months to consider.
    date_range : tuple[str]
        The date range to consider.
        
    Returns
    =======
    
    obs_df: pd.DataFrame
        The dataframe containing the observed wind power generation data.
    model_df: pd.DataFrame
        The dataframe containing the model wind power generation data.
    """

    # Load the clear heads data
    ch_df = pd.read_csv(ch_fpath)

    # Set up the installed capacities in GW
    onshore_cap_gw = onshore_cap / 1000
    offshore_cap_gw = offshore_cap / 1000

    # Set up the generation in CH
    ch_df["onshore_gen"] = ch_df["ons_cfs"] * onshore_cap_gw
    ch_df["offshore_gen"] = ch_df["ofs_cfs"] * offshore_cap_gw

    # Sum to give the total generation
    ch_df["total_gen"] = ch_df["onshore_gen"] + ch_df["offshore_gen"]

    # Make sure that date is a datetime
    ch_df["date"] = pd.to_datetime(ch_df["date"])

    # Set the date as the index and remove the title
    ch_df.set_index("date", inplace=True)

    # Subset the data to the relevant months
    ch_df = ch_df[ch_df.index.month.isin(months)]

    # Subset the data to the relevant date range
    ch_df = ch_df[(ch_df.index >= date_range[0]) & (ch_df.index <= date_range[1])]

    # Set up an initial guess for the parameters for the sigmoid fit
    p0 = [
        max(ch_df["total_gen"]),
        np.median(obs_df[obs_ws_col]),
        1,
        min(ch_df["total_gen"]),
    ]

    # Fit the sigmoid curve to the observed data
    popt, pcov = curve_fit(
        sigmoid, obs_df[obs_ws_col], ch_df["total_gen"], p0=p0, method="dogbox"
    )

    # Apply the sigmoid function to the observed data
    obs_df["sigmoid_total_wind_gen"] = sigmoid(obs_df[obs_ws_col], *popt)

    # Do the same for the model data
    model_df["sigmoid_total_wind_gen"] = sigmoid(model_df[model_ws_col], *popt)

    # If any of the sigmoid_total_wind_gen values are negative, set them to 0
    if any(obs_df["sigmoid_total_wind_gen"] < 0):
        print("Negative values in obs sigmoid_total_wind_gen, setting to 0")
        obs_df["sigmoid_total_wind_gen"] = obs_df["sigmoid_total_wind_gen"].clip(lower=0)

    if any(model_df["sigmoid_total_wind_gen"] < 0):
        print("Negative values in model sigmoid_total_wind_gen, setting to 0")
        model_df["sigmoid_total_wind_gen"] = model_df["sigmoid_total_wind_gen"].clip(lower=0)

    return obs_df, model_df

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
    df_obs_sfcWind["time"] = pd.to_datetime(df_obs_sfcWind["time"], origin="1952-01-01", unit="D")

    # subset the obs data to D, J, F
    df_obs_sfcWind = df_obs_sfcWind[df_obs_sfcWind["time"].dt.month.isin([12, 1, 2])]

    # Set time as the index for both dataframes
    df_obs_tas.set_index("time", inplace=True)
    df_obs_sfcWind.set_index("time", inplace=True)

    # Join the two dataframes with suffixes
    df_obs = df_obs_tas.join(df_obs_sfcWind, lsuffix="_tas", rsuffix="_sfcWind")

    # Reset the index of df_obs
    df_obs.reset_index(inplace=True)

    # Make sure that the time column is datetime
    df_obs["time"] = pd.to_datetime(df_obs["time"])

    # Apply the effective_dec_year to the df_obs
    df_obs["effective_dec_year"] = df_obs.apply(
        lambda row: gev_funcs.determine_effective_dec_year(row), axis=1
    )

    # Limit the obs data to the same years as the model data
    common_wyears = np.arange(1960, 2017 + 1)

    # Subset the obs data to the common_wyears
    df_obs = df_obs[df_obs["effective_dec_year"].isin(common_wyears)]

    # Subset the model data to the common_wyears
    df_model_djf = df_model_djf[df_model_djf["effective_dec_year"].isin(common_wyears)]

    # Create a new column for data_tas_c in df_model_full_djf
    df_model_djf["data_tas_c"] = df_model_djf["data_tas"] - 273.15

    # Apply the ws_to_wp_gen function to the obs and model data
    df_obs, df_model_djf = ws_to_wp_gen(
        obs_df=df_obs,
        model_df=df_model_djf,
        obs_ws_col="data_sfcWind",
        model_ws_col="data_sfcWind",
    )

    # print the head of the df_obs
    print(df_obs.head())

    # print the head of the df_model_djf
    print(df_model_djf.head())

    # print the time taken
    print(f"Time taken: {time.time() - start} seconds")

    return None

if __name__ == "__main__":
    main()