#!/usr/bin/env python
"""
process_temp_gev.py
===================

This script processes the temperature extremes for fidelity testing via GEV fitting.

"""
# %%
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
from functions import sigmoid, dot_plot

# Silence warnings
warnings.filterwarnings("ignore")


# Define the main function
def main():
    # Start the timer
    start_time = time.time()

    # Set up the directory in which the dfs are stored
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"

    # # Set up the path to the NAO data
    # nao_path = "/home/users/benhutch/unseen_multi_year/dfs/nao_delta_p_indices_1975_2015.csv"

    # # Load the NAO df
    # df_nao = pd.read_csv(nao_path)

    # # print the head of the nao data
    # print(df_nao.head())

    # # print the tail of the nao data
    # print(df_nao.tail())

    # sys.exit()

    # Load the model temperature data
    df_model_tas = pd.read_csv(
        os.path.join(
            dfs_dir,
            "HadGEM3-GC31-MM_dcppA-hindcast_tas_United_Kingdom_1960-2018_day.csv",
        )
    )

    # Process the model temperature data for DJF
    df_model_tas_djf = select_leads_wyears_DJF(
        df=df_model_tas,
        wyears=np.arange(1, 11 + 1),
    )

    # Set up the effective dec year column
    df_model_tas_djf["effective_dec_year"] = df_model_tas_djf["init_year"] + (
        df_model_tas_djf["winter_year"] - 1
    )

    # Load the observed data
    df_obs_tas = pd.read_csv(
        os.path.join(dfs_dir, "ERA5_tas_United_Kingdom_1960-2018_daily_2024-11-26.csv")
    )

    # Convert the 'time' column to datetime, assuming it represents days since "1950-01-01 00:00:00"
    df_obs_tas["time"] = pd.to_datetime(
        df_obs_tas["time"], origin="1950-01-01", unit="D"
    )

    # subset the obs data to D, J, F
    df_obs_tas = df_obs_tas[df_obs_tas["time"].dt.month.isin([12, 1, 2])]

    # new column for temp in C
    df_obs_tas["data_c"] = df_obs_tas["data"] - 273.15

    # Make sure that the time column is a datetime
    df_obs_tas["time"] = pd.to_datetime(df_obs_tas["time"])

    # # make sure time is a datetime in the NAO df
    # df_nao["time"] = pd.to_datetime(df_nao["time"])

    # Apply the effective dec year to the df obs tas
    df_obs_tas["effective_dec_year"] = df_obs_tas.apply(
        lambda row: gev_funcs.determine_effective_dec_year(row), axis=1
    )

    # calculate the return period for ver cold days
    gev_funcs.plot_return_periods_decades_obs(
        obs_df=df_obs_tas,
        obs_var_name="data_c",
        decades=np.arange(1960, 2020, 10),
        title="Decadal RPs, 1961-2017, DJF, UK, ERA5",
        year_col_name="effective_dec_year",
        num_samples=1000,
        figsize=(10, 5),
        bad_min=True,
    )

    sys.exit()

    # # make time the index for the obs data
    # df_obs_tas.set_index("time", inplace=True)

    # # make time the index for the nao data
    # df_nao.set_index("time", inplace=True)

    # # join the dataframes
    # df_obs_tas = df_obs_tas.join(df_nao, how="inner")

    # # print the head of df obs tas
    # print(df_obs_tas.head())

    # # print the tail of df obs tas
    # print(df_obs_tas.tail())

    # # calculate the correlations in the df
    # print(df_obs_tas.corr())

    # # Quantify the obs block min and max
    # block_minima_obs_tas = gev_funcs.obs_block_min_max(
    #     df=df_obs_tas,
    #     time_name="effective_dec_year",
    #     min_max_var_name="data_c",
    #     new_df_cols=["nao_index", "delta_p_index"],
    #     process_min=True,
    # )

    # # print the head of the block minima obs tas
    # print(block_minima_obs_tas.head())

    # # print the tail of the block minima obs tas
    # print(block_minima_obs_tas.tail())

    # # print the correlations in the block minima obs tas
    # print(block_minima_obs_tas.corr())

    # sys.exit()

    # Set up the common winter years
    # NOTE: Exclude 1960 as only 10 members initialised in 1960
    # available for this year
    common_wyears = np.arange(1961, 2017 + 1)

    # Subset the model data to the common winter years
    df_model_tas_djf = df_model_tas_djf[
        df_model_tas_djf["effective_dec_year"].isin(common_wyears)
    ]

    # Subset the obs data to the common winter years
    df_obs_tas = df_obs_tas[df_obs_tas["effective_dec_year"].isin(common_wyears)]

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
    block_minima_model_tas["effective_dec_year"] = block_minima_model_tas[
        "init_year"
    ] + (block_minima_model_tas["winter_year"] - 1)

    # print the model df for lead 2
    print(block_minima_model_tas[block_minima_model_tas["winter_year"] == 2])

    # print the effective dec years in the model df
    block_minima_model_tas_winter_2 = block_minima_model_tas[
        block_minima_model_tas["winter_year"] == 2
    ]

    # print the unique efefctive dec years
    print(block_minima_model_tas_winter_2["effective_dec_year"].unique())

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

    # Plot the lead time trends
    gev_funcs.lead_time_trends(
        model_df=block_minima_model_tas,
        obs_df=block_minima_obs_tas,
        model_var_name="data_tas_c_min",
        obs_var_name="data_c_min",
        lead_name="winter_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends, 1961-2017, DJF block min T",
        figsize=(15, 5),
    )

    # print the model df for lead 2
    print(block_minima_model_tas[block_minima_model_tas["winter_year"] == 2])

    # Use a function to correct the lead time dependent trends
    block_minima_model_tas_lead_dt = gev_funcs.lead_time_trend_corr(
        model_df=block_minima_model_tas,
        x_axis_name="effective_dec_year",
        y_axis_name="data_tas_c_min",
        lead_name="winter_year",
    )

    # print the head of the dataframe
    print(block_minima_model_tas_lead_dt.head())

    # print the tail of the dataframe
    print(block_minima_model_tas_lead_dt.tail())

    # print the unique effective dec year in block minima model tas lead dt
    print(block_minima_model_tas_lead_dt["effective_dec_year"].unique())

    # print the number of Nans in the model data
    # for the data tas c min dt column
    print(block_minima_model_tas_lead_dt["data_tas_c_min_dt"].isnull().sum())

    # pviot detrend the obs data
    block_minima_obs_tas_dt = gev_funcs.pivot_detrend_obs(
        df=block_minima_obs_tas,
        x_axis_name="effective_dec_year",
        y_axis_name="data_c_min",
    )

    # Compare the lead time corrected trends
    gev_funcs.lead_time_trends(
        model_df=block_minima_model_tas_lead_dt,
        obs_df=block_minima_obs_tas_dt,
        model_var_name="data_tas_c_min_dt",
        obs_var_name="data_c_min_dt",
        lead_name="winter_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends, 1961-2017, DJF block min T",
        figsize=(15, 5),
    )

    # Compare the trends with the full field data
    gev_funcs.compare_trends(
        model_df_full_field=df_model_tas_djf,
        obs_df_full_field=df_obs_tas,
        model_df_block=block_minima_model_tas_lead_dt,
        obs_df_block=block_minima_obs_tas_dt,
        model_var_name_full_field="data_tas_c",
        obs_var_name_full_field="data_c",
        model_var_name_block="data_tas_c_min_dt",
        obs_var_name_block="data_c_min_dt",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends (block min detrended obs, model lead time detrended)",
        figsize=(15, 5),
        window_size=10,
        centred_bool=True,
        min_periods=1,
    )

    # perform the lead time depdent bias correction
    # for the block minima
    block_minima_model_tas_lead_dt_bc = gev_funcs.lead_time_mean_bias_correct(
        model_df=block_minima_model_tas_lead_dt,
        obs_df=block_minima_obs_tas_dt,
        model_var_name="data_tas_c_min_dt",
        obs_var_name="data_c_min_dt",
        lead_name="winter_year",
    )

    # Set effective dec year as a datetime in years
    block_minima_obs_tas_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_tas_dt["effective_dec_year"], format="%Y"
    )

    # Set this as the index
    block_minima_obs_tas_dt.set_index("effective_dec_year", inplace=True)

    # Do the same for the model data
    block_minima_model_tas_lead_dt_bc["effective_dec_year"] = pd.to_datetime(
        block_minima_model_tas_lead_dt_bc["effective_dec_year"], format="%Y"
    )

    # plot the dot plot for the detrended obs
    dot_plot(
        obs_df=block_minima_obs_tas_dt,
        model_df=block_minima_model_tas_lead_dt_bc,
        obs_val_name="data_c_min_dt",
        model_val_name="data_tas_c_min_dt_bc",
        model_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        title="Lead time detrended model bc, 1961-2017, DJF block min T",
        ylims=(-12, 6),
        solid_line=np.min,
        dashed_quant=0.20,
    )

    block_minima_model_tas_lead_dt_bc["effective_dec_year"] = block_minima_model_tas_lead_dt_bc[
        "effective_dec_year"
    ].dt.year.astype(int)

    # # Test teh function for decadal RPd
    gev_funcs.plot_return_periods_decades(
        model_df=block_minima_model_tas_lead_dt_bc,
        model_var_name="data_tas_c_min_dt_bc",
        obs_df=block_minima_obs_tas_dt,
        obs_var_name="data_c_min_dt",
        decades=np.arange(1960, 2020, 10),
        title="Decadal RPs, 1961-2017, DJF block min T",
    )

    sys.exit()

    # Apply the linear detrend to the observations for block minima
    block_minima_obs_tas_dt = gev_funcs.pivot_detrend_obs(
        df=block_minima_obs_tas,
        x_axis_name="effective_dec_year",
        y_axis_name="data_c_min",
    )

    # Apply the ensmean mean rolling mean detrend to the model data
    block_minima_model_tas_dt = gev_funcs.pivot_detrend_model_rolling(
        df=block_minima_model_tas,
        x_axis_name="effective_dec_year",
        y_axis_name="data_tas_c_min",
        window=10,
        centred_bool=True,
        min_periods=1,
    )

    # remove the linear trend from the model data
    block_minima_model_tas_dt_linear = gev_funcs.pivot_detrend_model(
        df=block_minima_model_tas,
        x_axis_name="effective_dec_year",
        y_axis_name="data_tas_c_min",
    )

    # Now compare the trends
    gev_funcs.compare_trends(
        model_df_full_field=df_model_tas_djf,  # FF not detrended
        obs_df_full_field=df_obs_tas,  # FF not detrended
        model_df_block=block_minima_model_tas_dt,  # Block minima detrended
        obs_df_block=block_minima_obs_tas_dt,  # Block minima detrended
        model_var_name_full_field="data_tas_c",
        obs_var_name_full_field="data_c",
        model_var_name_block="data_tas_c_min_rm_dt",
        obs_var_name_block="data_c_min_dt",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends (block min detrended obs linear, model rolling)",
        figsize=(15, 5),
        window_size=10,
    )

    # Now compare the trends
    gev_funcs.compare_trends(
        model_df_full_field=df_model_tas_djf,  # FF not detrended
        obs_df_full_field=df_obs_tas,  # FF not detrended
        model_df_block=block_minima_model_tas_dt_linear,  # Block minima detrended
        obs_df_block=block_minima_obs_tas_dt,  # Block minima detrended
        model_var_name_full_field="data_tas_c",
        obs_var_name_full_field="data_c",
        model_var_name_block="data_tas_c_min_dt",
        obs_var_name_block="data_c_min_dt",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends (block min detrended obs linear, model linear)",
        figsize=(15, 5),
        window_size=10,
    )

    # Bias correct the rolling mean model obs
    block_minima_model_tas_dt_bc = gev_funcs.lead_time_mean_bias_correct(
        model_df=block_minima_model_tas_dt,
        obs_df=block_minima_obs_tas_dt,
        model_var_name="data_tas_c_min_rm_dt",
        obs_var_name="data_c_min_dt",
        lead_name="winter_year",
    )

    # bias correct the linear detrend model obs
    block_minima_model_tas_dt_linear_bc = gev_funcs.lead_time_mean_bias_correct(
        model_df=block_minima_model_tas_dt_linear,
        obs_df=block_minima_obs_tas_dt,
        model_var_name="data_tas_c_min_dt",
        obs_var_name="data_c_min_dt",
        lead_name="winter_year",
    )

    # Set effective dec year as a datetime for the obs data
    block_minima_obs_tas_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_tas_dt["effective_dec_year"], format="%Y"
    )

    # Set this as the index
    block_minima_obs_tas_dt.set_index("effective_dec_year", inplace=True)

    # For the model set effective dec year as a datetime
    block_minima_model_tas_dt_bc["effective_dec_year"] = pd.to_datetime(
        block_minima_model_tas_dt_bc["effective_dec_year"], format="%Y"
    )

    # For the other model data set effective dec year as a datetime
    block_minima_model_tas_dt_linear_bc["effective_dec_year"] = pd.to_datetime(
        block_minima_model_tas_dt_linear_bc["effective_dec_year"], format="%Y"
    )

    # Plot the dot plot for the linear detrended model obs
    # bias corrected
    dot_plot(
        obs_df=block_minima_obs_tas_dt,
        model_df=block_minima_model_tas_dt_linear_bc,
        obs_val_name="data_c_min_dt",
        model_val_name="data_tas_c_min_dt_bc",
        model_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        title="Linear dt model bc, 1961-2017, DJF block min T",
        ylims=(-12, 6),
        solid_line=np.min,
        dashed_quant=0.20,
    )

    # Do the same but for the rolling mean model obs
    dot_plot(
        obs_df=block_minima_obs_tas_dt,
        model_df=block_minima_model_tas_dt_bc,
        obs_val_name="data_c_min_dt",
        model_val_name="data_tas_c_min_rm_dt_bc",
        model_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        title="Rolling mean dt model bc, 1961-2017, DJF block min T",
        ylims=(-12, 6),
        solid_line=np.min,
        dashed_quant=0.20,
    )

    # make sure efefctive dec year is an int
    block_minima_model_tas_dt_bc["effective_dec_year"] = block_minima_model_tas_dt_bc[
        "effective_dec_year"
    ].dt.year.astype(int)


    # print how long the script took
    print(f"Script took {time.time() - start_time:.2f} seconds")
    print("Script complete!")

# If name is main
if __name__ == "__main__":
    main()
# %%
