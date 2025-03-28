#!/usr/bin/env python
"""
process_temp_monthly.py
=======================

This script processes the seasonal DJF temperature extremes in the model
ensemble to see if there is a signal for changing extremes over time.
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
from functions import sigmoid, dot_plot, plot_distributions_fidelity

# Set up a function to process seasonal means for given winter years
def process_seasonal_means(
    model_df: pd.DataFrame,
    model_var_name: str,
    winter_years: np.array,
    season: str = "DJF", # hard coded to DJF for now
) -> pd.DataFrame:
    """
    This function processes the seasonal means for the given winter years
    
    Parameters
    ==========

    model_df: pd.DataFrame
        The DataFrame containing the model data
    model_var_name: str
        The name of the variable in the model_df
    winter_years: np.array
        The array of winter years to process
    season: str
        The season to process the data for (default is DJF)

    Returns
    =======

    seasonal_means_df: pd.DataFrame
        The DataFrame containing the seasonal means for the given winter years

    """

    # Assert that season is djf
    assert season == "DJF", "Season must be DJF"

    # Set up a new dataframe to store the data
    seasonal_means_df = pd.DataFrame()

    # Loop over the unique init_years
    for init_year_this in tqdm(model_df["init_year"].unique()):
        for member_this in model_df["member"].unique():
            # Subset the data
            data_this = model_df[
                (model_df["init_year"] == init_year_this)
                & (model_df["member"] == member_this)
            ]

            # Loop over the winter years
            for i, winter_year_this in enumerate(winter_years):
                # Set up the leads to extract
                # e.g. for i == 0, would be 2,3,4
                # for i == 1, would be 14,15,16
                leads_this = np.arange(
                    (i * 12) + 2, (i * 12) + 5
                )

                # Subset the data
                data_this_winter = data_this[
                    data_this["lead"].isin(leads_this)
                ]

                # Calculate the seasonal mean
                seasonal_mean_this = data_this_winter[model_var_name].mean()

                # Create a new df
                new_df = pd.DataFrame(
                    {
                        "init_year": [init_year_this],
                        "member": [member_this],
                        "winter_year": [winter_year_this],
                        "effective_dec_year": [init_year_this + (winter_year_this - 1)],
                        "seasonal_mean": [seasonal_mean_this],
                    }
                )

                # Concat this to the dataframe
                seasonal_means_df = pd.concat(
                    [seasonal_means_df, new_df],
                )

    return seasonal_means_df


# Set up the main function
def main():
    # Start a timer for profiling
    start_time = time.time()

    # Hard code the model data path
    model_data_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/HadGEM3-GC31-MM_tas_United Kingdom_DJF_1960_2018_dcppA-hindcast_Amon.csv"
    obs_data_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/ERA5_obs_tas_United Kingdom_DJF_1960_2018.csv"

    # Load the model tas
    model_df = pd.read_csv(model_data_path)
    obs_df = pd.read_csv(obs_data_path)

    # Pre-process the model data
    # limit obs df between 1960-11-31 and 2018-03-01
    obs_df = obs_df[(obs_df['time'] >= '1961-09-30') & (obs_df['time'] < '2018-03-01')]    

    # Set up a column for mont
    obs_df['month'] = pd.to_datetime(obs_df['time']).dt.month

    # Subset to the months DJF
    obs_df = obs_df[obs_df['month'].isin([12, 1, 2])]

    # make sure time is datetime
    obs_df['time'] = pd.to_datetime(obs_df['time'])

    # apply the determine effective dec year function to ths obs
    obs_df['effective_dec_year'] = obs_df.apply(
        lambda row: gev_funcs.determine_effective_dec_year(row), axis=1
    )

    # Convert tas to tas c
    model_df['tas_c'] = model_df['data'] - 273.15

    # Do the same for the obs
    obs_df['tas_c'] = obs_df['obs'] - 273.15

    # Test the function for processing the model data DJF
    model_df_seasonal = process_seasonal_means(
        model_df=model_df,
        model_var_name="tas_c",
        winter_years=np.arange(1, 11 + 1, 1),
        season="DJF",
    )

    # Process the obs data into seasonal mkean
    obs_df_seasonal = obs_df.groupby(['effective_dec_year']).mean().reset_index()

    # Hard code the effective dec years
    effective_dec_years = np.arange(1961, 2017 + 1, 1)

    # Subset the model data to the effective dec years
    model_df_seasonal = model_df_seasonal[
        model_df_seasonal["effective_dec_year"].isin(effective_dec_years)
    ]

    # Subset the obs data to the effective dec years
    obs_df_seasonal = obs_df_seasonal[
        obs_df_seasonal["effective_dec_year"].isin(effective_dec_years)
    ]

    # Compare the lead time trends between the obs and the model
    gev_funcs.lead_time_trends(
        model_df=model_df_seasonal,
        obs_df=obs_df_seasonal,
        model_var_name="seasonal_mean",
        obs_var_name="tas_c",
        lead_name="winter_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends UK, DJF seasonal means, 1961-2017",
        figsize=(15, 5),
    )

    # Correct the lead time dependent trends
    model_df_seasonal_trend_corr = gev_funcs.lead_time_trend_corr(
        model_df=model_df_seasonal,
        x_axis_name="effective_dec_year",
        y_axis_name="seasonal_mean",
        lead_name="winter_year",
    )

    # Correct the trend in the obs data
    obs_df_seasonal_trend_corr = gev_funcs.pivot_detrend_obs(
        df=obs_df_seasonal,
        x_axis_name="effective_dec_year",
        y_axis_name="tas_c",
    )

    # Now compare the trends again
    gev_funcs.lead_time_trends(
        model_df=model_df_seasonal_trend_corr,
        obs_df=obs_df_seasonal_trend_corr,
        model_var_name="seasonal_mean_dt",
        obs_var_name="tas_c_dt",
        lead_name="winter_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends UK, DJF seasonal means, 1961-2017",
        figsize=(15, 5),
    )

    # Compare the overall trends
    gev_funcs.compare_trends(
        model_df_full_field=model_df_seasonal,
        obs_df_full_field=obs_df_seasonal,
        model_df_block=model_df_seasonal_trend_corr,
        obs_df_block=obs_df_seasonal_trend_corr,
        model_var_name_full_field="seasonal_mean",
        obs_var_name_full_field="tas_c",
        model_var_name_block="seasonal_mean_dt",
        obs_var_name_block="tas_c_dt",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends UK, DJF seasonal means, 1961-2017",
        figsize=(15, 5),
        window_size=10,
        centred_bool=True,
        min_periods=1,
    )

    # Apply the lead time dpeendent mean bias correction
    model_df_seasonal_dt_bc = gev_funcs.lead_time_mean_bias_correct(
        model_df=model_df_seasonal_trend_corr,
        obs_df=obs_df_seasonal_trend_corr,
        model_var_name="seasonal_mean_dt",
        obs_var_name="tas_c_dt",
        lead_name="winter_year",
    )

    # Now do the fidelity testing of the four moments
    plot_distributions_fidelity(
        obs_df=obs_df_seasonal_trend_corr,
        model_df=model_df_seasonal_dt_bc,
        obs_val_name="tas_c_dt",
        model_val_name="seasonal_mean_dt_bc",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        model_member_name="member",
        model_lead_name="winter_year",
        title="UK DJF seasonal mean temperature, 1961-2017",
        nboot=1000,
        figsize=(10, 6),
        nbins=40,
        fname_prefix="UK_DJF_seasonal_mean_temp_1961_2017",
    )

    # Set effective dec year as a datetime in years
    # for the obs
    obs_df_seasonal_trend_corr['effective_dec_year'] = pd.to_datetime(obs_df_seasonal_trend_corr['effective_dec_year'], format='%Y')

    # Set this as the index
    obs_df_seasonal_trend_corr.set_index('effective_dec_year', inplace=True)

    # Set effective dec year as a datetime in years
    # for the model
    model_df_seasonal_dt_bc['effective_dec_year'] = pd.to_datetime(model_df_seasonal_dt_bc['effective_dec_year'], format='%Y')

    # Make the dot plot
    dot_plot(
        obs_df=obs_df_seasonal_trend_corr,
        model_df=model_df_seasonal_dt_bc,
        obs_val_name="tas_c_dt",
        model_val_name="seasonal_mean_dt_bc",
        model_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        title="UK DJF seasonal mean temperature, 1961-2017",
        ylims=(-5, 12),
        solid_line=np.min,
        dashed_quant=0.20,
    )

    # Make sure that effective dec year is an int in the model data
    model_df_seasonal_dt_bc['effective_dec_year'] = model_df_seasonal_dt_bc['effective_dec_year'].dt.year.astype(int)

    # Plot the decadal return periods for cold winters
    gev_funcs.plot_return_periods_decades(
        model_df=model_df_seasonal_dt_bc,
        model_var_name="seasonal_mean_dt_bc",
        obs_df=obs_df_seasonal_trend_corr,
        obs_var_name="tas_c_dt",
        decades=np.arange(1960, 2020, 10),
        title="Decadal RPs, cold winters ~ 1962"
    )

    # print the head of the obs df
    print(obs_df_seasonal.head())

    # print the tails of the obs df
    print(obs_df_seasonal.tail())

    # print the head of the model df seasonal
    print(model_df_seasonal.head())

    # print the tail of the model df seasonal
    print(model_df_seasonal.tail())

    # print the ampout of time taken
    print("Time taken: ", time.time() - start_time)

    # print that the script has finished
    print("Script finished")

    return None


# If name is main
if __name__ == "__main__":
    main()
    
    




# %%
