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
import seaborn as sns

# Specific imports
from tqdm import tqdm
from matplotlib import gridspec
from datetime import datetime, timedelta
from matplotlib import cm
from scipy.optimize import curve_fit
from scipy.stats import linregress, percentileofscore, gaussian_kde
from scipy.stats import genextreme as gev
from sklearn.metrics import mean_squared_error, r2_score
from iris.util import equalise_attributes

# Local imports
import gev_functions as gev_funcs
# from process_dnw_gev import select_leads_wyears_DJF

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import sigmoid, dot_plot

# Silence warnings
warnings.filterwarnings("ignore")


# Set up a function to perform the model drift correction
def model_drift_corr_plot(
    model_df: pd.DataFrame,
    model_var_name: str,
    obs_df: pd.DataFrame,
    obs_var_name: str,
    lead_name: str,
    xlabel: str = "Temperature (C)",
    init_years_name: str = "init_year",
    eff_dec_years_name: str = "effective_dec_year",
    figsize: tuple = (10, 5),
    year1_year2_tuple: tuple = (1970, 2017),  # Constant forecast period
    lead_day_name: str = None,
    constant_period: bool = False,
) -> None:
    """
    Performs model drift correction by calculating anomalies
    as in Appendix E of doi: http://www.geosci-model-dev.net/9/3751/2016/

    Args:
    ======

    model_df : pd.DataFrame
        DataFrame of model data.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    lead_name : str
        Name of the column to use as the lead time axis in the model DataFrame.
    init_years_name : str
        Name of the column to use as the initial years axis in the model DataFrame.
    eff_dec_years_name : str
        Name of the column to use as the effective decadal years axis in the model DataFrame.
    figsize : tuple, optional
        Figure size, by default (10, 5).
    year1_year2_tuple : tuple, optional
        Tuple of years to use for the model drift correction, by default (1970, 2007).
    lead_day_name : str, optional
        Name of the column to use as the lead time in days, by default None.

    Returns:
    =======

    None

    """

    # Set up a copy of the model df
    model_df_copy = model_df.copy()
    obs_df_copy = obs_df.copy()

    model_df_copy = model_df_copy.reset_index(drop=True)
    obs_df_copy = obs_df_copy.reset_index(drop=True)

    effective_dec_years_constant = np.arange(
        year1_year2_tuple[0], year1_year2_tuple[1] + 1, 1
    )

    # Get the unique lead times
    unique_leads = sorted(model_df_copy[lead_name].unique())

    # Loop over the unique leads
    for i, lead in tqdm(
        enumerate(unique_leads),
        desc="Processing lead times",
        total=len(unique_leads),
        leave=False,
    ):
        # Subset the model data to the lead this
        model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

        # Set up the unique init years full
        unique_init_years_full = model_df_lead_this[init_years_name].unique()

        # obs_effective_dec_years_this = model_df_lead_this[eff_dec_years_name].unique()

        # # Subset the obs to this period
        # obs_df_lead_this = obs_df_copy[
        #     obs_df_copy[eff_dec_years_name].isin(obs_effective_dec_years_this)
        # ]

        # # Get the mean of the obs data
        # obs_mean_lead_this = obs_df_lead_this[obs_var_name].mean()

        # # limit to the effective dec years in the year 1 to year 2 period
        # model_df_lead_this = model_df_lead_this[
        #     model_df_lead_this[eff_dec_years_name].isin(effective_dec_years_constant)
        # ]

        # Extract the unique init years in this case
        unique_init_years = model_df_lead_this[init_years_name].unique()
        unique_members = model_df_lead_this["member"].unique()

        # if unique members does not have length 10 then raise an error
        if len(unique_members) != 10:
            raise ValueError(
                f"Unique members does not have length 10: {unique_members}"
            )

        if constant_period:
            # Set up the unique init years to be the effective dec years
            unique_init_years = effective_dec_years_constant

        # Set up an array to append the ensemble means to
        ensemble_means_this = np.zeros([len(unique_init_years)])

        # Loop over the unique effective dec years
        for j, eff_dec_year in enumerate(unique_init_years):
            # print(f"i: {i}")
            # print(f"j: {j}")
            # Subset the model data to the init year this
            model_df_lead_this_init = model_df_copy[
                (model_df_copy[eff_dec_years_name] == eff_dec_year)
                & (model_df_copy[lead_name] == lead)
            ]

            # if the df is empty
            # then raise an error
            if model_df_lead_this_init.empty:
                print(f"eff_dec_year: {eff_dec_year}")
                print(f"lead: {lead}")
                raise ValueError(
                    f"Model df lead this init is empty: {model_df_lead_this_init}"
                )

            # print(f"effective dec year: {eff_dec_year}")
            # # print the first row of model_df_lead_this_init
            # print(model_df_lead_this_init.head(1))

            # Calculate the ensemble mean this
            ensemble_mean_val_this = model_df_lead_this_init[model_var_name].mean()

            # Append the ensemble mean to the array
            ensemble_means_this[j] = ensemble_mean_val_this

        # if the ensemble means this does not have the same length
        # as effective dec years constant
        # then raise an error
        if len(ensemble_means_this) != len(unique_init_years):
            print(f"lead: {lead}")
            print(f"ensemble means this: {ensemble_means_this}")
            print(f"effective dec years constant: {effective_dec_years_constant}")

            # print the shape of the ensemble means this
            print(f"Shape of ensemble means this: {ensemble_means_this.shape}")

            # print the shape of the effective dec years constant
            print(f"Shape of effective dec years constant: {effective_dec_years_constant.shape}")

            raise ValueError(
                f"Ensemble means this does not have the same length as effective dec years constant"
            )

        # Calculate the mean of the ensemble means - forecast climatology
        forecast_clim_lead_this = np.mean(ensemble_means_this)

        # print the lead and the forecast climatology lead this
        print(f"lead: {lead} - forecast climatology lead this: {forecast_clim_lead_this}")

        # if the forecast clim lead this is nan
        if np.isnan(forecast_clim_lead_this):
            print(f"lead: {lead} - forecast climatology lead this is nan")
            print(f"first unique init years: {unique_init_years[0]}")
            print(f"last unique init years: {unique_init_years[-1]}")

            # print the ensemble means this
            print(f"Ensemble means this: {ensemble_means_this}")

        if lead_day_name is not None:
            print("Calculating anoms for days within a winter year")

            # Update only the rows corresponding to the current lead
            model_df_copy.loc[
                model_df_copy[lead_name] == lead, f"{model_var_name}_anomaly"
            ] = (
                model_df_copy.loc[model_df_copy[lead_name] == lead, model_var_name]
                - forecast_clim_lead_this
            )
        else:
            # Update only the rows corresponding to the current lead
            model_df_copy.loc[
                model_df_copy[lead_name] == lead, f"{model_var_name}_anomaly"
            ] = (
                model_df_copy.loc[model_df_copy[lead_name] == lead, model_var_name]
                - forecast_clim_lead_this
            )

    # Subset the model df copy to the effective dec years in the obs data
    model_df_copy_constant = model_df_copy[
        model_df_copy[eff_dec_years_name].isin(effective_dec_years_constant)
    ]

    # Set up the axes
    # Set up the figure size
    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=figsize,
        sharex=True,
        sharey=True,
        layout="compressed",
    )

    # Loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Subset the model data to the lead this
        model_df_lead_this = model_df_copy_constant[model_df_copy_constant[lead_name] == lead]

        # # print the lead
        # print(f"Lead time: {lead}")

        # # print the first and last unique effective dec years in this df
        # print(f"First unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[0]}")
        # print(f"Last unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[-1]}")

        # # print the len of the unique effective dec years in this df
        # print(f"Length of unique effective dec years: {len(model_df_lead_this[eff_dec_years_name].unique())}")

        # calculate the mean
        model_mean_this = model_df_lead_this[f"{model_var_name}"].mean()

        # if the omodel mean is nan
        if np.isnan(model_mean_this):
            print(model_df_lead_this)

        # include the mean in the title
        title = f"Lead {lead} - Model mean: {model_mean_this:.2f}"

        # Plot the data
        ax = axes.flatten()[i]

        # Plot the histograms using matplotlib
        ax.hist(
            model_df_lead_this[f"{model_var_name}"], 
            bins=30, 
            color="red", 
            edgecolor="black"
        )

        # include the title
        ax.set_title(title)

    # Add a suptitle including the min and max unique effective dec years
    min_eff_dec_year = model_df_copy_constant[eff_dec_years_name].min()
    max_eff_dec_year = model_df_copy_constant[eff_dec_years_name].max()
    plt.suptitle(
        f"Model raw block minima - {min_eff_dec_year} to {max_eff_dec_year}",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )

    # Set up the axes
    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=figsize,
        sharex=True,
        sharey=True,
        layout="compressed",
    )

    # loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Subset the model data to the lead this
        model_df_lead_this = model_df_copy_constant[model_df_copy_constant[lead_name] == lead]

        # print the lead
        print(f"Lead time: {lead}")

        # print the first and last unique effective dec years in this df
        print(f"First unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[0]}")
        print(f"Last unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[-1]}")

        # print the len of the unique effective dec years in this df
        print(f"Length of unique effective dec years: {len(model_df_lead_this[eff_dec_years_name].unique())}")

        # calculate the mean
        model_mean_this = model_df_lead_this[f"{model_var_name}_anomaly"].mean()

        # if the omodel mean is nan
        if np.isnan(model_mean_this):
            print(model_df_lead_this)

        # include the mean in the title
        title = f"Lead {lead} - Model mean: {model_mean_this:.2f}"

        # Plot the data
        ax = axes.flatten()[i]

        # Plot the histograms using matplotlib
        ax.hist(
            model_df_lead_this[f"{model_var_name}_anomaly"], 
            bins=30, 
            color="red", 
            edgecolor="black"
        )

        # include the title
        ax.set_title(title)

    # Add a suptitle including the min and max unique effective dec years
    min_eff_dec_year = model_df_copy_constant[eff_dec_years_name].min()
    max_eff_dec_year = model_df_copy_constant[eff_dec_years_name].max()

    plt.suptitle(
        f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )

    # do the same but not for a constant period
    # Set up the axes
    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=figsize,
        sharex=True,
        sharey=True,
        layout="compressed",
    )

    # loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Subset the model data to the lead this
        model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

        # print the lead
        # print(f"Lead time: {lead}")

        # # print the first and last unique effective dec years in this df
        # print(f"First unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[0]}")
        # print(f"Last unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[-1]}")

        # print the len of the unique effective dec years in this df
        print(f"Length of unique effective dec years: {len(model_df_lead_this[eff_dec_years_name].unique())}")

        # calculate the mean
        model_mean_this = model_df_lead_this[f"{model_var_name}_anomaly"].mean()

        # if the omodel mean is nan
        if np.isnan(model_mean_this):
            print(model_df_lead_this)

        # include the mean in the title
        title = f"Lead {lead} - Model mean: {model_mean_this:.2f}"

        # Plot the data
        ax = axes.flatten()[i]

        # Plot the histograms using matplotlib
        ax.hist(
            model_df_lead_this[f"{model_var_name}_anomaly"], 
            bins=30, 
            color="red", 
            edgecolor="black"
        )

        # include the title
        ax.set_title(title)

    # Add a suptitle including the min and max unique effective dec years
    min_eff_dec_year = model_df_copy[eff_dec_years_name].min()
    max_eff_dec_year = model_df_copy[eff_dec_years_name].max()
    plt.suptitle(
        f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )

    # Set up another figure
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        layout="compressed",
    )

    ax1 = axes[0]
    ax2 = axes[1]

    # Get the cmap
    cmap = cm.get_cmap("Blues", len(unique_leads))

    # Set up an array for means
    raw_means = np.zeros([len(unique_leads)])
    drift_bc_means = np.zeros([len(unique_leads)])

    # Loop over the leads
    for i, lead in enumerate(unique_leads):
        # Subset the model data to the lead this
        model_df_lead_this = model_df_copy_constant[model_df_copy_constant[lead_name] == lead]

        # Plot the density distribution with kde
        sns.kdeplot(
            model_df_lead_this[f"{model_var_name}"],
            ax=ax1,
            color=cmap(i),
        )

        # Calculate the mean
        raw_mean_this = model_df_lead_this[f"{model_var_name}"].mean()

        # Append the mean to the array
        raw_means[i] = raw_mean_this

        # Plot the density distribution with kde
        sns.kdeplot(
            model_df_lead_this[f"{model_var_name}_anomaly"],
            ax=ax2,
            label=f"Lead {lead}",
            color=cmap(i),
        )

        # Calculate the mean
        drift_bc_mean_this = model_df_lead_this[f"{model_var_name}_anomaly"].mean()

        # Append the mean to the array
        drift_bc_means[i] = drift_bc_mean_this

    # Calculate pairwise mean differences for raw means
    raw_mean_differences = [
        abs(raw_means[i] - raw_means[j])
        for i in range(len(raw_means))
        for j in range(i + 1, len(raw_means))
    ]
    raw_mean_difference_avg = np.mean(raw_mean_differences)

    # Calculate pairwise mean differences for drift corrected means
    drift_bc_mean_differences = [
        abs(drift_bc_means[i] - drift_bc_means[j])
        for i in range(len(drift_bc_means))
        for j in range(i + 1, len(drift_bc_means))
    ]
    drift_bc_mean_difference_avg = np.mean(drift_bc_mean_differences)

    # Set the titles
    ax1.set_title(f"Raw model data (mean diff = {raw_mean_difference_avg:.2f})")
    ax2.set_title(f"Drift corrected anomalies (mean diff = {drift_bc_mean_difference_avg:.2f})")

    # Include a legend in the top right of the right plot
    ax2.legend(
        loc="upper right",
        fontsize=8,
    )

    # include a sup title
    min_eff_dec_year = model_df_copy_constant[eff_dec_years_name].min()
    max_eff_dec_year = model_df_copy_constant[eff_dec_years_name].max()
    plt.suptitle(
        f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )

    # # Set up the axes
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        layout="compressed",
    )

    ax1 = axes[0]
    ax2 = axes[1]

    # Get the cmap
    cmap = cm.get_cmap("Blues", len(unique_leads))

    # Set up an array for means
    raw_means = np.zeros([len(unique_leads)])
    drift_bc_means = np.zeros([len(unique_leads)])

    # loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Subset the model data to the lead this
        model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

        # Plot the density distribution with kde
        sns.kdeplot(
            model_df_lead_this[f"{model_var_name}"],
            ax=ax1,
            color=cmap(i),
        )

        # Calculate the mean
        raw_mean_this = model_df_lead_this[f"{model_var_name}"].mean()

        # Append the mean to the array
        raw_means[i] = raw_mean_this

        # Plot the density distribution with kde
        sns.kdeplot(
            model_df_lead_this[f"{model_var_name}_anomaly"],
            ax=ax2,
            label=f"Lead {lead}",
            color=cmap(i),
        )

        # Calculate the mean
        drift_bc_mean_this = model_df_lead_this[f"{model_var_name}_anomaly"].mean()

        # Append the mean to the array
        drift_bc_means[i] = drift_bc_mean_this

    # Calculate pairwise mean differences for raw means
    raw_mean_differences = [
        abs(raw_means[i] - raw_means[j])
        for i in range(len(raw_means))
        for j in range(i + 1, len(raw_means))
    ]
    raw_mean_difference_avg = np.mean(raw_mean_differences)

    # Calculate pairwise mean differences for drift corrected means
    drift_bc_mean_differences = [
        abs(drift_bc_means[i] - drift_bc_means[j])
        for i in range(len(drift_bc_means))
        for j in range(i + 1, len(drift_bc_means))
    ]
    drift_bc_mean_difference_avg = np.mean(drift_bc_mean_differences)

    # Set the titles
    ax1.set_title(f"Raw model data (mean diff = {raw_mean_difference_avg:.2f})")
    ax2.set_title(f"Drift corrected anomalies (mean diff = {drift_bc_mean_difference_avg:.2f})")

    # Add a suptitle including the min and max unique effective dec years
    min_eff_dec_year = model_df_copy[eff_dec_years_name].min()
    max_eff_dec_year = model_df_copy[eff_dec_years_name].max()
    plt.suptitle(
        f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )

    # loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Extract the unique effective dec years in this case
        model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

        # Extract the unique effective dec years
        unique_eff_dec_years_this = model_df_lead_this[eff_dec_years_name].unique()

        # print the first and last unique effective dec years in this df
        print(f"First unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[0]}")
        print(f"Last unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[-1]}")

        # print the len of the unique effective dec years in this df
        print(f"Length of unique effective dec years: {len(model_df_lead_this[eff_dec_years_name].unique())}")

        # Subset the obs df to the same period
        obs_df_lead_this = obs_df_copy[
            obs_df_copy[eff_dec_years_name].isin(unique_eff_dec_years_this)
        ]

        # Get the mean of the obs data
        obs_mean_lead_this = obs_df_lead_this[obs_var_name].mean()

        # if the obs mean is nan
        if np.isnan(obs_mean_lead_this):
            print(f"Obs mean is nan for lead {lead}")
            print(f"Obs df lead this: {obs_df_lead_this}")
            raise ValueError(
                f"Obs mean is nan for lead {lead}: {obs_mean_lead_this}"
            )
        
        # print the obs mean
        print(f"Obs mean lead this: {obs_mean_lead_this}")

        # Add this back into the model df anoms for mean correction
        model_df_copy.loc[
            model_df_copy[lead_name] == lead, f"{model_var_name}_drift_bc"
        ] = (
            model_df_copy.loc[model_df_copy[lead_name] == lead, f"{model_var_name}_anomaly"]
            + obs_mean_lead_this
        )

    # Set up the axes
    fig, axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharex=True,
        sharey=True,
        layout="compressed",
    )

    ax1 = axes[0]
    ax2 = axes[1]

    # Get the cmap
    cmap = cm.get_cmap("Blues", len(unique_leads))

    # loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Subset the model data to the lead this
        model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

        # Plot the density distribution with kde
        sns.kdeplot(
            model_df_lead_this[f"{model_var_name}"],
            ax=ax1,
            color=cmap(i),
        )

        # Plot the density distribution with kde
        sns.kdeplot(
            model_df_lead_this[f"{model_var_name}_drift_bc"],
            ax=ax2,
            label=f"Lead {lead}",
            color=cmap(i),
        )

    # plot the observed disttibution
    sns.kdeplot(
        obs_df_copy[obs_var_name],
        ax=ax1,
        label="Observed",
        color="black",
        linestyle="--",
    )

    # plot the observed disttibution
    sns.kdeplot(
        obs_df_copy[obs_var_name],
        ax=ax2,
        label="Observed",
        color="black",
        linestyle="--",
    )

    # set up the titles
    ax1.set_title(f"No drift or bias corr. (no detrend)")
    ax2.set_title(f"Drift + bias corr. (no detrend)")

    # Include the legend in the top right of the right plot
    ax2.legend(
        loc="upper right",
        fontsize=8,
    )

    # Add a suptitle including the min and max unique effective dec years
    min_eff_dec_year = model_df_copy[eff_dec_years_name].min()
    max_eff_dec_year = model_df_copy[eff_dec_years_name].max()

    plt.suptitle(
        f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
        fontsize=14,
        fontweight="bold",
        y=1.08,
    )

    return model_df_copy

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

    # # Ste up the path to the test data
    # hannah_data_path = "/home/users/benhutch/energy-sotcr-2023/data/ERA5_UK_1940_2024_daily_SP_WP_demand_wind.csv"

    # # Load the test data
    # df_hannah_tas = pd.read_csv(hannah_data_path)

    # # format date column as a datetime
    # df_hannah_tas["date"] = pd.to_datetime(df_hannah_tas["date"])

    # # subset to months 12, 1, 2
    # df_hannah_tas = df_hannah_tas[df_hannah_tas["date"].dt.month.isin([12, 1, 2])]

    # # rename the date column to time
    # df_hannah_tas.rename(columns={"date": "time"}, inplace=True)

    # # create the effective dec year column
    # df_hannah_tas["effective_dec_year"] = df_hannah_tas.apply(
    #     lambda row: gev_funcs.determine_effective_dec_year(row), axis=1
    # )

    # # print the head and tail of the hannah data
    # print(df_hannah_tas.head())
    # print(df_hannah_tas.tail())

    # # restrict to just the UK_temp column and the effective dec year and time
    # df_hannah_tas = df_hannah_tas[["time", "UK_temp", "effective_dec_year"]]

    # # rmeove the nans
    # df_hannah_tas.dropna(inplace=True)

    # # calculate the return period for ver cold days
    # gev_funcs.plot_return_periods_decades_obs(
    #     obs_df=df_hannah_tas,
    #     obs_var_name="UK_temp",
    #     decades=np.arange(1940, 2030, 10),
    #     title="Decadal RPs, 1940-2024, DJF, UK, ERA5",
    #     year_col_name="effective_dec_year",
    #     num_samples=1000,
    #     figsize=(10, 5),
    #     bad_min=True,
    # )

    # sys.exit()

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

    # -------------------------------
    # Now load the model and obs 10m wind speed data
    # -------------------------------

    # Set up the model wind fname
    model_wind_fname = (
        "HadGEM3-GC31-MM_dcppA-hindcast_sfcWind_UK_wind_box_1960-2018_day.csv"
    )

    # Set up the path
    model_wind_path = os.path.join(dfs_dir, model_wind_fname)

    # if the path exists load the data
    if os.path.exists(model_wind_path):
        df_model_wind = pd.read_csv(model_wind_path)
    else:
        raise FileNotFoundError(f"File not found: {model_wind_path}")

    # Apply the function to select winter yeas
    df_model_wind = select_leads_wyears_DJF(
        df=df_model_wind,
        wyears=np.arange(1, 11 + 1),
    )

    # Set up the effective dec year column
    df_model_wind["effective_dec_year"] = df_model_wind["init_year"] + (
        df_model_wind["winter_year"] - 1
    )

    # Set up the path to the obs data
    obs_wind_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/ERA5_sfcWind_UK_wind_box_1960-2018_daily_2025-02-26.csv"

    # load the obs data
    df_obs_wind = pd.read_csv(obs_wind_path)

    # Convert the 'time' column to datetime, assuming it represents days since "1950-01-01 00:00:00"
    df_obs_wind["time"] = pd.to_datetime(
        df_obs_wind["time"], origin="1952-01-01", unit="D"
    )

    # Make sure time is a datetime
    df_obs_wind["time"] = pd.to_datetime(df_obs_wind["time"])

    # subset the obs data to D, J, F
    df_obs_wind = df_obs_wind[df_obs_wind["time"].dt.month.isin([12, 1, 2])]

    # Set up the effective dec year column
    df_obs_wind["effective_dec_year"] = df_obs_wind.apply(
        lambda row: gev_funcs.determine_effective_dec_year(row), axis=1
    )

    # # calculate the return period for ver cold days
    # gev_funcs.plot_return_periods_decades_obs(
    #     obs_df=df_obs_tas,
    #     obs_var_name="data_c",
    #     decades=np.arange(1960, 2020, 10),
    #     title="Decadal RPs, 1961-2017, DJF, UK, ERA5",
    #     year_col_name="effective_dec_year",
    #     num_samples=1000,
    #     figsize=(10, 5),
    #     bad_min=True,
    # )

    # sys.exit()

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

    # Subset the model wind data to the common winter years
    df_model_wind = df_model_wind[
        df_model_wind["effective_dec_year"].isin(common_wyears)
    ]

    # Subset the obs wind data to the common winter years
    df_obs_wind = df_obs_wind[df_obs_wind["effective_dec_year"].isin(common_wyears)]

    # print the head and tail of df model wind
    print(df_model_wind.head())
    print(df_model_wind.tail())

    # print the model tas head
    print(df_model_tas_djf.head())
    print(df_model_tas_djf.tail())

    # print the head and tail of df obs wind
    print(df_obs_wind.head())
    print(df_obs_wind.tail())

    # print the head and tail of df obs tas
    print(df_obs_tas.head())
    print(df_obs_tas.tail())

    # Create a new column for data tas c in df_model_full_djf
    df_model_tas_djf["data_tas_c"] = df_model_tas_djf["data"] - 273.15

    # Apply the block minima transform to the obs data
    block_minima_obs_tas = gev_funcs.obs_block_min_max(
        df=df_obs_tas,
        time_name="effective_dec_year",
        min_max_var_name="data_c",
        new_df_cols=["time"],
        process_min=True,
    )

    # Calculate the block minima for trhe obs wind speed
    block_minima_obs_wind = gev_funcs.obs_block_min_max(
        df=df_obs_wind,
        time_name="effective_dec_year",
        min_max_var_name="data",
        new_df_cols=["time"],
        process_min=True,
    )

    # print the heaad and tail of df obs tas
    print(df_obs_tas.head())
    print(df_obs_tas.tail())

    # print the head of the block minima obs tas
    print(block_minima_obs_tas.head())
    print(block_minima_obs_tas.tail())

    # Set up a fname for the dataframe
    fname = "block_minima_obs_tas_UK_1960-2017_DJF_2_April.csv"

    # Set up the dir to save to
    save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    # if the full path does not exist
    if not os.path.exists(os.path.join(save_dir, fname)):
        print(f"Saving {fname} to {save_dir}")
        block_minima_obs_tas.to_csv(os.path.join(save_dir, fname))

    # sys.exit()

    # print the head of the df_model_tas_djf
    print(df_model_tas_djf.head())
    print(df_model_tas_djf.tail())

    # Apply the block minima transform to the model data
    block_minima_model_tas = gev_funcs.model_block_min_max(
        df=df_model_tas_djf,
        time_name="init_year",
        min_max_var_name="data_tas_c",
        new_df_cols=["init_year", "member", "lead"],
        winter_year="winter_year",
        process_min=True,
    )

    # Apply the model block minima transform to the model wind data
    block_minima_model_wind = gev_funcs.model_block_min_max(
        df=df_model_wind,
        time_name="init_year",
        min_max_var_name="data",
        new_df_cols=["init_year", "member", "lead"],
        winter_year="winter_year",
        process_min=True,
    )

    # # print the head of the block minima model tas
    # print(block_minima_model_tas.head())
    # print(block_minima_model_tas.tail())

    # # print the unique init years in the model df
    # print(block_minima_model_tas["init_year"].unique())

    # Set up a fname for the dataframe
    fname = "block_minima_model_tas_UK_1960-2017_DJF.csv"

    # Set up the dir to save to
    save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    # if the full path does not exist
    if not os.path.exists(os.path.join(save_dir, fname)):
        print(f"Saving {fname} to {save_dir}")
        block_minima_model_tas.to_csv(os.path.join(save_dir, fname))

    # sys.exit()

    # # Ensure effective dec year is in the block minima model tas
    # block_minima_model_tas["effective_dec_year"] = block_minima_model_tas[
    #     "init_year"
    # ] + (block_minima_model_tas["winter_year"] - 1)

    # # print the model df for lead 2
    # print(block_minima_model_tas[block_minima_model_tas["winter_year"] == 2])

    # # print the effective dec years in the model df
    # block_minima_model_tas_winter_2 = block_minima_model_tas[
    #     block_minima_model_tas["winter_year"] == 2
    # ]

    # # print the unique efefctive dec years
    # print(block_minima_model_tas_winter_2["effective_dec_year"].unique())

    # # Compare the trends
    # gev_funcs.compare_trends(
    #     model_df_full_field=df_model_tas_djf,
    #     obs_df_full_field=df_obs_tas,
    #     model_df_block=block_minima_model_tas,
    #     obs_df_block=block_minima_obs_tas,
    #     model_var_name_full_field="data_tas_c",
    #     obs_var_name_full_field="data_c",
    #     model_var_name_block="data_tas_c_min",
    #     obs_var_name_block="data_c_min",
    #     model_time_name="effective_dec_year",
    #     obs_time_name="effective_dec_year",
    #     ylabel="Temperature (C)",
    #     suptitle="Temperature trends (no bias correction or detrend)",
    #     figsize=(15, 5),
    #     window_size=10,
    # )

    # # Plot the lead time trends
    # gev_funcs.lead_time_trends(
    #     model_df=block_minima_model_tas,
    #     obs_df=block_minima_obs_tas,
    #     model_var_name="data_tas_c_min",
    #     obs_var_name="data_c_min",
    #     lead_name="winter_year",
    #     ylabel="Temperature (C)",
    #     suptitle="Temperature trends, 1961-2017, DJF block min T",
    #     figsize=(15, 5),
    # )

    # # print the model df for lead 2
    # print(block_minima_model_tas[block_minima_model_tas["winter_year"] == 2])

    # # Use a function to correct the lead time dependent trends
    # block_minima_model_tas_lead_dt = gev_funcs.lead_time_trend_corr(
    #     model_df=block_minima_model_tas,
    #     x_axis_name="effective_dec_year",
    #     y_axis_name="data_tas_c_min",
    #     lead_name="winter_year",
    # )

    # print the head of block minima model tas
    print(block_minima_model_tas.head())

    # add the effective dec year to the block minima model tas
    block_minima_model_tas["effective_dec_year"] = block_minima_model_tas[
        "init_year"
    ] + (block_minima_model_tas["winter_year"] - 1)

    # do the same for wind speed
    block_minima_model_wind["effective_dec_year"] = block_minima_model_wind[
        "init_year"
    ] + (block_minima_model_wind["winter_year"] - 1)

    # Plot the lead pdfs
    gev_funcs.plot_lead_pdfs(
        model_df=block_minima_model_tas,
        obs_df=block_minima_obs_tas,
        model_var_name="data_tas_c_min",
        obs_var_name="data_c_min",
        lead_name="winter_year",
        xlabel="Temperature (C)",
        suptitle="Temperature PDFs, 1961-2017, DJF block min T (no drift, trend, or bias correction)",
        figsize=(10, 5),
    )

    # Plot the lead pdfs for wind speed pre drift/bc
    gev_funcs.plot_lead_pdfs(
        model_df=block_minima_model_wind,
        obs_df=block_minima_obs_wind,
        model_var_name="data_min",
        obs_var_name="data_min",
        lead_name="winter_year",
        xlabel="Wind speed (m/s)",
        suptitle="Wind speed PDFs, 1961-2017, DJF block min T (no drift, trend, or bias correction)",
        figsize=(10, 5),
    )

    # Plot the lead time depedent drift for the model and the corrected model
    block_minima_model_tas_drift_corr = model_drift_corr_plot(
        model_df=block_minima_model_tas,
        model_var_name="data_tas_c_min",
        obs_df=block_minima_obs_tas,
        obs_var_name="data_c_min",
        lead_name="winter_year",
        xlabel="Temperature (C)",
        year1_year2_tuple=(1970, 2017),
        constant_period=True,
    )

    # Apply the same drift correction to the wind data
    block_minima_model_wind_drift_corr = model_drift_corr_plot(
        model_df=block_minima_model_wind,
        model_var_name="data_min",
        obs_df=block_minima_obs_wind,
        obs_var_name="data_min",
        lead_name="winter_year",
        xlabel="Wind speed (m/s)",
        year1_year2_tuple=(1970, 2017),
        constant_period=True,
    )

    # PLOT THE LEAD pdfs post this
    gev_funcs.plot_lead_pdfs(
        model_df=block_minima_model_tas_drift_corr,
        obs_df=block_minima_obs_tas,
        model_var_name="data_tas_c_min_drift_bc",
        obs_var_name="data_c_min",
        lead_name="winter_year",
        xlabel="Temperature (C)",
        suptitle="Temperature PDFs, 1961-2017, DJF block min T (model drift corrected)",
        figsize=(10, 5),
    )

    # Plot the lead pdfs for wind speed post drift/bc
    gev_funcs.plot_lead_pdfs(
        model_df=block_minima_model_wind_drift_corr,
        obs_df=block_minima_obs_wind,
        model_var_name="data_min_drift_bc",
        obs_var_name="data_min",
        lead_name="winter_year",
        xlabel="Wind speed (m/s)",
        suptitle="Wind speed PDFs, 1961-2017, DJF block min T (model drift corrected)",
        figsize=(10, 5),
    )

    sys.exit()

    # # Loop over the unique lead times in block minima model tas drift corr
    # for lead in block_minima_model_tas_drift_corr["winter_year"].unique():
    #     # Print the lead time
    #     print(f"Lead time: {lead}")

    #     # Extract the data for this lead time, excluding NaN values
    #     data_this = block_minima_model_tas_drift_corr[
    #         (block_minima_model_tas_drift_corr["winter_year"] == lead)
    #         & (block_minima_model_tas_drift_corr["data_tas_c_min_drift_bc"].notna())
    #     ]

    #     # Check if there is any data left after filtering
    #     if not data_this.empty:
    #         # Print the unique init years
    #         print(f"First init year drift corr: {data_this['init_year'].unique()[0]}")
    #         print(f"Last init year drift corr: {data_this['init_year'].unique()[-1]}")
    #         print(f"second last init year drift corr: {data_this['init_year'].unique()[-2]}")
    #     else:
    #         print("No valid data for this lead time.")

    # # print the unique init years in block minima model tas
    # for lead in block_minima_model_tas["winter_year"].unique():
    #     # print the lead time
    #     print(f"Lead time: {lead}")

    #     data_this = block_minima_model_tas[
    #         (block_minima_model_tas["winter_year"] == lead)
    #         & (block_minima_model_tas["data_tas_c_min"].notna())
    #     ]
    #     # Check if there is any data left after filtering
    #     if not data_this.empty:
    #         # Print the unique init years
    #         print(f"First init year original: {data_this['init_year'].unique()[0]}")
    #         print(f"Last init year original: {data_this['init_year'].unique()[-1]}")
    #         print(f"second last init year original: {data_this['init_year'].unique()[-2]}")
    #     else:
    #         print("No valid data for this lead time.")

    # Use a function to correct the overall rolling mean trends
    block_minima_model_tas_drift_corr_dt = gev_funcs.pivot_detrend_model(
        model_df=block_minima_model_tas_drift_corr,
        obs_df=block_minima_obs_tas,
        model_x_axis_name="effective_dec_year",
        model_y_axis_name="data_tas_c_min_drift_bc",
        obs_x_axis_name="effective_dec_year",
        obs_y_axis_name="data_c_min",
        suffix="_dt",
    )

    # apply a detrend to the wind data
    block_minima_model_wind_drift_corr_dt = gev_funcs.pivot_detrend_model(
        model_df=block_minima_model_wind_drift_corr,
        obs_df=block_minima_obs_wind,
        model_x_axis_name="effective_dec_year",
        model_y_axis_name="data_min_drift_bc",
        obs_x_axis_name="effective_dec_year",
        obs_y_axis_name="data_min",
        suffix="_dt",
    )

    # print the head of the dataframe
    print(block_minima_model_tas_drift_corr_dt.head())

    # print the tail of the dataframe
    print(block_minima_model_tas_drift_corr_dt.tail())

    # print the unique effective dec year in block minima model tas lead dt
    print(block_minima_model_tas_drift_corr_dt["effective_dec_year"].unique())

    # print the number of Nans in the model data
    # for the data tas c min dt column
    print(block_minima_model_tas_drift_corr_dt["data_tas_c_min_drift_bc_dt"].isna().sum())

    # pviot detrend the obs data
    block_minima_obs_tas_dt = gev_funcs.pivot_detrend_obs(
        df=block_minima_obs_tas,
        x_axis_name="effective_dec_year",
        y_axis_name="data_c_min",
    )

    # pivot detrend the obs data for wind speed
    block_minima_obs_wind_dt = gev_funcs.pivot_detrend_obs(
        df=block_minima_obs_wind,
        x_axis_name="effective_dec_year",
        y_axis_name="data_min",
    )

    # # Compare the lead time corrected trends
    # gev_funcs.lead_time_trends(
    #     model_df=block_minima_model_tas_drift_corr_dt,
    #     obs_df=block_minima_obs_tas_dt,
    #     model_var_name="data_tas_c_min_drift_bc_dt",
    #     obs_var_name="data_c_min_dt",
    #     lead_name="winter_year",
    #     ylabel="Temperature (C)",
    #     suptitle="Temperature trends, 1961-2017, DJF block min T",
    #     figsize=(15, 5),
    # )

    # Compare the trends with the full field data
    gev_funcs.compare_trends(
        model_df_full_field=df_model_tas_djf,
        obs_df_full_field=df_obs_tas,
        model_df_block=block_minima_model_tas_drift_corr_dt,
        obs_df_block=block_minima_obs_tas_dt,
        model_var_name_full_field="data_tas_c",
        obs_var_name_full_field="data_c",
        model_var_name_block="data_tas_c_min_drift_bc",
        obs_var_name_block="data_c_min",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        suptitle="Temperature trends (block min detrended obs, model lead time detrended)",
        figsize=(15, 5),
        window_size=10,
        centred_bool=True,
        min_periods=1,
    )

    # compare trends for the wind data
    gev_funcs.compare_trends(
        model_df_full_field=df_model_wind,
        obs_df_full_field=df_obs_wind,
        model_df_block=block_minima_model_wind_drift_corr_dt,
        obs_df_block=block_minima_obs_wind_dt,
        model_var_name_full_field="data",
        obs_var_name_full_field="data",
        model_var_name_block="data_min_drift_bc",
        obs_var_name_block="data_min",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Wind speed (m/s)",
        suptitle="Wind speed trends (block min detrended obs, model lead time detrended)",
        figsize=(15, 5),
        window_size=10,
        centred_bool=True,
        min_periods=1,
    )

    # sys.exit()

    # Now plot the lead time dependent biases for the trend corrected data
    gev_funcs.plot_lead_pdfs(
        model_df=block_minima_model_tas_drift_corr_dt,
        obs_df=block_minima_obs_tas_dt,
        model_var_name="data_tas_c_min_drift_bc_dt",
        obs_var_name="data_c_min_dt",
        lead_name="winter_year",
        xlabel="Temperature (C)",
        suptitle="Temperature PDFs, 1961-2017, DJF block min T (model drift + trend corrected)",
        figsize=(10, 5),
    )

    # Plot the lead pdfs for wind speed post drift/bc
    gev_funcs.plot_lead_pdfs(
        model_df=block_minima_model_wind_drift_corr_dt,
        obs_df=block_minima_obs_wind_dt,
        model_var_name="data_min_drift_bc_dt",
        obs_var_name="data_min_dt",
        lead_name="winter_year",
        xlabel="Wind speed (m/s)",
        suptitle="Wind speed PDFs, 1961-2017, DJF block min T (model drift + trend corrected)",
        figsize=(10, 5),
    )

    # sys.exit()

    # perform the lead time depdent bias correction
    # for the block minima
    # block_minima_model_tas_lead_dt_bc = gev_funcs.lead_time_mean_bias_correct(
    #     model_df=block_minima_model_tas_lead_dt,
    #     obs_df=block_minima_obs_tas_dt,
    #     model_var_name="data_tas_c_min_dt",
    #     obs_var_name="data_c_min_dt",
    #     lead_name="winter_year",
    # )

    # # bias correct the block minima obs tas
    # block_minima_model_tas_lead_dt_bc = gev_funcs.lead_time_mean_bias_correct(
    #     model_df=block_minima_model_tas_lead_dt_bc,
    #     obs_df=block_minima_obs_tas_dt,
    #     model_var_name="data_tas_c_min",
    #     obs_var_name="data_c_min",
    #     lead_name="winter_year",
    # )

    # # perform the lead time dependent bias correction
    # # for the wind speed data
    # block_minima_model_wind_bc = gev_funcs.lead_time_mean_bias_correct(
    #     model_df=block_minima_model_wind,
    #     obs_df=block_minima_obs_wind,
    #     model_var_name="data_min",
    #     obs_var_name="data_min",
    #     lead_name="winter_year",
    # )

    # # print the head of the block minima model tas lead dt bc
    # print(block_minima_model_tas_lead_dt_bc.head())

    # # print the head of the block minima model wind bc
    # print(block_minima_model_wind_bc.head())

    # rename data_tas_c_min_dt_bc
    # to data_tas_c_min_bc_dt
    # block_minima_model_tas_lead_dt_bc.rename(
    #     columns={"data_tas_c_min_dt_bc": "data_tas_c_min_bc_dt"}, inplace=True
    # )

    # plot the plots
    gev_funcs.plot_detrend_ts_subplots(
        obs_df_left=block_minima_obs_tas_dt,
        model_df_left=block_minima_model_tas_drift_corr_dt,
        obs_df_right=block_minima_obs_wind_dt,
        model_df_right=block_minima_model_wind_drift_corr_dt,
        obs_var_name_left="data_c_min",
        model_var_name_left="data_tas_c_min_drift_bc",
        obs_var_name_right="data_min",
        model_var_name_right="data_min_drift_bc",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        ylabel_left="Temperature (C)",
        ylabel_right="Wind speed (m/s)",
        detrend_suffix_left="_dt",
        detrend_suffix_right="_dt",
    )

    # sys.exit()

    # Make sure effective dec year is a datetime in the model tas data
    block_minima_model_tas_drift_corr_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_model_tas_drift_corr_dt["effective_dec_year"], format="%Y"
    )

    # Make sure effective dec year is a datetime in the model wind data
    block_minima_model_wind_drift_corr_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_model_wind_drift_corr_dt["effective_dec_year"], format="%Y"
    )

    # Make sure effective dec year is a datetime in the obs tas data
    block_minima_obs_tas_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_tas_dt["effective_dec_year"], format="%Y"
    )

    # Make sure effective dec year is a datetime in the obs wind data
    block_minima_obs_wind_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_wind_dt["effective_dec_year"], format="%Y"
    )

    # Set this as the index in the obs tas data
    block_minima_obs_tas_dt.set_index("effective_dec_year", inplace=True)

    # Set this as the index in the obs wind data
    block_minima_obs_wind_dt.set_index("effective_dec_year", inplace=True)

    # Now test plotting the dot plots for temp and wind speed
    gev_funcs.dot_plot_subplots(
        obs_df_left=block_minima_obs_tas_dt,
        model_df_left=block_minima_model_tas_drift_corr_dt,
        obs_df_right=block_minima_obs_wind_dt,
        model_df_right=block_minima_model_wind_drift_corr_dt,
        obs_val_name_left="data_c_min_dt",
        model_val_name_left="data_tas_c_min_drift_bc_dt",
        obs_val_name_right="data_min_dt",
        model_val_name_right="data_min_drift_bc_dt",
        model_time_name="effective_dec_year",
        ylabel_left="Temperature (°C)",
        ylabel_right="Wind speed (m/s)",
        title_left="Block minima temperature (°C)",
        title_right="Block minima wind speed (m/s)",
        ylims_left=(-12, 8),
        ylims_right=(0, 8),
        dashed_quant=0.20,
        solid_line=np.min,
        figsize=(10, 5),
    )

    # sys.exit()

    # ---------------------------------------
    # Now process the GEV params for both temp and wind speed
    # ---------------------------------------

    # gev_params_raw_temp = gev_funcs.process_gev_params(
    #     obs_df=block_minima_obs_tas_dt,
    #     model_df=block_minima_model_tas_lead_dt_bc,
    #     obs_var_name="data_c_min_dt",
    #     model_var_name="data_tas_c_min_dt",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     nboot=1000,
    #     model_lead_name="winter_year",
    # )

    # set efefctive dec year back as an int in the model data
    block_minima_model_tas_drift_corr_dt["effective_dec_year"] = block_minima_model_tas_drift_corr_dt[
        "effective_dec_year"
    ].dt.year.astype(int)

    # set effective dec year back as an int in the obs data
    block_minima_model_wind_drift_corr_dt["effective_dec_year"] = block_minima_model_wind_drift_corr_dt[
        "effective_dec_year"
    ].dt.year.astype(int)

    # reset the index of the obs
    block_minima_obs_tas_dt.reset_index(inplace=True)

    # reset the index of the obs
    block_minima_obs_wind_dt.reset_index(inplace=True)

    # make sure effective dec year in the obs is a datetime
    block_minima_obs_tas_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_tas_dt["effective_dec_year"], format="%Y"
    )

    # make sure effective dec year in the obs is a datetime
    block_minima_obs_wind_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_wind_dt["effective_dec_year"], format="%Y"
    )

    # format the obs effective dec year as an int in years
    block_minima_obs_tas_dt["effective_dec_year"] = block_minima_obs_tas_dt[
        "effective_dec_year"
    ].dt.year.astype(int)

    # format the obs effective dec year as an int in years
    block_minima_obs_wind_dt["effective_dec_year"] = block_minima_obs_wind_dt[
        "effective_dec_year"
    ].dt.year.astype(int)

    # print the head of the block minima obs tas dt
    print(block_minima_obs_tas_dt.head())

    # print the head of the block minima model tas drift corr dt
    print(block_minima_model_tas_drift_corr_dt.head())

    # print the head of the block minima obs wind dt
    print(block_minima_obs_wind_dt.head())

    # print the head of the block minima model wind drift corr dt
    print(block_minima_model_wind_drift_corr_dt.head())

    # process the gev params for the bias corrected temp data
    gev_params_bc_temp = gev_funcs.process_gev_params(
        obs_df=block_minima_obs_tas_dt,
        model_df=block_minima_model_tas_drift_corr_dt,
        obs_var_name="data_c_min_dt",
        model_var_name="data_tas_c_min_drift_bc_dt",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        nboot=1000,
        model_lead_name="winter_year",
    )

    # Process the GEV params for the wind speed data
    gev_params_bc_wind = gev_funcs.process_gev_params(
        obs_df=block_minima_obs_wind_dt,
        model_df=block_minima_model_wind_drift_corr_dt,
        obs_var_name="data_min_dt",
        model_var_name="data_min_drift_bc_dt",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        nboot=1000,
        model_lead_name="winter_year",
    )

    # # process the GEV params for the bias corrected wind speed data
    # gev_params_bc_wind = gev_funcs.process_gev_params(
    #     obs_df=block_minima_obs_wind,
    #     model_df=block_minima_model_wind_bc,
    #     obs_var_name="data_min",
    #     model_var_name="data_min_bc",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     nboot=1000,
    #     model_lead_name="winter_year",
    # )

    # Now test the plotting function for these
    gev_funcs.plot_gev_params_subplots(
        gev_params_top_raw=gev_params_bc_temp,
        gev_params_top_bc=gev_params_bc_temp,
        gev_params_bottom_raw=gev_params_bc_wind,
        gev_params_bottom_bc=gev_params_bc_wind,
        obs_df_top=block_minima_obs_tas_dt,
        model_df_top=block_minima_model_tas_drift_corr_dt,
        obs_df_bottom=block_minima_obs_wind_dt,
        model_df_bottom=block_minima_model_wind_drift_corr_dt,
        obs_var_name_top="data_c_min_dt",
        model_var_name_top="data_tas_c_min_drift_bc_dt",
        obs_var_name_bottom="data_min_dt",
        model_var_name_bottom="data_min_drift_bc_dt",
        title_top="Distribution of DJF block minima temperature (°C)",
        title_bottom="Distribution of DJF block minima wind speed (m/s)",
        figsize=(15, 10),
    )

    sys.exit()

    # # plot the lead time dependent trends
    # gev_funcs.lead_time_trends(
    #     model_df=block_minima_model_tas_lead_dt_bc,
    #     obs_df=block_minima_obs_tas_dt,
    #     model_var_name="data_tas_c_min_rm_dt_bc",
    #     obs_var_name="data_c_min_dt",
    #     lead_name="winter_year",
    #     ylabel="Temperature (C)",
    #     suptitle="Temperature trends, 1961-2017, DJF block min T",
    #     figsize=(15, 5),
    # )

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

    # find the 20th percentile value in the obs
    obs_20th = np.percentile(block_minima_obs_tas_dt["data_c_min_dt"], 20)

    # subset the block minima obs tas dt to the 20th percentile
    block_minima_model_tas_lead_dt_bc = block_minima_model_tas_lead_dt_bc[
        block_minima_model_tas_lead_dt_bc["data_tas_c_min_rm_dt_bc"] <= obs_20th
    ]

    # reset the index of this dataframe
    block_minima_model_tas_lead_dt_bc.reset_index(drop=True, inplace=True)

    # Set up a fname for this dataframe
    fname = "block_minima_model_tas_dt_UK_1960-2017_DJF_yellow_dots.csv"

    # Set up the dir to save to
    save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    # if the full path does not exist
    if not os.path.exists(os.path.join(save_dir, fname)):
        print(f"Saving {fname} to {save_dir}")
        block_minima_model_tas_lead_dt_bc.to_csv(os.path.join(save_dir, fname))

    sys.exit()

    # # Find the min value of data_c_min_dt for the obs
    # obs_min = np.min(block_minima_obs_tas_dt["data_c_min_dt"])

    # # Subset the block minima model tas lead dt bc
    # block_minima_model_tas_lead_dt_bc = block_minima_model_tas_lead_dt_bc[
    #     block_minima_model_tas_lead_dt_bc["data_tas_c_min_dt_bc"] <= obs_min
    # ]

    # # reset the index of this dataframe
    # block_minima_model_tas_lead_dt_bc.reset_index(drop=True, inplace=True)

    # # print the length of this dataframe
    # print("Length of block_minima_model_tas_lead_dt_bc", len(block_minima_model_tas_lead_dt_bc))

    # # Set up a filename for the datraframe
    # fname = "block_minima_model_tas_lead_dt_bc_UK_1960-2017_DJF.csv"

    # #Set up the full path
    # save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    # if not os.path.exists(os.path.join(save_dir, fname)):
    #     print(f"Saving {fname} to {save_dir}")
    #     block_minima_model_tas_lead_dt_bc.to_csv(os.path.join(save_dir, fname))

    # # find the 20th percentil of the observations
    # obs_20th = np.percentile(block_minima_obs_tas_dt["data_c_min_dt"], 20)

    # # Subset the block minima obs tas dt to the 20th percentile
    # block_minima_obs_tas_dt = block_minima_obs_tas_dt[
    #     block_minima_obs_tas_dt["data_c_min_dt"] <= obs_20th
    # ]

    # # reset the index of this dataframe
    # block_minima_obs_tas_dt.reset_index(drop=True, inplace=True)

    # # Set up the filename for the dataframe
    # fname = "block_minima_obs_tas_dt_UK_1960-2017_DJF.csv"

    # # Set up the full path
    # save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    # if not os.path.exists(os.path.join(save_dir, fname)):
    #     print(f"Saving {fname} to {save_dir}")
    #     block_minima_obs_tas_dt.to_csv(os.path.join(save_dir, fname))

    # plot the dot plot for the detrended obs
    dot_plot(
        obs_df=block_minima_obs_tas_dt,
        model_df=block_minima_model_tas_lead_dt_bc,
        obs_val_name="data_c_min_dt",
        model_val_name="data_tas_c_min_rm_dt_bc",
        model_time_name="effective_dec_year",
        ylabel="Temperature (C)",
        title="Lead time detrended model bc, 1961-2017, DJF block min T",
        ylims=(-15, 6),
        solid_line=np.min,
        dashed_quant=0.20,
        figsize=(12, 5),
    )

    sys.exit()

    block_minima_model_tas_lead_dt_bc["effective_dec_year"] = (
        block_minima_model_tas_lead_dt_bc["effective_dec_year"].dt.year.astype(int)
    )

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
