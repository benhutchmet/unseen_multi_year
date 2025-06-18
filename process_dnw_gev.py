#!/usr/bin/env python

"""
process_dnw_gev.py
==================

This script processes daily obs and model data (all leads) into a dataframe containing demand net wind.

Methodology is still in development, so this script is a work in progress.

"""
# %%
# Local imports
import os
import sys
import glob
import shutil
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
from scipy.stats import linregress, percentileofscore, gaussian_kde, pearsonr
from scipy.stats import genextreme as gev
from sklearn.metrics import mean_squared_error, r2_score
from iris.util import equalise_attributes

# Local imports
import gev_functions as gev_funcs
from process_temp_gev import model_drift_corr_plot, plot_gev_rps, plot_emp_rps

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import sigmoid, dot_plot, plot_rp_extremes, empirical_return_level

# Silence warnings
warnings.filterwarnings("ignore")


# Set up a function to do pivoting for DnW
# As we detrend at the temp/wind stage pre-transformation
# and pre-block max identification
# we have to do this slightly differently
def pivot_emp_rps_dnw(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_var_name_wind: str,
    obs_var_name_tas: str,
    model_var_name_wind: str,
    model_var_name_tas: str,
    model_time_name: str,
    obs_time_name: str,
    nsamples: int = 1000,
    figsize: tuple[int, int] = (5, 5),
) -> None:
    """
    Pivots the entire ensemble for both temperature and wind speed around each
    year in turn and quantifies the likelihood of seeing an event worse than
    the worst observed extreme.

    Parameters
    ==========

        obs_df : pd.DataFrame
            The dataframe containing the observed data.
        model_df : pd.DataFrame
            The dataframe containing the model data.
        obs_var_name_wind : str
            The name of the observed wind speed variable.
        obs_var_name_tas : str
            The name of the observed temperature variable.
        model_var_name_wind : str
            The name of the model wind speed variable.
        model_var_name_tas : str
            The name of the model temperature variable.
        model_time_name : str
            The name of the time variable in the model data.
        obs_time_name : str
            The name of the time variable in the observed data.
        nsamples : int
            The number of samples to use for the pivoting.
        figsize : tuple[int, int]
            The size of the figure.

    Returns
    =======

        None

    """

    # Hard code the HDD and CDD bases
    hdd_base = 15.5
    cdd_base = 22.0
    demand_year = 2017

    # Make a copy of the dataframes
    obs_df_copy = obs_df.copy()
    model_df_copy = model_df.copy()

    # If the time column is not an int
    # then set as an int
    if obs_df_copy[obs_time_name].dtype != int:
        obs_df_copy[obs_time_name] = obs_df_copy[obs_time_name].astype(int)
    if model_df_copy[model_time_name].dtype != int:
        model_df_copy[model_time_name] = model_df_copy[model_time_name].astype(int)

    # Extract the unique time points from the model
    model_time_points = model_df_copy[model_time_name].unique()
    obs_time_points = obs_df_copy[obs_time_name].unique()

    # Set up the model vals and obs vals
    model_vals_tas = model_df_copy.groupby(model_time_name)[model_var_name_tas].mean()
    model_vals_wind = model_df_copy.groupby(model_time_name)[model_var_name_wind].mean()
    obs_vals_tas = obs_df_copy.groupby(obs_time_name)[obs_var_name_tas].mean()
    obs_vals_wind = obs_df_copy.groupby(obs_time_name)[obs_var_name_wind].mean()

    # Calculate the model trend
    slope_model_tas, intercept_model_tas, _, _, _ = linregress(
        model_time_points, model_vals_tas
    )
    slope_model_wind, intercept_model_wind, _, _, _ = linregress(
        model_time_points, model_vals_wind
    )

    # Calculate the obs trend
    slope_obs_tas, intercept_obs_tas, _, _, _ = linregress(
        obs_time_points, obs_vals_tas
    )
    slope_obs_wind, intercept_obs_wind, _, _, _ = linregress(
        obs_time_points, obs_vals_wind
    )

    # Detrend the temperature data
    model_df_copy[f"{model_var_name_tas}_dt"] = model_df_copy[model_var_name_tas] - (
        slope_model_tas * model_df_copy[model_time_name] + intercept_model_tas
    )
    model_df_copy[f"{model_var_name_wind}_dt"] = model_df_copy[model_var_name_wind] - (
        slope_model_wind * model_df_copy[model_time_name] + intercept_model_wind
    )

    # Detrend the observed data
    obs_df_copy[f"{obs_var_name_tas}_dt"] = obs_df_copy[obs_var_name_tas] - (
        slope_obs_tas * obs_df_copy[obs_time_name] + intercept_obs_tas
    )
    obs_df_copy[f"{obs_var_name_wind}_dt"] = obs_df_copy[obs_var_name_wind] - (
        slope_obs_wind * obs_df_copy[obs_time_name] + intercept_obs_wind
    )

    # Set up the final point for ths obs trend line
    final_point_obs_trend_tas = (
        slope_obs_tas * obs_df_copy[obs_time_name].max() + intercept_obs_tas
    )
    final_point_obs_trend_wind = (
        slope_obs_wind * obs_df_copy[obs_time_name].max() + intercept_obs_wind
    )

    # Set up a new column for _dt_pivot
    obs_df_copy[f"{obs_var_name_tas}_dt_pivot"] = (
        obs_df_copy[f"{obs_var_name_tas}_dt"] + final_point_obs_trend_tas
    )
    obs_df_copy[f"{obs_var_name_wind}_dt_pivot"] = (
        obs_df_copy[f"{obs_var_name_wind}_dt"] + final_point_obs_trend_wind
    )

    # Quantify the block maxima for the obs
    # To identify the worst eevent/day
    # Translate the wind speed to wind power generation
    df_obs, _ = ws_to_wp_gen(
        obs_df=obs_df_copy,
        model_df=model_df_copy,
        obs_ws_col=f"{obs_var_name_wind}_dt_pivot",
        model_ws_col=f"{model_var_name_wind}",
        date_range=("1961-12-01", "2018-03-01"),
    )

    # Convert the temperature to demand
    df_obs, _ = temp_to_demand(
        obs_df=df_obs,
        model_df=model_df_copy,
        obs_temp_col=f"{obs_var_name_tas}_dt_pivot",
        model_temp_col=f"{model_var_name_tas}",
    )

    # Set up the pivot names
    obs_var_name_tas = f"{obs_var_name_tas}_dt_pivot"
    obs_var_name_wind = f"{obs_var_name_wind}_dt_pivot"

    # Calculate the demand net wind
    df_obs["dnw"] = (
        df_obs[f"{obs_var_name_tas}_UK_demand"]
        - df_obs[f"{obs_var_name_wind}_sigmoid_total_wind_gen"]
    )

    # Calculate the block maxima for the obs
    obs_block_maxima = gev_funcs.obs_block_min_max(
        df=df_obs,
        time_name=obs_time_name,
        min_max_var_name="dnw",
        new_df_cols=["time"],
        process_min=False,
    )

    # Find the max dnw value in obs block maxima
    obs_max_dnw = obs_block_maxima["dnw_max"].max()

    # Find the time at which this occurs
    obs_max_dnw_time = obs_block_maxima.loc[
        obs_block_maxima["dnw_max"] == obs_max_dnw, obs_time_name
    ].values[0]

    # print the obs max dnw time
    print(f"Obs max dnw time: {obs_max_dnw_time}")

    # Print the obs max dnw value
    print(f"Obs max dnw value: {obs_max_dnw}")

    # Set up a new dataframe to append values to
    model_df_plume = pd.DataFrame()

    # Set up the sigmoid fit for wind speed pre-iterable
    ch_df = pd.read_csv(
        "/home/users/benhutch/unseen_multi_year/dfs/UK_clearheads_data_daily_1960_2018_ONDJFM.csv"
    )

    # Set up the onshore and offshore capacities in gw
    onshore_cap_gw = 15710.69 / 1000
    offshore_cap_gw = 14733.02 / 1000

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
    ch_df = ch_df[ch_df.index.month.isin([12, 1, 2])]

    # Subset the data to the relevant date range
    ch_df = ch_df[(ch_df.index >= "1961-12-01") & (ch_df.index <= "2018-03-01")]

    # Set up an initial guess for the parameters for the sigmoid fit
    p0 = [
        max(ch_df["total_gen"]),
        np.median(obs_df_copy[obs_var_name_wind]),
        1,
        min(ch_df["total_gen"]),
    ]

    # Extract the first and last years (i.e., YYYY) from the date range
    start_year = int("1961-12-01".split("-")[0])
    end_year = int("2018-03-01".split("-")[0])

    # set up the obs df copy subset
    obs_df_copy_subset = obs_df_copy[
        (obs_df_copy["effective_dec_year"] >= start_year)
        & (obs_df_copy["effective_dec_year"] < end_year)
    ]

    # Fit the sigmoid curve to the observed data
    popt, pcov = curve_fit(
        sigmoid,
        obs_df_copy_subset[obs_var_name_wind],
        ch_df["total_gen"],
        p0=p0,
        method="dogbox",
    )

    # DO the same for demand
    df_regr = pd.read_csv(
        "/home/users/benhutch/ERA5_energy_update/ERA5_Regression_coeffs_demand_model.csv"
    )

    # Set the index
    df_regr.set_index("Unnamed: 0", inplace=True)

    # Rename the columns by splitting by _ and extracting the second element
    df_regr.columns = [col.split("_")[0] for col in df_regr.columns]

    # If there is a column called "United" replace it with "United_Kingdom"
    if "United" in df_regr.columns:
        df_regr.rename(columns={"United": "United_Kingdom"}, inplace=True)

    # Set up the regression coefficients
    time_coeff_uk = df_regr.loc["time", "United_Kingdom"]
    hdd_coeff_uk = df_regr.loc["HDD", "United_Kingdom"]
    cdd_coeff_uk = df_regr.loc["CDD", "United_Kingdom"]

    # Loop through the model time points
    for i, time_point in tqdm(
        enumerate(model_time_points), desc="Looping through model time points"
    ):
        # Applu this index to the obs trend
        obs_trend_point_this_tas = slope_obs_tas * time_point + intercept_obs_tas
        obs_trend_point_this_wind = slope_obs_wind * time_point + intercept_obs_wind

        # Set up the trend value this
        trend_val_this_tas = slope_model_tas * time_point + intercept_model_tas
        trend_val_this_wind = slope_model_wind * time_point + intercept_model_wind

        # calculayte the trend point bias
        trend_point_bias_this_tas = obs_trend_point_this_tas - trend_val_this_tas
        trend_point_bias_this_wind = obs_trend_point_this_wind - trend_val_this_wind

        # Bias correct the trend val this tas bc
        trend_val_this_tas_bc = trend_val_this_tas + trend_point_bias_this_tas
        trend_val_this_wind_bc = trend_val_this_wind + trend_point_bias_this_wind

        # Adjust the model data for this time point
        model_adjusted_this_tas = np.array(
            model_df_copy[f"{model_var_name_tas}_dt"] + trend_val_this_tas_bc
        )
        model_adjusted_this_wind = np.array(
            model_df_copy[f"{model_var_name_wind}_dt"] + trend_val_this_wind_bc
        )

        # Set up a new column in the dataframes for this data
        model_df_copy[f"{model_var_name_tas}_dt_this_{time_point}"] = (
            model_adjusted_this_tas
        )
        model_df_copy[f"{model_var_name_wind}_dt_this_{time_point}"] = (
            model_adjusted_this_wind
        )

        # Set up the obs var names
        model_var_name_wind_this = f"{model_var_name_wind}_dt_this_{time_point}"
        model_var_name_tas_this = f"{model_var_name_tas}_dt_this_{time_point}"

        # Convert the wind speed to wind power generation
        # _, model_df_copy_this = ws_to_wp_gen(
        #     obs_df=obs_df_copy,
        #     model_df=model_df_copy,
        #     obs_ws_col=obs_var_name_wind,
        #     model_ws_col=model_var_name_wind_this,
        #     date_range=("1961-12-01", "2018-03-01"),
        # )

        # Apply the sigmoid fit to the model data
        model_df_copy[f"{model_var_name_wind_this}_sigmoid_total_wind_gen"] = sigmoid(
            model_df_copy[model_var_name_wind_this], *popt
        )

        # Convert the temperature to demand
        # _, model_df_copy_this = temp_to_demand(
        #     obs_df=obs_df_copy,
        #     model_df=model_df_copy,
        #     obs_temp_col=obs_var_name_tas,
        #     model_temp_col=model_var_name_tas_this,
        # )

        # Calculate hdd and cdd
        model_df_copy[f"{model_var_name_tas_this}_HDD"] = model_df_copy[
            model_var_name_tas_this
        ].apply(lambda x: max(0, hdd_base - x))
        model_df_copy[f"{model_var_name_tas_this}_CDD"] = model_df_copy[
            model_var_name_tas_this
        ].apply(lambda x: max(0, x - cdd_base))

        # Calculate the demand
        model_df_copy[f"{model_var_name_tas_this}_UK_demand"] = (
            (time_coeff_uk * demand_year)
            + (hdd_coeff_uk * model_df_copy[f"{model_var_name_tas_this}_HDD"])
            + (cdd_coeff_uk * model_df_copy[f"{model_var_name_tas_this}_CDD"])
        )

        # Calculate the demand net wind
        model_df_copy[f"dnw_{time_point}"] = (
            model_df_copy[f"{model_var_name_tas_this}_UK_demand"]
            - model_df_copy[f"{model_var_name_wind_this}_sigmoid_total_wind_gen"]
        )

        # Calculate the block maxima for the model
        model_block_maxima = gev_funcs.model_block_min_max(
            df=model_df_copy,
            time_name="init_year",
            min_max_var_name=f"dnw_{time_point}",
            new_df_cols=["lead"],
            winter_year="winter_year",
            process_min=False,
        )

        # Add in the effective dec year column
        model_block_maxima["effective_dec_year"] = model_block_maxima["init_year"] + (
            model_block_maxima["winter_year"] - 1
        )

        # Extract the model block maxima vals
        model_block_maxima_vals = model_block_maxima[f"dnw_{time_point}_max"].values

        # if this is not an array, then format it as an array
        if not isinstance(model_block_maxima_vals, np.ndarray):
            model_block_maxima_vals = np.array(model_block_maxima_vals)

        # Set up the central return levels
        model_df_central_rps_this = empirical_return_level(
            data=model_block_maxima_vals,
            high_values_rare=True,
        )

        # Set up the bootstrap to append to
        model_df_bootstrap_this = np.zeros(
            [nsamples, len(model_df_central_rps_this["sorted"])]
        )

        # Loop through the samples
        for j in range(nsamples):
            # Resample the model block max data
            model_vals_this = np.random.choice(
                model_block_maxima_vals,
                size=len(model_block_maxima_vals),
                replace=True,
            )

            # Calculate the empirical return levels
            model_df_rps_this = empirical_return_level(
                data=model_vals_this,
                high_values_rare=True,
            )

            # Append the values to the bootstrap
            model_df_bootstrap_this[j, :] = model_df_rps_this["sorted"]

        # Find the dnw value closest in value to the obs max
        obs_extreme_index_central = np.abs(
            model_df_central_rps_this["sorted"] - obs_max_dnw
        ).argmin()

        # Set up the 0025 and 0975 quantiles
        model_df_central_rps_this["0025"] = np.quantile(
            model_df_bootstrap_this, 0.025, axis=0
        )
        model_df_central_rps_this["0975"] = np.quantile(
            model_df_bootstrap_this, 0.975, axis=0
        )

        # Find the index of the row, where "sorted" is closest to the observed
        # extreme value
        obs_extreme_index_0025 = np.abs(
            model_df_central_rps_this["0025"] - obs_max_dnw
        ).argmin()
        obs_extreme_index_0975 = np.abs(
            model_df_central_rps_this["0975"] - obs_max_dnw
        ).argmin()

        # Set up a new dataframe to append to
        model_df_this = pd.DataFrame(
            {
                "model_time": [time_point],
                "central_rp": [
                    model_df_central_rps_this.iloc[obs_extreme_index_central]["period"]
                ],
                "0025_rp": [
                    model_df_central_rps_this.iloc[obs_extreme_index_0025]["period"]
                ],
                "0975_rp": [
                    model_df_central_rps_this.iloc[obs_extreme_index_0975]["period"]
                ],
            }
        )

        # Append the model df to the model df plume
        model_df_plume = pd.concat([model_df_plume, model_df_this], ignore_index=True)

    # translate these return periods in years into percentages
    model_df_plume["central_rp_%"] = 1 / (model_df_plume["central_rp"] / 100)
    model_df_plume["0025_rp_%"] = 1 / (model_df_plume["0025_rp"] / 100)
    model_df_plume["0975_rp_%"] = 1 / (model_df_plume["0975_rp"] / 100)

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the central return levels as a red line
    ax.plot(
        model_df_plume["model_time"],
        model_df_plume["central_rp_%"],
        color="red",
        label="Central return period",
    )

    # Plot the 0.025 and 0.975 quantiles as dashed red lines
    # Shade the area between the 0.025 and 0.975 quantiles
    ax.fill_between(
        model_df_plume["model_time"],
        model_df_plume["0025_rp_%"],
        model_df_plume["0975_rp_%"],
        color="red",
        alpha=0.3,  # Adjust transparency
        label="Return period range (0.025 - 0.975)",
    )

    # Limit the y-axis to between 0 and 4
    # Set new tick labels
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    # Set new tick labels for the primary y-axis
    # Set up yticks for the primary y-axis
    ax.set_yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_yticklabels(
        [
            "0%",
            "1%",
            "2%",
            "3%",
            "4%",
            "5%",
            "6%",
            "7%",
            "8%",
            "9%",
            "10%",
            "11%",
            "12%",
        ]
    )

    # Set up yticks for the second y-axis
    ax2 = ax.twinx()

    # Synchronize the tick positions of the second y-axis with the first y-axis
    ax2.set_yticks(ax.get_yticks())  # Use the same tick positions as the primary y-axis

    # Set tick labels for the second y-axis (ensure the number of labels matches the number of ticks)
    ax2.set_yticklabels(
        ["", "100", "50", "33", "25", "20", "17", "14", "13", "11", "10", "9", "8"]
    )  # 13 labels

    # Set the y-axis limits for both axes to ensure alignment
    ax2.set_ylim(ax.get_ylim())  # Match the limits of the primary y-axis

    # Set the y axis labels
    ax2.set_ylabel(
        "Return period (years)",
        fontsize=12,
    )

    # Set up the xlabel
    ax.set_xlabel(
        "Year",
        fontsize=12,
    )

    # Set up the ylabel
    ax.set_ylabel(
        "Chance of event",
        fontsize=12,
    )

    # Set the xlims as the min and max of the model time
    ax.set_xlim(
        model_df_plume["model_time"].min(),
        model_df_plume["model_time"].max(),
    )

    # Set up the title
    ax.set_title(
        f"Chance of >2010 DnW by year",
        fontsize=12,
    )

    # include faint gridlines
    ax.grid(
        color="gray",
        linestyle="-",
        linewidth=0.5,
        alpha=0.5,
    )

    # Show the plot
    plt.show()

    return None


# Set up a function for plotting the distributions
def plot_distributions_extremes(
    model_df_full_field: pd.DataFrame,
    obs_df_full_field: pd.DataFrame,
    model_df_block: pd.DataFrame,
    obs_df_block: pd.DataFrame,
    model_var_name_full_field: str,
    obs_var_name_full_field: str,
    model_var_name_block: str,
    obs_var_name_block: str,
    xlabels: tuple[str, str],
    percentile: float = 0.05,
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """
    Plots two distributions of the model and obs data
    for the full field and the block min/max, along with
    the 5th percentile of the full distribution.

    Parameters
    ==========

    model_df_full_field : pd.DataFrame
        The dataframe containing the model data for the full field.
    obs_df_full_field : pd.DataFrame
        The dataframe containing the observed data for the full field.
    model_df_block : pd.DataFrame
        The dataframe containing the model data for the block.
    obs_df_block : pd.DataFrame
        The dataframe containing the observed data for the block.
    model_var_name_full_field : str
        The name of the model variable for the full field.
    obs_var_name_full_field : str
        The name of the observed variable for the full field.
    model_var_name_block : str
        The name of the model variable for the block.
    obs_var_name_block : str
        The name of the observed variable for the block.
    xlabels : tuple[str, str]
        The labels for the x-axis of the plots.
    percentile : float
        The percentile to plot the full field distribution.
    figsize : tuple[int, int]
        The size of the figure.

    Returns
    =======

    None
    """

    # Set up the figure
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharey=True,
        layout="constrained",
    )

    # Set up the axes
    ax1 = axs[0]
    ax2 = axs[1]

    # Plot the full field data distribution for the model in red
    ax1.hist(
        model_df_full_field[model_var_name_full_field],
        bins=50,
        color="red",
        alpha=0.5,
        label="Model full field",
        density=True,
    )

    # Plot the block data distribution for the model in orange
    ax1.hist(
        model_df_block[model_var_name_block],
        bins=50,
        color="orange",
        alpha=0.5,
        label="Model block",
        density=True,
    )

    # plot the 5th percentile of the full distribution as a red dashed line
    full_field_5th_percentile = np.percentile(
        model_df_full_field[model_var_name_full_field], percentile * 100
    )
    full_field_1th_percentile = np.percentile(
        model_df_full_field[model_var_name_full_field], 0.01 * 100
    )

    # plot the 5th percentile of the full distribution as a red dashed line
    ax1.axvline(
        full_field_5th_percentile,
        color="red",
        linestyle="--",
        label=f"Full field {percentile * 100:.0f}th percentile",
    )

    # plot the 1th percentile of the full distribution as a red dot dahsed line
    ax1.axvline(
        full_field_1th_percentile,
        color="red",
        linestyle="-.",
        label=f"Full field 1th percentile",
    )

    # Count the number of the block values which exceed the 5th percentile
    block_exceedances_5 = np.sum(
        model_df_block[model_var_name_block] > full_field_5th_percentile
    )
    block_exceedances_1 = np.sum(
        model_df_block[model_var_name_block] > full_field_1th_percentile
    )

    # Print the number of exceedances
    print(
        f"Number of model block exceedances for {percentile * 100:.0f}th percentile: {block_exceedances_5}"
    )
    print(
        f"Number of model block exceedances for 1th percentile: {block_exceedances_1}"
    )

    # Include the exceedences in a textbox in the top left
    ax1.text(
        0.05,
        0.95,
        f"Model block exceedances:\n{percentile * 100:.0f}th percentile: {block_exceedances_5}\n1th percentile: {block_exceedances_1}",
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Plot the full field data distribution for the obs in blue
    ax2.hist(
        obs_df_full_field[obs_var_name_full_field],
        bins=50,
        color="grey",
        alpha=0.5,
        label="Obs full field",
        density=True,
    )

    # Plot the block data distribution for the obs in blue
    ax2.hist(
        obs_df_block[obs_var_name_block],
        bins=50,
        color="blue",
        alpha=0.5,
        label="Obs block",
        density=True,
    )

    # plot the 5th percentile of the full distribution as a blue dashed line
    obs_full_field_5th_percentile = np.percentile(
        obs_df_full_field[obs_var_name_full_field], percentile * 100
    )
    obs_full_field_1th_percentile = np.percentile(
        obs_df_full_field[obs_var_name_full_field], 0.01 * 100
    )

    # plot the 5th percentile of the full distribution as a blue dashed line
    ax2.axvline(
        obs_full_field_5th_percentile,
        color="blue",
        linestyle="--",
        label=f"Full field {percentile * 100:.0f}th percentile",
    )

    # plot the 1th percentile of the full distribution as a blue dot dahsed line
    ax2.axvline(
        obs_full_field_1th_percentile,
        color="blue",
        linestyle="-.",
        label=f"Full field 1th percentile",
    )

    # Count the number of the block values which exceed the 5th percentile
    obs_block_exceedances_5 = np.sum(
        obs_df_block[obs_var_name_block] > obs_full_field_5th_percentile
    )
    obs_block_exceedances_1 = np.sum(
        obs_df_block[obs_var_name_block] > obs_full_field_1th_percentile
    )

    # Print the number of exceedances
    print(
        f"Number of obs block exceedances for {percentile * 100:.0f}th percentile: {obs_block_exceedances_5}"
    )
    print(
        f"Number of obs block exceedances for 1th percentile: {obs_block_exceedances_1}"
    )

    # Include the exceedences in a textbox in the top left
    ax2.text(
        0.05,
        0.95,
        f"Obs block exceedances:\n{percentile * 100:.0f}th percentile: {obs_block_exceedances_5}\n1th percentile: {obs_block_exceedances_1}",
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )

    # Include legends in the top right
    ax1.legend(
        loc="upper right",
        fontsize=10,
    )
    ax2.legend(
        loc="upper right",
        fontsize=10,
    )

    # Set the xlabels
    ax1.set_xlabel(xlabels[0])
    ax2.set_xlabel(xlabels[1])

    # Set the titles
    ax1.set_title("DePreSys")
    ax2.set_title("ERA5")

    # Remove the y labels
    ax1.set_ylabel("")
    ax2.set_ylabel("")

    # remove the yticks
    ax1.set_yticks([])
    ax2.set_yticks([])

    return None


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
    offshore_cap: float = 14733.02,  # https://www.renewableuk.com/energypulse/ukwed/
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

    # create a copy of the obs df
    obs_df_copy = obs_df.copy()
    model_df_copy = model_df.copy()

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
        np.median(obs_df_copy[obs_ws_col]),
        1,
        min(ch_df["total_gen"]),
    ]

    # Extract the first and last years (i.e., YYYY) from the date range
    start_year = int(date_range[0].split("-")[0])
    end_year = int(date_range[1].split("-")[0])

    # set up the obs df copy subset
    obs_df_copy_subset = obs_df_copy[
        (obs_df_copy["effective_dec_year"] >= start_year)
        & (obs_df_copy["effective_dec_year"] < end_year)
    ]

    # Fit the sigmoid curve to the observed data
    popt, pcov = curve_fit(
        sigmoid,
        obs_df_copy_subset[obs_ws_col],
        ch_df["total_gen"],
        p0=p0,
        method="dogbox",
    )

    # Apply the sigmoid function to the observed data
    obs_df_copy[f"{obs_ws_col}_sigmoid_total_wind_gen"] = sigmoid(
        obs_df_copy[obs_ws_col], *popt
    )

    # Do the same for the model data
    model_df_copy[f"{model_ws_col}_sigmoid_total_wind_gen"] = sigmoid(
        model_df_copy[model_ws_col], *popt
    )

    # If any of the sigmoid_total_wind_gen values are negative, set them to 0
    if any(obs_df_copy[f"{obs_ws_col}_sigmoid_total_wind_gen"] < 0):
        print("Negative values in obs sigmoid_total_wind_gen, setting to 0")
        obs_df_copy[f"{obs_ws_col}_sigmoid_total_wind_gen"] = obs_df_copy[
            f"{obs_ws_col}_sigmoid_total_wind_gen"
        ].clip(lower=0)

    if any(model_df_copy[f"{model_ws_col}_sigmoid_total_wind_gen"] < 0):
        print("Negative values in model sigmoid_total_wind_gen, setting to 0")
        model_df_copy[f"{model_ws_col}_sigmoid_total_wind_gen"] = model_df_copy[
            f"{model_ws_col}_sigmoid_total_wind_gen"
        ].clip(lower=0)

    return obs_df_copy, model_df_copy


# Write a function to convert the temp (C) to weather dependent electricity demand
def temp_to_demand(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_temp_col: str,
    model_temp_col: str,
    hdd_base: float = 15.5,
    cdd_base: float = 22.0,
    regr_coeffs_fpath: str = "/home/users/benhutch/ERA5_energy_update/ERA5_Regression_coeffs_demand_model.csv",
    country: str = "United_Kingdom",
    demand_year: int = 2017,
) -> pd.DataFrame:
    """
    Converts temperature to electricity demand using regression coefficients
    as quantified from the observations.

    Parameters
    ==========

    obs_df : pd.DataFrame
        The dataframe containing the observed temperature data.
    model_df : pd.DataFrame
        The dataframe containing the model temperature data.
    obs_temp_col : str
        The name of the observed temperature column.
    model_temp_col : str
        The name of the model temperature column.
    hdd_base : float
        The base temperature for heating degree days.
    cdd_base : float
        The base temperature for cooling degree days.
    regr_coeffs_fpath : str
        The file path to the regression coefficients.
    country : str
        The country for which to calculate the demand.
    demand_year : int
        The year for which to calculate the demand.

    Returns
    =======

    obs_df: pd.DataFrame
        The dataframe containing the observed electricity demand data.
    model_df: pd.DataFrame
        The dataframe containing the model electricity demand data.

    """

    # Create a copy of the obs and model dfs
    obs_df_copy = obs_df.copy()
    model_df_copy = model_df.copy()

    # assertr that temperature is in C
    assert obs_df_copy[obs_temp_col].max() < 100, "Temperature is not in C"
    assert model_df_copy[model_temp_col].max() < 100, "Temperature is not in C"

    # Process the observed data
    obs_df_copy["hdd"] = obs_df_copy[obs_temp_col].apply(lambda x: max(0, hdd_base - x))
    obs_df_copy["cdd"] = obs_df_copy[obs_temp_col].apply(lambda x: max(0, x - cdd_base))

    # Process the model data in the same way
    model_df_copy["hdd"] = model_df_copy[model_temp_col].apply(
        lambda x: max(0, hdd_base - x)
    )
    model_df_copy["cdd"] = model_df_copy[model_temp_col].apply(
        lambda x: max(0, x - cdd_base)
    )

    # Load the regression coefficients
    df_regr = pd.read_csv(regr_coeffs_fpath)

    # Set the index
    df_regr.set_index("Unnamed: 0", inplace=True)

    # Rename the columns by splitting by _ and extracting the second element
    df_regr.columns = [x.split("_")[0] for x in df_regr.columns]

    # if there is a column called "United" replace it with "United Kingdom"
    if "United" in df_regr.columns:
        df_regr.rename(columns={"United": "United_Kingdom"}, inplace=True)

    # Extract the regression coefficients for the country
    time_coeff_uk = df_regr.loc["time", country]
    hdd_coeff_uk = df_regr.loc["HDD", country]
    cdd_coeff_uk = df_regr.loc["CDD", country]

    # Calculate the observed demand
    obs_df_copy[f"{obs_temp_col}_UK_demand"] = (
        (time_coeff_uk * demand_year)
        + (hdd_coeff_uk * obs_df_copy["hdd"])
        + (cdd_coeff_uk * obs_df_copy["cdd"])
    )

    # Calculate the model demand
    model_df_copy[f"{model_temp_col}_UK_demand"] = (
        (time_coeff_uk * demand_year)
        + (hdd_coeff_uk * model_df_copy["hdd"])
        + (cdd_coeff_uk * model_df_copy["cdd"])
    )

    # Set up the range of hdd_coefs
    hdd_coeffs = np.linspace(0.60, 0.90, num=1000)

    # print the min and max of hdd coeffs
    print(f"Min HDD coeff: {hdd_coeffs.min():.2f}")
    print(f"Max HDD coeff: {hdd_coeffs.max():.2f}")

    # loop over the hdd coeffs and calculate the demand
    for hdd_coeff in hdd_coeffs:
        # Calculate the demand using the hdd coeff
        model_df_copy[f"{model_temp_col}_UK_demand_{hdd_coeff}"] = (
            (time_coeff_uk * demand_year)
            + (hdd_coeff * model_df_copy["hdd"])
            + (cdd_coeff_uk * model_df_copy["cdd"])
        )

    # # Set up a figure
    # fig = plt.figure(figsize=(6, 6))

    # # For the model data, plot HDD on the x-axis and demand on the y-axis
    # plt.scatter(
    #     model_df_copy["hdd"],
    #     model_df_copy[f"{model_temp_col}_UK_demand"],
    #     color="red",
    #     alpha=0.5,
    #     label="Model HDD vs Demand",
    # )

    # # Plot the fit used in the model data
    # plt.plot(
    #     model_df_copy["hdd"],
    #     (time_coeff_uk * demand_year)
    #     + (hdd_coeff_uk * model_df_copy["hdd"])
    #     + (cdd_coeff_uk * model_df_copy["cdd"]),
    #     color="black",
    #     linestyle="--",
    #     label="Model fit",
    # )

    # # loop over and plot the fit usin all of the hdd coeffs
    # for hdd_coeff in hdd_coeffs:
    #     plt.plot(
    #         model_df_copy["hdd"],
    #         (time_coeff_uk * demand_year)
    #         + (hdd_coeff * model_df_copy["hdd"])
    #         + (cdd_coeff_uk * model_df_copy["cdd"]),
    #         color="grey",
    #         alpha=0.01,
    #     )

    # # Include the coefficients in a textbox in the bottom right of the plot
    # plt.text(
    #     0.95,
    #     0.05,
    #     f"Model coefficients:\n"
    #     f"Time: {time_coeff_uk:.2f}\n"
    #     f"HDD: {hdd_coeff_uk:.2f}\n"
    #     f"CDD: {cdd_coeff_uk:.2f}",
    #     transform=fig.transFigure,
    #     fontsize=10,
    #     verticalalignment="bottom",
    #     horizontalalignment="right",
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    # )

    # # include a legend in the top left
    # plt.legend(
    #     loc="upper left",
    #     fontsize=10,
    # )

    # # include labels and title
    # plt.xlabel("HDD (C)", fontsize=12)
    # plt.ylabel("Electricity demand (GW)", fontsize=12)

    return obs_df_copy, model_df_copy


# Define a function to plot the distribution by percentiles
def plot_multi_var_perc(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    x_var_name_obs: str,
    y_var_name_obs: str,
    x_var_name_model: str,
    y_var_name_model: str,
    xlabel: str,
    ylabel: str,
    title: str,
    legend_y1: str = None,
    legend_y2: str = None,
    y_var_name_model_2: str = None,
    ylabel_2: str = None,
    y2_var_name_model: str = None,
    y2_label: str = None,
    figsize: tuple[int, int] = (5, 10),
    inverse_flag = False,
    y1_zero_line = False,
    xlims: tuple[float, float] = None,
    ylims: tuple[float, float] = None,
    y2_lims: tuple[float, float] = None,
    x2_var_name_model: str = None,
):
    """
    Plots the relationship between variables as percentiles. E.g., binned by
    percentiles of temperature, what is the wind speed doing.

    Parameters
    ==========

        obs_df : pd.DataFrame
            The dataframe containing the observed data.
        model_df : pd.DataFrame
            The dataframe containing the model data.
        x_var_name_obs : str
            The name of the observed x variable.
        y_var_name_obs : str
            The name of the observed y variable.
        x_var_name_model : str
            The name of the model x variable.
        y_var_name_model : str
            The name of the model y variable.
        xlabel : str
            The label for the x-axis.
        ylabel : str
            The label for the y-axis.
        title : str
            The title for the plot.
        y2_var_name_model : str
            The name of the model y2 variable (optional).
        figsize : tuple[int, int]
            The size of the figure.

    Returns
    =======

        None

    """

    # Create copies of the df to work from here
    obs_df_copy = obs_df.copy()
    model_df_copy = model_df.copy()

    # Hard code the percentiles in increments of 5
    percentiles_5 = np.arange(0, 99 + 1, 1)  # 0, 5, 10, ..., 95

    # Set up new dataframes for the observed and model percentiles
    obs_percs_5 = pd.DataFrame()
    model_percs_5 = pd.DataFrame()
    model_percs_5_x2 = pd.DataFrame()

    # Loop through the percentiles
    for perc_this in percentiles_5:
        # Find the lower and upper bound
        lower_bound_this = perc_this / 100
        if perc_this + 5 >= 100:
            upper_bound_this = 1.0
        else:
            upper_bound_this = (perc_this + 5) / 100  # Increment by 5%

        # # Set up the lower_bound_this inverse
        # lower_bound_this_inverse = 1 - upper_bound_this
        # upper_bound_this_inverse = 1 - lower_bound_this

        # Find the lower bound for the obs
        obs_lower_bound_this = obs_df_copy[x_var_name_obs].quantile(lower_bound_this)
        obs_upper_bound_this = obs_df_copy[x_var_name_obs].quantile(upper_bound_this)

        # Find the lower bound for the model
        model_lower_bound_this = model_df_copy[x_var_name_model].quantile(lower_bound_this)
        model_upper_bound_this = model_df_copy[x_var_name_model].quantile(upper_bound_this)
        
        # if x2_var_name_model is not None, find the lower and upper bounds for it
        if x2_var_name_model is not None:
            model_lower_bound_this_x2 = model_df_copy[x2_var_name_model].quantile(lower_bound_this)
            model_upper_bound_this_x2 = model_df_copy[x2_var_name_model].quantile(upper_bound_this)

            # Subset the dataframes to the lower and upper bounds
            model_df_this_x2 = model_df_copy[
                (model_df_copy[x2_var_name_model] >= model_lower_bound_this_x2)
                & (model_df_copy[x2_var_name_model] < model_upper_bound_this_x2)
            ]

            # Set up a new dataframe for the model with x2 variable
            model_perc_df_this = pd.DataFrame(
                {
                    "percentile": [perc_this],
                    "lower_bound": [model_lower_bound_this_x2],
                    "upper_bound": [model_upper_bound_this_x2],
                    "n_days": [model_df_this_x2.shape[0]],
                    f"{y_var_name_model}_mean": [model_df_this_x2[y_var_name_model].mean()],
                    f"{y_var_name_model}_lower": [
                        model_df_this_x2[y_var_name_model].quantile(0.10)
                    ],
                    f"{y_var_name_model}_upper": [
                        model_df_this_x2[y_var_name_model].quantile(0.90)
                    ],
                }
            )

            # if there is an x2 variable
            if y2_var_name_model is not None:
                # Add the y2 variable to the model dataframe
                model_perc_df_this[f"{y2_var_name_model}_mean"] = model_df_this_x2[
                    y2_var_name_model
                ].mean()
                model_perc_df_this[f"{y2_var_name_model}_lower"] = model_df_this_x2[
                    y2_var_name_model
                ].quantile(0.10)
                model_perc_df_this[f"{y2_var_name_model}_upper"] = model_df_this_x2[
                    y2_var_name_model
                ].quantile(0.90)

            # concat to the df
            model_percs_5_x2 = pd.concat([model_percs_5_x2, model_perc_df_this])

        # Subset the dataframes to the lower and upper bounds
        obs_df_this = obs_df_copy[
            (obs_df_copy[x_var_name_obs] >= obs_lower_bound_this)
            & (obs_df_copy[x_var_name_obs] < obs_upper_bound_this)
        ]
        model_df_this = model_df_copy[
            (model_df_copy[x_var_name_model] >= model_lower_bound_this)
            & (model_df_copy[x_var_name_model] < model_upper_bound_this)
        ]

        # Set up a new dataframe for the obs
        obs_perc_df_this = pd.DataFrame(
            {
                "percentile": [perc_this],
                "lower_bound": [obs_lower_bound_this],
                "upper_bound": [obs_upper_bound_this],
                "n_days": [obs_df_this.shape[0]],
                f"{y_var_name_obs}_mean": [obs_df_this[y_var_name_obs].mean()],
            }
        )

        # Set up a new dataframe for the model
        model_perc_df_this = pd.DataFrame(
            {
                "percentile": [perc_this],
                "lower_bound": [model_lower_bound_this],
                "upper_bound": [model_upper_bound_this],
                "n_days": [model_df_this.shape[0]],
                f"{y_var_name_model}_mean": [model_df_this[y_var_name_model].mean()],
                f"{y_var_name_model}_lower": [
                    model_df_this[y_var_name_model].quantile(0.10)
                ],
                f"{y_var_name_model}_upper": [
                    model_df_this[y_var_name_model].quantile(0.90)
                ],
            }
        )

        # if there is a y2 variable, add it to the model dataframe
        if y2_var_name_model is not None:
            model_perc_df_this[f"{y2_var_name_model}_mean"] = model_df_this[
                y2_var_name_model
            ].mean()
            model_perc_df_this[f"{y2_var_name_model}_lower"] = model_df_this[
                y2_var_name_model
            ].quantile(0.10)
            model_perc_df_this[f"{y2_var_name_model}_upper"] = model_df_this[
                y2_var_name_model
            ].quantile(0.90)

        # if there is a y2 variable, add it to the obs dataframe
        if y_var_name_model_2 is not None:
            model_perc_df_this[f"{y_var_name_model_2}_mean"] = model_df_this[
                y_var_name_model_2
            ].mean()

        # Concat these dataframes
        obs_percs_5 = pd.concat([obs_percs_5, obs_perc_df_this])
        model_percs_5 = pd.concat([model_percs_5, model_perc_df_this])

    save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    #  Set up the current time as YYYYMMDD-HHMMSS
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

    # set up fnames
    obs_fname = f"{save_dir}/obs_perc_5_{current_time}.csv"
    model_fname = f"{save_dir}/model_perc_5_{current_time}.csv"

    # if the obs_percs_5 is not empty, save it
    if not obs_percs_5.empty:
        obs_percs_5.to_csv(obs_fname, index=False)
        print(f"Saved obs_percs_5 to {obs_fname}")

    # if the model_percs_5 is not empty, save it
    if not model_percs_5.empty:
        model_percs_5.to_csv(model_fname, index=False)
        print(f"Saved model_percs_5 to {model_fname}")

    # Set up the figure
    fig, ax = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=figsize,
    )

    # if the inverse flag is set, invert the y axis
    if inverse_flag:
        # Do the same for the model
        ax.plot(
            100 - model_percs_5["percentile"],
            model_percs_5[f"{y_var_name_model}_mean"],
            color="red",
            label=f"{legend_y1}",
        )

        # plot the lower bounds as a dashed red line
        ax.plot(
            100 - model_percs_5["percentile"],
            model_percs_5[f"{y_var_name_model}_lower"],
            color="red",
            linestyle="--",
        )

        # plot the upper bounds as a dashed red line
        ax.plot(
            100 - model_percs_5["percentile"],
            model_percs_5[f"{y_var_name_model}_upper"],
            color="red",
            linestyle="--",
        )

        # if the x2 variable is not None, plot it
        if x2_var_name_model is not None:
            ax.plot(
                model_percs_5_x2["percentile"],
                model_percs_5_x2[f"{y_var_name_model}_mean"],
                color="orange",
                label=f"{legend_y1}",
            )

            # plot the lower bounds as a dashed orange line
            ax.plot(
                model_percs_5_x2["percentile"],
                model_percs_5_x2[f"{y_var_name_model}_lower"],
                color="orange",
                linestyle="--",
            )

            # plot the upper bounds as a dashed orange line
            ax.plot(
                model_percs_5_x2["percentile"],
                model_percs_5_x2[f"{y_var_name_model}_upper"],
                color="orange",
                linestyle="--",
            )
        # if the y2 variable is not None, plot it
        if y_var_name_model_2 is not None:
            ax.plot(
                100 - model_percs_5["percentile"],
                model_percs_5[f"{y_var_name_model_2}_mean"],
                color="red",
                linestyle="--",
                label=f"{legend_y2}",
            )

        # Plot the 5% percentiles for temperature for wind speed
        # ax.plot(
        #     100 - obs_percs_5["percentile"],
        #     obs_percs_5[f"{y_var_name_obs}_mean"],
        #     color="black",
        #     label=f"{ylabel} (5% temp bins)"
        # )
    else:
        # # Plot the 5% percentiles for temperature for wind speed
        # ax.plot(
        #     obs_percs_5["percentile"],
        #     obs_percs_5[f"{y_var_name_obs}_mean"],
        #     color="black",
        #     label=f"{ylabel} (5% temp bins)"
        # )

        # if the y2 variable is not None, plot it
        if y_var_name_model_2 is not None:
            ax.plot(
                model_percs_5["percentile"],
                model_percs_5[f"{y_var_name_model_2}_mean"],
                color="red",
                linestyle="--",
                label=f"{legend_y1}",
            )

        # Do the same for the model
        ax.plot(
            model_percs_5["percentile"],
            model_percs_5[f"{y_var_name_model}_mean"],
            color="red",
            label=f"{legend_y1}",
        )

        # plot the lower bounds as a dashed red line
        ax.plot(
            model_percs_5["percentile"],
            model_percs_5[f"{y_var_name_model}_lower"],
            color="red",
            linestyle="--",
        )

        # plot the upper bounds as a dashed red line
        ax.plot(
            model_percs_5["percentile"],
            model_percs_5[f"{y_var_name_model}_upper"],
            color="red",
            linestyle="--",
        )

        # if x2_var_name_model is not None:
        if x2_var_name_model is not None:
            ax.plot(
                model_percs_5_x2["percentile"],
                model_percs_5_x2[f"{y_var_name_model}_mean"],
                color="orange",
                label=f"{legend_y2}",
            )

            # plot the lower bounds as a dashed orange line
            ax.plot(
                model_percs_5_x2["percentile"],
                model_percs_5_x2[f"{y_var_name_model}_lower"],
                color="orange",
                linestyle="--",
            )

            # plot the upper bounds as a dashed orange line
            ax.plot(
                model_percs_5_x2["percentile"],
                model_percs_5_x2[f"{y_var_name_model}_upper"],
                color="orange",
                linestyle="--",
            )

    # if y1 zero line is True
    if y1_zero_line:
        # include a red dashed zero line
        ax.axhline(
            0,
            color="red",
            linestyle="--",
        )

    # If there is a y2 variable, plot it
    if y2_var_name_model is not None:
        # Create a second y-axis
        ax2 = ax.twinx()

        if inverse_flag:
            ax2.plot(
                100 - model_percs_5["percentile"],
                model_percs_5[f"{y2_var_name_model}_mean"],
                color="blue",
                label=f"{legend_y2}",
            )

            # plot the lower bounds as a dashed red line
            ax2.plot(
                100 - model_percs_5["percentile"],
                model_percs_5[f"{y2_var_name_model}_lower"],
                color="blue",
                linestyle="--",
            )

            # plot the upper bounds as a dashed red line
            ax2.plot(
                100 - model_percs_5["percentile"],
                model_percs_5[f"{y2_var_name_model}_upper"],
                color="blue",
                linestyle="--",
            )
        else:
            ax2.plot(
                model_percs_5["percentile"],
                model_percs_5[f"{y2_var_name_model}_mean"],
                color="blue",
                label=f"{legend_y2}",
            )

            # plot the lower bounds as a dashed red line
            ax2.plot(
                model_percs_5["percentile"],
                model_percs_5[f"{y2_var_name_model}_lower"],
                color="blue",
                linestyle="--",
            )

            # plot the upper bounds as a dashed red line
            ax2.plot(
                model_percs_5["percentile"],
                model_percs_5[f"{y2_var_name_model}_upper"],
                color="blue",
                linestyle="--",
            )

            if x2_var_name_model is not None:
                ax2.plot(
                    model_percs_5_x2["percentile"],
                    model_percs_5_x2[f"{y2_var_name_model}_mean"],
                    color="green",
                    label=f"{legend_y2}",
                )

                # plot the lower bounds as a dashed orange line
                ax2.plot(
                    model_percs_5_x2["percentile"],
                    model_percs_5_x2[f"{y2_var_name_model}_lower"],
                    color="green",
                    linestyle="--",
                )

                # plot the upper bounds as a dashed orange line
                ax2.plot(
                    model_percs_5_x2["percentile"],
                    model_percs_5_x2[f"{y2_var_name_model}_upper"],
                    color="green",
                    linestyle="--",
                )

        # incldue a blue dashed zero line
        # ax2.axhline(
        #     0,
        #     color="blue",
        #     linestyle="--",
        # )

        # Set the y2 label
        ax2.set_ylabel(f"{y2_label}", fontsize=12, color="blue")

        # make sure the ticks/labels are also blue
        ax2.tick_params(axis="y", labelcolor="blue")
        ax2.spines["right"].set_color("blue")

    # if xlims is not none
    if xlims is not None:
        # Set the xlims
        ax.set_xlim(xlims)

    # if ylims is not none
    if ylims is not None:
        # Set the ylims
        ax.set_ylim(ylims)

    # if y2_lims is not None
    if y2_lims is not None:
        # Set the y2lims
        ax2.set_ylim(y2_lims)

    # Set the x and y labels
    ax.set_xlabel(f"{xlabel}", fontsize=12)
    ax.set_ylabel(f"{ylabel}", fontsize=12, color="red")

    # Make sure the ticks and labels are also red
    ax.tick_params(axis="y", labelcolor="red")
    ax.spines["left"].set_color("red")

    # Include gridlines
    ax.grid()

    # Get handles and labels from both ax and ax2
    handles_ax, labels_ax = ax.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()

    # Combine the handles and labels
    handles = handles_ax + handles_ax2
    labels = labels_ax + labels_ax2

    # Add a single legend to the figure
    ax.legend(
        handles=handles,
        labels=labels,
        loc="upper left",
        fontsize=10,
    )

    # Set the title
    ax.set_title(title)

    # Set up the x ticks for the percentiel axis
    x_ticks_perc = [
        20, 40, 60, 80, 100
    ]

    x_tick_vals_model = []
    x_tick_vals_obs = []

    # Find the values corresponding to these percentiles
    # in both the mnodel and obs df
    for x_tick in x_ticks_perc:
        # do the percentile correctly for inverted T
        x_tick = 100 - x_tick
        
        # Find the closest value of percentile in the observed DataFrame
        obs_df_this = obs_percs_5.iloc[
            (np.abs(obs_percs_5["percentile"] - x_tick)).argmin()
        ]

        # Find the closest value of percentile in the model DataFrame
        model_df_this = model_percs_5.iloc[
            (np.abs(model_percs_5["percentile"] - x_tick)).argmin()
        ]

        # Get the values
        obs_val_this = (obs_df_this["lower_bound"] + obs_df_this["upper_bound"]) / 2
        model_val_this = (model_df_this["lower_bound"] + model_df_this["upper_bound"]) / 2

        # Print the values
        print("x tick this: ", x_tick)
        print("obs val this: ", obs_val_this)
        print("model val this: ", model_val_this)

        # Append to the lists
        x_tick_vals_obs.append(obs_val_this)
        x_tick_vals_model.append(model_val_this)

    # format the values to two sf
    x_tick_vals_obs = [f"{x:.2f}" for x in x_tick_vals_obs]
    x_tick_vals_model = [f"{x:.2f}" for x in x_tick_vals_model]

    # Create a second x-axis
    secax = ax.secondary_xaxis("top")
    secax.set_xticks(x_ticks_perc)
    secax.set_xticklabels(x_tick_vals_model)

    # Set a tight layour
    plt.tight_layout()

    # Show the plot
    plt.show()

    return None

# Define a function
# to plot the subplots of variable against variable
def plot_multi_var_scatter(
    subset_obs_dfs: list[pd.DataFrame],
    subset_model_dfs: list[pd.DataFrame],
    x_var_name_obs: str,
    y_var_name_obs: str,
    x_var_name_model: str,
    y_var_name_model: str,
    xlabel: str,
    ylabel: str,
    subtitles: list[str],
    colours: list[str] = ["grey", "orange", "red"],
    figsize: tuple[int, int] = (10, 5),
):
    """
    Plots scatter plots showing the relationship between variables.

    Parameters
    ==========

        subset_obs_dfs : list[pd.DataFrame]
            List of dataframes containing the observed data.
        subset_model_dfs : list[pd.DataFrame]
            List of dataframes containing the model data.
        x_var_name_obs : str
            Name of the x variable in the observed data.
        y_var_name_obs : str
            Name of the y variable in the observed data.
        x_var_name_model : str
            Name of the x variable in the model data.
        y_var_name_model : str
            Name of the y variable in the model data.
        xlabel : str
            Label for the x-axis.
        ylabel : str
            Label for the y-axis.
        subtitles : list[str]
            List of subtitles for each subplot.
        figsize : tuple[int, int], optional
            Size of the figure, by default (10, 5)

    Returns
    =======

        None

    """

    # Set up the figure
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize,
        layout="constrained",
        sharey=True,
        sharex=True,
    )

    # Loop through the dataframes
    for i, (obs_df, model_df) in enumerate(zip(subset_obs_dfs, subset_model_dfs)):
        # Set up the N_obs
        N_obs = obs_df.shape[0]
        N_model = model_df.shape[0]

        # print the n model
        print(f"N model: {N_model}")
        print(f"N obs: {N_obs}")

        # if the len of obs is greater than one
        if N_obs > 1:
            # Calculate the correlation for the obs
            r_obs, _ = pearsonr(obs_df[x_var_name_obs], obs_df[y_var_name_obs])
        else:
            r_obs = np.nan

        # Calculate the correlation for the model
        r_model, _ = pearsonr(model_df[x_var_name_model], model_df[y_var_name_model])

        # Plot the model data
        axs[i].scatter(
            model_df[x_var_name_model],
            model_df[y_var_name_model],
            color=colours[i],
            alpha=0.5,
            label=f"Model (r={r_model:.2f}, N={N_model})",
        )

        # Plot the observed data
        axs[i].scatter(
            obs_df[x_var_name_obs],
            obs_df[y_var_name_obs],
            color="black",
            marker="x",
            label=f"Obs (r={r_obs:.2f}, N={N_obs})",
        )

        # include a vertical line for the mean of the x var name
        # in the mode
        # in the correct colour
        axs[i].axvline(
            model_df[x_var_name_model].mean(),
            color=colours[i],
            linestyle="--",
        )

        # DO the same for a horizontal line for the mean of the y var name
        axs[i].axhline(
            model_df[y_var_name_model].mean(),
            color=colours[i],
            linestyle="--",
        )

        # do a vertical dot dashed line in black for the mean of the obs x var
        axs[i].axvline(
            obs_df[x_var_name_obs].mean(),
            color="black",
            linestyle="-.",
        )

        # do the same for the y var
        axs[i].axhline(
            obs_df[y_var_name_obs].mean(),
            color="black",
            linestyle="-.",
        )

        # Set the labels and title
        axs[i].set_xlabel(xlabel)

        if i == 0:
            axs[i].set_ylabel(ylabel)

        # Set up the titles
        axs[i].set_title(subtitles[i])

        # include a legend in the top right
        axs[i].legend(
            loc="upper right",
            fontsize=10,
        )

    return None


# Define the main function
def main():
    # Start the timer
    start = time.time()

    # Hardcode the paths
    obs_tas_block_min_path = "/home/users/benhutch/unseen_multi_year/dfs/block_minima_obs_tas_UK_1961-2024_DJF_detrended.csv_06-05-2025"
    obs_wind_block_min_path = "/home/users/benhutch/unseen_multi_year/dfs/block_minima_obs_wind_UK_1961-2024_DJF_detrended.csv_06-05-2025"
    model_tas_block_min_path = "/home/users/benhutch/unseen_multi_year/dfs/block_minima_model_tas_UK_1961-2024_DJF_detrended.csv_06-05-2025"
    model_wind_block_min_path = "/home/users/benhutch/unseen_multi_year/dfs/block_minima_model_wind_UK_1961-2024_DJF_detrended.csv_06-05-2025"

    # load the dfs
    df_obs_tas_block_min = pd.read_csv(obs_tas_block_min_path)
    df_obs_wind_block_min = pd.read_csv(obs_wind_block_min_path)
    df_model_tas_block_min = pd.read_csv(model_tas_block_min_path)
    df_model_wind_block_min = pd.read_csv(model_wind_block_min_path)

    # Set up the directory in which the dfs are stored
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"

    # Set up the years test
    test_years = np.arange(1960, 2018 + 1, 1)
    members = np.arange(1, 10 + 1, 1)

    # Set up a list to store the missing fnames
    missing_fnames = []
    missing_fname_years = []

    # Set up an empty dataframe
    df_delta_p_full = pd.DataFrame()
    df_uas_full = pd.DataFrame()
    df_vas_full = pd.DataFrame()

    # Loop over the years
    for year in test_years:
        for member in members:
            # Set up the test fname
            test_fname = f"HadGEM3-GC31-MM_dcppA-hindcast_psl_delta_p_{year}_{member}_day.csv"

            # Set up the test fname for the uas and vas
            test_fname_uas = f"HadGEM3-GC31-MM_dcppA-hindcast_uas_UK_wind_box_{year}_{member}_day.csv"
            test_fname_vas = f"HadGEM3-GC31-MM_dcppA-hindcast_vas_UK_wind_box_{year}_{member}_day.csv"

            # Set up thge output dir
            # Set up the new base dir
            base_dir_new = "/home/users/benhutch/unseen_data/saved_dfs"

            # Set up the new output directory
            new_output_dir = os.path.join(
                base_dir_new,
                "delta_p",
                str(year),
            )

            # Cehck if the file exists
            if os.path.exists(os.path.join(new_output_dir, test_fname)):
                # Load the df
                df_delta_p_this = pd.read_csv(
                    os.path.join(new_output_dir, test_fname)
                )

                # concat the df to the full df
                df_delta_p_full = pd.concat([df_delta_p_full, df_delta_p_this])
            else:
                missing_fnames.append(test_fname)
                missing_fname_years.append(year)

            # Check if the uas file exists
            if os.path.exists(os.path.join(new_output_dir, test_fname_uas)):
                # Load the df
                df_uas_this = pd.read_csv(
                    os.path.join(new_output_dir, test_fname_uas)
                )

                # concat the df to the full df
                df_uas_full = pd.concat([df_uas_full, df_uas_this])
            else:
                missing_fnames.append(test_fname_uas)
                missing_fname_years.append(year)

            # Check if the vas file exists
            if os.path.exists(os.path.join(new_output_dir, test_fname_vas)):
                # Load the df
                df_vas_this = pd.read_csv(
                    os.path.join(new_output_dir, test_fname_vas)
                )

                # concat the df to the full df
                df_vas_full = pd.concat([df_vas_full, df_vas_this])
            else:
                missing_fnames.append(test_fname_vas)
                missing_fname_years.append(year)

            # # Check if the file exists
            # if os.path.exists(os.path.join(dfs_dir, test_fname)):
            #     # create a new directory to save to
            #     new_dir = os.path.join(
            #         dfs_dir,
            #         "delta_p",
            #         str(year),
            #     )

            #     # if the directory does not exist, create it
            #     if not os.path.exists(new_dir):
            #         os.makedirs(new_dir)

            #     # Move the file to the new directory
            #     shutil.move(
            #         os.path.join(dfs_dir, test_fname),
            #         os.path.join(new_dir, test_fname),
            #     )
            # else:
            #     # Append the fname to the list
            #     missing_fnames.append(test_fname)
            #     missing_fname_years.append(year)

    # Print the missing fnames
    print(f"Missing files: {missing_fnames}")

    # print the len of the missing fnames
    print(f"Number of missing files: {len(missing_fnames)}")

    # print the unique years
    print(f"Unique years: {len(set(missing_fname_years))}")

    # print the unique years
    print(f"Unique years: {set(missing_fname_years)}")

    # create a new column for delta_p_hpa as the difference between 
    # "data_n" and "data_s"
    df_delta_p_full["delta_p_hpa"] = (df_delta_p_full["data_n"] - df_delta_p_full["data_s"]) / 100

    # print the ehad of the df
    print(df_delta_p_full.head())

    # print the tail of the df
    print(df_delta_p_full.tail())

    # print te head of the uas dataframe
    print(df_uas_full.head())
    # print the tail of the uas dataframe
    print(df_uas_full.tail())

    # print the head of the vas dataframe
    print(df_vas_full.head())
    # print the tail of the vas dataframe
    print(df_vas_full.tail())

    # print the statistics of the df
    print(df_delta_p_full.describe())

    # sys.exit()

    # Load the model temperature data
    df_model_tas = pd.read_csv(
        os.path.join(
            dfs_dir,
            "HadGEM3-GC31-MM_dcppA-hindcast_tas_United_Kingdom_1960-2018_day.csv",
        )
    )

    # Load the model wind spped data
    df_model_sfcWind = pd.read_csv(
        os.path.join(
            dfs_dir,
            "HadGEM3-GC31-MM_dcppA-hindcast_sfcWind_UK_wind_box_1960-2018_day.csv",
        )
    )

    # Merge the two model dataframes on init_year, member, and lead
    df_model = df_model_tas.merge(
        df_model_sfcWind,
        on=["init_year", "member", "lead"],
        suffixes=("_tas", "_sfcWind"),
    )

    # merge the df delta p here as well
    df_model = df_model.merge(
        df_delta_p_full,
        on=["init_year", "member", "lead"],
        suffixes=("", ""),
    )

    # join the uas and vas dataframes
    df_model = df_model.merge(
        df_uas_full,
        on=["init_year", "member", "lead"],
        suffixes=("", "_uas"),
    )

    # join the uas and vas dataframes
    df_model = df_model.merge(
        df_vas_full,
        on=["init_year", "member", "lead"],
        suffixes=("", "_vas"),
    )

    # rename data to data_uas
    df_model.rename(columns={"data": "data_uas"}, inplace=True)

    # print the head of df_model
    print(df_model.head())

    # print the tail of df_model
    print(df_model.tail())

    # sys.exit()

    # drop data_n and data_s
    df_model.drop(columns=["data_n", "data_s"], inplace=True)

    # sys.exit()

    # Subset the leads for the valid winter years
    winter_years = np.arange(1, 11 + 1)

    # Process the df_model_djf
    df_model_djf = select_leads_wyears_DJF(df_model, winter_years)

    # Add the column for effective dec year to the df_model_djf
    df_model_djf["effective_dec_year"] = df_model_djf["init_year"] + (
        df_model_djf["winter_year"] - 1
    )

    # Load the observed data
    df_obs_tas = pd.read_csv(
        os.path.join(dfs_dir, "ERA5_tas_United_Kingdom_1960-2025_daily_2025-04-24.csv")
    )

    # Convert the 'time' column to datetime, assuming it represents days since "1950-01-01 00:00:00"
    df_obs_tas["time"] = pd.to_datetime(
        df_obs_tas["time"], origin="1950-01-01", unit="D"
    )

    # subset the obs data to D, J, F
    df_obs_tas = df_obs_tas[df_obs_tas["time"].dt.month.isin([12, 1, 2])]

    # new column for temp in C
    df_obs_tas["data_c"] = df_obs_tas["data"] - 273.15

    # Load the obs wind data
    df_obs_sfcWind = pd.read_csv(
        os.path.join(dfs_dir, "ERA5_sfcWind_UK_wind_box_1960-2025_daily_2025-05-20.csv")
    )

    # Set up the start and end date to use
    start_date = pd.to_datetime("1960-01-01")
    end_date = pd.to_datetime("2025-02-28")

    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Make sure time is a datetime
    df_obs_sfcWind["time"] = date_range

    # rename the 'obs_mean' column to 'data'
    df_obs_sfcWind.rename(columns={"obs_mean": "data"}, inplace=True)

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
    common_wyears = np.arange(1961, 2024 + 1)  # test full period first

    # Subset the obs data to the common_wyears
    df_obs = df_obs[df_obs["effective_dec_year"].isin(common_wyears)]

    # Subset the model data to the common_wyears
    df_model_djf = df_model_djf[df_model_djf["effective_dec_year"].isin(common_wyears)]

    # Create a new column for data_tas_c in df_model_full_djf
    df_model_djf["data_tas_c"] = df_model_djf["data_tas"] - 273.15

    # Plot the lead pdfs to visualise the biases/drifts
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_tas_c",
    #     obs_var_name="data_c",
    #     lead_name="winter_year",
    #     xlabel="Temperature (°C)",
    #     suptitle="Lead dependent temperature PDFs, DJF all days, 1961-2017",
    #     figsize=(10, 5),
    # )

    # # Plot the lead pdfs to visualise the biases/drifts
    # # but for wind speed
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_sfcWind",
    #     obs_var_name="data_sfcWind",
    #     lead_name="winter_year",
    #     xlabel="10m Wind Speed (m/s)",
    #     suptitle="Lead dependent wind speed PDFs, DJF all days, 1961-2017",
    #     figsize=(10, 5),
    # )

    # Apply the dirft correction to the model data
    df_model_djf = model_drift_corr_plot(
        model_df=df_model_djf,
        model_var_name="data_tas_c",
        obs_df=df_obs,
        obs_var_name="data_c",
        lead_name="winter_year",
        xlabel="Temperature (°C)",
        year1_year2_tuple=(1970, 2017),
        lead_day_name="lead",
        constant_period=True,
    )

    # do the same for tjhe wind speed data
    df_model_djf = model_drift_corr_plot(
        model_df=df_model_djf,
        model_var_name="data_sfcWind",
        obs_df=df_obs,
        obs_var_name="data_sfcWind",
        lead_name="winter_year",
        xlabel="10m Wind Speed (m/s)",
        year1_year2_tuple=(1970, 2017),
        lead_day_name="lead",
        constant_period=True,
    )

    # plot the lead pdfs to visualise the biases/drifts
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_tas_c_drift_bc",
    #     obs_var_name="data_c",
    #     lead_name="winter_year",
    #     xlabel="Temperature (°C)",
    #     suptitle="Lead dependent temperature PDFs, DJF all days, 1961-2017 (model drift + bias corrected)",
    #     figsize=(10, 5),
    # )

    # # Plot the lead pdfs to visualise the biases/drifts
    # # but for wind speed
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_sfcWind_drift_bc",
    #     obs_var_name="data_sfcWind",
    #     lead_name="winter_year",
    #     xlabel="10m Wind Speed (m/s)",
    #     suptitle="Lead dependent wind speed PDFs, DJF all days, 1961-2017 (model drift + bias corrected)",
    #     figsize=(10, 5),
    # )

    # sys.exit()

    #     # Plot the lead pdfs to visualise the biases/drifts
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_tas_c",
    #     obs_var_name="data_c",
    #     lead_name="winter_year",
    #     xlabel="Temperature (°C)",
    #     suptitle="Lead dependent temperature PDFs, DJF all days, 1961-2017",
    #     figsize=(10, 5),
    # )

    # # Plot the lead pdfs to visualise the biases/drifts
    # # but for wind speed
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_sfcWind",
    #     obs_var_name="data_sfcWind",
    #     lead_name="winter_year",
    #     xlabel="10m Wind Speed (m/s)",
    #     suptitle="Lead dependent wind speed PDFs, DJF all days, 1961-2017",
    #     figsize=(10, 5),
    # )

    # # Test the new function before all detrending takes place
    # pivot_emp_rps_dnw(
    #     obs_df=df_obs,
    #     model_df=df_model_djf,
    #     obs_var_name_wind="data_sfcWind",
    #     obs_var_name_tas="data_c",
    #     model_var_name_wind="data_sfcWind_drift_bc",
    #     model_var_name_tas="data_tas_c_drift_bc",
    #     model_time_name="effective_dec_year",
    #     obs_time_name="effective_dec_year",
    #     nsamples=1000,
    #     figsize=(5, 5),
    # )

    # sys.exit()

    # Pivot detrend the obs for temperature
    df_obs = gev_funcs.pivot_detrend_obs(
        df=df_obs,
        x_axis_name="effective_dec_year",
        y_axis_name="data_c",
    )

    # Pivot detrend the obs for wind speed
    df_obs = gev_funcs.pivot_detrend_obs(
        df=df_obs,
        x_axis_name="effective_dec_year",
        y_axis_name="data_sfcWind",
    )

    # perform the detrending on the model data
    df_model_djf = gev_funcs.pivot_detrend_model(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_x_axis_name="effective_dec_year",
        model_y_axis_name="data_tas_c_drift_bc",
        obs_x_axis_name="effective_dec_year",
        obs_y_axis_name="data_c",
        suffix="_dt",
    )

    # perform detrending on the non bias corrected data
    df_model_djf = gev_funcs.pivot_detrend_model(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_x_axis_name="effective_dec_year",
        model_y_axis_name="data_tas_c",
        obs_x_axis_name="effective_dec_year",
        obs_y_axis_name="data_c",
        suffix="_dt",
    )

    # compare the biases between these
    # Plot the lead pdfs to visualise the biases/drifts
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_tas_c_dt",
        obs_var_name="data_c_dt",
        lead_name="winter_year",
        xlabel="Temperature (°C)",
        suptitle="Lead dependent temperature PDFs, DJF all days, 1961-2017, detrended (no BC)",
        figsize=(10, 5),
    )

    # Plot the lead pdfs to visualise the biases/drifts
    # but for wind speed
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_tas_c_drift_bc_dt",
        obs_var_name="data_c_dt",
        lead_name="winter_year",
        xlabel="Temperature (°C)",
        suptitle="Lead dependent wind speed PDFs, DJF all days, 1961-2017, detrended (BC)",
        figsize=(10, 5),
    )

    # sys.exit()

    # apply a detrend to the wind data
    df_model_djf = gev_funcs.pivot_detrend_model(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_x_axis_name="effective_dec_year",
        model_y_axis_name="data_sfcWind_drift_bc",
        obs_x_axis_name="effective_dec_year",
        obs_y_axis_name="data_sfcWind",
        suffix="_dt",
    )

    # perform the same for the non bias corrected data
    df_model_djf = gev_funcs.pivot_detrend_model(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_x_axis_name="effective_dec_year",
        model_y_axis_name="data_sfcWind",
        obs_x_axis_name="effective_dec_year",
        obs_y_axis_name="data_sfcWind",
        suffix="_dt",
    )

    # do the same for wind speed
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_sfcWind_dt",
        obs_var_name="data_sfcWind_dt",
        lead_name="winter_year",
        xlabel="10m Wind Speed (m/s)",
        suptitle="Lead dependent wind speed PDFs, DJF all days, 1961-2017, detrended (no BC)",
        figsize=(10, 5),
    )

    # do the same for wind speed
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_sfcWind_drift_bc_dt",
        obs_var_name="data_sfcWind_dt",
        lead_name="winter_year",
        xlabel="10m Wind Speed (m/s)",
        suptitle="Lead dependent wind speed PDFs, DJF all days, 1961-2017, detrended (BC)",
        figsize=(10, 5),
    )

    # Set up the directory to save to
    save_dir_dfs = "/home/users/benhutch/unseen_multi_year/dfs"

    # Set up a fname for the full field model data
    full_field_model_fname = (
        "full_field_model_tas_wind_UK_1961-2024_DJF_detrended_07-05-2025.csv"
    )
    full_field_obs_fname = (
        "full_field_obs_tas_wind_UK_1961-2024_DJF_detrended_07-05-2025.csv"
    )

    # Set up the paths
    full_field_model_path = os.path.join(save_dir_dfs, full_field_model_fname)
    full_field_obs_path = os.path.join(save_dir_dfs, full_field_obs_fname)

    # If the file does not exist
    if not os.path.exists(full_field_model_path):
        print("Saving the model data")
        # Save the model data
        df_model_djf.to_csv(full_field_model_path, index=False)

    # If the file does not exist
    if not os.path.exists(full_field_obs_path):
        print("Saving the obs data")
        # Save the obs data
        df_obs.to_csv(full_field_obs_path, index=False)

    # Plot teh block min distribution
    # relative to the full distribution
    # For termpatrue first
    plot_distributions_extremes(
        model_df_full_field=df_model_djf,
        obs_df_full_field=df_obs,
        model_df_block=df_model_tas_block_min,
        obs_df_block=df_obs_tas_block_min,
        model_var_name_full_field="data_tas_c_drift_bc_dt",
        obs_var_name_full_field="data_c_dt",
        model_var_name_block="data_tas_c_min_drift_bc_dt",
        obs_var_name_block="data_c_min_dt",
        xlabels=["Temperature (°C)", "Temperature (°C)"],
        percentile=0.05,
    )

    # DO the same for wind speed
    plot_distributions_extremes(
        model_df_full_field=df_model_djf,
        obs_df_full_field=df_obs,
        model_df_block=df_model_wind_block_min,
        obs_df_block=df_obs_wind_block_min,
        model_var_name_full_field="data_sfcWind_drift_bc_dt",
        obs_var_name_full_field="data_sfcWind_dt",
        model_var_name_block="data_min_drift_bc_dt",
        obs_var_name_block="data_min_dt",
        xlabels=["10m Wind Speed (m/s)", "10m Wind Speed (m/s)"],
        percentile=0.05,
    )

    # sys.exit()

    # apply the ws to wp gen function to the bias corrected wind
    # data
    df_obs, df_model_djf = ws_to_wp_gen(
        obs_df=df_obs,
        model_df=df_model_djf,
        obs_ws_col="data_sfcWind_dt",
        model_ws_col="data_sfcWind_drift_bc_dt",
        date_range=("1961-12-01", "2018-03-01"),
    )

    # apply the ws to wp gen function to the non bias corrected wind
    # data
    df_obs, df_model_djf = ws_to_wp_gen(
        obs_df=df_obs,
        model_df=df_model_djf,
        obs_ws_col="data_sfcWind_dt",
        model_ws_col="data_sfcWind_dt",
        date_range=("1961-12-01", "2018-03-01"),
    )

    # plot the lead pdfs to visualise the biases/drifts
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_sfcWind_dt_sigmoid_total_wind_gen",
    #     obs_var_name="data_sfcWind_dt_sigmoid_total_wind_gen",
    #     lead_name="winter_year",
    #     xlabel="Wind Power Generation (GW)",
    #     suptitle="Lead dependent wind power generation PDFs, DJF all days, 1961-2024 (detrended, no BC wind)",
    #     figsize=(10, 5),
    # )

    # plot the lead pdfs to visualise the biases/drifts
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen",
    #     obs_var_name="data_sfcWind_dt_sigmoid_total_wind_gen",
    #     lead_name="winter_year",
    #     xlabel="Wind Power Generation (GW)",
    #     suptitle="Lead dependent wind power generation PDFs, DJF all days, 1961-2024 (detrended, BC wind)",
    #     figsize=(10, 5),
    # )

    # sys.exit()

    # convert the temperature tyo demand for bias corrected T data
    df_obs, df_model_djf = temp_to_demand(
        obs_df=df_obs,
        model_df=df_model_djf,
        obs_temp_col="data_c_dt",
        model_temp_col="data_tas_c_drift_bc_dt",
    )

    # print the head of the df_model djf
    print(df_model_djf.head())
    # print the tail of the df_model djf
    print(df_model_djf.tail())

    # print the shape of the df
    print(f"Shape of df_model_djf: {df_model_djf.shape}")

    sys.exit()


    # convert the temperature tyo demand for non bias corrected T data
    df_obs, df_model_djf = temp_to_demand(
        obs_df=df_obs,
        model_df=df_model_djf,
        obs_temp_col="data_c_dt",
        model_temp_col="data_tas_c_dt",
    )


    # do the same for temperature
    # convert the temperature tyo demand for bias corrected T data
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_tas_c_dt_UK_demand",
    #     obs_var_name="data_c_dt_UK_demand",
    #     lead_name="winter_year",
    #     xlabel="Demand (GW)",
    #     suptitle="Lead dependent demand PDFs, DJF all days, 1961-2024 (detrended, no BC T)",
    #     figsize=(10, 5),
    # )

    # do the same for temperature
    # convert the temperature tyo demand for non bias corrected T data
    # gev_funcs.plot_lead_pdfs(
    #     model_df=df_model_djf,
    #     obs_df=df_obs,
    #     model_var_name="data_tas_c_drift_bc_dt_UK_demand",
    #     obs_var_name="data_c_dt_UK_demand",
    #     lead_name="winter_year",
    #     xlabel="Demand (GW)",
    #     suptitle="Lead dependent demand PDFs, DJF all days, 1961-2024 (detrended, BC T)",
    #     figsize=(10, 5),
    # )

    # sys.exit()

    # Calculate demand net wind for the observations
    df_obs["demand_net_wind"] = (
        df_obs["data_c_dt_UK_demand"] - df_obs["data_sfcWind_dt_sigmoid_total_wind_gen"]
    )

    # Calculate demand net wind for the NON-BIAS CORRECTED model data
    df_model_djf["demand_net_wind"] = (
        df_model_djf["data_tas_c_dt_UK_demand"]
        - df_model_djf["data_sfcWind_dt_sigmoid_total_wind_gen"]
    )

    # Calculate demand net wind for the BIAS CORRECTED model data
    df_model_djf["demand_net_wind_bc"] = (
        df_model_djf["data_tas_c_drift_bc_dt_UK_demand"]
        - df_model_djf["data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen"]
    )

    # set up the obs var names for plotting
    obs_var_names = [
        "data_c_dt",
        "data_c_dt_UK_demand",
        "data_sfcWind_dt",
        "data_sfcWind_dt_sigmoid_total_wind_gen",
        "demand_net_wind",
    ]

    # set up the model var names for plotting
    model_var_names = [
        "data_tas_c_dt",
        "data_tas_c_dt_UK_demand",
        "data_sfcWind_dt",
        "data_sfcWind_dt_sigmoid_total_wind_gen",
        "demand_net_wind",
    ]

    # set up the model var names for plotting
    model_var_names_bc = [
        "data_tas_c_drift_bc_dt",
        "data_tas_c_drift_bc_dt_UK_demand",
        "data_sfcWind_drift_bc_dt",
        "data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen",
        "demand_net_wind_bc",
    ]

    # Set up the subplot titles
    subplot_titles = [
        ("a", "b"),
        ("c", "d"),
        ("e", "f"),
        ("g", "h"),
        ("i", "j"),
    ]

    # plot the PDFs for multivariatie testing
    # gev_funcs.plot_multi_var_dist(
    #     obs_df=df_obs,
    #     model_df=df_model_djf,
    #     model_df_bc=df_model_djf,
    #     obs_var_names=obs_var_names,
    #     model_var_names=model_var_names,
    #     model_var_names_bc=model_var_names_bc,
    #     row_titles=[
    #         "Temp (°C)",
    #         "Demand (GW)",
    #         "10m wind speed (m/s)",
    #         "Wind power gen. (GW)",
    #         "Demand net wind (GW)",
    #     ],
    #     subplot_titles=subplot_titles,
    #     figsize=(15, 15),
    # )

    # # now plot the relationships between variables here
    # gev_funcs.plot_rel_var(
    #     obs_df=df_obs,
    #     model_df=df_model_djf,
    #     model_df_bc=df_model_djf,
    #     obs_var_names=("data_c_dt", "data_sfcWind_dt"),
    #     model_var_names=("data_tas_c_dt", "data_sfcWind_dt"),
    #     model_var_names_bc=("data_tas_c_drift_bc_dt", "data_sfcWind_drift_bc_dt"),
    #     row_title="T vs sfcWind",
    #     figsize=(15, 5),
    # )

    # Ensure the 'time' column is in datetime format
    df_obs["time"] = pd.to_datetime(df_obs["time"])

    # Print the row in df_obs where time = 2025-01-08
    print(df_obs[df_obs["time"] == "2025-01-08"])

    # Save this row
    jan_8_2025 = df_obs[df_obs["time"] == "2025-01-08"]

    # Print the row
    print(jan_8_2025)

    # Extract all of the data for January 2025
    jan_2025 = df_obs[(df_obs["time"].dt.month == 1) & (df_obs["time"].dt.year == 2025)]

    feb_2006 = df_obs[(df_obs["time"].dt.month == 2) & (df_obs["time"].dt.year == 2006)]

    eff_dec_year_24 = df_obs[(df_obs["effective_dec_year"] == 2024)]

    # # create a new column for rank_dnw
    # rank dnw from highest to lowest
    # for eff_dec_year_24
    eff_dec_year_24["rank_dnw"] = eff_dec_year_24["demand_net_wind"].rank(
        ascending=False
    )

    # Print the January 2025 data
    print(jan_2025)

    # set us
    save_dir_jan = "/home/users/benhutch/unseen_multi_year/dfs"

    # set up the fname
    fname_jan = os.path.join(save_dir_jan, "jan_2025.csv")

    fname_feb = os.path.join(save_dir_jan, "feb_2006.csv")

    # save the data
    # if the file does not already exist
    if not os.path.exists(fname_jan):
        jan_2025.to_csv(fname_jan, index=False)
        print(f"Saved {fname_jan}")

    # if the file does not already exist
    if not os.path.exists(fname_feb):
        feb_2006.to_csv(fname_feb, index=False)
        print(f"Saved {fname_feb}")

    # if the file does not already exist
    if not os.path.exists(os.path.join(save_dir_jan, "eff_dec_year_24.csv")):
        eff_dec_year_24.to_csv(
            os.path.join(save_dir_jan, "eff_dec_year_24.csv"), index=False
        )
        print(f"Saved {os.path.join(save_dir_jan, 'eff_dec_year_24.csv')}")

    # sys.exit()

    # now quantify the seasonal block maxima for demand net wind
    # first for the observations
    block_max_obs_dnw = gev_funcs.obs_block_min_max(
        df=df_obs,
        time_name="effective_dec_year",
        min_max_var_name="demand_net_wind",
        new_df_cols=[
            "data_sfcWind_dt_sigmoid_total_wind_gen",
            "data_c_dt_UK_demand",
            "time",
            "data_c_dt",
            "data_sfcWind_dt",
        ],
        process_min=False,
    )

    # now for the model data
    # for the bias correctded data
    block_max_model_dnw = gev_funcs.model_block_min_max(
        df=df_model_djf,
        time_name="init_year",
        min_max_var_name="demand_net_wind_bc",
        new_df_cols=[
            "data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen",
            "data_tas_c_drift_bc_dt_UK_demand",
            "lead",
            "data_tas_c_drift_bc_dt",
            "data_sfcWind_drift_bc_dt",
            "delta_p_hpa",
            "data_uas",
            "data_vas",
        ],
        winter_year="winter_year",
        process_min=False,
    )

    # make sure effective dec year is in the block max obs data
    block_max_model_dnw["effective_dec_year"] = block_max_model_dnw["init_year"] + (
        block_max_model_dnw["winter_year"] - 1
    )

    # # Plot the biases in these
    # gev_funcs.plot_lead_pdfs(
    #     model_df=block_max_model_dnw,
    #     obs_df=block_max_obs_dnw,
    #     model_var_name="demand_net_wind_bc_max",
    #     obs_var_name="demand_net_wind_max",
    #     lead_name="winter_year",
    #     xlabel="Demand net wind (GW)",
    #     suptitle="Lead dependent demand net wind PDFs, DJF all days, 1961-2024 (detrended, BC T + sfcWind)",
    #     figsize=(10, 5),
    # )

    # apply a uniform bias correction to the block maxima from the model
    bias = (
        block_max_model_dnw["demand_net_wind_bc_max"].mean()
        - block_max_obs_dnw["demand_net_wind_max"].mean()
    )

    # print the bias
    print(f"Bias: {bias}")

    # apply the bias correction
    block_max_model_dnw["demand_net_wind_bc_max_bc"] = (
        block_max_model_dnw["demand_net_wind_bc_max"] - bias
    )

    # apply a uniform bias correct to the jan 8 df
    jan_8_2025["demand_net_wind_bc"] = jan_8_2025["demand_net_wind"] - bias

    # print the row
    print(jan_8_2025)

    # sys.exit()

    # # Plot the biases in these
    # gev_funcs.plot_lead_pdfs(
    #     model_df=block_max_model_dnw,
    #     obs_df=block_max_obs_dnw,
    #     model_var_name="demand_net_wind_bc_max_bc",
    #     obs_var_name="demand_net_wind_max",
    #     lead_name="winter_year",
    #     xlabel="Demand net wind (GW)",
    #     suptitle="Lead dependent demand net wind PDFs, DJF all days, 1961-2024 (detrended, BC T + sfcWind + BC)",
    #     figsize=(10, 5),
    # )

    # Compare the trends
    gev_funcs.compare_trends(
        model_df_full_field=df_model_djf,
        obs_df_full_field=df_obs,
        model_df_block=block_max_model_dnw,
        obs_df_block=block_max_obs_dnw,
        model_var_name_full_field="demand_net_wind_bc",
        obs_var_name_full_field="demand_net_wind",
        model_var_name_block="demand_net_wind_bc_max_bc",
        obs_var_name_block="demand_net_wind_max",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Demand net wind (GW)",
        suptitle="Lead dependent demand net wind PDFs, DJF all days, 1961-2024 (detrended, BC T + sfcWind + BC)",
        figsize=(15, 5),
    )

    # sys.exit()

    # format effective dec year as a datetime for the model data
    block_max_model_dnw["effective_dec_year"] = pd.to_datetime(
        block_max_model_dnw["effective_dec_year"], format="%Y"
    )

    # format effective dec year as a datetime for the obs data
    block_max_obs_dnw["effective_dec_year"] = pd.to_datetime(
        block_max_obs_dnw["effective_dec_year"], format="%Y"
    )

    # Set this as the index in the observations
    block_max_obs_dnw.set_index("effective_dec_year", inplace=True)

    # plot the dot plot
    gev_funcs.dot_plot_subplots(
        obs_df_left=block_max_obs_dnw,
        model_df_left=block_max_model_dnw,
        obs_df_right=block_max_obs_dnw,
        model_df_right=block_max_model_dnw,
        obs_val_name_left="demand_net_wind_max",
        model_val_name_left="demand_net_wind_bc_max",
        obs_val_name_right="demand_net_wind_max",
        model_val_name_right="demand_net_wind_bc_max_bc",
        model_time_name="effective_dec_year",
        ylabel_left="Demand net wind (GW)",
        ylabel_right="Demand net wind (GW)",
        title_left="Block maxima demand net wind (GW, no uniform BC)",
        title_right="Block maxima demand net wind (GW)",
        ylims_left=(30, 60),
        ylims_right=(30, 60),
        dashed_quant=0.80,
        solid_line=np.max,
        figsize=(10, 5),
    )

    # gev_funcs.dot_plot_subplots(
    #     obs_df_left=block_max_obs_dnw,
    #     model_df_left=block_max_model_dnw,
    #     obs_df_right=block_max_obs_dnw,
    #     model_df_right=block_max_model_dnw,
    #     obs_val_name_left="demand_net_wind_max",
    #     model_val_name_left="demand_net_wind_bc_max",
    #     obs_val_name_right="demand_net_wind_max",
    #     model_val_name_right="demand_net_wind_bc_max",
    #     model_time_name="effective_dec_year",
    #     ylabel_left="Demand net wind (GW)",
    #     ylabel_right="Demand net wind (GW)",
    #     title_left="Block maxima demand net wind (GW, no uniform BC)",
    #     title_right="Block maxima demand net wind (GW, no uniform BC)",
    #     ylims_left=(30, 60),
    #     ylims_right=(30, 60),
    #     dashed_quant=0.80,
    #     solid_line=np.max,
    #     figsize=(10, 5),
    # )

    # print the head and tail of the obs
    print("Block maxima obs data")
    print(block_max_obs_dnw.head())
    print(block_max_obs_dnw.tail())

    # Find the 80th percentile of the obs data
    obs_80th = block_max_obs_dnw["demand_net_wind_max"].quantile(0.80)

    # Subset the obs and model dfs to values below the 80th percentile
    block_max_obs_dnw_grey_dots = block_max_obs_dnw[
        block_max_obs_dnw["demand_net_wind_max"] < obs_80th
    ]
    block_max_model_dnw_grey_dots = block_max_model_dnw[
        block_max_model_dnw["demand_net_wind_bc_max"] < obs_80th
    ]

    # Find the maximum value in the obs data
    obs_max = block_max_obs_dnw["demand_net_wind_max"].max()

    # Subset the obs and model dfs to values above the 80th percentile
    # but below the max
    block_max_obs_dnw_yellow_dots = block_max_obs_dnw[
        (block_max_obs_dnw["demand_net_wind_max"] >= obs_80th)
        & (block_max_obs_dnw["demand_net_wind_max"] < obs_max)
    ]
    block_max_model_dnw_yellow_dots = block_max_model_dnw[
        (block_max_model_dnw["demand_net_wind_bc_max"] >= obs_80th)
        & (block_max_model_dnw["demand_net_wind_bc_max"] < obs_max)
    ]

    # Subset the obs data to the max value
    block_max_obs_dnw_red_dots = block_max_obs_dnw[
        block_max_obs_dnw["demand_net_wind_max"] == obs_max
    ]

    # Subset the model data to values above the max value
    block_max_model_dnw_red_dots = block_max_model_dnw[
        block_max_model_dnw["demand_net_wind_bc_max"] >= obs_max
    ]

    # Set up the subtitles
    subplot_titles = [
        "Grey dots",
        "Yellow dots",
        "Red dots",
    ]

    # print the columns in the obs data
    print(block_max_obs_dnw.columns)
    print(block_max_model_dnw.columns)

    # plot the multi var scatter
    # plot_multi_var_scatter(
    #     subset_obs_dfs=[
    #         block_max_obs_dnw_grey_dots,
    #         block_max_obs_dnw_yellow_dots,
    #         block_max_obs_dnw_red_dots,
    #     ],
    #     subset_model_dfs=[
    #         block_max_model_dnw_grey_dots,
    #         block_max_model_dnw_yellow_dots,
    #         block_max_model_dnw_red_dots,
    #     ],
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature (°C)",
    #     ylabel="10m Wind Speed (m/s)",
    #     subtitles=subplot_titles,
    #     figsize=(10, 5),
    # )

    # # do the same thing, but for different variables
    # plot_multi_var_scatter(
    #     subset_obs_dfs=[
    #         block_max_obs_dnw_grey_dots,
    #         block_max_obs_dnw_yellow_dots,
    #         block_max_obs_dnw_red_dots,
    #     ],
    #     subset_model_dfs=[
    #         block_max_model_dnw_grey_dots,
    #         block_max_model_dnw_yellow_dots,
    #         block_max_model_dnw_red_dots,
    #     ],
    #     x_var_name_obs="data_c_dt_UK_demand",
    #     y_var_name_obs="data_sfcWind_dt_sigmoid_total_wind_gen",
    #     x_var_name_model="data_tas_c_drift_bc_dt_UK_demand",
    #     y_var_name_model="data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen",
    #     xlabel="Demand (GW)",
    #     ylabel="Wind Power Generation (GW)",
    #     subtitles=subplot_titles,
    #     figsize=(10, 5),
    # )

    # Plot the percentiles of temp against wind speed
    # for subset data
    plot_multi_var_perc(
        obs_df=block_max_obs_dnw,
        model_df=block_max_model_dnw,
        x_var_name_obs="data_c_dt",
        y_var_name_obs="data_sfcWind_dt",
        x_var_name_model="data_tas_c_drift_bc_dt",
        y_var_name_model="data_sfcWind_drift_bc_dt",
        xlabel="Temperature",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of (inverted) temperature vs 10m wind speed, DnW days",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    # Do the same but with uas
    plot_multi_var_perc(
        obs_df=block_max_obs_dnw,
        model_df=block_max_model_dnw,
        x_var_name_obs="data_c_dt",
        y_var_name_obs="data_sfcWind_dt",
        x_var_name_model="data_tas_c_drift_bc_dt",
        y_var_name_model="data_uas",
        xlabel="Temperature",
        ylabel="U10m (m/s)",
        title="Percentiles of (inverted) temperature vs 10m wind speed, DnW days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_sfcWind_drift_bc_dt",
        y2_label="10m Wind Speed (m/s)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    plot_multi_var_perc(
        obs_df=block_max_obs_dnw,
        model_df=block_max_model_dnw,
        x_var_name_obs="data_c_dt",
        y_var_name_obs="data_sfcWind_dt",
        x_var_name_model="data_tas_c_drift_bc_dt",
        y_var_name_model="data_uas",
        xlabel="Temperature",
        ylabel="U10m (m/s)",
        title="Percentiles of (inverted) temperature vs 10m wind speed, DnW days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    # sys.exit()

    # Do the same but with vas
    # plot_multi_var_perc(
    #     obs_df=block_max_obs_dnw,
    #     model_df=block_max_model_dnw,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, DnW days",
    #     y2_var_name_model="data_vas",
    #     y2_label="V10m (m/s)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # # do the same for electricity demand against wind power generation
    # plot_multi_var_perc(
    #     obs_df=block_max_obs_dnw,
    #     model_df=block_max_model_dnw,
    #     x_var_name_obs="data_c_dt_UK_demand",
    #     y_var_name_obs="data_sfcWind_dt_sigmoid_total_wind_gen",
    #     x_var_name_model="data_tas_c_drift_bc_dt_UK_demand",
    #     y_var_name_model="data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen",
    #     xlabel="Demand (GW)",
    #     ylabel="Wind Power Generation (GW)",
    #     title="Percentiles of demand vs wind power generation, DnW days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    # )

    # # do the same but for greyb dots
    # plot_multi_var_perc(
    #     obs_df=block_max_obs_dnw_grey_dots,
    #     model_df=block_max_model_dnw_grey_dots,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, DnW days < 80th percentile",
    #     figsize=(5, 6),
    # )

    # # do the same but for yellow dots
    # plot_multi_var_perc(
    #     obs_df=block_max_obs_dnw_yellow_dots,
    #     model_df=block_max_model_dnw_yellow_dots,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, DnW days > 80th percentile",
    #     figsize=(5, 6),
    # )

    # # do the same but for red dots
    # plot_multi_var_perc(
    #     obs_df=block_max_obs_dnw_red_dots,
    #     model_df=block_max_model_dnw_red_dots,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, DnW days > obs max",
    #     figsize=(5, 6),
    # )

    # Do the same but for the full distribution
    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_djf,
        x_var_name_obs="data_c_dt",
        y_var_name_obs="data_sfcWind_dt",
        x_var_name_model="data_tas_c_drift_bc_dt",
        y_var_name_model="data_sfcWind_drift_bc_dt",
        xlabel="Temperature",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_djf,
        x_var_name_obs="data_c_dt",
        y_var_name_obs="data_sfcWind_dt",
        x_var_name_model="data_tas_c_drift_bc_dt",
        y_var_name_model="data_uas",
        xlabel="Temperature",
        ylabel="U10m (m/s)",
        title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_sfcWind_drift_bc_dt",
        y2_label="10m Wind Speed (m/s)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_djf,
        x_var_name_obs="data_c_dt",
        y_var_name_obs="data_sfcWind_dt",
        x_var_name_model="data_tas_c_drift_bc_dt",
        y_var_name_model="data_uas",
        xlabel="Temperature",
        ylabel="U10m (m/s)",
        title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    # do the same for electricity demand against wind power generation
    # plot_multi_var_perc(
    #     obs_df=df_obs,
    #     model_df=df_model_djf,
    #     x_var_name_obs="data_c_dt_UK_demand",
    #     y_var_name_obs="data_sfcWind_dt_sigmoid_total_wind_gen",
    #     x_var_name_model="data_tas_c_drift_bc_dt_UK_demand",
    #     y_var_name_model="data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen",
    #     xlabel="Demand (GW)",
    #     ylabel="Wind Power Generation (GW)",
    #     title="Percentiles of demand vs wind power generation, all winter days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    # )

    # DO the same but for uas on the second y-axis
    # plot_multi_var_perc(
    #     obs_df=df_obs,
    #     model_df=df_model_djf,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days",
    #     y2_var_name_model="data_uas",
    #     y2_label="U10m (m/s)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # # DO the same but for vas on the second y-axis
    # plot_multi_var_perc(
    #     obs_df=df_obs,
    #     model_df=df_model_djf,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days",
    #     y2_var_name_model="data_vas",
    #     y2_label="V10m (m/s)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # # find the 90th percentile of demand net wind
    # model_90th = df_model_djf["demand_net_wind_bc"].quantile(0.90)
    # obs_90th = df_obs["demand_net_wind"].quantile(0.90)

    # # subset the data to values above the 90th percentile
    # full_df_obs_subset = df_obs[
    #     df_obs["demand_net_wind"] > obs_90th
    # ]
    # full_df_model_subset = df_model_djf[
    #     df_model_djf["demand_net_wind_bc"] > model_90th
    # ]

    # # Plot the percentiles for these subset data
    # plot_multi_var_perc(
    #     obs_df=full_df_obs_subset,
    #     model_df=full_df_model_subset,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days > 90th percentile",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # # plot the percentiles of demand and supply for these subset data
    # plot_multi_var_perc(
    #     obs_df=full_df_obs_subset,
    #     model_df=full_df_model_subset,
    #     x_var_name_obs="data_c_dt_UK_demand",
    #     y_var_name_obs="data_sfcWind_dt_sigmoid_total_wind_gen",
    #     x_var_name_model="data_tas_c_drift_bc_dt_UK_demand",
    #     y_var_name_model="data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen",
    #     xlabel="Demand (GW)",
    #     ylabel="Wind Power Generation (GW)",
    #     title="Percentiles of demand vs wind power generation, all winter days > 90th percentile",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    # )

    # # find the 10th percentile for temperatur data
    # model_10th = df_model_djf["data_tas_c_drift_bc_dt"].quantile(0.10)
    # obs_10th = df_obs["data_c_dt"].quantile(0.10)

    # # subset the data to values beneath the 10th percentile
    # full_df_obs_subset_temp = df_obs[
    #     df_obs["data_c_dt"] < obs_10th
    # ]
    # full_df_model_subset_temp = df_model_djf[
    #     df_model_djf["data_tas_c_drift_bc_dt"] < model_10th
    # ]

    # # plot the relationships between temperature and wind speed for these subset data
    # plot_multi_var_perc(
    #     obs_df=full_df_obs_subset_temp,
    #     model_df=full_df_model_subset_temp,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days < 10th percentile t2m",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # # calculate the 10th percentile of wind speed
    # model_10th_ws = df_model_djf["data_sfcWind_drift_bc_dt"].quantile(0.10)
    # obs_10th_ws = df_obs["data_sfcWind_dt"].quantile(0.10)

    # # subset the data to values beneath the 10th percentile
    # full_df_obs_subset_ws = df_obs[
    #     df_obs["data_sfcWind_dt"] < obs_10th_ws
    # ]
    # full_df_model_subset_ws = df_model_djf[
    #     df_model_djf["data_sfcWind_drift_bc_dt"] < model_10th_ws
    # ]

    # # plot the relationships between temperature and wind speed for these subset data
    # plot_multi_var_perc(
    #     obs_df=full_df_obs_subset_ws,
    #     model_df=full_df_model_subset_ws,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days < 10th percentile si10",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # # # subset the data to values beneath 3 *C
    # # full_df_obs_subset = df_obs[
    # #     df_obs["data_c_dt"] < 3
    # # ]

    # # # subset the data to values beneath 3 *C
    # # full_df_model_subset = df_model_djf[
    # #     df_model_djf["data_tas_c_drift_bc_dt"] < 3
    # # ]

    # # # DO the same but for the subset distribution
    # # plot_multi_var_perc(
    # #     obs_df=full_df_obs_subset,
    # #     model_df=full_df_model_subset,
    # #     x_var_name_obs="data_c_dt",
    # #     y_var_name_obs="data_sfcWind_dt",
    # #     x_var_name_model="data_tas_c_drift_bc_dt",
    # #     y_var_name_model="data_sfcWind_drift_bc_dt",
    # #     xlabel="Temperature",
    # #     ylabel="10m Wind Speed (m/s)",
    # #     title="Percentiles of (inverted) temperature vs 10m wind speed, all winter days < 3 *C",
    # #     figsize=(5, 6),
    # # )

    sys.exit()

    # reset the index of the obs data
    block_max_obs_dnw.reset_index(inplace=True)

    # # plot the return period plots here
    # # first the empirical return periods
    plot_emp_rps(
        obs_df=block_max_obs_dnw,
        model_df=block_max_model_dnw,
        obs_val_name="demand_net_wind_max",
        model_val_name="demand_net_wind_bc_max",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        ylabel="Demand net wind (GW)",
        nsamples=1000,
        ylims=(44, 52),
        blue_line=np.max,
        high_values_rare=True,
        figsize=(5, 5),
        wind_2005_toggle=False,
    )

    # # plot the GEV fitted return periods
    # plot_gev_rps(
    #     obs_df=block_max_obs_dnw,
    #     model_df=block_max_model_dnw,
    #     obs_val_name="demand_net_wind_max",
    #     model_val_name="demand_net_wind_bc_max_bc",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     ylabel="Demand net wind (GW)",
    #     nsamples=1000,
    #     ylims=(jan_8_2025["demand_net_wind_bc"].values[0] - 1, 52),
    #     blue_line=np.max,
    #     high_values_rare=True,
    #     figsize=(5, 5),
    #     bonus_line=jan_8_2025["demand_net_wind_bc"].values[0],
    # )

    sys.exit()

    # ensure the effective dec year is a datetime and is just the year in the
    # model
    block_max_model_dnw["effective_dec_year"] = pd.to_datetime(
        block_max_model_dnw["effective_dec_year"], format="%Y"
    )

    # convert this to an int
    block_max_model_dnw["effective_dec_year"] = block_max_model_dnw[
        "effective_dec_year"
    ].dt.year.astype(int)

    # # plot the return periods over decades
    # gev_funcs.plot_return_periods_decades(
    #     model_df=block_max_model_dnw,
    #     model_var_name="demand_net_wind_bc_max_bc",
    #     obs_df=block_max_obs_dnw,
    #     obs_var_name="demand_net_wind_max",
    #     decades=np.arange(1960, 2020, 10),
    #     title="Return period of 1 in 100-year event",
    #     num_samples=1000,
    #     figsize=(10, 5),
    #     bad_min=False,
    # )

    # set up a fname for the obs dnw df
    obs_dnw_fpath = os.path.join(
        dfs_dir, "block_maxima_obs_demand_net_wind_07-05-2025.csv"
    )
    # set up a fname for the model dnw df
    model_dnw_fpath = os.path.join(
        dfs_dir, "block_maxima_model_demand_net_wind_07-05-2025.csv"
    )

    # if the fpath does not exist, svae the dtaa
    if not os.path.exists(obs_dnw_fpath):
        # Save the obs data
        block_max_obs_dnw.to_csv(obs_dnw_fpath, index=True)

    # if the fpath does not exist, svae the dtaa
    if not os.path.exists(model_dnw_fpath):
        # Save the model data
        block_max_model_dnw.to_csv(model_dnw_fpath, index=True)

    sys.exit()

    # Apply the lead time dependent mean bias correction
    # For temperature
    df_model_djf_bc = gev_funcs.lead_time_mean_bias_correct(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_tas_c",
        obs_var_name="data_c",
        lead_name="winter_year",
    )

    # print the columns in df_model_djf_bc
    print(df_model_djf_bc.columns)

    # Apply the lead time dependent mean bias correction
    # For wind speed
    df_model_djf_bc = gev_funcs.lead_time_mean_bias_correct(
        model_df=df_model_djf_bc,
        obs_df=df_obs,
        model_var_name="data_sfcWind",
        obs_var_name="data_sfcWind",
        lead_name="winter_year",
    )

    # Pivot detrend the obs
    df_obs_dt = gev_funcs.pivot_detrend_obs(
        df=df_obs,
        x_axis_name="effective_dec_year",
        y_axis_name="data_c",
    )

    # print the columns in df_model_djf_bc
    print(df_model_djf_bc.columns)

    # # Pivot detrend the model
    df_model_djf_bc_dt = gev_funcs.pivot_detrend_model(
        df=df_model_djf_bc,
        x_axis_name="effective_dec_year",
        y_axis_name="data_tas_c_bc",
    )

    # pivot detrend the non bias corrected model
    df_model_djf_dt = gev_funcs.pivot_detrend_model(
        df=df_model_djf,
        x_axis_name="effective_dec_year",
        y_axis_name="data_tas_c",
    )

    # print the columns in df_model_djf_bc_dt
    print("columns in df_model_djf_dt")
    print(df_model_djf_dt.columns)

    # # print the head of the df_obs
    # print(df_obs.columns)
    # print(df_obs_dt.columns)

    # # Apply the ws_to_wp_gen function to the obs and model data
    # df_obs, df_model_djf = ws_to_wp_gen(
    #     obs_df=df_obs,
    #     model_df=df_model_djf,
    #     obs_ws_col="data_sfcWind",
    #     model_ws_col="data_sfcWind_bc",
    # )

    # # Apply the ws_to_wp_gen function to the detrended obs and model data
    df_obs_dt, df_model_djf_bc_dt = ws_to_wp_gen(
        obs_df=df_obs_dt,
        model_df=df_model_djf_bc_dt,
        obs_ws_col="data_sfcWind",
        model_ws_col="data_sfcWind_bc",
    )

    # convert the non bias corrected model data to wind power generation
    _, df_model_djf_dt = ws_to_wp_gen(
        obs_df=df_obs_dt,
        model_df=df_model_djf_dt,
        obs_ws_col="data_sfcWind",
        model_ws_col="data_sfcWind",
    )

    # # Convert the temperature to demand
    # df_obs, df_model_djf = temp_to_demand(
    #     obs_df=df_obs,
    #     model_df=df_model_djf,
    #     obs_temp_col="data_c",
    #     model_temp_col="data_tas_c_bc",
    # )

    # # Convert the dt temperature to demand
    df_obs_dt, df_model_djf_bc_dt = temp_to_demand(
        obs_df=df_obs_dt,
        model_df=df_model_djf_bc_dt,
        obs_temp_col="data_c_dt",
        model_temp_col="data_tas_c_bc_dt",
    )

    # calculate the bias in bc demand data
    demand_bias = df_model_djf_bc_dt["UK_demand"].mean() - df_obs_dt["UK_demand"].mean()

    # print the demand bias
    print(f"Demand bias: {demand_bias}")
    # print the bc demand mean
    print(f"BC demand mean: {df_model_djf_bc_dt['UK_demand'].mean()}")
    # print the obs demand mean
    print(f"Obs demand mean: {df_obs_dt['UK_demand'].mean()}")

    # print the head of the df_obs_dt
    print(df_obs_dt.head())

    # print the head of the df_model_djf_bc_dt
    print(df_model_djf_dt.head())

    # # Convert the non bias corrected model temperature to demand
    _, df_model_djf_dt = temp_to_demand(
        obs_df=df_obs_dt,
        model_df=df_model_djf_dt,
        obs_temp_col="data_c",
        model_temp_col="data_tas_c_dt",
    )

    # calculate the bias in bc demand data
    demand_bias = df_model_djf_bc_dt["UK_demand"].mean() - df_obs_dt["UK_demand"].mean()

    # print the demand bias
    print(f"Demand bias: {demand_bias}")
    # print the bc demand mean
    print(f"BC demand mean: {df_model_djf_bc_dt['UK_demand'].mean()}")
    # print the obs demand mean
    print(f"Obs demand mean: {df_obs_dt['UK_demand'].mean()}")

    # Plot the lead pdfs for the demand data (temperature has been detrended)
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf_bc_dt,
        obs_df=df_obs_dt,
        model_var_name="UK_demand",
        obs_var_name="UK_demand",
        lead_name="winter_year",
        xlabel="Demand (GW)",
        suptitle="Lead dependent demand PDFs (detrended temp), DJF, 1960-2017",
    )

    # Plot the lead pdfs for the wind power generation data
    # all winter days
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf_bc_dt,
        obs_df=df_obs_dt,
        model_var_name="sigmoid_total_wind_gen",
        obs_var_name="sigmoid_total_wind_gen",
        lead_name="winter_year",
        xlabel="Wind Power Generation (GW)",
        suptitle="Lead dependent wind power generation PDFs, DJF, 1960-2017",
    )

    # Calculate demand net wind
    # for the detrended, but NON BIAS CORRECTED data
    df_obs_dt["demand_net_wind"] = (
        df_obs_dt["UK_demand"] - df_obs_dt["sigmoid_total_wind_gen"]
    )

    # Calculate demand net wind for the detrended, but NON BIAS CORRECTED model data
    df_model_djf_dt["demand_net_wind"] = (
        df_model_djf_dt["UK_demand"] - df_model_djf_dt["sigmoid_total_wind_gen"]
    )

    # Calculate demand net wind for the detrended, BIAS CORRECTED model data
    df_model_djf_bc_dt["demand_net_wind"] = (
        df_model_djf_bc_dt["UK_demand"] - df_model_djf_bc_dt["sigmoid_total_wind_gen"]
    )

    # -------------------------
    # Now do the generic fidelity testing
    # -------------------------

    # Plot the pdfs for multivariate testing
    # for all leads
    gev_funcs.plot_multi_var_dist(
        obs_df=df_obs_dt,
        model_df=df_model_djf_dt,
        model_df_bc=df_model_djf_bc_dt,
        obs_var_names=[
            "data_c_dt",
            "UK_demand",
            "data_sfcWind",
            "sigmoid_total_wind_gen",
            "demand_net_wind",
        ],
        model_var_names=[
            "data_tas_c_dt",
            "UK_demand",
            "data_sfcWind",
            "sigmoid_total_wind_gen",
            "demand_net_wind",
        ],
        model_var_names_bc=[
            "data_tas_c_bc_dt",
            "UK_demand",
            "data_sfcWind_bc",
            "sigmoid_total_wind_gen",
            "demand_net_wind",
        ],
        row_titles=[
            "Temp (°C)",
            "Demand (GW)",
            "10m wind speed (m/s)",
            "Wind power gen. (GW)",
            "Demand net wind (GW)",
        ],
        subplot_titles=[("a", "b"), ("c", "d"), ("e", "f"), ("g", "h"), ("i", "j")],
        figsize=(15, 15),
    )

    # Now plot the relationship between variables
    gev_funcs.plot_rel_var(
        obs_df=df_obs_dt,
        model_df=df_model_djf_dt,
        model_df_bc=df_model_djf_bc_dt,
        obs_var_names=("data_c_dt", "data_sfcWind"),
        model_var_names=("data_tas_c_dt", "data_sfcWind"),
        model_var_names_bc=("data_tas_c_bc_dt", "data_sfcWind_bc"),
        row_title="T vs sfcWind",
        figsize=(15, 5),
    )

    sys.exit()

    # print the head of df obs dt
    print(df_obs_dt.head())

    # Now quantify the seasonal block maxima for demand net wind
    # for the observations first
    block_maxima_obs_dnw = gev_funcs.obs_block_min_max(
        df=df_obs_dt,
        time_name="effective_dec_year",
        min_max_var_name="demand_net_wind",
        new_df_cols=["sigmoid_total_wind_gen", "UK_demand", "time"],
        process_min=False,
    )

    # Now quantify the seasonal block maxima for demand net wind
    # for the model data
    block_maxima_model_dnw = gev_funcs.model_block_min_max(
        df=df_model_djf_bc_dt,
        time_name="init_year",
        min_max_var_name="demand_net_wind",
        new_df_cols=["sigmoid_total_wind_gen", "UK_demand", "lead"],
        winter_year="winter_year",
        process_min=False,
    )

    # Make sure effective dec year exists for the model block max
    if "effective_dec_year" not in block_maxima_model_dnw.columns:
        block_maxima_model_dnw["effective_dec_year"] = block_maxima_model_dnw[
            "init_year"
        ] + (block_maxima_model_dnw["winter_year"] - 1)

    # Plot the detrend time series in this case
    gev_funcs.plot_detrend_ts(
        obs_df=block_maxima_obs_dnw,
        model_df=block_maxima_model_dnw,
        obs_var_name="demand_net_wind_max",
        model_var_name="demand_net_wind_max",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        title="Block maxima DJF demand net wind, 1960-2017",
        ylim=(35, 50),
        detrend_suffix=None,
    )

    # Check the slope in demand net wind
    # for the obs and model data
    gev_funcs.compare_trends(
        model_df_full_field=df_model_djf_bc_dt,
        obs_df_full_field=df_obs_dt,
        model_df_block=block_maxima_model_dnw,
        obs_df_block=block_maxima_obs_dnw,
        model_var_name_full_field="demand_net_wind",
        obs_var_name_full_field="demand_net_wind",
        model_var_name_block="demand_net_wind_max",
        obs_var_name_block="demand_net_wind_max",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        suptitle="Demand Net Wind trends (temp + wind lead BC)",
        figsize=(15, 5),
    )

    # Correct lead time dependent bias in demand net wind
    block_maxima_model_dnw = gev_funcs.lead_time_mean_bias_correct(
        model_df=block_maxima_model_dnw,
        obs_df=block_maxima_obs_dnw,
        model_var_name="demand_net_wind_max",
        obs_var_name="demand_net_wind_max",
        lead_name="winter_year",
    )

    # print the columns for block_maxima_model_dnw
    print(block_maxima_model_dnw.columns)

    # Set up effective dec year as a datetime for the model data
    block_maxima_model_dnw["effective_dec_year"] = pd.to_datetime(
        block_maxima_model_dnw["effective_dec_year"], format="%Y"
    )

    # Set effective dec year as a datetime for the obs data
    block_maxima_obs_dnw["effective_dec_year"] = pd.to_datetime(
        block_maxima_obs_dnw["effective_dec_year"], format="%Y"
    )

    # Set this as the index
    block_maxima_obs_dnw.set_index("effective_dec_year", inplace=True)

    # print the head of the df
    print(block_maxima_obs_dnw.head())

    # print the head of the model df
    print(block_maxima_model_dnw.head())

    # set up a fname for the obs dnw df
    obs_dnw_fpath = os.path.join(dfs_dir, "block_maxima_obs_demand_net_wind.csv")
    # set up a fname for the model dnw df
    model_dnw_fpath = os.path.join(dfs_dir, "block_maxima_model_demand_net_wind.csv")

    # if the fpath does not exist, svae the dtaa
    if not os.path.exists(obs_dnw_fpath):
        # Save the obs data
        block_maxima_obs_dnw.to_csv(obs_dnw_fpath, index=True)

    # if the fpath does not exist, svae the dtaa
    if not os.path.exists(model_dnw_fpath):
        # Save the model data
        block_maxima_model_dnw.to_csv(model_dnw_fpath, index=True)

    # ------------------------------------------
    # Do the new dot plot inline with the others
    # ------------------------------------------
    gev_funcs.dot_plot_subplots(
        obs_df_left=block_maxima_obs_dnw,
        model_df_left=block_maxima_model_dnw,
        obs_df_right=block_maxima_obs_dnw,
        model_df_right=block_maxima_model_dnw,
        obs_val_name_left="demand_net_wind_max",
        model_val_name_left="demand_net_wind_max",
        obs_val_name_right="demand_net_wind_max",
        model_val_name_right="demand_net_wind_max_bc",
        model_time_name="effective_dec_year",
        ylabel_left="Demand Net Wind (GW)",
        ylabel_right="Demand Net Wind (GW)",
        title_left="Block maxima demand net wind (no BC)",
        title_right="Block maxima demand net wind (GW)",
        ylims_left=(30, 60),
        ylims_right=(30, 60),
        dashed_quant=0.80,
        solid_line=np.max,
        figsize=(10, 5),
    )

    # Plot the dot plot the block maxima dnw extremes
    # Non bias corrected
    dot_plot(
        obs_df=block_maxima_obs_dnw,
        model_df=block_maxima_model_dnw,
        obs_val_name="demand_net_wind_max",
        model_val_name="demand_net_wind_max",
        model_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        title="Block maxima demand net wind, DJF, 1960-2017, no BC",
        ylims=(30, 60),
        solid_line=np.max,
        dashed_quant=0.80,
    )

    # Now do the same for the bias corrected data
    dot_plot(
        obs_df=block_maxima_obs_dnw,
        model_df=block_maxima_model_dnw,
        obs_val_name="demand_net_wind_max",
        model_val_name="demand_net_wind_max_bc",
        model_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        title="Block maxima demand net wind, DJF, 1960-2017, BC",
        ylims=(30, 60),
        solid_line=np.max,
        dashed_quant=0.80,
    )

    # reset the index of block_maxima_obs_dnw
    block_maxima_obs_dnw.reset_index(inplace=True)

    # turn effective dec year back into an int for block_maxima_obs_dnw
    block_maxima_obs_dnw["effective_dec_year"] = block_maxima_obs_dnw[
        "effective_dec_year"
    ].dt.year.astype(int)

    # remove the trend from the obs data
    block_maxima_obs_dnw_dt = gev_funcs.pivot_detrend_obs(
        df=block_maxima_obs_dnw,
        x_axis_name="effective_dec_year",
        y_axis_name="demand_net_wind_max",
    )

    # turn effective dec year back into an int for blco_maxima_model_dnw
    block_maxima_model_dnw["effective_dec_year"] = block_maxima_model_dnw[
        "effective_dec_year"
    ].dt.year.astype(int)

    # remove the trend from the model data
    block_maxima_model_dnw_dt = gev_funcs.pivot_detrend_model(
        df=block_maxima_model_dnw,
        x_axis_name="effective_dec_year",
        y_axis_name="demand_net_wind_max",
    )

    # remove the trend from the bias corrected model data
    block_maxima_model_dnw_bc_dt = gev_funcs.pivot_detrend_model(
        df=block_maxima_model_dnw,
        x_axis_name="effective_dec_year",
        y_axis_name="demand_net_wind_max_bc",
    )

    # Set the effective dec year as a datetime for the model data
    block_maxima_model_dnw_dt["effective_dec_year"] = pd.to_datetime(
        block_maxima_model_dnw_dt["effective_dec_year"], format="%Y"
    )

    # Set effective dec year as an index for the obs
    block_maxima_obs_dnw_dt.set_index("effective_dec_year", inplace=True)

    # do the dot plot for the detrended model and obs data
    dot_plot(
        obs_df=block_maxima_obs_dnw_dt,
        model_df=block_maxima_model_dnw_dt,
        obs_val_name="demand_net_wind_max",
        model_val_name="demand_net_wind_max_dt",
        model_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        title="Block maxima demand net wind, DJF, 1960-2017, detrended",
        ylims=(30, 60),
        solid_line=np.max,
        dashed_quant=0.80,
    )

    # do the dot plot for the detrended model and obs data
    dot_plot(
        obs_df=block_maxima_obs_dnw_dt,
        model_df=block_maxima_model_dnw_bc_dt,
        obs_val_name="demand_net_wind_max",
        model_val_name="demand_net_wind_max_bc_dt",
        model_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        title="Block maxima demand net wind, DJF, 1960-2017, detrended, BC",
        ylims=(30, 60),
        solid_line=np.max,
        dashed_quant=0.80,
    )

    # set the index of block_maxima_obs_dnw back to effective dec year
    block_maxima_obs_dnw.reset_index(inplace=True)

    # make sure effective dec year is a datetime
    block_maxima_obs_dnw["effective_dec_year"] = pd.to_datetime(
        block_maxima_obs_dnw["effective_dec_year"], format="%Y"
    )

    # set effective dec year as an int
    block_maxima_obs_dnw["effective_dec_year"] = block_maxima_obs_dnw[
        "effective_dec_year"
    ].dt.year.astype(int)

    # set back as the index
    block_maxima_obs_dnw.set_index("effective_dec_year", inplace=True)

    # Now plot the comparison for wind/demand
    # during demand net wind days
    # but standardised
    gev_funcs.plot_scatter_cmap(
        obs_df=block_maxima_obs_dnw,
        model_df=block_maxima_model_dnw,
        obs_x_var_name="sigmoid_total_wind_gen",
        obs_y_var_name="UK_demand",
        obs_cmap_var_name="demand_net_wind_max",
        model_x_var_name="sigmoid_total_wind_gen",
        model_y_var_name="UK_demand",
        model_cmap_var_name="demand_net_wind_max",
        xlabel="Normalised wind power generation anoms",
        ylabel="Normalised demand anoms",
        cmap_label="Normalised demand net wind anoms",
        sup_title=None,
        xlims=(-5, 5),
        model_title="Demand net wind anoms",
        cmap="viridis_r",
        figsize=(6, 6),
    )

    sys.exit()

    # # process the dict for standard fid testing
    # moments_dict = gev_funcs.process_moments_fidelity(
    #     obs_df=df_obs_dt,
    #     model_df=df_model_djf_dt,
    #     obs_var_name="demand_net_wind",
    #     model_var_name="demand_net_wind",
    #     obs_wyears_name="effective_dec_year",
    #     model_wyears_name="effective_dec_year",
    #     nboot=1000,
    #     model_member_name="member",
    #     model_lead_name="winter_year",
    # )

    # # Now plot the fidelity testing output
    # gev_funcs.plot_moments_fidelity(
    #     obs_df=df_obs_dt,
    #     model_df=df_model_djf_dt,
    #     obs_var_name="demand_net_wind",
    #     model_var_name="demand_net_wind",
    #     moments_fidelity=moments_dict,
    #     title="Fidelity testing for demand net wind (detrended temp), DJF, 1960-2017",
    #     figsize=(15, 5),
    # )

    # Now sys exit
    # sys.exit()

    # Calculate the block maxima demand net wind for the obs data
    block_maxima_obs_dnw = gev_funcs.obs_block_min_max(
        df=df_obs,
        time_name="effective_dec_year",
        min_max_var_name="demand_net_wind",
        new_df_cols=["sigmoid_total_wind_gen", "UK_demand"],
        process_min=False,
    )

    # Calculate the block maxima demand net wind for the obs data detrend
    block_maxima_obs_dnw_dt = gev_funcs.obs_block_min_max(
        df=df_obs_dt,
        time_name="effective_dec_year",
        min_max_var_name="demand_net_wind",
        new_df_cols=["sigmoid_total_wind_gen", "UK_demand"],
        process_min=False,
    )

    # print the head of df_model_djf
    print(df_model_djf.head())
    print(df_model_djf.tail())

    # Same for the model data
    block_maxima_model_dnw = gev_funcs.model_block_min_max(
        df=df_model_djf,
        time_name="init_year",
        min_max_var_name="demand_net_wind",
        new_df_cols=["sigmoid_total_wind_gen", "UK_demand", "init_year", "winter_year"],
        winter_year="winter_year",
        process_min=False,
    )

    # Same for the model data detrend
    block_maxima_model_dnw_dt = gev_funcs.model_block_min_max(
        df=df_model_djf_dt,
        time_name="init_year",
        min_max_var_name="demand_net_wind",
        new_df_cols=["sigmoid_total_wind_gen", "UK_demand", "init_year", "winter_year"],
        winter_year="winter_year",
        process_min=False,
    )

    # sys.exit()

    # print the head of the df_obs
    print(df_obs.head())

    # print the head of the df_model_djf
    print(df_model_djf.head())

    # make sure that effective_dec_year is in the block_maxima_model_dnw_dt
    if "effective_dec_year" not in block_maxima_model_dnw_dt.columns:
        block_maxima_model_dnw_dt["effective_dec_year"] = block_maxima_model_dnw_dt[
            "init_year"
        ] + (block_maxima_model_dnw_dt["winter_year"] - 1)

    if "effective_dec_year" not in block_maxima_model_dnw.columns:
        block_maxima_model_dnw["effective_dec_year"] = block_maxima_model_dnw[
            "init_year"
        ] + (block_maxima_model_dnw["winter_year"] - 1)

    # Now compare the trends
    gev_funcs.compare_trends(
        model_df_full_field=df_model_djf,
        obs_df_full_field=df_obs,
        model_df_block=block_maxima_model_dnw,
        obs_df_block=block_maxima_obs_dnw,
        model_var_name_full_field="demand_net_wind",
        obs_var_name_full_field="demand_net_wind",
        model_var_name_block="demand_net_wind_max",
        obs_var_name_block="demand_net_wind_max",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        suptitle="Demand Net Wind trends (temp + wind lead BC)",
        figsize=(15, 5),
    )

    # Now compare the trends for the detrended data
    gev_funcs.compare_trends(
        model_df_full_field=df_model_djf_dt,
        obs_df_full_field=df_obs_dt,
        model_df_block=block_maxima_model_dnw_dt,
        obs_df_block=block_maxima_obs_dnw_dt,
        model_var_name_full_field="demand_net_wind",
        obs_var_name_full_field="demand_net_wind",
        model_var_name_block="demand_net_wind_max",
        obs_var_name_block="demand_net_wind_max",
        model_time_name="effective_dec_year",
        obs_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        suptitle="Demand Net Wind trends (detrended, temp + wind lead BC)",
        figsize=(15, 5),
    )

    # print the columns in the block_maxima_model_dnw_dt
    print(block_maxima_model_dnw_dt.columns)

    # print the head of block_maxima_model_dnw_dt
    print(block_maxima_model_dnw_dt.head())

    # Process lead time dependent mean bias correction for demand net wind
    block_maxima_model_dnw_dt = gev_funcs.lead_time_mean_bias_correct(
        model_df=block_maxima_model_dnw_dt,
        obs_df=block_maxima_obs_dnw_dt,
        model_var_name="demand_net_wind_max",
        obs_var_name="demand_net_wind_max",
        lead_name="winter_year",
    )

    # bias correct the demand net wind (non)

    # If block maxima model dt does not have column:
    # "effective_dec_year", add it
    if "effective_dec_year" not in block_maxima_model_dnw_dt.columns:
        block_maxima_model_dnw_dt["effective_dec_year"] = block_maxima_model_dnw_dt[
            "init_year"
        ] + (block_maxima_model_dnw_dt["winter_year"] - 1)

    # Now process the GEV params
    # for the non biasw corrected data
    gev_params_no_bc = gev_funcs.process_gev_params(
        obs_df=block_maxima_obs_dnw,
        model_df=block_maxima_model_dnw,
        obs_var_name="demand_net_wind_max",
        model_var_name="demand_net_wind_max",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        nboot=1000,
        model_lead_name="winter_year",
    )

    # Process the GEV params for the bias corrected data
    gev_params_bc = gev_funcs.process_gev_params(
        obs_df=block_maxima_obs_dnw_dt,
        model_df=block_maxima_model_dnw_dt,
        obs_var_name="demand_net_wind_max",
        model_var_name="demand_net_wind_max_bc",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        nboot=1000,
        model_lead_name="winter_year",
    )

    # Now plot the GEV params - non bias corrected
    gev_funcs.plot_gev_params(
        gev_params=gev_params_no_bc,
        obs_df=block_maxima_obs_dnw,
        model_df=block_maxima_model_dnw,
        obs_var_name="demand_net_wind_max",
        model_var_name="demand_net_wind_max",
        title="Distribution of max DJF demand net wind (GW), no BC",
        obs_label="obs",
        model_label="model",
        figsize=(15, 5),
    )

    # Now plot the GEV params - bias corrected
    gev_funcs.plot_gev_params(
        gev_params=gev_params_bc,
        obs_df=block_maxima_obs_dnw_dt,
        model_df=block_maxima_model_dnw_dt,
        obs_var_name="demand_net_wind_max",
        model_var_name="demand_net_wind_max_bc",
        title="Distribution of max DJF demand net wind (GW), BC",
        obs_label="obs",
        model_label="model",
        figsize=(15, 5),
    )

    # Set effective dec year as a datetime for the model data
    block_maxima_model_dnw_dt["effective_dec_year"] = pd.to_datetime(
        block_maxima_model_dnw_dt["effective_dec_year"], format="%Y"
    )

    # Reset the index of the obs data
    block_maxima_obs_dnw.reset_index(inplace=True)

    # Format effective dec year as a datetime
    block_maxima_obs_dnw["effective_dec_year"] = pd.to_datetime(
        block_maxima_obs_dnw["effective_dec_year"], format="%Y"
    )

    # set effective dec year as the index in the obs df
    block_maxima_obs_dnw.set_index("effective_dec_year", inplace=True)

    # Plot the dot plot for the non bias corrected data
    dot_plot(
        obs_df=block_maxima_obs_dnw,
        model_df=block_maxima_model_dnw,
        obs_val_name="demand_net_wind_max",
        model_val_name="demand_net_wind_max",
        model_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        title="Observed vs modelled max DJF demand net wind (GW), no BC",
        ylims=(35, 50),
        solid_line=np.max,
        dashed_quant=0.80,
    )

    # Plot the dot plot for the bias corrected data
    dot_plot(
        obs_df=block_maxima_obs_dnw_dt,
        model_df=block_maxima_model_dnw_dt,
        obs_val_name="demand_net_wind_max",
        model_val_name="demand_net_wind_max_bc",
        model_time_name="effective_dec_year",
        ylabel="Demand Net Wind (GW)",
        title="Observed vs modelled max DJF demand net wind (GW), BC",
        ylims=(35, 50),
        solid_line=np.max,
        dashed_quant=0.80,
    )

    # print the time taken
    print(f"Time taken: {time.time() - start} seconds")

    return None


if __name__ == "__main__":
    main()
# %%
