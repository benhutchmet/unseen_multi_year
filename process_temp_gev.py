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
from process_dnw_gev import (
    select_leads_wyears_DJF,
    plot_distributions_extremes,
    plot_multi_var_perc,
)

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import sigmoid, dot_plot, plot_rp_extremes, empirical_return_level

# Silence warnings
warnings.filterwarnings("ignore")


# Define a function to do pivoting
# to each year and then calculating empirical return periods for this
def pivot_emp_rps(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    obs_time_name: str,
    model_time_name: str,
    var_name: str,
    nsamples: int = 10000,
    figsize: tuple = (5, 5),
    wind_2005_toggle=True,
) -> None:
    """
    Pivots the entire ensemble around each year in turn and quantifies
    the likelihood of seeing an event worse than the worst observed extreme.

    Parameters
    ==========

        obs_df : pd.DataFrame
            DataFrame of observed data.
        model_df : pd.DataFrame
            DataFrame of model data.
        obs_val_name : str
            Name of the column to use in the observed DataFrame.
        model_val_name : str
            Name of the column to use in the model DataFrame.
        obs_time_name : str
            Name of the column to use as the time axis in the observed DataFrame.
        model_time_name : str
            Name of the column to use as the time axis in the model DataFrame.
        var_name : str
            Name of the variable to use in the DataFrames.
        nsamples : int, optional
            Number of samples to use for the empirical return periods, by default 10000.
        figsize : tuple, optional
            Figure size, by default (5, 5).

    Returns
    =======

        None


    """

    # Make a copy of the dataframes
    obs_df_copy = obs_df.copy()
    model_df_copy = model_df.copy()

    # if the time column is not datetime
    if not isinstance(obs_df_copy[obs_time_name].values[0], int):
        obs_df_copy[obs_time_name].astype(int)

    # if the time column is not datetime
    if not isinstance(model_df_copy[model_time_name].values[0], int):
        model_df_copy[model_time_name].astype(int)

    # Extract the unique time points from the models
    unique_model_times = model_df_copy[model_time_name].unique()
    unique_obs_times = obs_df_copy[obs_time_name].unique()

    # Set up the model and obs vals
    model_vals = model_df_copy.groupby(model_time_name)[model_val_name].mean()
    obs_vals = obs_df_copy[obs_val_name].values

    # Calculate the model trend
    slope_model, intercept_model, _, _, _ = linregress(unique_model_times, model_vals)

    # Calculate the obs trend
    slope_obs, intercept_obs, _, _, _ = linregress(unique_obs_times, obs_vals)

    # Calculate the model trend line
    model_trend_line = slope_model * unique_model_times + intercept_model
    obs_trend_line = slope_obs * unique_obs_times + intercept_obs

    # Detrend the model and obs data
    model_df_copy[f"{model_val_name}_dt"] = model_df_copy[model_val_name] - (
        slope_model * model_df_copy[model_time_name] + intercept_model
    )

    # Detrend the obs data
    obs_df_copy[f"{obs_val_name}_dt"] = obs_df_copy[obs_val_name] - (
        slope_obs * obs_df_copy[obs_time_name] + intercept_obs
    )

    # Set up the trend value this for the obs
    obs_trend_val_this = slope_obs * unique_model_times[-1] + intercept_obs

    # Adjust the detrended data for this year
    obs_adjusted_this = np.array(obs_df_copy[f"{obs_val_name}_dt"] + obs_trend_val_this)

    obs_extreme_value = np.min(obs_adjusted_this)

    # # if the variable is tas
    # if var_name == "tas":
    #     # Worst observed extreme is min
    #     obs_extreme_value = np.min(obs_adjusted_this)
    # elif var_name == "sfcWind" and wind_2005_toggle:
    #     print("Identifying second worst observed extreme")

    #     # Find the second worst observed extreme
    #     sorted_values = np.sort(obs_adjusted_this)  # Sort the values in ascending order
    #     if len(sorted_values) > 1:
    #         obs_extreme_value = sorted_values[1]  # Second lowest value
    #     else:
    #         raise ValueError("Not enough data to determine the second worst extreme.")
    # elif var_name == "sfcWind" and not wind_2005_toggle:
    #     # Worst observed extreme is min
    #     obs_extreme_value = np.min(obs_adjusted_this)
    # else:
    #     raise ValueError(
    #         "Variable not recognised. Please use tas or sfcWind."
    #     )

    # find the index of the obs extreme value in obs_adjusted_this
    obs_extreme_index = np.where(obs_adjusted_this == obs_extreme_value)[0][0]

    # find the time in which this occurs
    # Find the time in which this occurs
    obs_extreme_time = obs_df_copy.iloc[obs_extreme_index][obs_time_name]

    # print the worst observed extreme
    print(f"Worst observed extreme: {obs_extreme_value}")

    # print the time in which this occurs
    print(f"Worst observed extreme time (year): {obs_extreme_time}")

    # Set up a new dataframe to append values to
    model_df_plume = pd.DataFrame()

    # Loop over the unique model times
    for i, model_time in tqdm(
        enumerate(unique_model_times),
        desc="Processing model times",
        total=len(unique_model_times),
        leave=False,
    ):
        # Find the index where the model time is equal to the obs time
        obs_index_this = np.where(obs_df_copy[obs_time_name] == model_time)[0][0]

        # Apply this index to the obs trend
        obs_trend_point_this = obs_trend_line[obs_index_this]

        # Set up the trend value this
        trend_val_this = slope_model * model_time + intercept_model

        # Calculate the final point bias
        trend_point_bias = obs_trend_point_this - trend_val_this

        # Apply this to the model data
        trend_val_this_bc = (
            slope_model * model_time + intercept_model + trend_point_bias
        )

        # Adjust the detrended data for this year
        model_adjusted_this = np.array(
            model_df_copy[f"{model_val_name}_dt"] + trend_val_this_bc
        )

        # Set up the central return levels
        model_df_central_rps_this = empirical_return_level(
            data=model_adjusted_this,
            high_values_rare=False,
        )

        # Set up the bootstrap
        model_df_bootstrap_rps_this = np.zeros(
            [nsamples, len(model_df_central_rps_this["sorted"])]
        )

        # Loop over the samples
        for j in tqdm(
            range(nsamples),
            desc="Processing samples",
            total=nsamples,
            leave=False,
        ):
            # Resample the model data
            model_vals_this = np.random.choice(
                model_adjusted_this,
                size=len(model_df_central_rps_this["sorted"]),
                replace=True,
            )

            # Calculate the empirical return levels
            model_df_rls_this = empirical_return_level(
                data=model_vals_this,
                high_values_rare=False,
            )

            # Append the return levels to the array
            model_df_bootstrap_rps_this[j, :] = model_df_rls_this["sorted"]

        # Now to find the rp for the worst obs event
        # using the central estimate
        # as well as the 0025 and 0975 quantiles
        # Find the index of the row, where "sorted" is closest to the observed
        # extreme value
        obs_extreme_index_central = np.abs(
            model_df_central_rps_this["sorted"] - obs_extreme_value
        ).argmin()

        # Print this row
        # in the dataframe
        # Corrected code using single quotes for column names
        # print(
        #     f"Row in the dataframe closest to obs extreme value: {model_df_central_rps_this.iloc[obs_extreme_index_central]['sorted']}"
        # )
        # print(
        #     f"Row in the dataframe closest to obs extreme value: {model_df_central_rps_this.iloc[obs_extreme_index_central]['period']}"
        # )

        # Set up the 0025 and 0975 quantiles
        model_df_central_rps_this["025"] = np.quantile(
            model_df_bootstrap_rps_this, 0.025, axis=0
        )
        model_df_central_rps_this["975"] = np.quantile(
            model_df_bootstrap_rps_this, 0.975, axis=0
        )

        # Find the index of the row, where "sorted" is closest to the observed
        # extreme value
        obs_extreme_index_025 = np.abs(
            model_df_central_rps_this["025"] - obs_extreme_value
        ).argmin()

        # Print this row
        # in the dataframe
        # print(
        #     f"Row in the dataframe closest to obs extreme value: {model_df_central_rps_this.iloc[obs_extreme_index_025]['025']}"
        # )
        # print(
        #     f"Row in the dataframe closest to obs extreme value: {model_df_central_rps_this.iloc[obs_extreme_index_025]['period']}"
        # )

        # Find the index of the row, where "sorted" is closest to the observed
        # extreme value
        obs_extreme_index_975 = np.abs(
            model_df_central_rps_this["975"] - obs_extreme_value
        ).argmin()

        # Print this row
        # in the dataframe
        # print(
        #     f"Row in the dataframe closest to obs extreme value: {model_df_central_rps_this.iloc[obs_extreme_index_975]['975']}"
        # )
        # print(
        #     f"Row in the dataframe closest to obs extreme value: {model_df_central_rps_this.iloc[obs_extreme_index_975]['period']}"
        # )

        # Set up a new dataframe with the values
        model_df_this = pd.DataFrame(
            {
                "model_time": [model_time],  # Wrap scalar in a list
                "central_rp": [
                    model_df_central_rps_this.iloc[obs_extreme_index_central]["period"]
                ],
                "025_rp": [
                    model_df_central_rps_this.iloc[obs_extreme_index_025]["period"]
                ],
                "975_rp": [
                    model_df_central_rps_this.iloc[obs_extreme_index_975]["period"]
                ],
            }
        )

        # Append this to the dataframe
        model_df_plume = pd.concat(
            [model_df_plume, model_df_this],
            ignore_index=True,
        )

        # # print the head of model_df_plume
        # print(model_df_plume.head())

    # translate these return periods in years into percentages
    model_df_plume["central_rp_%"] = 1 / (model_df_plume["central_rp"] / 100)
    model_df_plume["025_rp_%"] = 1 / (model_df_plume["025_rp"] / 100)
    model_df_plume["975_rp_%"] = 1 / (model_df_plume["975_rp"] / 100)

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
        model_df_plume["025_rp_%"],
        model_df_plume["975_rp_%"],
        color="red",
        alpha=0.3,  # Adjust transparency
        label="Return period range (0.025 - 0.975)",
    )

    # Limit the y-axis to between 0 and 4
    ax.set_ylim(0, 4)

    # Set up the ticks for the first y-axis using the ax object
    # ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5])  # Tick positions
    # ax.set_yticklabels(["0%", "0.5%", "1.0%", "1.5%", "2.0%", "2.5%", "3.0%", "3.5%"])  # Tick labels

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
        f"Chance of <{obs_extreme_time} by year",
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


# Define a function to subset extremes
# based on the full field data
def subset_extremes(
    model_df_full_field: pd.DataFrame,
    obs_df_full_field: pd.DataFrame,
    model_df_block: pd.DataFrame,
    obs_df_block: pd.DataFrame,
    model_var_name_full_field: str,
    obs_var_name_full_field: str,
    model_var_name_block: str,
    obs_var_name_block: str,
    percentile: float = 0.05,
) -> tuple:
    """
    Compares the block minima values to a percentile threshold
    of the full field values. If the block minima values exceed these
    thresholds, then the block minima values are removed.

    Parameters
    ==========

        model_df_full_field : pd.DataFrame
            DataFrame of model data.
        obs_df_full_field : pd.DataFrame
            DataFrame of observed data.
        model_df_block : pd.DataFrame
            DataFrame of model data.
        obs_df_block : pd.DataFrame
            DataFrame of observed data.
        model_var_name_full_field : str
            Name of the column to use in the model DataFrame.
        obs_var_name_full_field : str
            Name of the column to use in the observed DataFrame.
        model_var_name_block : str
            Name of the column to use in the model DataFrame.
        obs_var_name_block : str
            Name of the column to use in the observed DataFrame.
        percentile : float, optional
            Percentile threshold, by default 0.05.

    Returns
    =======

        tuple
            Tuple of DataFrames containing the extremes for model and obs data.

    """

    # Create copies of the DataFrames
    model_df_full_field_copy = model_df_full_field.copy()
    obs_df_full_field_copy = obs_df_full_field.copy()
    model_df_block_copy = model_df_block.copy()
    obs_df_block_copy = obs_df_block.copy()

    # reset the index of the DataFrames
    model_df_full_field_copy = model_df_full_field_copy.reset_index(drop=True)
    obs_df_full_field_copy = obs_df_full_field_copy.reset_index(drop=True)
    model_df_block_copy = model_df_block_copy.reset_index(drop=True)
    obs_df_block_copy = obs_df_block_copy.reset_index(drop=True)

    # Find the percentile threshold for the model df
    model_df_full_field_threshold = np.percentile(
        model_df_full_field_copy[model_var_name_full_field].values,
        percentile * 100,
    )
    obs_df_full_field_threshold = np.percentile(
        obs_df_full_field_copy[obs_var_name_full_field].values,
        percentile * 100,
    )

    # Count the number of block values which exceed this threshold
    model_df_block_exceed = np.sum(
        model_df_block_copy[model_var_name_block].values > model_df_full_field_threshold
    )
    obs_df_block_exceed = np.sum(
        obs_df_block_copy[obs_var_name_block].values > obs_df_full_field_threshold
    )

    # Print the number of block values which exceed this threshold
    print(f"Model block values exceeding threshold: {model_df_block_exceed}")
    print(f"Obs block values exceeding threshold: {obs_df_block_exceed}")

    # Find the indices of rows where the model block values exceed the threshold
    model_df_block_exceed_indices = model_df_block_copy[
        model_df_block_copy[model_var_name_block] > model_df_full_field_threshold
    ].index

    # Drop these rows
    model_df_block_copy = model_df_block_copy.drop(model_df_block_exceed_indices)

    # Find the indices of rows where the obs block values exceed the threshold
    obs_df_block_exceed_indices = obs_df_block_copy[
        obs_df_block_copy[obs_var_name_block] > obs_df_full_field_threshold
    ].index

    # Drop these rows
    obs_df_block_copy = obs_df_block_copy.drop(obs_df_block_exceed_indices)

    # Return the DataFrames as a tuple
    return (
        model_df_block_copy,
        obs_df_block_copy,
    )


# Define a function for plotting the return periods
# empirical
def plot_emp_rps(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    obs_time_name: str,
    model_time_name: str,
    ylabel: str,
    nsamples: int = 10000,
    ylims: tuple = (0, 1),
    blue_line: callable = np.min,
    high_values_rare: bool = False,
    figsize: tuple = (5, 5),
    bonus_line: float = None,
    wind_2005_toggle: bool = True,
) -> None:
    """
    Plot the empirical return periods for the model data.

    Parameters
    ==========

        obs_df : pd.DataFrame
            DataFrame of observed data.
        model_df : pd.DataFrame
            DataFrame of model data.
        obs_val_name : str
            Name of the column to use in the observed DataFrame.
        model_val_name : str
            Name of the column to use in the model DataFrame.
        obs_time_name : str
            Name of the column to use as the time axis in the observed DataFrame.
        model_time_name : str
            Name of the column to use as the time axis in the model DataFrame.
        ylabel : str
            Name of the y-axis label.
        nsamples : int, optional
            Number of samples to use for the empirical return periods, by default 10000.
        ylims : tuple, optional
            Y-axis limits, by default (0, 1).
        blue_line : callable, optional
            Function to use for the blue line, by default np.min.
        high_values_rare : bool, optional
            If True, the high values are rare, by default False.

    Returns
    =======

        None

    """

    # if the time column is not datetime
    if not isinstance(obs_df[obs_time_name].values[0], np.datetime64):
        obs_df[obs_time_name] = pd.to_datetime(obs_df[obs_time_name])

    # if the time column is not datetime
    if not isinstance(model_df[model_time_name].values[0], np.datetime64):
        model_df[model_time_name] = pd.to_datetime(model_df[model_time_name])

    # Set up the central return levels
    model_df_central_rps = empirical_return_level(
        data=model_df[model_val_name].values,
        high_values_rare=high_values_rare,
    )

    # Set up the bootstrap
    model_df_bootstrap_rps = np.zeros([nsamples, len(model_df_central_rps)])

    # Loop over the samples
    for i in tqdm(range(nsamples)):
        # Resample the model data
        model_vals_this = np.random.choice(
            model_df[model_val_name].values,
            size=len(model_df_central_rps["sorted"]),
            replace=True,
        )

        # Calculate the empirical return levels
        model_df_rls_this = empirical_return_level(
            data=model_vals_this,
            high_values_rare=high_values_rare,
        )

        # Append the return levels to the array
        model_df_bootstrap_rps[i, :] = model_df_rls_this["sorted"]

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the central return levels
    ax.plot(
        model_df_central_rps["period"],
        model_df_central_rps["sorted"],
        color="red",
        label="Rank",
    )

    _ = ax.fill_between(
        model_df_central_rps["period"],
        np.quantile(
            model_df_bootstrap_rps,
            0.025,
            axis=0,
        ),
        np.quantile(
            model_df_bootstrap_rps,
            0.975,
            axis=0,
        ),
        color="red",
        alpha=0.5,
        label="Rank uncertainty",
    )

    # Set up a logarithmic x-axis
    ax.set_xscale("log")

    # Limit to between 10 and 1000 years
    ax.set_xlim(10, 1000)

    # Set the xticks at 10, 20, 50, 100, 200, 500, 1000
    plt.xticks(
        [10, 20, 50, 100, 200, 500, 1000],
        ["10", "20", "50", "100", "200", "500", "1000"],
    )

    # Set the ylim
    ax.set_ylim(ylims)

    # Set the y labels
    ax.set_ylabel(
        ylabel,
        fontsize=12,
    )

    # Set the x labels
    ax.set_xlabel(
        "Return period (years)",
        fontsize=12,
    )

    # Extreme value as the min
    extreme_value = blue_line(obs_df[obs_val_name].values)

    # # If ylabel includes "wind" or "Wind", and wind_2005_toggle is True
    # if ("wind" in ylabel or "Wind" in ylabel) and wind_2005_toggle:
    #     # Rank the obs values from low to high
    #     obs_df["rank"] = obs_df[obs_val_name].rank(
    #         method="first", ascending=True
    #     )
    #     # Find the second worst observed extreme
    #     extreme_value = obs_df.loc[
    #         obs_df["rank"] == 2, obs_val_name
    #     ].values[0]
    # elif "wind" in ylabel or "Wind" in ylabel and not wind_2005_toggle:
    #     # Find the extreme value
    #     extreme_value = blue_line(
    #         obs_df[obs_val_name].values
    #     )
    # else:
    #     # Find the extreme value
    #     extreme_value = blue_line(
    #         obs_df[obs_val_name].values
    #     )

    # find the time in which this occurs
    extreme_time = obs_df.loc[
        obs_df[obs_val_name] == extreme_value, obs_time_name
    ].values[0]

    # Convert numpy.datetime64 to pandas.Timestamp and extract the year
    extreme_time = pd.Timestamp(extreme_time).year

    # Print the extreme value
    print(f"Extreme value: {extreme_value}")

    # Print the time in which this occurs
    print(f"Extreme time (year): {extreme_time}")

    # Plot the extreme value as a blue horizontal
    # dashed line
    ax.axhline(y=extreme_value, color="blue", linestyle="--", label=f"{extreme_time}")

    # if bonus line is not none, then mark it on the plot
    if bonus_line is not None:
        # Plot the bonus line as a red horizontal dashed line
        ax.axhline(
            y=bonus_line,
            color="red",
            linestyle="--",
        )

    # include a legend in the top right
    ax.legend(
        loc="upper right",
        fontsize=12,
    )

    return None


# Define a function to plot the gev return periods
def plot_gev_rps(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_val_name: str,
    model_val_name: str,
    obs_time_name: str,
    model_time_name: str,
    ylabel: str,
    nsamples: int = 10000,
    ylims: tuple = (0, 1),
    blue_line: callable = np.min,
    high_values_rare: bool = False,
    figsize: tuple = (5, 5),
    bonus_line: float = None,
) -> None:
    """
    Plots the return periods for the model and obs using a GEV.

    Parameters
    ==========

        obs_df : pd.DataFrame
            DataFrame of observed data.
        model_df : pd.DataFrame
            DataFrame of model data.
        obs_val_name : str
            Name of the column to use in the observed DataFrame.
        model_val_name : str
            Name of the column to use in the model DataFrame.
        obs_time_name : str
            Name of the column to use as the time axis in the observed DataFrame.
        model_time_name : str
            Name of the column to use as the time axis in the model DataFrame.
        ylabel : str
            Name of the y-axis label.
        nsamples : int, optional
            Number of samples to use for the empirical return periods, by default 10000.
        ylims : tuple, optional
            Y-axis limits, by default (0, 1).
        blue_line : callable, optional
            Function to use for the blue line, by default np.min.
        high_values_rare : bool, optional
            If True, the high values are rare, by default False.

    Returns
    =======

        None

    """

    # If the time column is not a datetime
    if not isinstance(obs_df[obs_time_name].values[0], np.datetime64):
        obs_df[obs_time_name] = pd.to_datetime(obs_df[obs_time_name])

    # If the time column is not a datetime
    if not isinstance(model_df[model_time_name].values[0], np.datetime64):
        model_df[model_time_name] = pd.to_datetime(model_df[model_time_name])

    # Set up the empirical return levels
    model_df_central_rps = empirical_return_level(
        data=model_df[model_val_name].values,
        high_values_rare=high_values_rare,
    )

    # set up the empirical bootstrap
    model_df_bootstrap_rps = np.zeros([nsamples, len(model_df_central_rps["sorted"])])

    # Loop over the samples
    for i in tqdm(range(nsamples)):
        # Resample the model data
        model_vals_this = np.random.choice(
            model_df[model_val_name].values,
            size=len(model_df_central_rps["sorted"]),
            replace=True,
        )

        # Calculate the empirical return levels
        model_df_rls_this = empirical_return_level(
            data=model_vals_this,
            high_values_rare=high_values_rare,
        )

        # Append the return levels to the array
        model_df_bootstrap_rps[i, :] = model_df_rls_this["sorted"]

    # Set up the probabilities and years
    probs = 1 / np.arange(1.1, 1000, 0.1) * 100
    years = np.arange(1.1, 1000, 0.1)

    # Set up the lists to store GEV params
    model_gev_params = []
    obs_gev_params = []

    # Set up the model gev params first
    model_gev_params_first = gev.fit(
        model_df[model_val_name].values,
    )

    # Loop over the samples
    for i in tqdm(range(nsamples)):
        # Resample the model data
        model_vals_this = np.random.choice(
            model_df[model_val_name].values,
            size=len(model_df[model_val_name]),
            replace=True,
        )

        # Resample the obs data
        obs_vals_this = np.random.choice(
            obs_df[obs_val_name].values,
            size=len(obs_df[obs_val_name]),
            replace=True,
        )

        # Quantify the model RLs using the GEV
        model_gev_params.append(
            gev.fit(
                model_vals_this,
            )
        )

        # Do the same for the obs data
        obs_gev_params.append(
            gev.fit(
                obs_vals_this,
            )
        )

    # Set up the return levels list for model and obs
    model_gev_rls = []
    obs_gev_rls = []

    if high_values_rare:
        years_ppf = 1 - 1 / years
    else:
        years_ppf = 1 / years

    # Loop over the nsamples
    for i in tqdm(range(nsamples)):
        # Calculate the return levels for the model
        model_gev_rls.append(
            np.array(
                gev.ppf(
                    years_ppf,
                    *model_gev_params[i],
                )
            )
        )

        # Calculate the return levels for the obs
        obs_gev_rls.append(
            np.array(
                gev.ppf(
                    years_ppf,
                    *obs_gev_params[i],
                )
            )
        )

    # generate the return levels from this
    model_gev_rls_first = np.array(
        gev.ppf(
            years_ppf,
            *model_gev_params_first,
        )
    )

    # Calculate the probability for a 1-in-100-year return period
    return_period = 100
    probability = 1 - 1 / return_period if high_values_rare else 1 / return_period

    # Calculate the value for the 1-in-100-year return period using the first model GEV parameters
    value_1_in_100 = gev.ppf(probability, *model_gev_params_first)

    # Print the value
    print(
        f"The value corresponding to a 1-in-100-year return period is: {value_1_in_100}"
    )

    # Convert the probs to return years
    return_years = 1 / (probs / 100)

    # Convert the model and obs GEV params to arrays
    model_gev_params = np.array(model_gev_params)
    obs_gev_params = np.array(obs_gev_params)

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # plot the empirical points as black dots
    ax.plot(
        model_df_central_rps["period"],
        model_df_central_rps["sorted"],
        "o",
        color="black",
        label="Empirical",
        linestyle="None",
        markersize=3,
    )

    # Plot the 0.025 and 0.975 quantiles of the empirical return levels
    # as dashed red lines
    _ = ax.plot(
        model_df_central_rps["period"],
        np.quantile(model_df_bootstrap_rps, 0.025, axis=0),
        color="red",
        linestyle="--",
        label="Empirical uncertainty",
    )
    _ = ax.plot(
        model_df_central_rps["period"],
        np.quantile(model_df_bootstrap_rps, 0.975, axis=0),
        color="red",
        linestyle="--",
    )

    # Plot the observed return levels
    _ = ax.fill_between(
        return_years,
        np.quantile(obs_gev_rls, 0.025, axis=0).T,
        np.quantile(obs_gev_rls, 0.975, axis=0).T,
        color="gray",
        alpha=0.5,
        label="ERA5",
    )

    # Plot the model return levels
    _ = ax.fill_between(
        return_years,
        np.quantile(model_gev_rls, 0.025, axis=0).T,
        np.quantile(model_gev_rls, 0.975, axis=0).T,
        color="red",
        alpha=0.5,
        label="DePreSys",
    )

    # Plot the model gev return levels first
    # as a solid red line
    _ = ax.plot(
        return_years,
        model_gev_rls_first,
        color="red",
    )

    # Set up a logarithmic x-axis
    ax.set_xscale("log")

    # Limit to between 10 and 1000 years
    ax.set_xlim(10, 1000)

    # Set the xticks at 10, 20, 50, 100, 200, 500, 1000
    plt.xticks(
        [10, 20, 50, 100, 200, 500, 1000],
        ["10", "20", "50", "100", "200", "500", "1000"],
    )

    # Set the ylim
    ax.set_ylim(ylims)

    # Set the y labels
    ax.set_ylabel(
        ylabel,
        fontsize=12,
    )

    # Set the x labels
    ax.set_xlabel(
        "Return period (years)",
        fontsize=12,
    )

    # Find the extreme value
    extreme_value = blue_line(obs_df[obs_val_name].values)

    # find the time in which this occurs
    extreme_time = obs_df.loc[
        obs_df[obs_val_name] == extreme_value, obs_time_name
    ].values[0]

    # Convert numpy.datetime64 to pandas.Timestamp and extract the year
    extreme_time = pd.Timestamp(extreme_time).year

    # Print the extreme value
    print(f"Extreme value: {extreme_value}")

    # Print the time in which this occurs
    print(f"Extreme time (year): {extreme_time}")

    # Plot the extreme value as a blue horizontal
    # dashed line
    ax.axhline(y=extreme_value, color="blue", linestyle="--", label=f"{extreme_time}")

    # if bonus line is not none, then mark it on the plot
    if bonus_line is not None:
        # Plot the bonus line as a red horizontal dashed line
        ax.axhline(
            y=bonus_line,
            color="red",
            linestyle="--",
        )

    # include a legend in the top right
    ax.legend(
        loc="upper right",
        fontsize=12,
    )

    # set up another figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot a histogram showing the distribution of points
    ax.hist(
        model_df[model_val_name].values,
        bins=40,
        color="red",
        alpha=0.5,
        label="Model distribution",
        density=True,
    )

    # PLot the GEV fit to this
    x = np.linspace(
        model_df[model_val_name].min(),
        model_df[model_val_name].max(),
        100,
    )

    # Calculate the GEV fit
    y = gev.pdf(
        x,
        *model_gev_params_first,
    )

    # Plot the GEV fit
    ax.plot(
        x,
        y,
        color="red",
        label="Model GEV fit",
        linestyle="--",
    )

    # include a legend in the top right
    ax.legend(
        loc="upper right",
        fontsize=12,
    )

    # Set the x label
    ax.set_xlabel(
        ylabel,
        fontsize=12,
    )

    # remove the y label
    ax.set_ylabel(
        "",
    )

    # remove the y ticks
    ax.set_yticks([])

    # calculate the 10th percentile of tthe variable
    model_10th_percentile = np.percentile(
        model_df[model_val_name].values,
        10,
    )

    # Set up a new figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot a histogram showing the distribution of points
    ax.hist(
        model_df[model_val_name].values,
        bins=40,
        color="red",
        alpha=0.5,
        label="Model distribution",
        density=True,
    )

    # Plot the GEV fit to this
    x = np.linspace(
        model_df[model_val_name].min(),
        model_df[model_val_name].max(),
        100,
    )

    # Calculate the GEV fit
    y = gev.pdf(
        x,
        *model_gev_params_first,
    )

    # Plot the GEV fit
    ax.plot(
        x,
        y,
        color="red",
        label="Model GEV fit",
        linestyle="--",
    )

    # Set the xlims between min and 10th percentile
    ax.set_xlim(
        model_df[model_val_name].min(),
        model_10th_percentile,
    )

    # Set the x label
    ax.set_xlabel(
        ylabel,
        fontsize=12,
    )

    # remove the y label
    ax.set_ylabel(
        "",
    )

    # remove the y ticks
    ax.set_yticks([])

    # Set up the legend
    ax.legend(
        loc="upper right",
        fontsize=12,
    )

    # Set up a new figure for the Q-Q plot
    fig, ax = plt.subplots(figsize=figsize)

    # Sort the model data (empirical quantiles)
    sorted_model_data = np.sort(model_df[model_val_name].values)

    # Calculate the theoretical quantiles using the GEV CDF
    theoretical_quantiles = gev.ppf(
        np.linspace(0, 1, len(sorted_model_data)),
        *model_gev_params_first,
    )

    # Plot the Q-Q plot
    ax.scatter(
        sorted_model_data,
        theoretical_quantiles,
        color="red",
        alpha=0.7,
        label="Q-Q Plot",
        marker="o",
        s=5,
    )

    # Add a 1:1 reference line
    # Filter out NaN and inf values from sorted_model_data and theoretical_quantiles
    valid_model_data = sorted_model_data[np.isfinite(sorted_model_data)]
    valid_theoretical_quantiles = theoretical_quantiles[
        np.isfinite(theoretical_quantiles)
    ]

    # Calculate min and max values ignoring NaN and inf
    min_val = min(np.nanmin(valid_model_data), np.nanmin(valid_theoretical_quantiles))
    max_val = max(np.nanmax(valid_model_data), np.nanmax(valid_theoretical_quantiles))
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="blue",
        linestyle="--",
        label="1:1 Line",
        linewidth=2,
    )

    # Set labels and legend
    ax.set_xlabel("Empirical Quantiles (Model Data)", fontsize=12)
    ax.set_ylabel("Theoretical Quantiles (GEV Fit)", fontsize=12)
    ax.legend(loc="upper left", fontsize=12)

    # # Show the plot
    # plt.show()

    # Set up a new figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the Q-Q plot
    ax.scatter(
        sorted_model_data,
        theoretical_quantiles,
        color="red",
        alpha=0.7,
        label="Q-Q Plot",
        marker="o",
        s=5,
    )

    # Add a 1:1 reference line
    # Filter out NaN and inf values from sorted_model_data and theoretical_quantiles
    valid_model_data = sorted_model_data[np.isfinite(sorted_model_data)]
    valid_theoretical_quantiles = theoretical_quantiles[
        np.isfinite(theoretical_quantiles)
    ]

    # Calculate min and max values ignoring NaN and inf
    min_val = min(np.nanmin(valid_model_data), np.nanmin(valid_theoretical_quantiles))
    max_val = max(np.nanmax(valid_model_data), np.nanmax(valid_theoretical_quantiles))
    print(f"min_val: {min_val}")
    print(f"max_val: {max_val}")
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="blue",
        linestyle="--",
        label="1:1 Line",
        linewidth=2,
    )

    # Set labels and legend
    ax.set_xlabel("Empirical Quantiles (Model Data)", fontsize=12)
    ax.set_ylabel("Theoretical Quantiles (GEV Fit)", fontsize=12)
    ax.legend(loc="upper left", fontsize=12)
    ax.set_title("Q-Q Plot of Model Data vs. GEV Fit", fontsize=14)

    # Set the xlims between min and 10th percentile
    ax.set_xlim(
        model_df[model_val_name].min(),
        model_10th_percentile,
    )

    # Set the ylims between min and 10th percentile
    ax.set_ylim(
        model_df[model_val_name].min(),
        model_10th_percentile,
    )

    return None


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
            print(
                f"Shape of effective dec years constant: {effective_dec_years_constant.shape}"
            )

            raise ValueError(
                f"Ensemble means this does not have the same length as effective dec years constant"
            )

        # Calculate the mean of the ensemble means - forecast climatology
        forecast_clim_lead_this = np.mean(ensemble_means_this)

        # print the lead and the forecast climatology lead this
        print(
            f"lead: {lead} - forecast climatology lead this: {forecast_clim_lead_this}"
        )

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

    # # Set up the axes
    # # Set up the figure size
    # fig, axes = plt.subplots(
    #     nrows=3,
    #     ncols=4,
    #     figsize=figsize,
    #     sharex=True,
    #     sharey=True,
    #     layout="compressed",
    # )

    # # Loop over the unique leads
    # for i, lead in enumerate(unique_leads):
    #     # Subset the model data to the lead this
    #     model_df_lead_this = model_df_copy_constant[model_df_copy_constant[lead_name] == lead]

    #     # # print the lead
    #     # print(f"Lead time: {lead}")

    #     # # print the first and last unique effective dec years in this df
    #     # print(f"First unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[0]}")
    #     # print(f"Last unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[-1]}")

    #     # # print the len of the unique effective dec years in this df
    #     # print(f"Length of unique effective dec years: {len(model_df_lead_this[eff_dec_years_name].unique())}")

    #     # calculate the mean
    #     model_mean_this = model_df_lead_this[f"{model_var_name}"].mean()

    #     # if the omodel mean is nan
    #     if np.isnan(model_mean_this):
    #         print(model_df_lead_this)

    #     # include the mean in the title
    #     title = f"Lead {lead} - Model mean: {model_mean_this:.2f}"

    #     # Plot the data
    #     ax = axes.flatten()[i]

    #     # Plot the histograms using matplotlib
    #     ax.hist(
    #         model_df_lead_this[f"{model_var_name}"],
    #         bins=30,
    #         color="red",
    #         edgecolor="black"
    #     )

    #     # include the title
    #     ax.set_title(title)

    # # Add a suptitle including the min and max unique effective dec years
    # min_eff_dec_year = model_df_copy_constant[eff_dec_years_name].min()
    # max_eff_dec_year = model_df_copy_constant[eff_dec_years_name].max()
    # plt.suptitle(
    #     f"Model raw block minima - {min_eff_dec_year} to {max_eff_dec_year}",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=1.08,
    # )

    # # Set up the axes
    # fig, axes = plt.subplots(
    #     nrows=3,
    #     ncols=4,
    #     figsize=figsize,
    #     sharex=True,
    #     sharey=True,
    #     layout="compressed",
    # )

    # # loop over the unique leads
    # for i, lead in enumerate(unique_leads):
    #     # Subset the model data to the lead this
    #     model_df_lead_this = model_df_copy_constant[model_df_copy_constant[lead_name] == lead]

    #     # print the lead
    #     print(f"Lead time: {lead}")

    #     # print the first and last unique effective dec years in this df
    #     print(f"First unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[0]}")
    #     print(f"Last unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[-1]}")

    #     # print the len of the unique effective dec years in this df
    #     print(f"Length of unique effective dec years: {len(model_df_lead_this[eff_dec_years_name].unique())}")

    #     # calculate the mean
    #     model_mean_this = model_df_lead_this[f"{model_var_name}_anomaly"].mean()

    #     # if the omodel mean is nan
    #     if np.isnan(model_mean_this):
    #         print(model_df_lead_this)

    #     # include the mean in the title
    #     title = f"Lead {lead} - Model mean: {model_mean_this:.2f}"

    #     # Plot the data
    #     ax = axes.flatten()[i]

    #     # Plot the histograms using matplotlib
    #     ax.hist(
    #         model_df_lead_this[f"{model_var_name}_anomaly"],
    #         bins=30,
    #         color="red",
    #         edgecolor="black"
    #     )

    #     # include the title
    #     ax.set_title(title)

    # # Add a suptitle including the min and max unique effective dec years
    # min_eff_dec_year = model_df_copy_constant[eff_dec_years_name].min()
    # max_eff_dec_year = model_df_copy_constant[eff_dec_years_name].max()

    # plt.suptitle(
    #     f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=1.08,
    # )

    # # do the same but not for a constant period
    # # Set up the axes
    # fig, axes = plt.subplots(
    #     nrows=3,
    #     ncols=4,
    #     figsize=figsize,
    #     sharex=True,
    #     sharey=True,
    #     layout="compressed",
    # )

    # # loop over the unique leads
    # for i, lead in enumerate(unique_leads):
    #     # Subset the model data to the lead this
    #     model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

    #     # print the lead
    #     # print(f"Lead time: {lead}")

    #     # # print the first and last unique effective dec years in this df
    #     # print(f"First unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[0]}")
    #     # print(f"Last unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[-1]}")

    #     # print the len of the unique effective dec years in this df
    #     print(f"Length of unique effective dec years: {len(model_df_lead_this[eff_dec_years_name].unique())}")

    #     # calculate the mean
    #     model_mean_this = model_df_lead_this[f"{model_var_name}_anomaly"].mean()

    #     # if the omodel mean is nan
    #     if np.isnan(model_mean_this):
    #         print(model_df_lead_this)

    #     # include the mean in the title
    #     title = f"Lead {lead} - Model mean: {model_mean_this:.2f}"

    #     # Plot the data
    #     ax = axes.flatten()[i]

    #     # Plot the histograms using matplotlib
    #     ax.hist(
    #         model_df_lead_this[f"{model_var_name}_anomaly"],
    #         bins=30,
    #         color="red",
    #         edgecolor="black"
    #     )

    #     # include the title
    #     ax.set_title(title)

    # # Add a suptitle including the min and max unique effective dec years
    # min_eff_dec_year = model_df_copy[eff_dec_years_name].min()
    # max_eff_dec_year = model_df_copy[eff_dec_years_name].max()
    # plt.suptitle(
    #     f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=1.08,
    # )

    # # Set up another figure
    # fig, axes = plt.subplots(
    #     nrows=1,
    #     ncols=2,
    #     figsize=figsize,
    #     sharex=True,
    #     sharey=True,
    #     layout="compressed",
    # )

    # ax1 = axes[0]
    # ax2 = axes[1]

    # # Get the cmap
    # cmap = cm.get_cmap("Blues", len(unique_leads))

    # # Set up an array for means
    # raw_means = np.zeros([len(unique_leads)])
    # drift_bc_means = np.zeros([len(unique_leads)])

    # # Loop over the leads
    # for i, lead in enumerate(unique_leads):
    #     # Subset the model data to the lead this
    #     model_df_lead_this = model_df_copy_constant[model_df_copy_constant[lead_name] == lead]

    #     # Plot the density distribution with kde
    #     sns.kdeplot(
    #         model_df_lead_this[f"{model_var_name}"],
    #         ax=ax1,
    #         color=cmap(i),
    #     )

    #     # Calculate the mean
    #     raw_mean_this = model_df_lead_this[f"{model_var_name}"].mean()

    #     # Append the mean to the array
    #     raw_means[i] = raw_mean_this

    #     # Plot the density distribution with kde
    #     sns.kdeplot(
    #         model_df_lead_this[f"{model_var_name}_anomaly"],
    #         ax=ax2,
    #         label=f"Lead {lead}",
    #         color=cmap(i),
    #     )

    #     # Calculate the mean
    #     drift_bc_mean_this = model_df_lead_this[f"{model_var_name}_anomaly"].mean()

    #     # Append the mean to the array
    #     drift_bc_means[i] = drift_bc_mean_this

    # # Calculate pairwise mean differences for raw means
    # raw_mean_differences = [
    #     abs(raw_means[i] - raw_means[j])
    #     for i in range(len(raw_means))
    #     for j in range(i + 1, len(raw_means))
    # ]
    # raw_mean_difference_avg = np.mean(raw_mean_differences)

    # # Calculate pairwise mean differences for drift corrected means
    # drift_bc_mean_differences = [
    #     abs(drift_bc_means[i] - drift_bc_means[j])
    #     for i in range(len(drift_bc_means))
    #     for j in range(i + 1, len(drift_bc_means))
    # ]
    # drift_bc_mean_difference_avg = np.mean(drift_bc_mean_differences)

    # # Set the titles
    # ax1.set_title(f"Raw model data (mean diff = {raw_mean_difference_avg:.2f})")
    # ax2.set_title(f"Drift corrected anomalies (mean diff = {drift_bc_mean_difference_avg:.2f})")

    # # Include a legend in the top right of the right plot
    # ax2.legend(
    #     loc="upper right",
    #     fontsize=8,
    # )

    # # include a sup title
    # min_eff_dec_year = model_df_copy_constant[eff_dec_years_name].min()
    # max_eff_dec_year = model_df_copy_constant[eff_dec_years_name].max()
    # plt.suptitle(
    #     f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=1.08,
    # )

    # # # Set up the axes
    # fig, axes = plt.subplots(
    #     nrows=1,
    #     ncols=2,
    #     figsize=figsize,
    #     sharex=True,
    #     sharey=True,
    #     layout="compressed",
    # )

    # ax1 = axes[0]
    # ax2 = axes[1]

    # # Get the cmap
    # cmap = cm.get_cmap("Blues", len(unique_leads))

    # # Set up an array for means
    # raw_means = np.zeros([len(unique_leads)])
    # drift_bc_means = np.zeros([len(unique_leads)])

    # # loop over the unique leads
    # for i, lead in enumerate(unique_leads):
    #     # Subset the model data to the lead this
    #     model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

    #     # Plot the density distribution with kde
    #     sns.kdeplot(
    #         model_df_lead_this[f"{model_var_name}"],
    #         ax=ax1,
    #         color=cmap(i),
    #     )

    #     # Calculate the mean
    #     raw_mean_this = model_df_lead_this[f"{model_var_name}"].mean()

    #     # Append the mean to the array
    #     raw_means[i] = raw_mean_this

    #     # Plot the density distribution with kde
    #     sns.kdeplot(
    #         model_df_lead_this[f"{model_var_name}_anomaly"],
    #         ax=ax2,
    #         label=f"Lead {lead}",
    #         color=cmap(i),
    #     )

    #     # Calculate the mean
    #     drift_bc_mean_this = model_df_lead_this[f"{model_var_name}_anomaly"].mean()

    #     # Append the mean to the array
    #     drift_bc_means[i] = drift_bc_mean_this

    # # Calculate pairwise mean differences for raw means
    # raw_mean_differences = [
    #     abs(raw_means[i] - raw_means[j])
    #     for i in range(len(raw_means))
    #     for j in range(i + 1, len(raw_means))
    # ]
    # raw_mean_difference_avg = np.mean(raw_mean_differences)

    # # Calculate pairwise mean differences for drift corrected means
    # drift_bc_mean_differences = [
    #     abs(drift_bc_means[i] - drift_bc_means[j])
    #     for i in range(len(drift_bc_means))
    #     for j in range(i + 1, len(drift_bc_means))
    # ]
    # drift_bc_mean_difference_avg = np.mean(drift_bc_mean_differences)

    # # Set the titles
    # ax1.set_title(f"Raw model data (mean diff = {raw_mean_difference_avg:.2f})")
    # ax2.set_title(f"Drift corrected anomalies (mean diff = {drift_bc_mean_difference_avg:.2f})")

    # # Add a suptitle including the min and max unique effective dec years
    # min_eff_dec_year = model_df_copy[eff_dec_years_name].min()
    # max_eff_dec_year = model_df_copy[eff_dec_years_name].max()
    # plt.suptitle(
    #     f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=1.08,
    # )

    # loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Extract the unique effective dec years in this case
        model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

        # Extract the unique effective dec years
        unique_eff_dec_years_this = model_df_lead_this[eff_dec_years_name].unique()

        # print the first and last unique effective dec years in this df
        print(
            f"First unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[0]}"
        )
        print(
            f"Last unique effective dec years: {model_df_lead_this[eff_dec_years_name].unique()[-1]}"
        )

        # print the len of the unique effective dec years in this df
        print(
            f"Length of unique effective dec years: {len(model_df_lead_this[eff_dec_years_name].unique())}"
        )

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
            raise ValueError(f"Obs mean is nan for lead {lead}: {obs_mean_lead_this}")

        # print the obs mean
        print(f"Obs mean lead this: {obs_mean_lead_this}")

        # Add this back into the model df anoms for mean correction
        model_df_copy.loc[
            model_df_copy[lead_name] == lead, f"{model_var_name}_drift_bc"
        ] = (
            model_df_copy.loc[
                model_df_copy[lead_name] == lead, f"{model_var_name}_anomaly"
            ]
            + obs_mean_lead_this
        )

    # # Set up the axes
    # fig, axes = plt.subplots(
    #     nrows=1,
    #     ncols=2,
    #     figsize=figsize,
    #     sharex=True,
    #     sharey=True,
    #     layout="compressed",
    # )

    # ax1 = axes[0]
    # ax2 = axes[1]

    # # Get the cmap
    # cmap = cm.get_cmap("Blues", len(unique_leads))

    # # loop over the unique leads
    # for i, lead in enumerate(unique_leads):
    #     # Subset the model data to the lead this
    #     model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

    #     # Plot the density distribution with kde
    #     sns.kdeplot(
    #         model_df_lead_this[f"{model_var_name}"],
    #         ax=ax1,
    #         color=cmap(i),
    #     )

    #     # Plot the density distribution with kde
    #     sns.kdeplot(
    #         model_df_lead_this[f"{model_var_name}_drift_bc"],
    #         ax=ax2,
    #         label=f"Lead {lead}",
    #         color=cmap(i),
    #     )

    # # plot the observed disttibution
    # sns.kdeplot(
    #     obs_df_copy[obs_var_name],
    #     ax=ax1,
    #     label="Observed",
    #     color="black",
    #     linestyle="--",
    # )

    # # plot the observed disttibution
    # sns.kdeplot(
    #     obs_df_copy[obs_var_name],
    #     ax=ax2,
    #     label="Observed",
    #     color="black",
    #     linestyle="--",
    # )

    # # set up the titles
    # ax1.set_title(f"No drift or bias corr. (no detrend)")
    # ax2.set_title(f"Drift + bias corr. (no detrend)")

    # # Include the legend in the top right of the right plot
    # ax2.legend(
    #     loc="upper right",
    #     fontsize=8,
    # )

    # # Add a suptitle including the min and max unique effective dec years
    # min_eff_dec_year = model_df_copy[eff_dec_years_name].min()
    # max_eff_dec_year = model_df_copy[eff_dec_years_name].max()

    # plt.suptitle(
    #     f"Model drift corrected anomalies - {min_eff_dec_year} to {max_eff_dec_year}",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=1.08,
    # )

    return model_df_copy


# Define the main function
def main():
    # Start the timer
    start_time = time.time()

    # set iup the delta p filepath
    # delta_p_fpath = "/home/users/benhutch/unseen_multi_year/dfs/ERA5_delta_p_1961_2024_DJF_day.csv"

    # # # Set up the test path
    # arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/"
    # test_fname = "HadGEM3-GC31-MM_vas_Europe_1960_DJF_day_20250507_124217.npy"

    # # if the path exists load the data
    # if os.path.exists(os.path.join(arrs_dir, test_fname)):
    #     # Load the data
    #     model_arr = np.load(os.path.join(arrs_dir, test_fname))

    #     # Print the shape of the model arr
    #     print(f"Shape of model arr: {model_arr.shape}")

    # sys.exit()

    # # Set up the years
    # years_test = np.arange(1960, 2018 + 1, 1)

    # # Set up the dir
    # arrs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/model/"

    # # loop over the years
    # for year in years_test:
    #     print(f"Year: {year}")
    #     # Set up the test path
    #     test_fname_uas = f"HadGEM3-GC31-MM_uas_Europe_{year}_DJF_day_*.npy"
    #     test_fname_vas = f"HadGEM3-GC31-MM_vas_Europe_{year}_DJF_day_*.npy"

    #     # Globally search for the files
    #     uas_files = glob.glob(os.path.join(arrs_dir, test_fname_uas))
    #     vas_files = glob.glob(os.path.join(arrs_dir, test_fname_vas))

    #     # assert that the len of files is 1
    #     if len(uas_files) != 1:
    #         print(f"Found {len(uas_files)} uas files for year {year}")
    #         print(f"Files: {uas_files}")

    #     if len(vas_files) != 1:
    #         print(f"Found {len(vas_files)} vas files for year {year}")
    #         print(f"Files: {vas_files}")

    # sys.exit()

    # Set up the directory in which to store the dfs
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
            test_fname = (
                f"HadGEM3-GC31-MM_dcppA-hindcast_psl_delta_p_{year}_{member}_day.csv"
            )

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
                df_delta_p_this = pd.read_csv(os.path.join(new_output_dir, test_fname))

                # concat the df to the full df
                df_delta_p_full = pd.concat([df_delta_p_full, df_delta_p_this])
            else:
                missing_fnames.append(test_fname)
                missing_fname_years.append(year)

            # Check if the uas file exists
            if os.path.exists(os.path.join(new_output_dir, test_fname_uas)):
                # Load the df
                df_uas_this = pd.read_csv(os.path.join(new_output_dir, test_fname_uas))

                # concat the df to the full df
                df_uas_full = pd.concat([df_uas_full, df_uas_this])
            else:
                missing_fnames.append(test_fname_uas)
                missing_fname_years.append(year)

            # Check if the vas file exists
            if os.path.exists(os.path.join(new_output_dir, test_fname_vas)):
                # Load the df
                df_vas_this = pd.read_csv(os.path.join(new_output_dir, test_fname_vas))

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
    df_delta_p_full["delta_p_hpa"] = (
        df_delta_p_full["data_n"] - df_delta_p_full["data_s"]
    ) / 100

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
        os.path.join(dfs_dir, "ERA5_tas_United_Kingdom_1960-2025_daily_2025-04-24.csv")
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

    # Merge the two dfs
    df_model = df_model_tas_djf.merge(
        df_model_wind,
        on=["init_year", "member", "lead", "winter_year", "effective_dec_year"],
        suffixes=("_tas", "_sfcWind"),
    )

    # merge in the delta P
    df_model = df_model.merge(
        df_delta_p_full,
        on=["init_year", "member", "lead"],
        suffixes=("", ""),
    )

    # Merge the uas data in
    df_model = df_model.merge(
        df_uas_full,
        on=["init_year", "member", "lead"],
        suffixes=("", "_uas"),
    )

    # Merge the vas data in
    df_model = df_model.merge(
        df_vas_full,
        on=["init_year", "member", "lead"],
        suffixes=("", "_vas"),
    )

    # rename data as data_uas
    df_model.rename(columns={"data": "data_uas"}, inplace=True)

    # Set up the path to the obs data
    obs_wind_path = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/ERA5_sfcWind_UK_wind_box_1960-2025_daily_2025-05-20.csv"

    # load the obs data
    df_obs_wind = pd.read_csv(obs_wind_path)

    # Convert the 'time' column to datetime, assuming it represents days since "1950-01-01 00:00:00"
    # df_obs_wind["time"] = pd.to_datetime(
    #     df_obs_wind["time"], origin="1952-01-01", unit="D"
    # )

    # Set up the start and end date to use
    start_date = pd.to_datetime("1960-01-01")
    end_date = pd.to_datetime("2025-02-28")

    # Create a date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="D")

    # Make sure time is a datetime
    df_obs_wind["time"] = date_range

    # subset the obs data to D, J, F
    df_obs_wind = df_obs_wind[df_obs_wind["time"].dt.month.isin([12, 1, 2])]

    # Set up the effective dec year column
    df_obs_wind["effective_dec_year"] = df_obs_wind.apply(
        lambda row: gev_funcs.determine_effective_dec_year(row), axis=1
    )

    # join the two obs dfs on the time column
    df_obs = df_obs_tas.merge(
        df_obs_wind,
        on=["time", "effective_dec_year"],
        suffixes=("_tas", "_sfcWind"),
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

    # drop the data column from the df obs
    df_obs.drop(columns=["data"], inplace=True)

    # rename the data_c column as data_tas_c
    df_obs.rename(columns={"data_c": "data_tas_c"}, inplace=True)

    # create a new column in the model df for data_tas_c
    df_model["data_tas_c"] = df_model["data_tas"] - 273.15

    # rename the obs_mean column in the df obs as data_sfcWind
    df_obs.rename(columns={"obs_mean": "data_sfcWind"}, inplace=True)

    # extract the unique leads in df model
    unique_leads = df_model["lead"].unique()

    # # rpint the unique leads
    # print(f"Unique leads: {unique_leads}")

    # # ptin the number of unique leads
    # print(f"Number of unique leads: {len(unique_leads)}")

    # print the len of the df obs
    print(f"Length of df obs: {len(df_obs)}")

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
    common_wyears = np.arange(1961, 2024 + 1)

    # # Set up the common w years for wind
    # common_wyears_wind = np.arange(1961, 2023 + 1)

    # Subset the model data to the common winter years
    # df_model_tas_djf = df_model_tas_djf[
    #     df_model_tas_djf["effective_dec_year"].isin(common_wyears)
    # ]

    # Do the same for df model
    df_model = df_model[df_model["effective_dec_year"].isin(common_wyears)]

    # Subset the obs data to the common winter years
    # df_obs_tas = df_obs_tas[df_obs_tas["effective_dec_year"].isin(common_wyears)]

    # Do the same for df obs
    df_obs = df_obs[df_obs["effective_dec_year"].isin(common_wyears)]

    # print the head of df model
    print(df_model.head())

    # print the tail of df model
    print(df_model.tail())

    # print the head of df obs
    print(df_obs.head())

    # print the tail of df obs
    print(df_obs.tail())

    # load in the delta p file
    if os.path.exists(delta_p_fpath):
        df_delta_p = pd.read_csv(delta_p_fpath)

    # convert df delta p time to datetime
    df_delta_p["time"] = pd.to_datetime(
        df_delta_p["time"]
    )

    # print the type of trhe time column in df delta p
    print(f"Type of time column in df_delta_p: {type(df_delta_p['time'].iloc[0])}")

    # print the tyupe of the time colum in df obs
    print(f"Type of time column in df_obs: {type(df_obs['time'].iloc[0])}")

    df_obs = df_obs.merge(
            df_delta_p,
            on=["time"],
            suffixes=("", "delta_p"),
        )
    
    # print the head of df obs
    print(df_obs.head())

    # print the tail of df obs
    print(df_obs.tail())

    # divide delta p by 100
    df_obs["delta_p_index"] = df_obs["delta_p_index"] / 100
    
    # sys.exit()

    # Subset the model wind data to the common winter years
    # df_model_wind = df_model_wind[
    #     df_model_wind["effective_dec_year"].isin(common_wyears)
    # ]

    # df_model_wind_short = df_model_wind[
    #     df_model_wind["effective_dec_year"].isin(common_wyears_wind)
    # ]

    # Subset the obs wind data to the common winter years
    # df_obs_wind = df_obs_wind[df_obs_wind["effective_dec_year"].isin(common_wyears)]

    # # do the same for df obs wind short
    # df_obs_wind_short = df_obs_wind[
    #     df_obs_wind["effective_dec_year"].isin(common_wyears_wind)
    # ]

    # print the head and tail of df model wind
    # print(df_model_wind.head())
    # print(df_model_wind.tail())

    # # print the model tas head
    # print(df_model_tas_djf.head())
    # print(df_model_tas_djf.tail())

    # # print the head and tail of df obs wind
    # print(df_obs_wind.head())
    # print(df_obs_wind.tail())

    # # print the head and tail of df obs tas
    # print(df_obs_tas.head())
    # print(df_obs_tas.tail())

    # # Create a new column for data tas c in df_model_full_djf
    # df_model_tas_djf["data_tas_c"] = df_model_tas_djf["data"] - 273.15

    # Apply the block minima transform to the obs data
    block_minima_obs_tas = gev_funcs.obs_block_min_max(
        df=df_obs,
        time_name="effective_dec_year",
        min_max_var_name="data_tas_c",
        new_df_cols=["time", "data_sfcWind", "delta_p_index"],
        process_min=True,
    )

    # # rename "obs_mean" to "data" in df_obs_wind
    # df_obs_wind.rename(columns={"obs_mean": "data"}, inplace=True)

    # Calculate the block minima for trhe obs wind speed
    block_minima_obs_wind = gev_funcs.obs_block_min_max(
        df=df_obs,
        time_name="effective_dec_year",
        min_max_var_name="data_sfcWind",
        new_df_cols=["time", "data_tas_c", "delta_p_index"],
        process_min=True,
    )

    # # do the same for obs wind short
    # df_obs_wind_short.rename(columns={"obs_mean": "data"}, inplace=True)

    # # block minima obs wind short
    # block_minima_obs_wind_short = gev_funcs.obs_block_min_max(
    #     df=df_obs_wind_short,
    #     time_name="effective_dec_year",
    #     min_max_var_name="data",
    #     new_df_cols=["time"],
    #     process_min=True,
    # )

    # # print the heaad and tail of df obs tas
    # print(df_obs_tas.head())
    # print(df_obs_tas.tail())

    # print the head of the block minima obs tas
    print(block_minima_obs_tas.head())
    print(block_minima_obs_tas.tail())

    # # Set up a fname for the dataframe
    # fname = "block_minima_obs_tas_UK_1961-2024_DJF_2_April.csv"

    # # Set up the dir to save to
    # save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    # # if the full path does not exist
    # if not os.path.exists(os.path.join(save_dir, fname)):
    #     print(f"Saving {fname} to {save_dir}")
    #     block_minima_obs_tas.to_csv(os.path.join(save_dir, fname))

    # # sys.exit()

    # # print the head of the df_model_tas_djf
    # print(df_model_tas_djf.head())
    # print(df_model_tas_djf.tail())

    # Apply the block minima transform to the model data
    block_minima_model_tas = gev_funcs.model_block_min_max(
        df=df_model,
        time_name="init_year",
        min_max_var_name="data_tas_c",
        new_df_cols=["init_year", "member", "lead", "data_sfcWind", "delta_p_hpa", "data_uas", "data_vas"],
        winter_year="winter_year",
        process_min=True,
    )

    # Apply the model block minima transform to the model wind data
    block_minima_model_wind = gev_funcs.model_block_min_max(
        df=df_model,
        time_name="init_year",
        min_max_var_name="data_sfcWind",
        new_df_cols=["init_year", "member", "lead", "data_tas_c", "delta_p_hpa", "data_uas", "data_vas"],
        winter_year="winter_year",
        process_min=True,
    )

    # --------------------------------------------------
    # Now plot the percentile plots
    # First for temperature, then for wind
    # --------------------------------------------------

    # print the columns in the df obs
    print(df_obs.columns)

    # print the columns in the df model
    print(df_model.columns)

    # sys.exit()

    # For all days, plot the percentiles of T against sfcWind
    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_obs,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_sfcWind",
        xlabel="100 - temperature percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of temperature vs 10m wind speed, all DJF days",
        y2_var_name_model="delta_p_index",
        y2_label="Delta P N-S Index (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    # find the 10th percentile of sfcWind in the obs data
    tenth_percentile_obs = df_obs["data_sfcWind"].quantile(0.10)

    # subset the df obs to values beneath the 10th percentile
    df_obs_low_wind = df_obs[df_obs["data_sfcWind"] < tenth_percentile_obs]

    # Plot the percentiles of T against sfcWind for the low wind days
    plot_multi_var_perc(
        obs_df=df_obs_low_wind,
        model_df=df_obs_low_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_sfcWind",
        xlabel="100 - temperature percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of temperature vs 10m wind speed, low wind days",
        y2_var_name_model="delta_p_index",
        y2_label="delta P N-S Index (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
        xlims=(0, 105),
        ylims=(3, 5),
    )


    # Set up the path to hazel demand data
    hazel_path = "/home/users/benhutch/NGrid_demand/csv_files/gas_electricity_demand_data.csv"

    # import the hazel data
    if os.path.exists(hazel_path):
        df_hazel = pd.read_csv(hazel_path)

    df_hazel["date"] = pd.to_datetime(df_hazel["date"])

    # Set this as the index
    df_hazel.set_index("date", inplace=True)

    # Create a copy of the df_obs
    df_obs_copy = df_obs.copy()

    # Set date as the index
    df_obs_copy.set_index("time", inplace=True)

    # find the min and max date in hazel df
    min_date_hazel = df_hazel.index.min()
    max_date_hazel = df_hazel.index.max()

    # limit the df_obs_copy to the min and max date in hazel df
    df_obs_copy = df_obs_copy[
        (df_obs_copy.index >= min_date_hazel) & (df_obs_copy.index <= max_date_hazel)
    ]

    # join the two dataframes
    df_obs_copy = df_obs_copy.join(df_hazel, how="inner")

    # print the head and tail of this df
    print(df_obs_copy.head())
    print(df_obs_copy.tail())

    # # For all days, plot the percentiles of T against sfcWind
    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="100 - temperature percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of temperature vs 10m wind speed, all DJF days",
    #     y2_var_name_model="delta_p_index",
    #     y2_label="Delta P N-S Index (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     x2_var_name_model="elec_demand_5yrRmean_nohols",
    # )

    # sys.exit()
    
    # # plot the percentiles of demand against wind speed
    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="elec_demand_5yrRmean_nohols",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="elec_demand_5yrRmean_nohols",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="demand percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of demand vs 10m wind speed, all DJF days",
    #     y2_var_name_model="delta_p_index",
    #     y2_label="Delta P N-S Index (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     ylims=(4, 12),
    #     y2_lims=(-40, 20),
    # )

    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="elec_demand_5yrRmean_nohols",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="elec_demand_5yrRmean_nohols",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="demand percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of demand vs 10m wind speed, all DJF days",
    #     y2_var_name_model="data_tas_c",
    #     y2_label="Temperature (C)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     ylims=(4, 12),
    #     y2_lims=(-5, 11),
    # )

    # # For all days, plot the percentiles of T against sfcWind
    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="100 - temperature percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of temperature vs 10m wind speed, all DJF days",
    #     y2_var_name_model="delta_p_index",
    #     y2_label="Delta P N-S Index (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    #     ylims=(4, 12),
    #     y2_lims=(-40, 20),
    # )

    # # For all days, plot the percentiles of T against sfcWind
    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="100 - temperature percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of temperature vs 10m wind speed, all DJF days",
    #     y2_var_name_model="data_tas_c",
    #     y2_label="Temperature (C)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    #     ylims=(4, 12),
    #     y2_lims=(-5, 11),
    # )

    # drop the rows which are nans in the  elec_demand_5yrRmean_nohols column
    df_obs_copy.dropna(subset=["elec_demand_5yrRmean_nohols"], inplace=True)

    # Find the 90th percentile value of demand
    ninety_percentile_demand = df_obs_copy["elec_demand_5yrRmean_nohols"].quantile(0.90)

    # subset the df obs copy to high demand
    df_obs_high_demand = df_obs_copy[
        df_obs_copy["elec_demand_5yrRmean_nohols"] >= ninety_percentile_demand
    ]

    # Finbd the 10th percentile of temperature
    tenth_percentile_temp = df_obs_copy["data_tas_c"].quantile(0.10)

    # subset the df obs copy to low temperature
    df_obs_low_temp = df_obs_copy[
        df_obs_copy["data_tas_c"] <= tenth_percentile_temp
    ]

    # print the shape of df obs high demand
    print(f"Shape of df_obs_high_demand: {df_obs_high_demand.shape}")

    # print the shape of df obs low temp
    print(f"Shape of df_obs_low_temp: {df_obs_low_temp.shape}")

    # print the head of df obs high demand
    print(df_obs_high_demand.head())
    print(df_obs_high_demand.tail())

    # print the head of df obs low temp
    print(df_obs_low_temp.head())
    print(df_obs_low_temp.tail())

    # Set up the directory in which to save theese
    save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    # Set up the fnames
    fname_high_demand = "df_obs_high_demand_2025-05-28.csv"
    fname_low_temp = "df_obs_low_temp_2025-05-28.csv"

    # if the save dir does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # if the full paths do not exist
    if not os.path.exists(os.path.join(save_dir, fname_high_demand)):
        print(f"Saving {fname_high_demand} to {save_dir}")
        df_obs_high_demand.to_csv(os.path.join(save_dir, fname_high_demand))

    if not os.path.exists(os.path.join(save_dir, fname_low_temp)):
        print(f"Saving {fname_low_temp} to {save_dir}")
        df_obs_low_temp.to_csv(os.path.join(save_dir, fname_low_temp))

    sys.exit()


    plot_multi_var_perc(
            obs_df=df_obs_copy,
            model_df=df_obs_copy,
            x_var_name_obs="data_tas_c",
            y_var_name_obs="data_sfcWind",
            x_var_name_model="data_tas_c",
            y_var_name_model="data_sfcWind",
            xlabel="100 - temperature percentiles",
            ylabel="10m Wind Speed (m/s)",
            title="Percentiles of temperature vs 10m wind speed, all DJF days",
            y2_var_name_model="delta_p_index",
            y2_label="Delta P N-S Index (hPa)",
            figsize=(5, 6),
            inverse_flag=False,
            xlims=(80, 105),
            x2_var_name_model="elec_demand_5yrRmean_nohols",
        )

    plot_multi_var_perc(
            obs_df=df_obs_copy,
            model_df=df_obs_copy,
            x_var_name_obs="data_tas_c",
            y_var_name_obs="data_sfcWind",
            x_var_name_model="data_tas_c",
            y_var_name_model="data_sfcWind",
            xlabel="100 - temperature percentiles",
            ylabel="10m Wind Speed (m/s)",
            title="Percentiles of temperature vs 10m wind speed, all DJF days",
            y2_var_name_model="data_tas_c",
            y2_label="Temperature (C)",
            figsize=(5, 6),
            inverse_flag=False,
            xlims=(80, 105),
            x2_var_name_model="elec_demand_5yrRmean_nohols",
        )
    
    sys.exit()

    # # plot the percentiles of demand against wind speed
    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="elec_demand_5yrRmean_nohols",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="elec_demand_5yrRmean_nohols",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="demand percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of demand vs 10m wind speed, all DJF days",
    #     y2_var_name_model="delta_p_index",
    #     y2_label="Delta P N-S Index (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     xlims=(75, 100),
    #     ylims=(4, 12),
    #     y2_lims=(-40, 20),
    # )

    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="elec_demand_5yrRmean_nohols",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="elec_demand_5yrRmean_nohols",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="demand percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of demand vs 10m wind speed, all DJF days",
    #     y2_var_name_model="data_tas_c",
    #     y2_label="Temperature (C)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     xlims=(75, 100),
    #     ylims=(4, 12),
    #     y2_lims=(-5, 11),
    # )

    # # For all days, plot the percentiles of T against sfcWind
    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="100 - temperature percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of temperature vs 10m wind speed, all DJF days",
    #     y2_var_name_model="delta_p_index",
    #     y2_label="Delta P N-S Index (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    #     xlims=(75, 100),
    #     ylims=(4, 12),
    #     y2_lims=(-40, 20),
    # )

    # # For all days, plot the percentiles of T against sfcWind
    # plot_multi_var_perc(
    #     obs_df=df_obs_copy,
    #     model_df=df_obs_copy,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="100 - temperature percentiles",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of temperature vs 10m wind speed, all DJF days",
    #     y2_var_name_model="data_tas_c",
    #     y2_label="Temperature (C)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    #     xlims=(75, 100),
    #     ylims=(4, 12),
    #     y2_lims=(-5, 11),
    # )

    # Find the 80th percentile value of elec_demand_5yrRmean_nohols
    eighty_percentile_demand = df_obs_copy["elec_demand_5yrRmean_nohols"].quantile(0.80)

    # Subset the df_obs_copy to values above the 80th percentile
    df_obs_high_demand = df_obs_copy[df_obs_copy["elec_demand_5yrRmean_nohols"] > eighty_percentile_demand]

    # Plot the percentiles of demand against wind speed for the high demand days
    plot_multi_var_perc(
        obs_df=df_obs_high_demand,
        model_df=df_obs_high_demand,
        x_var_name_obs="elec_demand_5yrRmean_nohols",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="elec_demand_5yrRmean_nohols",
        y_var_name_model="data_sfcWind",
        xlabel="demand percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of demand vs 10m wind speed, all DJF days",
        y2_var_name_model="delta_p_index",
        y2_label="Delta P N-S Index (hPa)",
        figsize=(5, 6),
        inverse_flag=False,
        ylims=(4, 12),
        y2_lims=(-40, 20),
    )

    plot_multi_var_perc(
        obs_df=df_obs_high_demand,
        model_df=df_obs_high_demand,
        x_var_name_obs="elec_demand_5yrRmean_nohols",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="elec_demand_5yrRmean_nohols",
        y_var_name_model="data_sfcWind",
        xlabel="demand percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of demand vs 10m wind speed, all DJF days",
        y2_var_name_model="data_tas_c",
        y2_label="Temperature (C)",
        figsize=(5, 6),
        inverse_flag=False,
        ylims=(4, 12),
        y2_lims=(-5, 11),
    )

    # Find the median value of temperature for high demand
    median_temp_high_demand = df_obs_high_demand["data_tas_c"].quantile(0.20)

    # Split the df_obs_high_demand into two parts based on the median temperature
    df_obs_high_demand_below_median = df_obs_high_demand[
        df_obs_high_demand["data_tas_c"] < median_temp_high_demand
    ]
    df_obs_high_demand_above_median = df_obs_high_demand[
        df_obs_high_demand["data_tas_c"] >= median_temp_high_demand
    ]

    # Plot the percentiles of demand against wind speed for the high demand days below median temperature
    plot_multi_var_perc(
        obs_df=df_obs_high_demand_below_median,
        model_df=df_obs_high_demand_below_median,
        x_var_name_obs="elec_demand_5yrRmean_nohols",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="elec_demand_5yrRmean_nohols",
        y_var_name_model="data_sfcWind",
        xlabel="demand percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of demand vs 10m wind speed, below median temperature",
        y2_var_name_model="delta_p_index",
        y2_label="Delta P N-S Index (hPa)",
        figsize=(5, 6),
        inverse_flag=False,
        ylims=(1, 12),
        y2_lims=(-40, 20),
    )

    plot_multi_var_perc(
        obs_df=df_obs_high_demand_below_median,
        model_df=df_obs_high_demand_below_median,
        x_var_name_obs="elec_demand_5yrRmean_nohols",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="elec_demand_5yrRmean_nohols",
        y_var_name_model="data_sfcWind",
        xlabel="demand percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of demand vs 10m wind speed, below median temperature",
        y2_var_name_model="data_tas_c",
        y2_label="Temperature (C)",
        figsize=(5, 6),
        inverse_flag=False,
        ylims=(1, 12),
        y2_lims=(-5, 11),
    )

    # sys.exit()

    # Plot the percentiles of demand against wind speed for the high demand days above median temperature
    plot_multi_var_perc(
        obs_df=df_obs_high_demand_above_median,
        model_df=df_obs_high_demand_above_median,
        x_var_name_obs="elec_demand_5yrRmean_nohols",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="elec_demand_5yrRmean_nohols",
        y_var_name_model="data_sfcWind",
        xlabel="demand percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of demand vs 10m wind speed, above median temperature",
        y2_var_name_model="delta_p_index",
        y2_label="Delta P N-S Index (hPa)",
        figsize=(5, 6),
        inverse_flag=False,
        ylims=(4, 12),
        y2_lims=(-40, 20),
    )

    plot_multi_var_perc(
        obs_df=df_obs_high_demand_above_median,
        model_df=df_obs_high_demand_above_median,
        x_var_name_obs="elec_demand_5yrRmean_nohols",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="elec_demand_5yrRmean_nohols",
        y_var_name_model="data_sfcWind",
        xlabel="demand percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of demand vs 10m wind speed, above median temperature",
        y2_var_name_model="data_tas_c",
        y2_label="Temperature (C)",
        figsize=(5, 6),
        inverse_flag=False,
        ylims=(4, 12),
        y2_lims=(-5, 11),
    )

    sys.exit()

    # # Plot percentiles of sfcWind against T
    # plot_multi_var_perc(
    #     obs_df=df_obs,
    #     model_df=df_model,
    #     x_var_name_obs="data_sfcWind",
    #     y_var_name_obs="data_tas_c",
    #     x_var_name_model="data_sfcWind",
    #     y_var_name_model="data_tas_c",
    #     xlabel="100 - wind speed percentiles",
    #     ylabel="Temperature (C)",
    #     title="Percentiles of wind speed vs temperature, all DJF days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    # )

    # plot_multi_var_perc(
    #     obs_df=df_obs,
    #     model_df=df_model,
    #     x_var_name_obs="data_sfcWind",
    #     y_var_name_obs="data_tas_c",
    #     x_var_name_model="data_sfcWind",
    #     y_var_name_model="data_tas_c",
    #     xlabel="100 - wind speed percentiles",
    #     ylabel="Temperature (C)",
    #     title="Percentiles of wind speed vs temperature, all DJF days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     xlims=(-5, 20),
    #     ylims=(1.8, 2.8),
    # )

    # plot_multi_var_perc(
    #     obs_df=df_obs,
    #     model_df=df_model,
    #     x_var_name_obs="data_sfcWind",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_sfcWind",
    #     y_var_name_model="data_uas",
    #     xlabel="Wind speed percentiles",
    #     ylabel="U/V-component of Wind at 10m (m/s)",
    #     title="Percentiles of temperature vs U10m wind speed, all DJF days",
    #     y_var_name_model_2="data_vas",
    #     ylabel_2="V10m (m/s)",
    #     y2_var_name_model="data_sfcWind",
    #     y2_label="10m Wind Speed (m/s)",
    #     figsize=(5, 6),
    #     inverse_flag=False,
    #     y1_zero_line=True,
    # )

    # sys.exit()

    # # For all days, sense check by plotting the uas against delta P
    # plot_multi_var_perc(
    #     obs_df=df_obs,
    #     model_df=df_model,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_uas",
    #     xlabel="100 - temperature percentiles",
    #     ylabel="U-component of Wind at 10m (m/s)",
    #     title="Percentiles of temperature vs U10m wind speed, all DJF days",
    #     y_var_name_model_2="data_vas",
    #     ylabel_2="V10m (m/s)",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    #     y1_zero_line=True,
    # )

    # Find the median wind speed
    # median_wind_speed = df_model["data_sfcWind"].median()

    # Find the 10th percentile value of data_sfcWind
    tenth_percentile = df_model["data_sfcWind"].quantile(0.1)

    central_lower = df_model["data_sfcWind"].quantile(0.40)
    central_upper = df_model["data_sfcWind"].quantile(0.60)

    # # subset the ensemble to force the wind speed to be less than 5 m/s
    df_model_low_wind = df_model[df_model["data_sfcWind"] < tenth_percentile]
    df_model_higher_wind = df_model[
        (df_model["data_sfcWind"] > central_lower) & (df_model["data_sfcWind"] < central_upper)
    ]

    # Set up the directory which we save to
    save_dir_dfs_transfer = "/home/users/benhutch/unseen_multi_year/dfs"

    # Set up the fname
    fname_low_wind = "model_all_DJF_days_lowest_0-10_percentile_wind_speed.csv"
    fname_higher_wind = "model_all_DJF_days_40-60_percentile_wind_speed.csv"

    # Form the full paths
    full_path_low_wind = os.path.join(save_dir_dfs_transfer, fname_low_wind)
    full_path_higher_wind = os.path.join(save_dir_dfs_transfer, fname_higher_wind)

    # If full path low wind does not exist, save the df
    if not os.path.exists(full_path_low_wind):
        print(f"Saving {fname_low_wind} to {save_dir_dfs_transfer}")
        df_model_low_wind.to_csv(full_path_low_wind)

    # If full path higher wind does not exist, save the df
    if not os.path.exists(full_path_higher_wind):
        print(f"Saving {fname_higher_wind} to {save_dir_dfs_transfer}")
        df_model_higher_wind.to_csv(full_path_higher_wind)

    sys.exit()

    # # find at which percentile/qunatile the value of 5 m/s is in the data
    # # print the quantile of the wind speed
    # value = 5

    # percentile = percentileofscore(
    #     df_model["data_sfcWind"],
    #     value,
    #     kind="rank",
    # )

    # # print the percentile
    # print(f"Percentile of {value} m/s: {percentile:.2f}%")

    # sys.exit()

    # For all days, plot the percentiles of T against sfcWind
    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_low_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_sfcWind",
        xlabel="100 - temperature percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of temperature vs 10m wind speed, low wind speed days",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    # Do the same, but for U and V, and then wind speed on the second y-axis
    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_low_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_uas",
        xlabel="100 - temperature percentiles",
        ylabel="U-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U/V 10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_sfcWind",
        y2_label="10m Wind Speed (m/s)",
        figsize=(5, 6),
        inverse_flag=True,
        y1_zero_line=True,
    )

    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_low_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_uas",
        xlabel="100 - temperature percentiles",
        ylabel="U-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U/V 10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_sfcWind",
        y2_label="10m Wind Speed (m/s)",
        figsize=(5, 6),
        inverse_flag=True,
        y1_zero_line=True,
        xlims=(80, 100),
    )

    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_low_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_uas",
        xlabel="100 - temperature percentiles",
        ylabel="U-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U/V 10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
        y1_zero_line=True,
        xlims=(80, 100),
    )

    # do the same but with temperature on the oteher y-axis
    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_low_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_uas",
        xlabel="100 - temperature percentiles",
        ylabel="U-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U/V 10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_tas_c",
        y2_label="Temperature (C)",
        figsize=(5, 6),
        inverse_flag=True,
        y1_zero_line=True,
        xlims=(80, 100),
    )


    # For all days, plot the percentiles of T against sfcWind
    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_higher_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_sfcWind",
        xlabel="100 - temperature percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of temperature vs 10m wind speed, higher wind speed days",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
    )

    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_higher_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_uas",
        xlabel="100 - temperature percentiles",
        ylabel="U-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U/V 10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_sfcWind",
        y2_label="10m Wind Speed (m/s)",
        figsize=(5, 6),
        inverse_flag=True,
        y1_zero_line=True,
    )

    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_higher_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_uas",
        xlabel="100 - temperature percentiles",
        ylabel="U-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U/V 10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_sfcWind",
        y2_label="10m Wind Speed (m/s)",
        figsize=(5, 6),
        inverse_flag=True,
        y1_zero_line=True,
        xlims=(80, 100),
    )

    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_higher_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_uas",
        xlabel="100 - temperature percentiles",
        ylabel="U-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U/V 10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=True,
        y1_zero_line=True,
        xlims=(80, 100),
    )

    # Do the same but with temperature on the other y-axis
    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model_higher_wind,
        x_var_name_obs="data_tas_c",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_tas_c",
        y_var_name_model="data_uas",
        xlabel="100 - temperature percentiles",
        ylabel="U-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U/V 10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_tas_c",
        y2_label="Temperature (C)",
        figsize=(5, 6),
        inverse_flag=True,
        y1_zero_line=True,
        xlims=(80, 100),
    )

    sys.exit()

    # Sense check by plotting temp percentiles against temperature
    # plot_multi_var_perc(
    #     obs_df=df_obs,
    #     model_df=df_model,
    #     x_var_name_obs="data_tas_c",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c",
    #     y_var_name_model="data_tas_c",
    #     xlabel="100 - temperature percentiles",
    #     ylabel="Temperature (C)",
    #     title="Percentiles of temperature vs temperature, all DJF days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    #     y1_zero_line=True,
    # )

    # Sense check by plotting wind percentiles against wind speed
    plot_multi_var_perc(
        obs_df=df_obs,
        model_df=df_model,
        x_var_name_obs="data_sfcWind",
        y_var_name_obs="data_sfcWind",
        x_var_name_model="data_sfcWind",
        y_var_name_model="data_sfcWind",
        xlabel="100 - wind speed percentiles",
        ylabel="10m Wind Speed (m/s)",
        title="Percentiles of wind speed vs wind speed, all DJF days",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=False,
    )

    # Do the same, but for block minima T days
    # plot_multi_var_perc(
    #     obs_df=block_minima_obs_tas,
    #     model_df=block_minima_model_tas,
    #     x_var_name_obs="data_tas_c_min",
    #     y_var_name_obs="data_sfcWind",
    #     x_var_name_model="data_tas_c_min",
    #     y_var_name_model="data_sfcWind",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of temperature vs 10m wind speed, block min T days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )

    # Do the same but for block minima wind days
    plot_multi_var_perc(
        obs_df=block_minima_obs_wind,
        model_df=block_minima_model_wind,
        x_var_name_obs="data_sfcWind_min",
        y_var_name_obs="data_tas_c",
        x_var_name_model="data_sfcWind_min",
        y_var_name_model="data_tas_c",
        xlabel="10m Wind Speed (m/s)",
        ylabel="Temperature",
        title="Percentiles of 10m wind speed vs temperature, block min wind days",
        y2_var_name_model="delta_p_hpa",
        y2_label="delta P N-S (hPa)",
        figsize=(5, 6),
        inverse_flag=False,
    )

    plot_multi_var_perc(
        obs_df=block_minima_obs_wind,
        model_df=block_minima_model_wind,
        x_var_name_obs="data_sfcWind_min",
        y_var_name_obs="data_sfcWind_min",
        x_var_name_model="data_sfcWind_min",
        y_var_name_model="data_uas",
        xlabel="Wind speed percentiles",
        ylabel="U/V-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_sfcWind_min",
        y2_label="10m Wind Speed (m/s)",
        figsize=(5, 6),
        inverse_flag=False,
        y1_zero_line=True,
    )

    plot_multi_var_perc(
        obs_df=block_minima_obs_wind,
        model_df=block_minima_model_wind,
        x_var_name_obs="data_sfcWind_min",
        y_var_name_obs="data_sfcWind_min",
        x_var_name_model="data_sfcWind_min",
        y_var_name_model="data_uas",
        xlabel="Wind speed percentiles",
        ylabel="U/V-component of Wind at 10m (m/s)",
        title="Percentiles of temperature vs U10m wind speed, all DJF days",
        y_var_name_model_2="data_vas",
        ylabel_2="V10m (m/s)",
        y2_var_name_model="data_tas_c",
        y2_label="Temperature (C)",
        figsize=(5, 6),
        inverse_flag=False,
        y1_zero_line=True,
    )

    sys.exit()

    # obs_df=block_max_obs_dnw,
    #     model_df=block_max_model_dnw,
    #     x_var_name_obs="data_c_dt",
    #     y_var_name_obs="data_sfcWind_dt",
    #     x_var_name_model="data_tas_c_drift_bc_dt",
    #     y_var_name_model="data_sfcWind_drift_bc_dt",
    #     xlabel="Temperature",
    #     ylabel="10m Wind Speed (m/s)",
    #     title="Percentiles of (inverted) temperature vs 10m wind speed, DnW days",
    #     y2_var_name_model="delta_p_hpa",
    #     y2_label="delta P N-S (hPa)",
    #     figsize=(5, 6),
    #     inverse_flag=True,
    # )


    # # Set up the block minima model wind short
    # block_minima_model_wind_short = gev_funcs.model_block_min_max(
    #     df=df_model_wind_short,
    #     time_name="init_year",
    #     min_max_var_name="data",
    #     new_df_cols=["init_year", "member", "lead"],
    #     winter_year="winter_year",
    #     process_min=True,
    # )

    # # print the head of the block minima model tas
    # print(block_minima_model_tas.head())
    # print(block_minima_model_tas.tail())

    # # print the unique init years in the model df
    # print(block_minima_model_tas["init_year"].unique())

    # Set up a fname for the dataframe
    fname = "block_minima_model_tas_UK_1961-2024_DJF.csv"

    # Set up the dir to save to
    save_dir = "/home/users/benhutch/unseen_multi_year/dfs"

    # if the full path does not exist
    if not os.path.exists(os.path.join(save_dir, fname)):
        print(f"Saving {fname} to {save_dir}")
        block_minima_model_tas.to_csv(os.path.join(save_dir, fname))

    # sys.exit()

    # # # Ensure effective dec year is in the block minima model tas
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

    # Compare the trends
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

    # print the columns in the block minima model tas
    print(block_minima_model_tas.columns)

    # print the columns in the block minima model wind
    print(block_minima_model_wind.columns)

    # Check for duplicate column names in block_minima_model_tas
    if block_minima_model_tas.columns.duplicated().any():
        print("Duplicate column names in block minima model tas")
        print(
            block_minima_model_tas.columns[block_minima_model_tas.columns.duplicated()]
        )

        # Drop the duplicate columns
        block_minima_model_tas = block_minima_model_tas.loc[
            :, ~block_minima_model_tas.columns.duplicated()
        ]

    # Check for duplicate column names in block_minima_model_wind
    if block_minima_model_wind.columns.duplicated().any():
        print("Duplicate column names in block minima model wind")
        print(
            block_minima_model_wind.columns[
                block_minima_model_wind.columns.duplicated()
            ]
        )

        # Drop the duplicate columns
        block_minima_model_wind = block_minima_model_wind.loc[
            :, ~block_minima_model_wind.columns.duplicated()
        ]

    # if anything is duplicate in model block wind short
    if block_minima_model_wind_short.columns.duplicated().any():
        print("Duplicate column names in block minima model wind short")
        print(
            block_minima_model_wind_short.columns[
                block_minima_model_wind_short.columns.duplicated()
            ]
        )

        # Drop the duplicate columns
        block_minima_model_wind_short = block_minima_model_wind_short.loc[
            :, ~block_minima_model_wind_short.columns.duplicated()
        ]

    # add the effective dec year to the block minima model tas
    block_minima_model_tas["effective_dec_year"] = block_minima_model_tas[
        "init_year"
    ] + (block_minima_model_tas["winter_year"] - 1)

    # do the same for wind speed
    block_minima_model_wind["effective_dec_year"] = block_minima_model_wind[
        "init_year"
    ] + (block_minima_model_wind["winter_year"] - 1)

    # do the same for wind speed short
    block_minima_model_wind_short["effective_dec_year"] = block_minima_model_wind_short[
        "init_year"
    ] + (block_minima_model_wind_short["winter_year"] - 1)

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

    # print the head of the block minima model tas drift corr
    block_minima_model_wind_short_drift_corr = model_drift_corr_plot(
        model_df=block_minima_model_wind_short,
        model_var_name="data_min",
        obs_df=block_minima_obs_wind_short,
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

    # plot the lead pdfs for model wind drift corr short
    gev_funcs.plot_lead_pdfs(
        model_df=block_minima_model_wind_short_drift_corr,
        obs_df=block_minima_obs_wind_short,
        model_var_name="data_min_drift_bc",
        obs_var_name="data_min",
        lead_name="winter_year",
        xlabel="Wind speed (m/s)",
        suptitle="Wind speed PDFs, 1961-2017, DJF block min T (model drift corrected)",
        figsize=(10, 5),
    )

    # sys.exit()

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

    # Test the new function
    pivot_emp_rps(
        obs_df=block_minima_obs_tas,
        model_df=block_minima_model_tas_drift_corr,
        obs_val_name="data_c_min",
        model_val_name="data_tas_c_min_drift_bc",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        var_name="tas",
        nsamples=1000,
        figsize=(5, 5),
    )

    # DO the same for wind speed
    pivot_emp_rps(
        obs_df=block_minima_obs_wind,
        model_df=block_minima_model_wind_drift_corr,
        obs_val_name="data_min",
        model_val_name="data_min_drift_bc",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        var_name="sfcWind",
        nsamples=1000,
        figsize=(5, 5),
        wind_2005_toggle=True,
    )

    # Do the same for wind speed short
    # pivot_emp_rps(
    #     obs_df=block_minima_obs_wind_short,
    #     model_df=block_minima_model_wind_short_drift_corr,
    #     obs_val_name="data_min",
    #     model_val_name="data_min_drift_bc",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     var_name="sfcWind",
    #     nsamples=100,
    #     figsize=(5, 5),
    #     wind_2005_toggle=True,
    # )

    # sys.exit()

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

    # Apply a detrend to thje wind data short
    block_minima_model_wind_short_drift_corr_dt = gev_funcs.pivot_detrend_model(
        model_df=block_minima_model_wind_short_drift_corr,
        obs_df=block_minima_obs_wind_short,
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
    print(
        block_minima_model_tas_drift_corr_dt["data_tas_c_min_drift_bc_dt"].isna().sum()
    )

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

    # pivot detrend the obs data for wind speed short
    block_minima_obs_wind_short_dt = gev_funcs.pivot_detrend_obs(
        df=block_minima_obs_wind_short,
        x_axis_name="effective_dec_year",
        y_axis_name="data_min",
    )

    # Set up a name for the block min obs
    fname_tas = "block_minima_obs_tas_UK_1961-2024_DJF_detrended.csv"
    fname_wind = "block_minima_obs_wind_UK_1961-2024_DJF_detrended.csv"

    # Do the same for the model data
    fname_tas_model = "block_minima_model_tas_UK_1961-2024_DJF_detrended.csv"
    fname_wind_model = "block_minima_model_wind_UK_1961-2024_DJF_detrended.csv"

    # Set up the directory to save to
    save_dir_dfs = "/home/users/benhutch/unseen_multi_year/dfs"

    # Set up thne current date in dd-mm-yyyy format
    current_date = datetime.now().strftime("%d-%m-%Y")

    # Set up the full path for the obs
    obs_path_tas = os.path.join(save_dir_dfs, f"{fname_tas}_{current_date}")
    obs_path_wind = os.path.join(save_dir_dfs, f"{fname_wind}_{current_date}")
    model_path_tas = os.path.join(save_dir_dfs, f"{fname_tas_model}_{current_date}")
    model_path_wind = os.path.join(save_dir_dfs, f"{fname_wind_model}_{current_date}")

    # # if the file does not exist
    # if not os.path.exists(obs_path_tas):
    #     print(f"Saving {fname_tas} to {save_dir_dfs}")
    #     block_minima_obs_tas_dt.to_csv(obs_path_tas)

    # # if the file does not exist
    # if not os.path.exists(obs_path_wind):
    #     print(f"Saving {fname_wind} to {save_dir_dfs}")
    #     block_minima_obs_wind_dt.to_csv(obs_path_wind)

    # # if the file does not exist
    # if not os.path.exists(model_path_tas):
    #     print(f"Saving {fname_tas_model} to {save_dir_dfs}")
    #     block_minima_model_tas_drift_corr_dt.to_csv(model_path_tas)

    # # if the file does not exist
    # if not os.path.exists(model_path_wind):
    #     print(f"Saving {fname_wind_model} to {save_dir_dfs}")
    #     block_minima_model_wind_drift_corr_dt.to_csv(model_path_wind)

    # # Set up the paths to the full field model data
    # model_full_field_path = os.path.join(
    #     save_dir_dfs, "full_field_model_tas_wind_UK_1961-2024_DJF_detrended_07-05-2025.csv"
    # )
    # obs_full_field_path = os.path.join(
    #     save_dir_dfs, "full_field_obs_tas_wind_UK_1961-2024_DJF_detrended_07-05-2025.csv"
    # )

    # # Load these dataframes
    # df_model_full_field = pd.read_csv(model_full_field_path)
    # df_obs_full_field = pd.read_csv(obs_full_field_path)

    # plot_distributions_extremes(
    #     model_df_full_field=df_model_full_field,
    #     obs_df_full_field=df_obs_full_field,
    #     model_df_block=block_minima_model_tas_drift_corr_dt,
    #     obs_df_block=block_minima_obs_tas_dt,
    #     model_var_name_full_field="data_tas_c_drift_bc_dt",
    #     obs_var_name_full_field="data_c_dt",
    #     model_var_name_block="data_tas_c_min_drift_bc_dt",
    #     obs_var_name_block="data_c_min_dt",
    #     xlabels=["Temperature (°C)", "Temperature (°C)"],
    #     percentile=0.05,
    # )

    # # DO the same for wind speed
    # plot_distributions_extremes(
    #     model_df_full_field=df_model_full_field,
    #     obs_df_full_field=df_obs_full_field,
    #     model_df_block=block_minima_model_wind_drift_corr_dt,
    #     obs_df_block=block_minima_obs_wind_dt,
    #     model_var_name_full_field="data_sfcWind_drift_bc_dt",
    #     obs_var_name_full_field="data_sfcWind_dt",
    #     model_var_name_block="data_min_drift_bc_dt",
    #     obs_var_name_block="data_min_dt",
    #     xlabels=["10m Wind Speed (m/s)", "10m Wind Speed (m/s)"],
    #     percentile=0.05,
    # )

    # # print the len of block minima model tas drift corr dt
    # print(f"len of block minima model tas drift corr dt pre subset: {len(block_minima_model_tas_drift_corr_dt)}")

    # # print the len of block minima obs tas dt pre subset
    # print(f"len of block minima obs tas dt pre subset: {len(block_minima_obs_tas_dt)}")

    # # Test the new function for trimming the distribution
    # # Subset the extremes for temperature
    # block_min_model_tas_extremes, block_min_obs_tas_extremes = subset_extremes(
    #     model_df_full_field=df_model_full_field,
    #     obs_df_full_field=df_obs_full_field,
    #     model_df_block=block_minima_model_tas_drift_corr_dt,
    #     obs_df_block=block_minima_obs_tas_dt,
    #     model_var_name_full_field="data_tas_c_drift_bc_dt",
    #     obs_var_name_full_field="data_c_dt",
    #     model_var_name_block="data_tas_c_min_drift_bc_dt",
    #     obs_var_name_block="data_c_min_dt",
    #     percentile=0.05,
    # )

    # # print the len of block minima model tas drift corr dt
    # print(f"len of block minima model tas drift corr dt post subset: {len(block_min_model_tas_extremes)}")

    # # print the len of block minima obs tas dt post subset
    # print(f"len of block minima obs tas dt post subset: {len(block_min_obs_tas_extremes)}")

    # # print the len of block minima model wind drift corr dt
    # print(f"len of block minima model wind drift corr dt pre subset: {len(block_minima_model_wind_drift_corr_dt)}")

    # # print the len of block minima obs wind dt pre subset
    # print(f"len of block minima obs wind dt pre subset: {len(block_minima_obs_wind_dt)}")

    # # Do the same but for wind speed
    # block_min_model_wind_extremes, block_min_obs_wind_extremes = subset_extremes(
    #     model_df_full_field=df_model_full_field,
    #     obs_df_full_field=df_obs_full_field,
    #     model_df_block=block_minima_model_wind_drift_corr_dt,
    #     obs_df_block=block_minima_obs_wind_dt,
    #     model_var_name_full_field="data_sfcWind_drift_bc_dt",
    #     obs_var_name_full_field="data_sfcWind_dt",
    #     model_var_name_block="data_min_drift_bc_dt",
    #     obs_var_name_block="data_min_dt",
    #     percentile=0.05,
    # )

    # # print the len of block minima model wind drift corr dt
    # print(f"len of block minima model wind drift corr dt post subset: {len(block_min_model_wind_extremes)}")

    # # print the len of block minima obs wind dt post subset
    # print(f"len of block minima obs wind dt post subset: {len(block_min_obs_wind_extremes)}")

    # sys.exit()

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

    # # Compare the trends with the full field data
    # gev_funcs.compare_trends(
    #     model_df_full_field=df_model_tas_djf,
    #     obs_df_full_field=df_obs_tas,
    #     model_df_block=block_minima_model_tas_drift_corr_dt,
    #     obs_df_block=block_minima_obs_tas_dt,
    #     model_var_name_full_field="data_tas_c",
    #     obs_var_name_full_field="data_c",
    #     model_var_name_block="data_tas_c_min_drift_bc",
    #     obs_var_name_block="data_c_min",
    #     model_time_name="effective_dec_year",
    #     obs_time_name="effective_dec_year",
    #     ylabel="Temperature (C)",
    #     suptitle="Temperature trends (block min detrended obs, model lead time detrended)",
    #     figsize=(15, 5),
    #     window_size=10,
    #     centred_bool=True,
    #     min_periods=1,
    # )

    # # # compare trends for the wind data
    # gev_funcs.compare_trends(
    #     model_df_full_field=df_model_wind,
    #     obs_df_full_field=df_obs_wind,
    #     model_df_block=block_minima_model_wind_drift_corr_dt,
    #     obs_df_block=block_minima_obs_wind_dt,
    #     model_var_name_full_field="data",
    #     obs_var_name_full_field="data",
    #     model_var_name_block="data_min_drift_bc",
    #     obs_var_name_block="data_min",
    #     model_time_name="effective_dec_year",
    #     obs_time_name="effective_dec_year",
    #     ylabel="Wind speed (m/s)",
    #     suptitle="Wind speed trends (block min detrended obs, model lead time detrended)",
    #     figsize=(15, 5),
    #     window_size=10,
    #     centred_bool=True,
    #     min_periods=1,
    # )

    # sys.exit()

    # # Now plot the lead time dependent biases for the trend corrected data
    # gev_funcs.plot_lead_pdfs(
    #     model_df=block_minima_model_tas_drift_corr_dt,
    #     obs_df=block_minima_obs_tas_dt,
    #     model_var_name="data_tas_c_min_drift_bc_dt",
    #     obs_var_name="data_c_min_dt",
    #     lead_name="winter_year",
    #     xlabel="Temperature (C)",
    #     suptitle="Temperature PDFs, 1961-2017, DJF block min T (model drift + trend corrected)",
    #     figsize=(10, 5),
    # )

    # # Plot the lead pdfs for wind speed post drift/bc
    # gev_funcs.plot_lead_pdfs(
    #     model_df=block_minima_model_wind_drift_corr_dt,
    #     obs_df=block_minima_obs_wind_dt,
    #     model_var_name="data_min_drift_bc_dt",
    #     obs_var_name="data_min_dt",
    #     lead_name="winter_year",
    #     xlabel="Wind speed (m/s)",
    #     suptitle="Wind speed PDFs, 1961-2017, DJF block min T (model drift + trend corrected)",
    #     figsize=(10, 5),
    # )

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

    # # plot the plots
    # gev_funcs.plot_detrend_ts_subplots(
    #     obs_df_left=block_minima_obs_tas_dt,
    #     model_df_left=block_minima_model_tas_drift_corr_dt,
    #     obs_df_right=block_minima_obs_wind_dt,
    #     model_df_right=block_minima_model_wind_drift_corr_dt,
    #     obs_var_name_left="data_c_min",
    #     model_var_name_left="data_tas_c_min_drift_bc",
    #     obs_var_name_right="data_min",
    #     model_var_name_right="data_min_drift_bc",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     ylabel_left="Temperature (C)",
    #     ylabel_right="Wind speed (m/s)",
    #     detrend_suffix_left="_dt",
    #     detrend_suffix_right="_dt",
    # )

    # sys.exit()

    # # Set up the names of the new dataframes
    # block_minima_model_tas_drift_corr_dt = block_min_model_tas_extremes
    # block_minima_model_wind_drift_corr_dt = block_min_model_wind_extremes
    # block_minima_obs_tas_dt = block_min_obs_tas_extremes
    # block_minima_obs_wind_dt = block_min_obs_wind_extremes

    # Make sure effective dec year is a datetime in the model tas data
    block_minima_model_tas_drift_corr_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_model_tas_drift_corr_dt["effective_dec_year"], format="%Y"
    )

    # Make sure effective dec year is a datetime in the model wind data
    block_minima_model_wind_drift_corr_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_model_wind_drift_corr_dt["effective_dec_year"], format="%Y"
    )

    # Make sure effective dec year is a datetime in the model wind data short
    block_minima_model_wind_short_drift_corr_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_model_wind_short_drift_corr_dt["effective_dec_year"], format="%Y"
    )

    # Make sure effective dec year is a datetime in the obs tas data
    block_minima_obs_tas_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_tas_dt["effective_dec_year"], format="%Y"
    )

    # Make sure effective dec year is a datetime in the obs wind data
    block_minima_obs_wind_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_wind_dt["effective_dec_year"], format="%Y"
    )

    # Make sure effective dec year is a datetime in the obs wind data short
    block_minima_obs_wind_short_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_wind_short_dt["effective_dec_year"], format="%Y"
    )

    # Set this as the index in the obs tas data
    block_minima_obs_tas_dt.set_index("effective_dec_year", inplace=True)

    # Set this as the index in the obs wind data
    block_minima_obs_wind_dt.set_index("effective_dec_year", inplace=True)

    # Set this as the index in the obs wind data short
    block_minima_obs_wind_short_dt.set_index("effective_dec_year", inplace=True)

    # print the head and tail ofg block minima obs wind dt
    print(block_minima_obs_wind_dt.head())
    print(block_minima_obs_wind_dt.tail())

    # # Now test plotting the dot plots for temp and wind speed
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
        title_right="Block minima wind speed 1960-2023 (m/s)",
        ylims_left=(-12, 8),
        ylims_right=(0, 8),
        dashed_quant=0.20,
        solid_line=np.min,
        figsize=(10, 5),
    )

    # # Dot plot subplots for tas and wind speed short
    # gev_funcs.dot_plot_subplots(
    #     obs_df_left=block_minima_obs_tas_dt,
    #     model_df_left=block_minima_model_tas_drift_corr_dt,
    #     obs_df_right=block_minima_obs_wind_short_dt,
    #     model_df_right=block_minima_model_wind_short_drift_corr_dt,
    #     obs_val_name_left="data_c_min_dt",
    #     model_val_name_left="data_tas_c_min_drift_bc_dt",
    #     obs_val_name_right="data_min_dt",
    #     model_val_name_right="data_min_drift_bc_dt",
    #     model_time_name="effective_dec_year",
    #     ylabel_left="Temperature (°C)",
    #     ylabel_right="Wind speed (m/s)",
    #     title_left="Block minima temperature (°C)",
    #     title_right="Block minima wind speed 1960-2023 (m/s)",
    #     ylims_left=(-12, 8),
    #     ylims_right=(0, 8),
    #     dashed_quant=0.20,
    #     solid_line=np.min,
    #     figsize=(10, 5),
    # )

    # sys.exit()

    # reset the index of the obs data
    block_minima_obs_tas_dt.reset_index(inplace=True)
    # reset the index of the obs data
    block_minima_obs_wind_dt.reset_index(inplace=True)
    # reset the index of the obs data
    block_minima_obs_wind_short_dt.reset_index(inplace=True)

    # test the return period extremes function
    # For temperature extremes first
    # plot_rp_extremes(
    #     obs_df=block_minima_obs_tas_dt,
    #     model_df=block_minima_model_tas_drift_corr_dt,
    #     obs_val_name="data_c_min_dt",
    #     model_val_name="data_tas_c_min_drift_bc_dt",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     ylim=(-9, -2.5),
    #     percentile=0.01,
    #     n_samples=10000,
    #     high_values_rare=False,
    # )

    # # test the new function doing the same
    # # thing but using GEVs
    # plot_gev_rps(
    #     obs_df=block_minima_obs_tas_dt,
    #     model_df=block_minima_model_tas_drift_corr_dt,
    #     obs_val_name="data_c_min_dt",
    #     model_val_name="data_tas_c_min_drift_bc_dt",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     ylabel="Temperature (°C)",
    #     nsamples=1000,
    #     ylims=(-9, -2.5),
    #     blue_line=np.min,
    #     high_values_rare=False,
    #     figsize=(5, 5),
    # )

    # test the funcion for doing the same thing
    plot_emp_rps(
        obs_df=block_minima_obs_tas_dt,
        model_df=block_minima_model_tas_drift_corr_dt,
        obs_val_name="data_c_min_dt",
        model_val_name="data_tas_c_min_drift_bc_dt",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        ylabel="Temperature (°C)",
        nsamples=1000,
        ylims=(-9, -2.5),
        blue_line=np.min,
        high_values_rare=False,
        figsize=(5, 5),
    )

    # # do thye same thing fo wind speed
    # plot_gev_rps(
    #     obs_df=block_minima_obs_wind_dt,
    #     model_df=block_minima_model_wind_drift_corr_dt,
    #     obs_val_name="data_min_dt",
    #     model_val_name="data_min_drift_bc_dt",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     ylabel="Wind speed (m/s)",
    #     nsamples=1000,
    #     ylims=(2, 3.5),
    #     blue_line=np.min,
    #     high_values_rare=False,
    #     figsize=(5, 5),
    # )

    # plot empirical return periods for wind speed
    plot_emp_rps(
        obs_df=block_minima_obs_wind_dt,
        model_df=block_minima_model_wind_drift_corr_dt,
        obs_val_name="data_min_dt",
        model_val_name="data_min_drift_bc_dt",
        obs_time_name="effective_dec_year",
        model_time_name="effective_dec_year",
        ylabel="Wind speed (m/s)",
        nsamples=1000,
        ylims=(2, 3.5),
        blue_line=np.min,
        high_values_rare=False,
        figsize=(5, 5),
    )

    # # # plot the empirical return periods for wind speed short
    # plot_emp_rps(
    #     obs_df=block_minima_obs_wind_short_dt,
    #     model_df=block_minima_model_wind_short_drift_corr_dt,
    #     obs_val_name="data_min_dt",
    #     model_val_name="data_min_drift_bc_dt",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     ylabel="Wind speed (m/s)",
    #     nsamples=1000,
    #     ylims=(2, 3.5),
    #     blue_line=np.min,
    #     high_values_rare=False,
    #     figsize=(5, 5),
    #     wind_2005_toggle=False,
    # )

    # make sure that effective dec year is a datetime year
    block_minima_model_tas_drift_corr_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_model_tas_drift_corr_dt["effective_dec_year"], format="%Y"
    )

    # format effective dec year as an int
    block_minima_model_tas_drift_corr_dt["effective_dec_year"] = (
        block_minima_model_tas_drift_corr_dt["effective_dec_year"].dt.year.astype(int)
    )

    # do the same conversion for wind speed
    block_minima_model_wind_drift_corr_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_model_wind_drift_corr_dt["effective_dec_year"], format="%Y"
    )

    # format effective dec year as an int
    block_minima_model_wind_drift_corr_dt["effective_dec_year"] = (
        block_minima_model_wind_drift_corr_dt["effective_dec_year"].dt.year.astype(int)
    )

    # Make sure that effective dec year is a datetime year
    block_minima_model_wind_short_drift_corr_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_model_wind_short_drift_corr_dt["effective_dec_year"], format="%Y"
    )

    # format effective dec year as an int
    block_minima_model_wind_short_drift_corr_dt["effective_dec_year"] = (
        block_minima_model_wind_short_drift_corr_dt[
            "effective_dec_year"
        ].dt.year.astype(int)
    )

    # # plot the return periods by decade for wind speed
    # gev_funcs.plot_return_periods_decades(
    #     model_df=block_minima_model_wind_drift_corr_dt,
    #     model_var_name="data_min_drift_bc_dt",
    #     obs_df=block_minima_obs_wind_dt,
    #     obs_var_name="data_min_dt",
    #     decades=np.arange(1960, 2030, 10),
    #     title="Wind speed (m/s)",
    #     num_samples=1000,
    #     figsize=(10, 5),
    #     bad_min=True,
    # )

    # # plot the return periods by decade for temperature
    # gev_funcs.plot_return_periods_decades(
    #     model_df=block_minima_model_tas_drift_corr_dt,
    #     model_var_name="data_tas_c_min_drift_bc_dt",
    #     obs_df=block_minima_obs_tas_dt,
    #     obs_var_name="data_c_min_dt",
    #     decades=np.arange(1960, 2030, 10),
    #     title="Temperature (°C)",
    #     num_samples=1000,
    #     figsize=(10, 5),
    #     bad_min=True,
    # )

    # sys.exit()

    # test the same function for temperature
    # plot_emp_rps(
    #     obs_df=block_minima_obs_wind_dt,
    #     model_df=block_minima_model_wind_drift_corr_dt,
    #     obs_val_name="data_min_dt",
    #     model_val_name="data_min_drift_bc_dt",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     ylabel="Wind speed (m/s)",
    #     nsamples=10000,
    #     ylims=(2, 3.5),
    #     blue_line=np.min,
    #     high_values_rare=False,
    #     figsize=(5, 5),
    # )

    # # do the same for the wind speed extremes
    # plot_rp_extremes(
    #     obs_df=block_minima_obs_wind_dt,
    #     model_df=block_minima_model_wind_drift_corr_dt,
    #     obs_val_name="data_min_dt",
    #     model_val_name="data_min_drift_bc_dt",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     ylim=(2, 3.5),
    #     percentile=0.01,
    #     n_samples=10000,
    #     high_values_rare=False,
    # )

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

    # # set efefctive dec year back as an int in the model data
    # block_minima_model_tas_drift_corr_dt["effective_dec_year"] = block_minima_model_tas_drift_corr_dt[
    #     "effective_dec_year"
    # ].dt.year.astype(int)

    # # set effective dec year back as an int in the obs data
    # block_minima_model_wind_drift_corr_dt["effective_dec_year"] = block_minima_model_wind_drift_corr_dt[
    #     "effective_dec_year"
    # ].dt.year.astype(int)

    # reset the index of the obs
    block_minima_obs_tas_dt.reset_index(inplace=True)

    # reset the index of the obs
    block_minima_obs_wind_dt.reset_index(inplace=True)

    # reset the index of the obs
    block_minima_obs_wind_short_dt.reset_index(inplace=True)

    # make sure effective dec year in the obs is a datetime
    block_minima_obs_tas_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_tas_dt["effective_dec_year"], format="%Y"
    )

    # make sure effective dec year in the obs is a datetime
    block_minima_obs_wind_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_wind_dt["effective_dec_year"], format="%Y"
    )

    # make sure effective dec year in the obs is a datetime
    block_minima_obs_wind_short_dt["effective_dec_year"] = pd.to_datetime(
        block_minima_obs_wind_short_dt["effective_dec_year"], format="%Y"
    )

    # format the obs effective dec year as an int in years
    block_minima_obs_tas_dt["effective_dec_year"] = block_minima_obs_tas_dt[
        "effective_dec_year"
    ].dt.year.astype(int)

    # format the obs effective dec year as an int in years
    block_minima_obs_wind_dt["effective_dec_year"] = block_minima_obs_wind_dt[
        "effective_dec_year"
    ].dt.year.astype(int)

    # format the obs effective dec year as an int in years
    block_minima_obs_wind_short_dt["effective_dec_year"] = (
        block_minima_obs_wind_short_dt["effective_dec_year"].dt.year.astype(int)
    )

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

    # # # process the GEV params for the short wind data
    # gev_params_bc_wind_short = gev_funcs.process_gev_params(
    #     obs_df=block_minima_obs_wind_short_dt,
    #     model_df=block_minima_model_wind_short_drift_corr_dt,
    #     obs_var_name="data_min_dt",
    #     model_var_name="data_min_drift_bc_dt",
    #     obs_time_name="effective_dec_year",
    #     model_time_name="effective_dec_year",
    #     nboot=100,
    #     model_lead_name="winter_year",
    # )

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
        title_bottom="Distribution of DJF block minima wind speed 1960-2024 (m/s)",
        figsize=(15, 10),
    )

    # # Now test the plotting function for these
    # gev_funcs.plot_gev_params_subplots(
    #     gev_params_top_raw=gev_params_bc_temp,
    #     gev_params_top_bc=gev_params_bc_temp,
    #     gev_params_bottom_raw=gev_params_bc_wind_short,
    #     gev_params_bottom_bc=gev_params_bc_wind_short,
    #     obs_df_top=block_minima_obs_tas_dt,
    #     model_df_top=block_minima_model_tas_drift_corr_dt,
    #     obs_df_bottom=block_minima_obs_wind_short_dt,
    #     model_df_bottom=block_minima_model_wind_short_drift_corr_dt,
    #     obs_var_name_top="data_c_min_dt",
    #     model_var_name_top="data_tas_c_min_drift_bc_dt",
    #     obs_var_name_bottom="data_min_dt",
    #     model_var_name_bottom="data_min_drift_bc_dt",
    #     title_top="Distribution of DJF block minima temperature (°C)",
    #     title_bottom="Distribution of DJF block minima wind speed 1960-2023 (m/s)",
    #     figsize=(15, 10),
    # )

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


# If name is main
if __name__ == "__main__":
    main()
# %%
