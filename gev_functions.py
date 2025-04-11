"""
gev_functions.py

Functions for chapter 2 daily extremes analysis.

Author: Ben Hutchins, 2025
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
from typing import Any, Callable, Union, List, Any, Tuple

from scipy.optimize import curve_fit
from scipy.stats import linregress, percentileofscore, gaussian_kde, skew, kurtosis
from scipy.stats import genextreme as gev
from sklearn.metrics import mean_squared_error, r2_score
from iris.util import equalise_attributes

# # Suppress warnings
# warnings.filterwarnings('ignore')

sys.path.append("/home/users/benhutch/unseen_functions")
from functions import estimate_period


def determine_effective_dec_year(row):
    year = row["time"].year
    month = row["time"].month
    if month in [1, 2, 3]:
        return year - 1
    elif month in [10, 11, 12]:
        return year
    else:
        return None


# do the same but for canari
def determine_effective_dec_year_canari(row):
    year = row["time"].split("-")[0]
    month = row["time"].split("-")[1]

    if month in ["01", "02", "03"]:
        return int(year) - 1
    elif month in ["10", "11", "12"]:
        return int(year)
    else:
        return None


def month_col_canari(row):
    return int(row["time"].split("-")[1])


def year_col_canari(row):
    return int(row["time"].split("-")[0])


# Define a function to do the pivot detrending
def pivot_detrend_obs(
    df: pd.DataFrame,
    x_axis_name: str,
    y_axis_name: str,
    suffix: str = "_dt",
) -> pd.DataFrame:
    """
    Pivot detrend a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to detrend.
    x_axis_name : str
        Name of the column to use as the x-axis.
    y_axis_name : str
        Name of the column to use as the y-axis.
    suffix : str, optional
        Suffix to append to the detrended column, by default "_dt".

    Returns
    -------
    pd.DataFrame
        Detrended DataFrame.
    """

    # Make a copy of the DataFrame
    df_copy = df.copy()

    # Define the function to fit
    slope, intercept, r_value, p_value, std_err = linregress(
        df_copy[x_axis_name], df_copy[y_axis_name]
    )

    # Calculate the trend line
    trend = slope * df_copy[x_axis_name] + intercept

    # Determine the final point on the trend line
    final_point = trend.iloc[-1]

    # Create a new column with the detrended values
    df_copy[y_axis_name + suffix] = final_point - trend + df_copy[y_axis_name]

    return df_copy


def pivot_detrend_model(
    df: pd.DataFrame,
    x_axis_name: str,
    y_axis_name: str,
    suffix: str = "_dt",
    member_name: str = "member",
) -> pd.DataFrame:
    """
    Pivot detrend a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to detrend.
    x_axis_name : str
        Name of the column to use as the x-axis.
    y_axis_name : str
        Name of the column to use as the y-axis.
    suffix : str, optional
        Suffix to append to the detrended column, by default "_dt".
    member_name : str, optional
        Name of the column to use as the member identifier, by default "member".

    Returns
    -------
    pd.DataFrame
        Detrended DataFrame.
    """

    # Make a copy of the DataFrame
    df_copy = df.copy()

    # Set up the n members
    members = df_copy[member_name].unique()
    n_members = len(members)

    # Set up the slopes
    slopes = np.zeros(n_members)
    intercepts = np.zeros(n_members)

    # Loop over the members
    for i, member in enumerate(members):
        # Get the data for this member
        data = df_copy[df_copy[member_name] == member]

        # Define the function to fit
        slope, intercept, r_value, p_value, std_err = linregress(
            data[x_axis_name], data[y_axis_name]
        )

        # Store the slope and intercept
        slopes[i] = slope
        intercepts[i] = intercept

    # Calculate the trend line
    slopes_mean = np.mean(slopes)
    intercepts_mean = np.mean(intercepts)

    # Calculate the trend line
    trend = intercepts_mean + slopes_mean * df_copy[x_axis_name]

    # Determine the final point on the trend line
    final_point = trend.iloc[-1]

    # Create a new column with the detrended values
    df_copy[y_axis_name + suffix] = final_point - trend + df_copy[y_axis_name]

    return df_copy

# Write a function to pivot detrend the model
# but using the rolling mean methodology
def pivot_detrend_model_rolling(
    df: pd.DataFrame,
    x_axis_name: str,
    y_axis_name: str,
    suffix: str = "_rm_dt",
    member_name: str = "member",
    window: int = 10,
    centred_bool: bool = True,
    min_periods: int = 1,
) -> pd.DataFrame:
    """
    Pivot detrend a DataFrame by taking the 10 year rolling mean
    of the ensemble mean

    Parameters
    ==========

    df : pd.DataFrame
        DataFrame to detrend
    x_axis_name : str
        Name of the column to use as the x-axis
    y_axis_name : str
        Name of the column to use as the y-axis
    suffix : str, optional
        Suffix to append to the detrended column, by default "_rm_dt"
    member_name : str, optional
        Name of the column to use as the member identifier, by default "member"
    window : int, optional
        The size of the moving window, by default 10
    centred_bool : bool, optional
        Whether to set the window to be centred, by default True
    min_periods : int, optional
        The minimum number of periods to use, by default 1

    Returns
    =======

    pd.DataFrame
        Detrended DataFrame

    """

    # Make a copy of the DataFrame
    df_copy = df.copy()

    # Calculate the model ensemble mean
    model_ensmean = df_copy.groupby(x_axis_name)[y_axis_name].mean()

    # Calculate the rolling trend line
    trend = model_ensmean.rolling(
        window=window, center=centred_bool, min_periods=min_periods
    ).mean()

    # Determine the final point on the trend line
    final_point = trend.iloc[-1]

    # Set up the detrended valued
    values = df_copy[y_axis_name].values

    # Create a new column with the detrended values
    df_copy[y_axis_name + suffix] = (
        values - trend.loc[df_copy[x_axis_name].values].values + final_point
    )

    return df_copy

# Define a function to calculate the obs block minima/maxima
def obs_block_min_max(
    df: pd.DataFrame,
    time_name: str,
    min_max_var_name: str,
    new_df_cols: list[str] = [],
    process_min: bool = True,
) -> pd.DataFrame:
    """
    Calculate the block minima/maxima for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to calculate block minima/maxima for.
    time_name : str
        Name of the column to use as the time axis.
    min_max_var_name : str
        Name of the column to calculate the block minima/maxima for.
    new_df_cols : list[str]
        List of new column names to add to the DataFrame.
    process_min : bool, optional
        Whether to calculate the block minima (True) or maxima (False), by default True.


    Returns
    -------
    pd.DataFrame
        New dataframe with the block minima/maxima added.
    """
    # Initialize the new DataFrame
    block_df = pd.DataFrame()

    # Loop over the unique time names
    for time in df[time_name].unique():
        # Get the data for this time
        time_data = df[df[time_name] == time]

        # Get the min/max value
        if process_min:
            min_max_value = time_data[min_max_var_name].idxmin()
            name = "min"
        else:
            min_max_value = time_data[min_max_var_name].idxmax()
            name = "max"

        # Create a new dataframe
        df_this = pd.DataFrame(
            {
                time_name: [time],
                f"{min_max_var_name}_{name}": [
                    time_data.loc[min_max_value, min_max_var_name]
                ],
            }
        )

        # if the cols are not empty
        if new_df_cols:
            # Add the new columns
            for col in new_df_cols:
                df_this[col] = time_data.loc[min_max_value, col]

        # Concat to the block df
        block_df = pd.concat([block_df, df_this])

    return block_df


# write a function to calculate the block minima/maxima
# for a given percentile for the obs
def obs_block_min_max_percentile(
    df: pd.DataFrame,
    time_name: str,
    min_max_var_name: str,
    percentile: float = 5,
) -> pd.DataFrame:
    """
    Calculate the block minima/maxima for a DataFrame.

    But for the lowest 5th percentile of the data

    Parameters
    ----------

    df : pd.DataFrame
        DataFrame to calculate block minima/maxima for.
    time_name : str
        Name of the column to use as the time axis.
    min_max_var_name : str
        Name of the column to calculate the block minima/maxima for.
    percentile : float, optional
        The percentile to use, by default 5.

    Returns
    -------

    pd.DataFrame
        New dataframe with the block minima/maxima added.

    """

    # Initialize the new dataframe
    block_df = pd.DataFrame()

    # Loop over the unique time names
    for time in df[time_name].unique():
        # Get the data for this time
        time_data = df[df[time_name] == time]

        # if percentile is less than 50
        if percentile < 50:
            # Find the rows that are less than the threshold
            time_data = time_data[
                time_data[min_max_var_name]
                < np.percentile(time_data[min_max_var_name], percentile)
            ]
        else:
            # Find the rows that are greater than the threshold
            time_data = time_data[
                time_data[min_max_var_name]
                > np.percentile(time_data[min_max_var_name], percentile)
            ]

        # Concat to the block df
        block_df = pd.concat([block_df, time_data])

    return block_df


# Define a function to calculate the model block minima/maxima
def model_block_min_max(
    df: pd.DataFrame,
    time_name: str,
    min_max_var_name: str,
    new_df_cols: list[str],
    winter_year: str = None,
    process_min: bool = True,
    member_name: str = "member",
) -> pd.DataFrame:
    """
    Calculate the block minima/maxima for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to calculate block minima/maxima for.
    time_name : str
        Name of the column to use as the time axis.
    min_max_var_name : str
        Name of the column to calculate the block minima/maxima for.
    new_df_cols : list[str]
        List of new column names to add to the DataFrame.
    process_min : bool, optional
        Whether to calculate the block minima (True) or maxima (False), by default True.
    member_name : str, optional
        Name of the column to use as the member identifier, by default "member".

    Returns
    -------
    pd.DataFrame
        New dataframe with the block minima/maxima added.
    """
    # Initialize the new DataFrame
    block_df = pd.DataFrame()

    if winter_year is not None:
        print(f"Assuming winter year column name: {winter_year}")

        # Loop over the unique time names
        for time in df[time_name].unique():
            for wyear in df[winter_year].unique():
                for member in df[member_name].unique():
                    # Get the data for this time
                    time_data = df[
                        (df[time_name] == time)
                        & (df[winter_year] == wyear)
                        & (df[member_name] == member)
                    ]

                    # if time_data is empty
                    if time_data.empty:
                        print(f"Empty data for {time}, {wyear}, {member}")
                        continue

                    # Get the min/max value
                    if process_min:
                        min_max_value = time_data[min_max_var_name].idxmin()
                        name = "min"
                    else:
                        min_max_value = time_data[min_max_var_name].idxmax()
                        name = "max"

                    # Create a new dataframe
                    df_this = pd.DataFrame(
                        {
                            time_name: [time],
                            winter_year: [wyear],
                            member_name: [member],
                            f"{min_max_var_name}_{name}": [
                                time_data.loc[min_max_value, min_max_var_name]
                            ],
                        }
                    )

                    # Add the new columns
                    for col in new_df_cols:
                        df_this[col] = time_data.loc[min_max_value, col]

                    # Concat to the block df
                    block_df = pd.concat([block_df, df_this])

    else:
        print("Assuming first winter year only")
        # Loop over the unique time names
        for time in df[time_name].unique():
            for member in df[member_name].unique():
                # Get the data for this time
                time_data = df[(df[time_name] == time) & (df[member_name] == member)]

                # Get the min/max value
                if process_min:
                    min_max_value = time_data[min_max_var_name].idxmin()
                    name = "min"
                else:
                    min_max_value = time_data[min_max_var_name].idxmax()
                    name = "max"

                # Create a new dataframe
                df_this = pd.DataFrame(
                    {
                        time_name: [time],
                        member_name: [member],
                        f"{min_max_var_name}_{name}": [
                            time_data.loc[min_max_value, min_max_var_name]
                        ],
                    }
                )

                # Add the new columns
                for col in new_df_cols:
                    df_this[col] = time_data.loc[min_max_value, col]

                # Concat to the block df
                block_df = pd.concat([block_df, df_this])

    return block_df


# Define a function to calculate the block minima/maxima
# for the model data
# but for the lowest 5th percentile of the data
# for each independent season
def model_block_min_max_percentile(
    df: pd.DataFrame,
    time_name: str,
    min_max_var_name: str,
    percentile: float = 5,
    winter_year: str = None,
    member_name: str = "member",
) -> pd.DataFrame:
    """
    Calculate the block minima/maxima for a DataFrame.

    But for the lowest 5th percentile of the data

    Parameters
    ----------

    df : pd.DataFrame
        DataFrame to calculate block minima/maxima for.
    time_name : str
        Name of the column to use as the time axis.
    min_max_var_name : str
        Name of the column to calculate the block minima/maxima for.
    percentile : float, optional
        The percentile to use, by default 5.
    winter_year : str, optional
        Name of the column to use as the winter year, by default None.
    member_name : str, optional
        Name of the column to use as the member identifier, by default "member".

    Returns
    -------

    pd.DataFrame
        New dataframe with the block minima/maxima added.

    """

    # Initialize the new dataframe
    block_df = pd.DataFrame()

    # quantify the percentile threshold
    threshold = np.percentile(df[min_max_var_name], percentile)

    if winter_year is not None:
        print(f"Assuming winter year column name: {winter_year}")

        # Loop over the unique time names
        for time in df[time_name].unique():
            for wyear in df[winter_year].unique():
                for member in df[member_name].unique():
                    # Get the data for this time
                    time_data = df[
                        (df[time_name] == time)
                        & (df[winter_year] == wyear)
                        & (df[member_name] == member)
                    ]

                    # if time_data is empty
                    if time_data.empty:
                        print(f"Empty data for {time}, {wyear}, {member}")
                        continue

                    # if percentile is less than 50
                    if percentile < 50:
                        # Find the rows that are less than the threshold
                        time_data = time_data[time_data[min_max_var_name] < threshold]
                    else:
                        # Find the rows that are greater than the threshold
                        time_data = time_data[time_data[min_max_var_name] > threshold]

                    # Concat to the block df
                    block_df = pd.concat([block_df, time_data])

    else:
        print("Assuming first winter year only")
        # Loop over the unique time names
        for time in df[time_name].unique():
            for member in df[member_name].unique():
                # Get the data for this time
                time_data = df[(df[time_name] == time) & (df[member_name] == member)]

                # if time_data is empty
                if time_data.empty:
                    print(f"Empty data for {time}, {member}")
                    continue

                # if percentile is less than 50
                if percentile < 50:
                    # Find the rows that are less than the threshold
                    time_data = time_data[time_data[min_max_var_name] < threshold]
                else:
                    # Find the rows that are greater than the threshold
                    time_data = time_data[time_data[min_max_var_name] > threshold]

                # Concat to the block df
                block_df = pd.concat([block_df, time_data])

    return block_df


# Define a function for simple mean bias correct
def mean_bias_correct(
    model_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    model_var_name: str,
    obs_var_name: str,
    suffix: str = "_bc",
) -> pd.DataFrame:
    """
    Perform a simple mean bias correction.

    Parameters
    ----------
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_var_name : str
        Name of the column to correct in the model DataFrame.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    suffix : str, optional
        Suffix to append to the corrected column, by default "_bc".

    Returns
    -------
    pd.DataFrame
        Corrected DataFrame.
    """
    # Calculate the bias
    bias = model_df[model_var_name].mean() - obs_df[obs_var_name].mean()

    # Print the size of the bias
    print(f"Mean bias correction: {bias}")

    # Correct the model data
    model_df[model_var_name + suffix] = model_df[model_var_name] - bias

    return model_df


# Define a function which performs a lead-time dependent
# mean bias correction
def lead_time_mean_bias_correct(
    model_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    model_var_name: str,
    obs_var_name: str,
    lead_name: str,
    suffix: str = "_bc",
) -> pd.DataFrame:
    """
    Perform a lead-time dependent mean bias correction.

    Parameters
    ----------
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_var_name : str
        Name of the column to correct in the model DataFrame.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    lead_name : str
        Name of the column to use as the lead time axis.
    suffix : str, optional
        Suffix to append to the corrected column, by default "_bc".

    Returns
    -------
    pd.DataFrame
        Corrected DataFrame.
    """

    # Set up the corrected df
    corrected_df = pd.DataFrame()

    # Get the unique lead times from the model df
    lead_times = sorted(model_df[lead_name].unique())

    # Loop over the lead times
    for lead in lead_times:
        # Get the data for this lead time
        data_this = model_df[model_df[lead_name] == lead]

        # Extract the unique effective dec years for the model
        unique_eff_dec_years_model = data_this["effective_dec_year"].unique()

        # Extract the unique effective dec years for the obs
        unique_eff_dec_years_obs = obs_df["effective_dec_year"].unique()

        obs_df_lead_this = None

        # if the unique eff dec years for the model and obs are not the same
        if not np.array_equal(unique_eff_dec_years_model, unique_eff_dec_years_obs):
            print("Unique effective decadal years for model and obs are not the same")
            # Subset the obs this to the unique eff dec years for the model
            obs_df_lead_this = obs_df[
                obs_df["effective_dec_year"].isin(unique_eff_dec_years_model)
            ]

        # If obs_df_lead_this is None
        if obs_df_lead_this is not None:
            # Calculate the mean bias
            bias = (
                data_this[model_var_name].mean() - obs_df_lead_this[obs_var_name].mean()
            )
        else:
            # Calculate the mean bias
            bias = data_this[model_var_name].mean() - obs_df[obs_var_name].mean()

        # Print the size of the bias
        print(f"Lead {lead} mean bias correction: {bias}")

        # Correct the model data
        data_this[model_var_name + suffix] = data_this[model_var_name] - bias

        # Concat to the corrected df
        corrected_df = pd.concat([corrected_df, data_this])

    return corrected_df


# Define a function to process the GEV params
def process_gev_params(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_var_name: str,
    model_var_name: str,
    obs_time_name: str,
    model_time_name: str,
    nboot: int = 1000,
    model_member_name: str = "member",
    model_lead_name: str = None,
) -> dict:
    """
    Process the GEV parameters.

    Parameters
    ----------
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    obs_time_name : str
        Name of the column to use as the time axis in the observed DataFrame.
    model_time_name : str
        Name of the column to use as the time axis in the model DataFrame.
    nboot : int, optional
        Number of bootstrap samples to use, by default 1000.
    model_member_name : str, optional
        Name of the column to use as the member identifier in the model DataFrame, by default "member".

    Returns
    -------
    dict
        Dictionary of GEV parameters.
    """

    mdi = -999.0

    # Initialize the dictionary
    gev_params = {
        "obs_shape": mdi,
        "obs_loc": mdi,
        "obs_scale": mdi,
        "model_shape": [np.zeros(nboot)],
        "model_loc": [np.zeros(nboot)],
        "model_scale": [np.zeros(nboot)],
    }

    # Fit the GEV distribution to the observed data
    shape_obs, loc_obs, scale_obs = gev.fit(obs_df[obs_var_name], 0)

    # Store the observed parameters
    gev_params["obs_shape"] = shape_obs
    gev_params["obs_loc"] = loc_obs
    gev_params["obs_scale"] = scale_obs

    # Loop over the nboot
    for i in tqdm(range(nboot)):
        # Set up the psuedo-observed data
        pseudo_obs_this = np.zeros_like(obs_df[obs_var_name].values)
        for t, time in enumerate(obs_df[obs_time_name].unique()):
            # Subset the data
            df_model_this_time = model_df[model_df[model_time_name] == time]

            # Pick a random member
            member_this = np.random.choice(
                df_model_this_time[model_member_name].unique()
            )

            # Get the data for this member
            data_this = df_model_this_time[
                df_model_this_time[model_member_name] == member_this
            ]

            # if model_lead_name is not None
            if model_lead_name is not None:
                # Pick a random lead time
                lead_this = np.random.choice(data_this[model_lead_name].unique())

                # Get the data for this lead time
                data_this = data_this[data_this[model_lead_name] == lead_this]

            # Extract the values
            model_value_this = data_this[model_var_name].values

            # if model_value_this.size > 0:
            # assert that the len of model_value_this is one
            assert model_value_this.size == 1, "model_value_this should have length 1"

            # Add the data to the pseudo-observed data
            pseudo_obs_this[t] = model_value_this

        # Fit the GEV distribution to the pseudo-observed data
        shape_model, loc_model, scale_model = gev.fit(pseudo_obs_this, 0)

        # Store the model parameters
        gev_params["model_shape"][0][i] = shape_model
        gev_params["model_loc"][0][i] = loc_model
        gev_params["model_scale"][0][i] = scale_model

    return gev_params


# Write a function for processing full field fidelity
def process_moments_fidelity(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_var_name: str,
    model_var_name: str,
    obs_wyears_name: str,
    model_wyears_name: str,
    nboot: int = 1000,
    model_member_name: str = "member",
    model_lead_name: str = None,
) -> dict:
    """
    Process the moments of fidelity.

    Parameters
    ----------
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    model_var_name : str
        Name of the column to use in the mode DataFrame.
    obs_wyears_name : str
        Name of the column to use as the winter years in the observed DataFrame.
    model_wyears_name : str
        Name of the column to use as the winter years in the model DataFrame.
    nboot : int, optional
        Number of bootstrap samples to use, by default 1000.
    model_member_name : str, optional
        Name of the column to use as the member identifier in the model DataFrame, by default "member".
    model_lead_name : str, optional
        Name of the column to use as the lead time in the model DataFrame, by default None.

    Returns
    -------
    dict
        Dictionary of moments of fidelity.
    """

    mdi = -999.0

    # Initialize the dictionary
    moments_fidelity = {
        "obs_mean": mdi,
        "obs_std": mdi,
        "obs_skew": mdi,
        "obs_kurt": mdi,
        "model_mean": [np.zeros(nboot)],
        "model_std": [np.zeros(nboot)],
        "model_skew": [np.zeros(nboot)],
        "model_kurt": [np.zeros(nboot)],
    }

    # Calculate the observed mean and standard deviation
    obs_mean = np.mean(obs_df[obs_var_name])
    obs_std = np.std(obs_df[obs_var_name])
    obs_skew = skew(obs_df[obs_var_name])  # from scipy
    obs_kurt = kurtosis(obs_df[obs_var_name])  # ditto

    # Store the observed moments
    moments_fidelity["obs_mean"] = obs_mean
    moments_fidelity["obs_std"] = obs_std
    moments_fidelity["obs_skew"] = obs_skew
    moments_fidelity["obs_kurt"] = obs_kurt

    # Set up the obs years
    n_obs_years = len(obs_df[obs_wyears_name].unique())
    obs_years = obs_df[obs_wyears_name].unique()

    # Set up the model winters
    if model_lead_name is not None:
        print("Assuming multiple winters are present")
        model_years = model_df[model_wyears_name].unique()
        model_members = model_df[model_member_name].unique()
        model_winters = model_df[model_lead_name].unique()

        # Select the model df for the first winter, member and year
        model_df_this = model_df[
            (model_df[model_wyears_name] == model_years[0])
            & (model_df[model_member_name] == model_members[0])
            & (model_df[model_lead_name] == model_winters[0])
        ]

        # Set up the length of this
        n_model_winter_days = len(model_df_this)
    else:
        print("Assuming only one winter is present")
        model_years = model_df[model_wyears_name].unique()
        model_members = model_df[model_member_name].unique()

        # Select the model df for the first winter and member
        model_df_this = model_df[
            (model_df[model_wyears_name] == model_years[0])
            & (model_df[model_member_name] == model_members[0])
        ]

        # Set up the length of this
        n_model_winter_days = len(model_df_this)

    # print the n obs years
    print(f"Number of observed years: {n_obs_years}")
    print(f"Number of model winter days: {n_model_winter_days}")

    # assert that n_model_winter days is less than 200
    assert n_model_winter_days < 200, "n_model_winter_days should be less than 200"

    # Loop over the nboot
    for i in tqdm(range(nboot)):
        # Set up the psuedo-observed data
        pseudo_obs_this = np.zeros((n_obs_years, n_model_winter_days))
        for y, obs_year in enumerate(obs_years):
            # Subset the data
            df_model_this_year = model_df[model_df[model_wyears_name] == obs_year]

            # Pick a random member
            member_this = np.random.choice(
                df_model_this_year[model_member_name].unique()
            )

            # Get the data for this member
            data_this = df_model_this_year[
                df_model_this_year[model_member_name] == member_this
            ]

            # if model_lead_name is not None
            if model_lead_name is not None:
                # Pick a random lead time
                lead_this = np.random.choice(data_this[model_lead_name].unique())

                # Get the data for this lead time
                data_this = data_this[data_this[model_lead_name] == lead_this]

            # Extract the values
            model_values_this = data_this[model_var_name].values

            # assert that the size of values is greater than 1
            assert (
                model_values_this.size > 1
            ), "model_values_this should have length greater than 1"

            # assert that the size of values is less than 200
            assert (
                model_values_this.size < 200
            ), "model_values_this should have length less than 200"

            # Add the data to the pseudo-observed data
            pseudo_obs_this[y, :] = model_values_this

        # Flatten the pseudo-observed data
        pseudo_obs_this_flat = pseudo_obs_this.flatten()

        # Calculate the mean, standard deviation, skew and kurtosis
        model_mean_this = np.mean(pseudo_obs_this_flat)
        model_std_this = np.std(pseudo_obs_this_flat)
        model_skew_this = skew(pseudo_obs_this_flat)
        model_kurt_this = kurtosis(pseudo_obs_this_flat)

        # Store the model moments
        moments_fidelity["model_mean"][0][i] = model_mean_this
        moments_fidelity["model_std"][0][i] = model_std_this
        moments_fidelity["model_skew"][0][i] = model_skew_this
        moments_fidelity["model_kurt"][0][i] = model_kurt_this

    return moments_fidelity


# define a function to plot the distributions
def plot_multi_var_dist(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    model_df_bc: pd.DataFrame,
    obs_var_names: list[str],
    model_var_names: list[str],
    model_var_names_bc: list[str],
    row_titles: list[str],
    subplot_titles: list[str],
    figsize: tuple = (15, 5),
) -> None:
    """
    Plot the distributions of multiple variables.

    Parameters
    ----------
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    model_df_bc : pd.DataFrame
        DataFrame of model data after bias correction.
    obs_var_names : list[str]
        List of names of the columns to use in the observed DataFrame.
    model_var_names : list[str]
        List of names of the columns to use in the model DataFrame.
    model_var_names_bc : list[str]
        List of names of the columns to use in the model DataFrame after bias correction.
    row_titles : list[str]
        List of titles for each row.
    subplot_titles : list[tuple]
        List of titles for each subplot.
    figsize : tuple, optional
        Figure size, by default (15, 5).

    Returns
    -------
    None
    """

    # Set up the nrows
    nrows = len(obs_var_names)

    # Set up the figure
    fig, axs = plt.subplots(nrows=nrows, ncols=2, figsize=figsize)

    # loop over the nrows
    for r, row in enumerate(range(nrows)):
        # Plot the raw distribution on the first axis
        axs[r, 0].hist(
            model_df[model_var_names[r]],
            color="red",
            alpha=0.5,
            bins=30,
            label="model",
            density=True,
        )

        # Same for the observations
        axs[r, 0].hist(
            obs_df[obs_var_names[r]],
            color="black",
            alpha=0.5,
            bins=30,
            label="observed",
            density=True,
        )

        # remove the numbers from the y axis
        axs[r, 0].set_yticks([])

        # Set the row title to the left, rotated 90 degrees and larger font size
        axs[r, 0].set_ylabel(row_titles[r], rotation=90, fontsize=14)

        # # Set up the subplot title
        # axs[r, 0].set_title(subplot_titles[r][0])

        axs[r, 0].text(
            0.05,
            0.05,
            subplot_titles[r][0],
            transform=axs[r, 0].transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # do the same for the bias corrected distribution
        axs[r, 1].text(
            0.05,
            0.05,
            subplot_titles[r][1],
            transform=axs[r, 1].transAxes,
            fontsize=12,
            verticalalignment="bottom",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Plot the bias corrected distribution on the second axis
        axs[r, 1].hist(
            model_df_bc[model_var_names_bc[r]],
            color="red",
            alpha=0.5,
            bins=30,
            label="model",
            density=True,
        )

        # Same for the observations
        axs[r, 1].hist(
            obs_df[obs_var_names[r]],
            color="black",
            alpha=0.5,
            bins=30,
            label="observed",
            density=True,
        )

        # if r is 0
        if r == 0:
            # Set the title
            axs[r, 0].set_title("raw")
            axs[r, 1].set_title("bias corrected")

            # include a legend
            axs[r, 1].legend(loc="upper right")

        # calculate the bias for the raw data
        bias_raw = model_df[model_var_names[r]].mean() - obs_df[obs_var_names[r]].mean()
        bias_bc = (
            model_df_bc[model_var_names_bc[r]].mean() - obs_df[obs_var_names[r]].mean()
        )

        # include these in a textbox in the top left
        axs[r, 0].text(
            0.05,
            0.95,
            f"raw bias = {bias_raw:.2f}",
            transform=axs[r, 0].transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        axs[r, 1].text(
            0.05,
            0.95,
            f"BC bias = {bias_bc:.2f}",
            transform=axs[r, 1].transAxes,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

    # Specify a tight layout
    fig.tight_layout()

    # Show the plot
    plt.show()

    return None


# Define a function for plotting the moments fidelity
def plot_moments_fidelity(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_var_name: str,
    model_var_name: str,
    moments_fidelity: dict,
    title: str,
    figsize: tuple = (15, 5),
) -> None:
    """
    Plot the moments of fidelity.

    Parameters
    ----------

    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    moments_fidelity : dict
        Dictionary of moments of fidelity.
    title : str
        Title of the plot.
    figsize : tuple, optional
        Figure size, by default (15, 5).

    Returns
    -------

    None

    """

    # Set up the figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(nrows=2, ncols=3, width_ratios=[2, 1, 1])

    ax_main = fig.add_subplot(gs[:, 0])  # Span all rows with first col

    # Plot the distribution on the first axis
    ax_main.hist(
        model_df[model_var_name], color="red", alpha=0.5, label="model", density=True
    )

    # Same for the observations
    ax_main.hist(
        obs_df[obs_var_name], color="black", alpha=0.5, label="observed", density=True
    )

    # Include a textbox with the sample size
    ax_main.text(
        0.95,
        0.90,
        f"model N = {len(model_df)}\nobs N = {len(obs_df)}",
        transform=ax_main.transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
        horizontalalignment="right",
    )

    # Include a legend
    ax_main.legend(loc="upper right")

    ax_main.tick_params(
        axis="y", which="both", left=False, right=False, labelleft=False
    )

    # Remove the numbers from the y axis
    ax_main.set_yticks([])

    # Add a title
    ax_main.set_title(title)

    # Set up the list of stat names
    stat_names = ["mean", "std", "skew", "kurt"]

    # Set up the list of axes labels
    ax_labels = ["a", "b", "c", "d"]

    # Additional subplots for metrics
    ax_mean = fig.add_subplot(gs[0, 1])
    ax_skew = fig.add_subplot(gs[0, 2])
    ax_stddev = fig.add_subplot(gs[1, 1])
    ax_kurtosis = fig.add_subplot(gs[1, 2])

    axes = [ax_mean, ax_skew, ax_stddev, ax_kurtosis]

    # Set up the model stats
    model_stats = [
        moments_fidelity["model_mean"][0],
        moments_fidelity["model_skew"][0],
        moments_fidelity["model_std"][0],
        moments_fidelity["model_kurt"][0],
    ]

    # set up the obs stats
    obs_stats = [
        moments_fidelity["obs_mean"],
        moments_fidelity["obs_skew"],
        moments_fidelity["obs_std"],
        moments_fidelity["obs_kurt"],
    ]

    # Loop over the stats
    for i, ax in enumerate(axes):
        # Plot the historgaram of the model stats
        ax.hist(model_stats[i], bins=100, density=True, color="red", label="model")

        # Plot the obs stats
        ax.axvline(obs_stats[i], color="black", linestyle="-", label="ERA5")

        # Calculate the percentile of score for the obs
        obs_pos = percentileofscore(model_stats[i], obs_stats[i])

        # Plot vertical black dashed lines for the 2.5% and 97.5% quantiles of the model stats
        ax.axvline(np.quantile(model_stats[i], 0.025), color="black", linestyle="--")

        ax.axvline(np.quantile(model_stats[i], 0.975), color="black", linestyle="--")

        # rmeove the yticks
        ax.set_yticks([])

        # Add a title in bold with obs_pos rounded to the closest integer
        ax.set_title(f"{stat_names[i]}, {round(obs_pos)}%", fontweight="bold")

        # add the axes labels
        # in the top left
        ax.text(
            0.05,
            0.95,
            ax_labels[i],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="top",
            ha="left",
            bbox=dict(facecolor="white", alpha=0.5),
            zorder=100,
        )

    # Specify a tight layout
    fig.tight_layout()

    # Show the plot
    plt.show()

    return None


# Define the gev plotting function
def plot_gev_params(
    gev_params: dict,
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_var_name: str,
    model_var_name: str,
    title: str,
    obs_label: str = "Observed",
    model_label: str = "Model",
    figsize: tuple = (12, 8),
) -> None:
    """
    Plot the GEV parameters.

    Parameters
    ----------
    gev_params : dict
        Dictionary of GEV parameters.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    title : str
        Title of the plot.
    obs_label : str, optional
        Label for the observed data, by default "Observed".
    model_label : str, optional
        Label for the model data, by default "Model".
    figsize : tuple, optional
        Figure size, by default (12, 8).

    Returns
    -------
    None
    """
    # Set up the figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.5, 1, 1, 1])

    # Create the subplots
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    ax3 = plt.subplot(gs[3])

    # Plot the model dsitribution
    ax0.hist(
        model_df[model_var_name],
        bins=20,
        color="red",
        alpha=0.5,
        label=model_label,
        density=True,
    )

    # Plot the distributions
    ax0.hist(
        obs_df[obs_var_name],
        bins=20,
        color="black",
        alpha=0.5,
        label=obs_label,
        density=True,
    )

    # plot the model mean gev
    xvals_model = np.linspace(
        np.min(model_df[model_var_name]), np.max(model_df[model_var_name]), 100
    )

    # Plot the GEV distribution
    ax0.plot(
        xvals_model,
        gev.pdf(
            xvals_model,
            gev_params["model_shape"][0].mean(),
            gev_params["model_loc"][0].mean(),
            gev_params["model_scale"][0].mean(),
        ),
        color="red",
        linestyle="--",
    )

    # print the mean of the model shape
    print(f"Model shape mean: {gev_params['model_shape'][0].mean()}")
    print(f"Model shape std: {gev_params['model_shape'][0].std()}")
    print(f"Model loc mean: {gev_params['model_loc'][0].mean()}")

    # plot the obs mean gev
    ax0.plot(
        xvals_model,
        gev.pdf(
            xvals_model,
            gev_params["obs_shape"],
            gev_params["obs_loc"],
            gev_params["obs_scale"],
        ),
        color="black",
        linestyle="--",
    )

    # print the obs shape
    print(f"Obs shape: {gev_params['obs_shape']}")
    print(f"Obs loc: {gev_params['obs_loc']}")
    print(f"Obs scale: {gev_params['obs_scale']}")

    # Set the title
    ax0.set_title(title)

    # Include a legend
    ax0.legend(loc="upper right")

    # Plot the histogram of the loc values in red
    ax1.hist(gev_params["model_loc"][0], bins=30, color="red", alpha=0.5)

    # Mark the 2.5%tile as a dashed vertical line
    ax1.axvline(
        np.percentile(gev_params["model_loc"][0], 2.5),
        color="red",
        linestyle="--",
        label="2.5%tile",
    )

    # Mark the 97.5%tile as a dashed vertical line
    ax1.axvline(
        np.percentile(gev_params["model_loc"][0], 97.5),
        color="red",
        linestyle="--",
        label="97.5%tile",
    )

    # Plot the observed line as a blue vertical line
    ax1.axvline(gev_params["obs_loc"], color="blue", lw=3, label="Observed")

    # Include a title for the loc
    obs_percentile_loc = percentileofscore(
        gev_params["model_loc"][0], gev_params["obs_loc"]
    )

    # Set the title
    ax1.set_title(f"location, {obs_percentile_loc:.2f}%")

    # Plot the scale values
    ax2.hist(gev_params["model_scale"][0], bins=30, color="red", alpha=0.5)

    # Mark the 2.5%tile as a dashed vertical line
    ax2.axvline(
        np.percentile(gev_params["model_scale"][0], 2.5),
        color="red",
        linestyle="--",
        label="2.5%tile",
    )

    # Mark the 97.5%tile as a dashed vertical line
    ax2.axvline(
        np.percentile(gev_params["model_scale"][0], 97.5),
        color="red",
        linestyle="--",
        label="97.5%tile",
    )

    # Plot the observed line as a blue vertical line
    ax2.axvline(gev_params["obs_scale"], color="blue", lw=3, label="Observed")

    # Include a title for the scale
    obs_percentile_scale = percentileofscore(
        gev_params["model_scale"][0], gev_params["obs_scale"]
    )

    # Set the title
    ax2.set_title(f"scale, {obs_percentile_scale:.2f}%")

    # Plot the shape values
    ax3.hist(gev_params["model_shape"][0], bins=30, color="red", alpha=0.5)

    # Mark the 2.5%tile as a dashed vertical line
    ax3.axvline(
        np.percentile(gev_params["model_shape"][0], 2.5),
        color="red",
        linestyle="--",
        label="2.5%tile",
    )

    # Mark the 97.5%tile as a dashed vertical line
    ax3.axvline(
        np.percentile(gev_params["model_shape"][0], 97.5),
        color="red",
        linestyle="--",
        label="97.5%tile",
    )

    # Plot the observed line as a blue vertical line
    ax3.axvline(gev_params["obs_shape"], color="blue", lw=3, label="Observed")

    # Include a title for the shape
    obs_percentile_shape = percentileofscore(
        gev_params["model_shape"][0], gev_params["obs_shape"]
    )

    # Set the title
    ax3.set_title(f"shape, {obs_percentile_shape:.2f}%")

    # remove the y-axis ticks
    for ax in [ax0, ax1, ax2, ax3]:
        ax.yaxis.set_ticks([])

    # Set up a tight layout
    plt.tight_layout()

    return None


# Define the gev plotting function
def plot_gev_params_subplots(
    gev_params_top_raw: dict,
    gev_params_top_bc: dict,
    gev_params_bottom_raw: dict,
    gev_params_bottom_bc: dict,
    obs_df_top: pd.DataFrame,
    model_df_top: pd.DataFrame,
    obs_df_bottom: pd.DataFrame,
    model_df_bottom: pd.DataFrame,
    obs_var_name_top: str,
    model_var_name_top: str,
    obs_var_name_bottom: str,
    model_var_name_bottom: str,
    title_top: str,
    title_bottom: str,
    obs_label: str = "Observed",
    model_label: str = "Model",
    figsize: tuple = (15, 10),
) -> None:
    """
    Plot the GEV parameters.

    Parameters
    ----------
    gev_params : dict
        Dictionary of GEV parameters.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    title : str
        Title of the plot.
    obs_label : str, optional
        Label for the observed data, by default "Observed".
    model_label : str, optional
        Label for the model data, by default "Model".
    figsize : tuple, optional
        Figure size, by default (12, 8).

    Returns
    -------
    None
    """
    # Set up the figure
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(2, 4, width_ratios=[1.5, 1, 1, 1])

    # Create the subplots for the first row
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[0, 2])
    ax3 = plt.subplot(gs[0, 3])

    # Create the subplots for the second row
    ax4 = plt.subplot(gs[1, 0])
    ax5 = plt.subplot(gs[1, 1])
    ax6 = plt.subplot(gs[1, 2])
    ax7 = plt.subplot(gs[1, 3])

    # Set up the list of axes
    axes = [[ax0, ax1, ax2, ax3], [ax4, ax5, ax6, ax7]]
    # Set up the list of gev params
    gev_params_list = [
        [gev_params_top_raw, gev_params_top_bc],
        [gev_params_bottom_raw, gev_params_bottom_bc],
    ]
    # Set up the list of obs df
    obs_dfs = [obs_df_top, obs_df_bottom]
    # Set up the list of model df
    model_dfs = [model_df_top, model_df_bottom]

    # Set up the list of obs var names
    obs_var_names = [obs_var_name_top, obs_var_name_bottom]
    # Set up the list of model var names
    model_var_names = [model_var_name_top, model_var_name_bottom]

    # Set up the list of titles
    titles = [title_top, title_bottom]

    # Loop over the axes
    for i, ax_row in enumerate(axes):
        # Set up the axes
        ax0 = ax_row[0]
        ax1 = ax_row[1]
        ax2 = ax_row[2]
        ax3 = ax_row[3]

        # Set up the model df
        model_df = model_dfs[i]

        # Set up the obs df
        obs_df = obs_dfs[i]

        # Set up the obs var name
        obs_var_name = obs_var_names[i]

        # Set up the model var name
        model_var_name = model_var_names[i]

        # Set up the gev params
        gev_params_non_bc = gev_params_list[i][0]

        # Set up the gev params bc
        gev_params = gev_params_list[i][1] # Set default as BC

        # Set up the title
        title = titles[i]

        # Plot the model dsitribution
        ax0.hist(
            model_df[model_var_name],
            bins=20,
            color="red",
            alpha=0.5,
            label=model_label,
            density=True,
        )

        # Plot the distributions
        ax0.hist(
            obs_df[obs_var_name],
            bins=20,
            color="black",
            alpha=0.5,
            label=obs_label,
            density=True,
        )

        # plot the model mean gev
        xvals_model = np.linspace(
            np.min(model_df[model_var_name]), np.max(model_df[model_var_name]), 100
        )

        # # Plot the GEV distribution
        # ax0.plot(
        #     xvals_model,
        #     gev.pdf(
        #         xvals_model,
        #         gev_params["model_shape"][0].mean(),
        #         gev_params["model_loc"][0].mean(),
        #         gev_params["model_scale"][0].mean(),
        #     ),
        #     color="red",
        #     linestyle="--",
        # )

        # # print the mean of the model shape
        # print(f"Model shape mean: {gev_params['model_shape'][0].mean()}")
        # print(f"Model shape std: {gev_params['model_shape'][0].std()}")
        # print(f"Model loc mean: {gev_params['model_loc'][0].mean()}")

        # # plot the obs mean gev
        # ax0.plot(
        #     xvals_model,
        #     gev.pdf(
        #         xvals_model,
        #         gev_params["obs_shape"],
        #         gev_params["obs_loc"],
        #         gev_params["obs_scale"],
        #     ),
        #     color="black",
        #     linestyle="--",
        # )

        # # print the obs shape
        # print(f"Obs shape: {gev_params['obs_shape']}")
        # print(f"Obs loc: {gev_params['obs_loc']}")
        # print(f"Obs scale: {gev_params['obs_scale']}")

        # Set the title
        ax0.set_title(title)

        # Include a legend
        ax0.legend(loc="upper right")

        # Plot the histogram of the loc values in red
        ax1.hist(gev_params["model_loc"][0], bins=30, color="red", alpha=0.5)

        # Mark the 2.5%tile as a dashed vertical line
        ax1.axvline(
            np.percentile(gev_params["model_loc"][0], 2.5),
            color="red",
            linestyle="--",
            label="2.5%tile",
        )

        # Mark the 97.5%tile as a dashed vertical line
        ax1.axvline(
            np.percentile(gev_params["model_loc"][0], 97.5),
            color="red",
            linestyle="--",
            label="97.5%tile",
        )

        # Plot the observed line as a blue vertical line
        ax1.axvline(gev_params["obs_loc"], color="blue", lw=3, label="Observed")

        # include a dashed red vertical line
        # to show the mean location parameter without bias adjustment
        ax1.axvline(
            gev_params_non_bc["model_loc"][0].mean(), color="red", lw=3, linestyle="--"
        )

        # Include a title for the loc
        obs_percentile_loc = percentileofscore(
            gev_params["model_loc"][0], gev_params["obs_loc"]
        )

        # Set the title
        ax1.set_title(f"location, {obs_percentile_loc:.2f}%")

        # Plot the scale values
        ax2.hist(gev_params["model_scale"][0], bins=30, color="red", alpha=0.5)

        # Mark the 2.5%tile as a dashed vertical line
        ax2.axvline(
            np.percentile(gev_params["model_scale"][0], 2.5),
            color="red",
            linestyle="--",
            label="2.5%tile",
        )

        # Mark the 97.5%tile as a dashed vertical line
        ax2.axvline(
            np.percentile(gev_params["model_scale"][0], 97.5),
            color="red",
            linestyle="--",
            label="97.5%tile",
        )

        # Plot the observed line as a blue vertical line
        ax2.axvline(gev_params["obs_scale"], color="blue", lw=3, label="Observed")

        # Include a title for the scale
        obs_percentile_scale = percentileofscore(
            gev_params["model_scale"][0], gev_params["obs_scale"]
        )

        # Set the title
        ax2.set_title(f"scale, {obs_percentile_scale:.2f}%")

        # Plot the shape values
        ax3.hist(gev_params["model_shape"][0], bins=30, color="red", alpha=0.5)

        # Mark the 2.5%tile as a dashed vertical line
        ax3.axvline(
            np.percentile(gev_params["model_shape"][0], 2.5),
            color="red",
            linestyle="--",
            label="2.5%tile",
        )

        # Mark the 97.5%tile as a dashed vertical line
        ax3.axvline(
            np.percentile(gev_params["model_shape"][0], 97.5),
            color="red",
            linestyle="--",
            label="97.5%tile",
        )

        # Plot the observed line as a blue vertical line
        ax3.axvline(gev_params["obs_shape"], color="blue", lw=3, label="Observed")

        # Include a title for the shape
        obs_percentile_shape = percentileofscore(
            gev_params["model_shape"][0], gev_params["obs_shape"]
        )

        # Set the title
        ax3.set_title(f"shape, {obs_percentile_shape:.2f}%")

        # remove the y-axis ticks
        for ax in [ax0, ax1, ax2, ax3]:
            ax.yaxis.set_ticks([])

    # Set up a tight layout
    plt.tight_layout()

    return None

# Define a function for plotting the time series
def plot_detrend_ts(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_var_name: str,
    model_var_name: str,
    obs_time_name: str,
    model_time_name: str,
    ylabel: str,
    title: str,
    ylim: tuple = None,
    detrend_suffix: str = "_dt",
    plot_min: bool = True,
    model_member_name: str = "member",
    figsize: tuple = (10, 5),
) -> None:
    """
    Plot the detrended time series.

    Parameters
    ----------
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    obs_time_name : str
        Name of the column to use as the time axis in the observed DataFrame.
    model_time_name : str
        Name of the column to use as the time axis in the model DataFrame.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    detrend_suffix : str, optional
        Suffix of the detrended column, by default "_dt".
    plot_min : bool, optional
        Whether to plot the minima (True) or maxima (False), by default True.
    model_member_name : str, optional
        Name of the column to use as the member identifier in the model DataFrame, by default "member".
    figsize : tuple, optional
        Figure size, by default (10, 5).

    Returns
    -------
    None
    """
    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Loop ovr the unique members
    for i, member in enumerate(model_df[model_member_name].unique()):
        # Get the data for this member
        data_this = model_df[model_df[model_member_name] == member]

        # if i = 0
        if i == 0:
            if detrend_suffix is not None:
                # plot the data detrended in grey with a label
                ax.plot(
                    data_this[model_time_name],
                    data_this[f"{model_var_name}{detrend_suffix}"],
                    color="grey",
                    alpha=0.2,
                    label="Model ens dtr",
                )
            else:
                # plot the data in grey with a label
                ax.plot(
                    data_this[model_time_name],
                    data_this[model_var_name],
                    color="grey",
                    alpha=0.2,
                    label="Model ens",
                )
        else:
            if detrend_suffix is not None:
                # plot the data detrended in grey
                ax.plot(
                    data_this[model_time_name],
                    data_this[f"{model_var_name}{detrend_suffix}"],
                    color="grey",
                    alpha=0.2,
                )
            else:
                # plot the data in grey
                ax.plot(
                    data_this[model_time_name],
                    data_this[model_var_name],
                    color="grey",
                    alpha=0.2,
                )

    # Plot the observed data
    ax.plot(
        obs_df[obs_time_name],
        obs_df[obs_var_name],
        color="black",
        linestyle="--",
        label="Obs",
    )

    # if detrend suffix is not none
    if detrend_suffix is not None:
        ax.plot(
            obs_df[obs_time_name],
            obs_df[f"{obs_var_name}{detrend_suffix}"],
            color="black",
            label="Obs dtr",
        )

    if plot_min:
        # Include a solid black line for the min value of the observed data (no dt)
        ax.axhline(obs_df[obs_var_name].min(), color="black", linestyle="--")

        # if detrend suffix is not None
        if detrend_suffix is not None:
            # Include a solid black line for the min value of the observed data (dt)
            ax.axhline(obs_df[f"{obs_var_name}{detrend_suffix}"].min(), color="black")
    else:
        # Include a solid black line for the max value of the observed data (dt)
        ax.axhline(obs_df[f"{obs_var_name}"].max(), color="black", linestyle="--")

        if detrend_suffix is not None:
            # Include a solid black line for the max value of the observed data (dt)
            ax.axhline(obs_df[f"{obs_var_name}{detrend_suffix}"].max(), color="black")

    # # Include text on these lines
    # ax.text(
    #     obs_df[obs_time_name].min(),
    #     obs_df[obs_var_name].max() - 0.1,
    #     "Obs max",
    #     color="black",
    #     verticalalignment="top",
    # )

    # ax.text(
    #     obs_df[obs_time_name].min(),
    #     obs_df[f"{obs_var_name}{detrend_suffix}"].max() - 0.1,
    #     "Obs max dtr",
    #     color="black",
    #     verticalalignment="top",
    # )

    # Add a red line for the ensemble mean of the model data (no dt)
    ax.plot(
        model_df[model_time_name].unique(),
        model_df.groupby(model_time_name)[model_var_name].mean(),
        color="red",
        linestyle="--",
        label="Model ensmean",
    )

    if detrend_suffix is not None:
        # Add a red line for the ensemble mean of the model data (dt)
        ax.plot(
            model_df[model_time_name].unique(),
            model_df.groupby(model_time_name)[
                f"{model_var_name}{detrend_suffix}"
            ].mean(),
            color="red",
            label="Model ensmean dtr",
        )

    # Include gridlines
    ax.grid(True)

    # if ylim is not None
    if ylim is not None:
        # set the y-axis limits
        ax.set_ylim(ylim)

    # Include the y label
    ax.set_ylabel(ylabel)

    # Include a legend
    ax.legend(loc="upper center", ncol=3)

    # Set up the title
    ax.set_title(title)

    return None


# Define a function for plotting the time series
def plot_detrend_ts_subplots(
    obs_df_left: pd.DataFrame,
    model_df_left: pd.DataFrame,
    obs_df_right: pd.DataFrame,
    model_df_right: pd.DataFrame,
    obs_var_name_left: str,
    model_var_name_left: str,
    obs_var_name_right: str,
    model_var_name_right: str,
    obs_time_name: str,
    model_time_name: str,
    ylabel_left: str,
    ylabel_right: str,
    ylim: tuple = None,
    detrend_suffix_left: str = "_dt",
    detrend_suffix_right: str = "_dt",
    plot_min: bool = True,
    model_member_name: str = "member",
    figsize: tuple = (15, 5),
) -> None:
    """
    Plot the detrended time series for two different variables

    Parameters
    ----------
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    obs_time_name : str
        Name of the column to use as the time axis in the observed DataFrame.
    model_time_name : str
        Name of the column to use as the time axis in the model DataFrame.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.
    detrend_suffix : str, optional
        Suffix of the detrended column, by default "_dt".
    plot_min : bool, optional
        Whether to plot the minima (True) or maxima (False), by default True.
    model_member_name : str, optional
        Name of the column to use as the member identifier in the model DataFrame, by default "member".
    figsize : tuple, optional
        Figure size, by default (10, 5).

    Returns
    -------
    None
    """
    # Set up the figure
    fig, axs = plt.subplots(ncols=2, nrows=1, figsize=figsize, layout="compressed")

    # Set up lists of the dataframes to iterate over
    obs_dfs = [obs_df_left, obs_df_right]
    model_dfs = [model_df_left, model_df_right]
    obs_var_names = [obs_var_name_left, obs_var_name_right]
    model_var_names = [model_var_name_left, model_var_name_right]
    ylabels = [ylabel_left, ylabel_right]
    detrend_suffixes = [detrend_suffix_left, detrend_suffix_right]

    # Loop over the axes
    for i, ax in enumerate(axs):
        # Get the observed and model dataframes
        obs_df_this = obs_dfs[i]
        model_df_this = model_dfs[i]
        obs_var_name_this = obs_var_names[i]
        model_var_name_this = model_var_names[i]
        ylabel_this = ylabels[i]
        detrend_suffix = detrend_suffixes[i]

        # loop over the unique init years
        for y, init_year in enumerate(model_df_this["init_year"].unique()):
            # Loop ovr the unique members
            for i, member in enumerate(model_df_this[model_member_name].unique()):
                # Get the data for this member and init year
                data_this = model_df_this[
                    (model_df_this["init_year"] == init_year)
                    & (model_df_this[model_member_name] == member)
                ]

                # if i = 0
                if i == 0 and y == 0:
                    if detrend_suffix is not None:
                        # plot the data detrended in grey with a label
                        ax.plot(
                            data_this[model_time_name],
                            data_this[f"{model_var_name_this}{detrend_suffix}"],
                            color="grey",
                            alpha=0.2,
                            label="Model ens dtr",
                        )
                    else:
                        # plot the data in grey with a label
                        ax.plot(
                            data_this[model_time_name],
                            data_this[model_var_name_this],
                            color="grey",
                            alpha=0.2,
                            label="Model ens",
                        )
                else:
                    if detrend_suffix is not None:
                        # plot the data detrended in grey
                        ax.plot(
                            data_this[model_time_name],
                            data_this[f"{model_var_name_this}{detrend_suffix}"],
                            color="grey",
                            alpha=0.2,
                        )
                    else:
                        # plot the data in grey
                        ax.plot(
                            data_this[model_time_name],
                            data_this[model_var_name_this],
                            color="grey",
                            alpha=0.2,
                        )

        # Plot the observed data
        ax.plot(
            obs_df_this[obs_time_name],
            obs_df_this[obs_var_name_this],
            color="black",
            linestyle="--",
            label="Obs",
        )

        # if detrend suffix is not none
        if detrend_suffix is not None:
            ax.plot(
                obs_df_this[obs_time_name],
                obs_df_this[f"{obs_var_name_this}{detrend_suffix}"],
                color="black",
                label="Obs dtr",
            )

        if plot_min:
            # Include a solid black line for the min value of the observed data (no dt)
            ax.axhline(obs_df_this[obs_var_name_this].min(), color="black", linestyle="--")

            # if detrend suffix is not None
            if detrend_suffix is not None:
                # Include a solid black line for the min value of the observed data (dt)
                ax.axhline(obs_df_this[f"{obs_var_name_this}{detrend_suffix}"].min(), color="black")
        else:
            # Include a solid black line for the max value of the observed data (dt)
            ax.axhline(obs_df_this[f"{obs_var_name_this}"].max(), color="black", linestyle="--")

            if detrend_suffix is not None:
                # Include a solid black line for the max value of the observed data (dt)
                ax.axhline(obs_df_this[f"{obs_var_name_this}{detrend_suffix}"].max(), color="black")

        # # Include text on these lines
        # ax.text(
        #     obs_df[obs_time_name].min(),
        #     obs_df[obs_var_name].max() - 0.1,
        #     "Obs max",
        #     color="black",
        #     verticalalignment="top",
        # )

        # ax.text(
        #     obs_df[obs_time_name].min(),
        #     obs_df[f"{obs_var_name}{detrend_suffix}"].max() - 0.1,
        #     "Obs max dtr",
        #     color="black",
        #     verticalalignment="top",
        # )

        # Add a red line for the ensemble mean of the model data (no dt)
        ax.plot(
            model_df_this[model_time_name].unique(),
            model_df_this.groupby(model_time_name)[model_var_name_this].mean(),
            color="red",
            linestyle="--",
            label="Model ensmean",
        )

        if detrend_suffix is not None:
            # Add a red line for the ensemble mean of the model data (dt)
            ax.plot(
                model_df_this[model_time_name].unique(),
                model_df_this.groupby(model_time_name)[
                    f"{model_var_name_this}{detrend_suffix}"
                ].mean(),
                color="red",
                label="Model ensmean dtr",
            )

        # Include gridlines
        ax.grid(True)

        # if ylim is not None
        if ylim is not None:
            # set the y-axis limits
            ax.set_ylim(ylim)

        # Include the y label
        ax.set_ylabel(ylabel_this)

        # Include a legend
        ax.legend(loc="upper center", ncol=3)

    return None

# Define a function to plot the scatter cmap plots
def plot_scatter_cmap(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    obs_x_var_name: str,
    obs_y_var_name: str,
    obs_cmap_var_name: str,
    model_x_var_name: str,
    model_y_var_name: str,
    model_cmap_var_name: str,
    xlabel: str,
    ylabel: str,
    cmap_label: str,
    sup_title: str,
    xlims: tuple = None,
    obs_title="Observed",
    model_title="Model",
    cmap: str = "viridis_r",
    figsize: tuple = (10, 5),
) -> None:
    """
    Plots a colormap scatter plot.

    Parameters
    ----------
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    obs_x_var_name : str
        Name of the column to use as the x-axis in the observed DataFrame.
    obs_y_var_name : str
        Name of the column to use as the y-axis in the observed DataFrame.
    obs_cmap_var_name : str
        Name of the column to use as the colormap in the observed DataFrame.
    model_x_var_name : str
        Name of the column to use as the x-axis in the model DataFrame.
    model_y_var_name : str
        Name of the column to use as the y-axis in the model DataFrame.
    model_cmap_var_name : str
        Name of the column to use as the colormap in the model DataFrame.
    xlabel : str
        Label for the x-axis.
    ylabel : str
        Label for the y-axis.
    cmap_label : str
        Label for the colormap.
    sup_title : str
        Title of the plot.
    xlims : tuple, optional
        Limits for the x-axis, by default None.
    obs_title : str, optional
        Title for the observed data, by default "Observed".
    model_title : str, optional
        Title for the model data, by default "Model".
    cmap : str, optional
        Colormap to use, by default "viridis_r".
    figsize : tuple, optional
        Figure size, by default (10, 5).

    Returns
    -------

    None

    """

    # Set up the figure
    # as 1 row and 2 columns
    fig, ax0 = plt.subplots(
        nrows=1,
        ncols=1,
        figsize=figsize,
    )

    # process the data for the obs as standardised anomalies
    obs_x_stand_anoms = (
        obs_df[obs_x_var_name] - obs_df[obs_x_var_name].mean()
    ) / obs_df[obs_x_var_name].std()
    obs_y_stand_anoms = (
        obs_df[obs_y_var_name] - obs_df[obs_y_var_name].mean()
    ) / obs_df[obs_y_var_name].std()

    # Process the obs cmap variable
    obs_cmap_stand_anoms = (
        obs_df[obs_cmap_var_name] - obs_df[obs_cmap_var_name].mean()
    ) / obs_df[obs_cmap_var_name].std()

    # Include a vertical dashed line for the mean x variable
    ax0.axvline(0, color="black", linestyle="--")

    # Include a horizontal dashed line for the mean y variable
    ax0.axhline(0, color="black", linestyle="--")

    # Set the title
    ax0.set_title(obs_title)

    # Set the x label
    ax0.set_xlabel(xlabel)

    # Set the y label
    ax0.set_ylabel(ylabel)

    # if ylims is not None
    if xlims is not None:
        # set the y-axis limits
        ax0.set_xlim(xlims)

    # Set up the x_var and y_var for the model
    x_var_model = model_df[model_x_var_name]
    y_var_model = model_df[model_y_var_name]

    # standardise the model data
    x_var_model = (x_var_model - x_var_model.mean()) / x_var_model.std()
    y_var_model = (y_var_model - y_var_model.mean()) / y_var_model.std()

    # standardise the cmap variable
    cmap_var_model = model_df[model_cmap_var_name]
    cmap_var_model = (cmap_var_model - cmap_var_model.mean()) / cmap_var_model.std()

    # Perform kernel density estimate
    xy = np.vstack([x_var_model, y_var_model])
    kde = gaussian_kde(xy)
    xmin, xmax = x_var_model.min(), x_var_model.max()
    ymin, ymax = y_var_model.min(), y_var_model.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    # Plot the scatter for the model data
    sc1 = ax0.scatter(
        x_var_model,
        y_var_model,
        c=cmap_var_model,
        cmap=cmap,
        s=10,
        alpha=0.5,
        label="model",
    )

    # Plot the density contours
    ax0.contour(X, Y, Z, levels=5, colors="black")

    # Create the scatter plot
    sc0 = ax0.scatter(
        obs_x_stand_anoms,
        obs_y_stand_anoms,
        c=obs_cmap_stand_anoms,
        cmap=cmap,
        s=100,
        marker="*",
        edgecolors="black",
        label="obs",
    )

    # find the index of the 2010 value
    idx_2010 = obs_df.index.get_loc(2010)

    # apply the index to the stand anoms
    obs_x_stand_anoms_2010 = obs_x_stand_anoms.iloc[idx_2010]
    obs_y_stand_anoms_2010 = obs_y_stand_anoms.iloc[idx_2010]

    # Include text for the 2010 point
    ax0.text(
        obs_x_stand_anoms_2010 + 0.2,
        obs_y_stand_anoms_2010 + 0.1,
        "2010",
        color="red",
        verticalalignment="top",
    )

    # Include a vertical dashed line for the mean x variable
    ax0.axvline(0, color="black", linestyle="--")

    # Include a horizontal dashed line for the mean y variable
    ax0.axhline(0, color="black", linestyle="--")

    # Set the title
    ax0.set_title(model_title)

    # Set the x label
    ax0.set_xlabel(xlabel)

    # if ylims is not None
    if xlims is not None:
        # set the y-axis limits
        ax0.set_xlim(xlims)

    # # Set up a tight layout before adding the colorbar
    # fig.tight_layout()

    # Add the colorbar after setting up the tight layout
    cbar = fig.colorbar(sc0, ax=ax0, orientation="vertical", pad=0.02)

    # Set the label for the colorbar
    cbar.set_label(cmap_label)

    # Set the super title
    fig.suptitle(sup_title, y=1.02)

    # Adjust the layout again if necessary
    plt.subplots_adjust(top=0.9)

    # Include a legend in the top left
    ax0.legend(loc="upper left")

    return None


# Define a function for extracting cubes for a dataframe
def extract_sel_cubes(
    model_df: pd.DataFrame,
    model_var_name: str,
    model_time_name: str = "init_year",
    model_lead_name: str = "lead",
    model_member_name: str = "member",
    freq: str = "day",
):
    """
    Extracts the cubes for a DataFrame.

    Parameters
    ----------

    model_df : pd.DataFrame
        DataFrame of model data.
    model_var_name : str
        Name of the column to use in the model DataFrame.
        E.g. "tas", "sfcWind", etc.
    model_time_name : str, optional
        Name of the column to use as the time axis in the model DataFrame, by default "init_year".
    model_lead_name : str, optional
        Name of the column to use as the lead time axis in the model DataFrame, by default "lead".
    model_member_name : str, optional
        Name of the column to use as the member identifier in the model DataFrame, by default "member".
    freq : str, optional
        Frequency of the data, by default "day".

    Returns
    -------

    iris.cube
        Cube of the data.

    """

    # Hard code the base path
    base_path = "/badc/cmip6/data/CMIP6/DCPP/MOHC/HadGEM3-GC31-MM/dcppA-hindcast/"

    # Extract the unique times
    unique_times = model_df[model_time_name].unique()

    # Set up an empty cube list
    cube_list = []

    # set up an empty dates list
    dates = []

    # Loop over the unique times
    for i, iyear_this in tqdm(enumerate(unique_times)):
        # Subset the df to this init year
        model_df_subset_this = model_df[model_df[model_time_name] == iyear_this]
        for j, row in model_df_subset_this.iterrows():
            # Extract the init_year, member and lead_time
            init_year = int(row[model_time_name])
            member = int(row[model_member_name])
            lead_time = int(row[model_lead_name])

            # Set up the path
            path_this = (
                f"s{init_year}-r{member}i1p1f2/{freq}/{model_var_name}/gn/files/d*/"
            )

            # If the lead time is 60 or less
            if lead_time <= 60:
                # Set up the fname
                fname = f"{model_var_name}_{freq}_HadGEM3-GC31-MM_dcppA-hindcast_s{init_year}-r{member}i1p1f2_gn_{init_year}1101-{init_year}1230.nc"
            else:
                # Set up the lead time - base
                lead_time_minus_base = lead_time - 60

                # Divide this by 360
                nyears = lead_time_minus_base // 360

                # Set up the year to extract
                year_to_extract = init_year + nyears

                # Set up the base lead year
                base_lead_year = 60 + (nyears * 360)

                # Find the lead time this
                lead_time_this = lead_time - base_lead_year

                # if nyears == 10
                if nyears == 10:
                    # Set up the fname
                    fname = f"{model_var_name}_{freq}_HadGEM3-GC31-MM_dcppA-hindcast_s{init_year}-r{member}i1p1f2_gn_{year_to_extract + 1}0101-{year_to_extract + 1}0330.nc"
                else:
                    # Set up the fname
                    fname = f"{model_var_name}_{freq}_HadGEM3-GC31-MM_dcppA-hindcast_s{init_year}-r{member}i1p1f2_gn_{year_to_extract + 1}0101-{year_to_extract + 1}1230.nc"

            # Set up the full path
            full_path = os.path.join(base_path, path_this, fname)

            # glob the files
            files_this = glob.glob(full_path)

            # if the length of files_this is not 1
            if len(files_this) != 1:
                print(f"files: {files_this}")
                print(f"full_path: {full_path}")
                print(f"init_year: {init_year}")
                print(f"member: {member}")
                print(f"lead_time: {lead_time}")
                print(f"nyears: {nyears}")
                print(f"year_this: {year_to_extract}")

                print("Length of files_this is not 1")
                print("Exiting")
                sys.exit(1)

            # Load the cube
            cube = iris.load_cube(files_this[0], model_var_name)

            # Set up the realization coord
            realization_coord = iris.coords.AuxCoord(
                np.int32(member),
                "realization",
                units="1",
            )

            # add aux coords for the member
            cube.add_aux_coord(realization_coord)

            # Set up the index to extract
            if lead_time <= 60:
                # Extract the index lead - 1
                index = lead_time - 1
            else:
                # extract the date
                index = lead_time_this - 1

            # Extract the date
            date_this = cube.coord("time").points[index]
            date_this = cftime.num2date(
                date_this,
                cube.coord("time").units.origin,
                cube.coord("time").units.calendar,
            )

            # if the date_this is not in the dates list, append it
            if date_this not in dates:
                dates.append(date_this)
            else:
                print(f"Date {date_this} already in the dates list")

            # apply this to the cube
            cube_subset = cube[index]

            # Append the cube to the list
            cube_list.append(cube_subset)

    # Set up the list of cubes as a cube list
    cubes = iris.cube.CubeList(cube_list)

    # equalise the attributes
    attrs = equalise_attributes(cubes)

    # Merge the cube list
    cubes_merged = cubes.merge_cube()

    return cubes_merged


# Write a function to plot the lead time dependent drift
def plot_lead_drift(
    model_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    model_var_name: str,
    obs_var_name: str,
    lead_name: str,
    ylabel: str,
    title: str,
    figsize: tuple = (10, 5),
) -> None:
    """
    Plots the lead time dependent drift for the model data. Also plots the observed data to the side.

    Parameters
    ----------

    model_df : pd.DataFrame
        DataFrame of model data.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    lead_name : str
        Name of the column to use as the lead time axis in the model DataFrame.
    ylabel : str
        Label for the y-axis.
    title : str
        Title of the plot.

    Returns
    -------

    None

    """

    # Set up the figure and the axes
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [8, 1]},
    )

    # Set up the axes
    ax0 = axs[0]
    ax1 = axs[1]

    # Get the unique lead times
    unique_leads = sorted(model_df[lead_name].unique())

    # Loop over the lead names
    for lead in unique_leads:
        # Subset the model data
        model_df_lead_this = model_df[model_df[lead_name] == lead]

        # Plot the boxplot
        ax0.boxplot(
            model_df_lead_this[model_var_name],
            positions=[lead],
            widths=0.8,
            showfliers=True,
            boxprops=dict(color="red", facecolor="none"),
            capprops=dict(color="red"),
            whiskerprops=dict(color="red"),
            flierprops=dict(markerfacecolor="red", marker="."),
            medianprops=dict(color="red"),
            whis=[1, 99],  # the 0th and 100th percentiles (i.e. min and max)
            patch_artist=True,
        )

    # Set the x-axis label
    ax0.set_xlabel("Lead year")

    # Set the y-axis label
    ax0.set_ylabel(ylabel)

    # Set the title
    ax0.set_title(title)

    # Plot the observed data
    ax1.boxplot(
        obs_df[obs_var_name],
        positions=[1],
        widths=0.8,
        showfliers=True,
        boxprops=dict(color="black", facecolor="none"),
        capprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        flierprops=dict(markerfacecolor="black", marker="."),
        medianprops=dict(color="black"),
        whis=[1, 99],  # the 0th and 100th percentiles (i.e. min and max)
        patch_artist=True,
    )

    # Set a title for ax1 for the obs
    ax1.set_title("Obs")

    # Specify a tight layout
    plt.tight_layout()

    return None


# Define a function for plotting the distributions for each lead time relative
# to the observatrions
def plot_lead_pdfs(
    model_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    model_var_name: str,
    obs_var_name: str,
    lead_name: str,
    xlabel: str,
    suptitle: str,
    figsize: tuple = (10, 10),
) -> None:
    """
    Plots the probability density functions for each lead time relative to the observations.

    Parameters
    ----------

    model_df : pd.DataFrame
        DataFrame of model data.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    lead_name : str
        Name of the column to use as the lead time axis in the model DataFrame.
    xlabel : str
        Label for the x-axis.
    suptitle : str
        Title of the plot.

    Returns
    -------

    None

    """

    # Set up the figure with three rows and 4 columns
    fig, axs = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    # Flatten the axes
    axs_flat = axs.flatten()

    # Get the unique lead times
    unique_leads = sorted(model_df[lead_name].unique())

    # Loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Subset the model data
        model_df_lead_this = model_df[model_df[lead_name] == lead]

        # Extract the unique effective dec years for the model
        unique_eff_dec_years_model = model_df_lead_this["effective_dec_year"].unique()

        # Extract the unique effective dec years for the obs
        unique_eff_dec_years_obs = obs_df["effective_dec_year"].unique()

        obs_df_lead_this = None

        # if the unique eff dec years for the model and obs are not the same
        if not np.array_equal(unique_eff_dec_years_model, unique_eff_dec_years_obs):
            print("Unique effective decadal years for model and obs are not the same")
            # Subset the obs this to the unique eff dec years for the model
            obs_df_lead_this = obs_df[
                obs_df["effective_dec_year"].isin(unique_eff_dec_years_model)
            ]

        # Plot the model distribution
        axs_flat[i].hist(
            model_df_lead_this[model_var_name],
            bins=20,
            color="red",
            alpha=0.5,
            label="model",
            density=True,
        )

        # if obs_df_lead_this is not None
        if obs_df_lead_this is not None:
            # Plot the observed distribution
            axs_flat[i].hist(
                obs_df_lead_this[obs_var_name],
                bins=20,
                color="black",
                alpha=0.5,
                label="obs",
                density=True,
            )

            # Calculate the mean bias
            bias = (
                model_df_lead_this[model_var_name].mean()
                - obs_df_lead_this[obs_var_name].mean()
            )
        else:
            # Plot the observed distribution
            axs_flat[i].hist(
                obs_df[obs_var_name],
                bins=20,
                color="black",
                alpha=0.5,
                label="obs",
                density=True,
            )

            # Calculate the mean bias
            bias = (
                model_df_lead_this[model_var_name].mean() - obs_df[obs_var_name].mean()
            )

        # Set the title
        axs_flat[i].set_title(f"lead {lead}, bias: {bias:.2f}")

        # Remove the y-axis ticks
        axs_flat[i].yaxis.set_ticks([])

        # if the axis is on the bottom row, then set the xlabel
        if i >= 8:
            axs_flat[i].set_xlabel(xlabel)

        # if i is 11, remove the plot
        if i == 11:
            axs_flat[i].axis("off")

    # Set up the suptitle
    fig.suptitle(suptitle)

    # Set a tight layout
    plt.tight_layout()

    return None


# Write a function to compare the trends in full field
# and block minima or maxima
def compare_trends(
    model_df_full_field: pd.DataFrame,
    obs_df_full_field: pd.DataFrame,
    model_df_block: pd.DataFrame,
    obs_df_block: pd.DataFrame,
    model_var_name_full_field: str,
    obs_var_name_full_field: str,
    model_var_name_block: str,
    obs_var_name_block: str,
    model_time_name: str,
    obs_time_name: str,
    ylabel: str,
    suptitle: str,
    member_name: str = "member",
    figsize: tuple = (10, 5),
    window_size: int = 5,
    centred_bool: bool = True,
    min_periods: int = 1,
) -> None:
    """
    Compares the trends in the full field and block minima or maxima.

    Parameters
    ----------

    model_df_full_field : pd.DataFrame
        DataFrame of model data for the full field.
    obs_df_full_field : pd.DataFrame
        DataFrame of observed data for the full field.
    model_df_block : pd.DataFrame
        DataFrame of model data for the block minima or maxima.
    obs_df_block : pd.DataFrame
        DataFrame of observed data for the block minima or maxima.
    model_var_name_full_field : str
        Name of the column to use in the model DataFrame for the full field.
    obs_var_name_full_field : str
        Name of the column to use in the observed DataFrame for the full field.
    model_var_name_block : str
        Name of the column to use in the model DataFrame for the block minima or maxima.
    obs_var_name_block : str
        Name of the column to use in the observed DataFrame for the block minima or maxima.
    model_time_name : str
        Name of the column to use as the time axis in the model DataFrame.
    obs_time_name : str
        Name of the column to use as the time axis in the observed DataFrame.
    ylabel : str
        Label for the y-axis.
    suptitle : str
        Title of the plot.
    member_name : str, optional
        Name of the column to use as the member identifier in the model DataFrame, by default "member".
    figsize : tuple, optional
        Figure size, by default (10, 5).
    window_size : int, optional
        Window size for the rolling mean, by default 5.
    centred_bool : bool, optional
        Whether to centre the rolling mean, by default True.
    min_periods : int, optional
        Minimum number of periods for the rolling mean, by default 1.

    Returns
    -------

    None

    """

    # Set up the figure as one row and two columns
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [1, 1]},
    )

    # Set up the axes
    ax0 = axs[0]
    ax1 = axs[1]

    # group by effecteive dec yuear and take the mean
    obs_df_full_field_grouped = obs_df_full_field.groupby("effective_dec_year")[
        obs_var_name_full_field
    ].mean()

    # print the had of the opbs df full field grouped
    print(obs_df_full_field_grouped.head())

    # Plot the full field data for the observations
    ax0.plot(
        obs_df_full_field[obs_time_name].unique(),
        obs_df_full_field_grouped,
        color="black",
        label="obs",
    )

    # quantify linear trend for the obs
    slope_obs_ff, intercept_obs_ff, _, _, _ = linregress(
        obs_df_full_field[obs_time_name].unique(), obs_df_full_field_grouped
    )

    # print the slope and intercept
    print(f"obs slope: {slope_obs_ff}, obs intercept: {intercept_obs_ff}")

    # calclate the trend line
    trend_line_obs_ff = (
        slope_obs_ff * obs_df_full_field[obs_time_name].unique() + intercept_obs_ff
    )

    # Calculate the obs 5 year rolling trendline
    rolling_trend_line_obs_ff = obs_df_full_field_grouped.rolling(
        window=window_size, center=centred_bool, min_periods=min_periods
    ).mean()

    # Plot this line as a black dot dashed line
    ax0.plot(
        obs_df_full_field[obs_time_name].unique(),
        rolling_trend_line_obs_ff,
        color="black",
        linestyle="-.",
    )

    # # calculate the final point
    # final_point_obs_ff = trend_line_obs_ff.iloc[-1]

    # # pivot the trend line
    # trend_line_obs_ff_pivot = final_point_obs_ff - trend_line_obs_ff + obs_df_full_field[obs_var_name_full_field]

    # plot this line as a dashed black line
    ax0.plot(
        obs_df_full_field[obs_time_name].unique(),
        trend_line_obs_ff,
        color="black",
        linestyle="--",
    )

    # plot the full field data for the model
    # taking the ensemble mean
    ax0.plot(
        model_df_full_field[model_time_name].unique(),
        model_df_full_field.groupby(model_time_name)[model_var_name_full_field].mean(),
        color="red",
        label="model ensmean",
    )

    # Set up the slopes
    slopes_model_ff = np.zeros(model_df_full_field[member_name].nunique())
    intercepts_model_ff = np.zeros(model_df_full_field[member_name].nunique())

    # Loop over the members
    for i, member in enumerate(model_df_full_field[member_name].unique()):
        # Get the data for this member
        data_this = model_df_full_field[model_df_full_field[member_name] == member]

        # quantify linear trend for the model
        slope_model_ff, intercept_model_ff, _, _, _ = linregress(
            data_this[model_time_name], data_this[model_var_name_full_field]
        )

        # store the slopes and intercepts
        slopes_model_ff[i] = slope_model_ff
        intercepts_model_ff[i] = intercept_model_ff

    # calculate the mean slope and intercept
    slope_model_ff_mean = np.mean(slopes_model_ff)
    intercept_model_ff_mean = np.mean(intercepts_model_ff)

    # calculate the 5th and 95th percentiles
    slope_model_ff_5th = np.percentile(slopes_model_ff, 5)
    slope_model_ff_95th = np.percentile(slopes_model_ff, 95)

    # calculate the CI
    ci_model_ff = (slope_model_ff_95th - slope_model_ff_5th) / 2

    # plot the ensemble mean slope
    ax0.plot(
        model_df_full_field[model_time_name].unique(),
        slope_model_ff_mean * model_df_full_field[model_time_name].unique()
        + intercept_model_ff_mean,
        color="red",
        linestyle="--",
    )

    # Calculate the model ensemble mean
    model_ensmean_ff = model_df_full_field.groupby(model_time_name)[
        model_var_name_full_field
    ].mean()

    # Calculate the rolling mean of this trend line
    rolling_trend_line_model_ff = model_ensmean_ff.rolling(
        window=window_size, center=centred_bool, min_periods=min_periods
    ).mean()

    # Plot this as a red dot dashed line
    ax0.plot(
        model_df_full_field[model_time_name].unique(),
        rolling_trend_line_model_ff,
        color="red",
        linestyle="-.",
    )

    # Set the title for ax0
    ax0.set_title(
        f"Full field, obs slope: {slope_obs_ff:.3f}, model slope: {slope_model_ff_mean:.3f} (+/- {ci_model_ff:.3f})",
        fontweight="bold",
    )

    # Plot the block data for the observations
    ax1.plot(
        obs_df_block[obs_time_name],
        obs_df_block[obs_var_name_block],
        color="black",
        label="obs",
    )

    # quantify linear trend for the obs
    slope_obs_block, intercept_obs_block, _, _, _ = linregress(
        obs_df_block[obs_time_name], obs_df_block[obs_var_name_block]
    )

    # print the slope and intercept
    print(f"obs slope: {slope_obs_block}, obs intercept: {intercept_obs_block}")

    # calclate the trend line
    trend_line_obs_block = (
        slope_obs_block * obs_df_block[obs_time_name] + intercept_obs_block
    )

    # plot the trend line
    ax1.plot(
        obs_df_block[obs_time_name], trend_line_obs_block, color="black", linestyle="--"
    )

    # quantify the 5 year rolling trend for the observatsion
    rolling_trend_line_obs_block = (
        obs_df_block[obs_var_name_block]
        .rolling(window=window_size, center=centred_bool, min_periods=min_periods)
        .mean()
    )

    # plot this line as a black dot dashed line
    ax1.plot(
        obs_df_block[obs_time_name],
        rolling_trend_line_obs_block,
        color="black",
        linestyle="-.",
    )

    # plot the block data for the model
    # taking the ensemble mean
    ax1.plot(
        model_df_block[model_time_name].unique(),
        model_df_block.groupby(model_time_name)[model_var_name_block].mean(),
        color="red",
        label="model ensmean",
    )

    # Set up the slopes
    slopes_model_block = np.zeros(model_df_block[member_name].nunique())
    intercepts_model_block = np.zeros(model_df_block[member_name].nunique())

    # Loop over the members
    for i, member in enumerate(model_df_block[member_name].unique()):
        # Get the data for this member
        data_this = model_df_block[model_df_block[member_name] == member]

        # quantify linear trend for the model
        slope_model_block, intercept_model_block, _, _, _ = linregress(
            data_this[model_time_name], data_this[model_var_name_block]
        )

        # store the slopes
        slopes_model_block[i] = slope_model_block
        intercepts_model_block[i] = intercept_model_block

    # calculate the mean slope and intercept
    slope_model_block_mean = np.mean(slopes_model_block)
    intercept_model_block_mean = np.mean(intercepts_model_block)

    # calculate the 5th and 95th percentiles
    slope_model_block_5th = np.percentile(slopes_model_block, 5)
    slope_model_block_95th = np.percentile(slopes_model_block, 95)

    # calculate the CI
    ci_model_block = (slope_model_block_95th - slope_model_block_5th) / 2

    # plot the ensemble mean slope
    ax1.plot(
        model_df_block[model_time_name].unique(),
        slope_model_block_mean * model_df_block[model_time_name].unique()
        + intercept_model_block_mean,
        color="red",
        linestyle="--",
    )

    # Calculate the model ensemble mean
    model_ensmean_block = model_df_block.groupby(model_time_name)[
        model_var_name_block
    ].mean()

    # Calculate the rolling mean of this trend line
    rolling_trend_line_model_block = model_ensmean_block.rolling(
        window=window_size, center=centred_bool, min_periods=min_periods
    ).mean()

    # Plot this as a red dot dashed line
    ax1.plot(
        model_df_block[model_time_name].unique(),
        rolling_trend_line_model_block,
        color="red",
        linestyle="-.",
    )

    # Set the title for ax1
    ax1.set_title(
        f"Block minima/maxima, obs slope: {slope_obs_block:.3f}, model slope: {slope_model_block_mean:.3f} (+/- {ci_model_block:.3f})",
        fontweight="bold",
    )

    # Set the ylabel
    ax0.set_ylabel(ylabel)

    # Set up t6he suptitle
    fig.suptitle(suptitle, fontweight="bold")

    # Set a tight layout
    plt.tight_layout()

    return None


# Set up a function for plotting the relation between variables
def plot_rel_var(
    obs_df: pd.DataFrame,
    model_df: pd.DataFrame,
    model_df_bc: pd.DataFrame,
    obs_var_names: Tuple[str, ...],
    model_var_names: Tuple[str, ...],
    model_var_names_bc: Tuple[str, ...],
    row_title: str,
    figsize: tuple = (10, 5),
) -> None:
    """
    Function which plots the relationship between varaibles.

    Parameters
    ==========

    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_df : pd.DataFrame
        DataFrame of model data.
    model_df_bc : pd.DataFrame
        DataFrame of model data for the block minima or maxima.
    obs_var_names : tuple(str)
        Name of the column to use in the observed DataFrame.
    model_var_names : tuple(str)
        Name of the column to use in the model DataFrame.
    model_var_names_bc : tuple(str)
        Name of the column to use in the model DataFrame for the block minima or maxima.
    row_title : str
        Title for the row.
    subplot_titles : tuple(str)
        Titles for the subplots.
    figsize : tuple, optional
        Figure size, by default (10, 5).

    Returns
    =======

    None

    """

    # Set up the figure with nrows = 1 and ncols = 3
    fig, axs = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=figsize,
        sharey=True,
        sharex=True,
        gridspec_kw={"width_ratios": [1, 1, 1]},
    )

    # Set up the axes
    ax0 = axs[0]
    ax1 = axs[1]
    ax2 = axs[2]

    # Plot the scatter of tuple[0] against tuple[1] for the observations
    # Plot the hexbin of tuple[0] against tuple[1] for the observations
    ax0.hexbin(
        obs_df[obs_var_names[0]],
        obs_df[obs_var_names[1]],
        gridsize=30,
        cmap="winter_r",
        mincnt=1,
    )

    # # Add a color bar to show the density scale
    # cb = fig.colorbar(hb, ax=ax0, label='Counts')

    # Set the title for ax0
    ax0.set_title("a) ERA5")

    # Set up the row label
    ax0.set_ylabel(row_title, fontweight="bold", rotation=90, fontsize=12)

    # Plot the scatter of tuple[0] against tuple[1] for the model
    ax1.hexbin(
        model_df[model_var_names[0]],
        model_df[model_var_names[1]],
        gridsize=50,
        cmap="autumn_r",
        mincnt=1,
    )

    # Set the title for ax1
    ax1.set_title("b) Raw model")

    # Plot the scatter of tuple[0] against tuple[1] for the model block minima or maxima
    ax2.hexbin(
        model_df_bc[model_var_names_bc[0]],
        model_df_bc[model_var_names_bc[1]],
        gridsize=50,
        cmap="autumn_r",
        mincnt=1,
    )

    # Set the title for ax2
    ax2.set_title("c) Bias corrected model")

    # Set a tight layout
    plt.tight_layout()

    return None

# Define a function for plotting the return periods for a given decade
def plot_return_periods_decades(
    model_df: pd.DataFrame,
    model_var_name: str,
    obs_df: pd.DataFrame,
    obs_var_name: str,
    decades: np.ndarray,
    title: str,
    model_year_name: str = "effective_dec_year",
    num_samples: int = 1000,
    figsize: tuple = (10, 5),
    bad_min: bool = True,
) -> None:
    """
    Estimates the return period of extremes for a given decade.
    
    Parameters
    ----------
    
    model_df : pd.DataFrame
        DataFrame of model data.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame
    decades : np.ndarray
        Array of decades to use.
    title : str
        Title of the plot.
    model_year_name : str, optional
        Name of the column to use as the year axis in the model DataFrame, by default "effective_dec_year".
    num_samples : int, optional
        Number of samples to use for the bootstrapping, by default 1000.
    figsize : tuple, optional
        Figure size, by default (10, 5).
    bad_min : bool, optional
        Whether to use the block minima or maxima, by default True.
        True for minima, False for maxima.
        
    Returns
    -------
    
    None
    
    """

    # Sort out the decade years
    decade_years = []

    # Set up the decade years
    for i, decade in enumerate(decades):
        if i == 0:
            decade_years.append(np.arange(decade, decade + 11))
        else:
            decade_years.append(np.arange(decade + 1, decade + 11))

    # Flatten the list of arrays into a single list
    decade_years_flat = np.concatenate(decade_years)

    # Filter out the years not in the model data
    filtered_years = [year for year in decade_years_flat if year in model_df[model_year_name].unique()]

    # Reshape the filtered list back into the original decade structure
    filtered_decade_years = []
    for i, decade in enumerate(decades):
        if i == 0:
            filtered_decade_years.append(np.array([year for year in filtered_years if 1961 <= year <= 1970]))
        else:
            filtered_decade_years.append(np.array([year for year in filtered_years if decade + 1 <= year <= decade + 10]))

    # Extract the unique winter years from the model df
    unique_winter_years = model_df[model_year_name].unique()

    # Set up the min or max bad yea
    if bad_min:
        bad_year = obs_df.loc[
            obs_df[obs_var_name].idxmin()
        ]
    else:
        bad_year = obs_df.loc[
            obs_df[obs_var_name].idxmax()
        ]

    # Print the bad year
    print(f"Bad year: {bad_year}")

    # print the value on this bad year
    print(f"Value: {bad_year[obs_var_name]}")

    # Set up the params for the decade
    decade_params = np.zeros([
        len(filtered_decade_years),
        num_samples,
        3
    ])

    # print the filtered decade years
    print(f"Filtered decade years: {filtered_decade_years}")

    # Loop over the unique winter years
    for i, decade in tqdm(enumerate(filtered_decade_years)):
        # Subset the model data to this decade
        model_df_decade = model_df[model_df[model_year_name].isin(decade)]

        # Set up the params for this decade
        params_decade_this = np.zeros([num_samples, 3])

        # # print the decade
        # print(f"Decade: {decade}")

        # # print the model df decade head
        # print(model_df_decade.head())

        # Loop over the number of samples
        for j in range(num_samples):
            params_decade_this[j, :] = gev.fit(
            np.random.choice(
                model_df_decade[model_var_name],
                size=len(model_df_decade),
                replace=True,
                )
            )
        
        # Append the params to the model year
        decade_params[i, :, :] = params_decade_this

    # Set up arrays for the mean, 2.5th and 97.5th percentiles
    period_decade_mean = np.zeros([len(filtered_decade_years)])
    period_decade_0025 = np.zeros([len(filtered_decade_years)])
    period_decade_0975 = np.zeros([len(filtered_decade_years)])

    # Find the value for the obs worst event
    worst_obs_event_value = bad_year[obs_var_name]

    # Loop over the unique years to fit the ppfs
    for i, decade in tqdm(enumerate(filtered_decade_years)):
        # Subset to the params for this decade
        params_this = decade_params[i, :, :]

        # Estimate the period for the mean
        period_decade_mean[i] = estimate_period(
            return_level=worst_obs_event_value,
            loc=np.mean(params_this[:, 1]),
            scale=np.mean(params_this[:, 2]),
            shape=np.mean(params_this[:, 0]),
        )

        # Estimate the period for the 2.5th percentile
        period_decade_0025[i] = estimate_period(
            return_level=worst_obs_event_value,
            loc=np.percentile(params_this[:, 1], 2.5),
            scale=np.percentile(params_this[:, 2], 2.5),
            shape=np.percentile(params_this[:, 0], 2.5),
        )

        # Estimate the period for the 97.5th percentile
        period_decade_0975[i] = estimate_period(
            return_level=worst_obs_event_value,
            loc=np.percentile(params_this[:, 1], 97.5),
            scale=np.percentile(params_this[:, 2], 97.5),
            shape=np.percentile(params_this[:, 0], 97.5),
        )

    # Put these into a dataframe with the decades
    model_decades_rls = pd.DataFrame(
        {
            "decade": decades,
            "mean": period_decade_mean,
            "0025": period_decade_0025,
            "0975": period_decade_0975,
        }
    )

    # for each row of "mean", "025", "975" calculate the return period
    # 1 / (1 - (100 - value))
    # model_decades_rls["mean_rp (%)"] = 1 - (model_decades_rls["mean"] / 100)
    # model_decades_rls["0025_rp (%)"] = 1 - (model_decades_rls["0025"] / 100)
    # model_decades_rls["0975_rp (%)"] = 1 - (model_decades_rls["0975"] / 100)

    # calculate the return period in years of the observed worst event
    model_decades_rls["mean_rp (years)"] = 1 / (model_decades_rls["mean"] / 100)
    model_decades_rls["0025_rp (years)"] = 1 / (model_decades_rls["0025"] / 100)
    model_decades_rls["0975_rp (years)"] = 1 / (model_decades_rls["0975"] / 100)

    # save the model decades rls
    save_idr = "/home/users/benhutch/unseen_multi_year/dfs"

    # save the model decades rls
    model_decades_rls.to_csv(f"{save_idr}/model_decades_rls.csv", index=False)

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot teh mean as a scatter
    ax.scatter(
        model_decades_rls["decade"],
        model_decades_rls["mean_rp (years)"],
        color="blue",
        label="mean",
    )

    # plot the 025 and 975 percentiles
    plt.errorbar(
        model_decades_rls["decade"],
        model_decades_rls["mean_rp (years)"],
        yerr=[
            model_decades_rls["mean_rp (years)"] - model_decades_rls["0975_rp (years)"], # lower errors
            model_decades_rls["0025_rp (years)"] - model_decades_rls["mean_rp (years)"], # upper errors
        ],
        fmt="none",
        ecolor="blue",
        capsize=5,
    )

    # set up a log yscale
    plt.yscale("log")

    # Set up the y ticks
    plt.yticks([10, 100, 500, 1000], ["10", "100", "500", "1000"])

    # set up the y axis labels
    plt.ylabel("Return period (years)")

    # Set up the title
    plt.title(title)

    return None

# Write a function to compare the return period in the obs
# using all of the winter days
def plot_return_periods_decades_obs(
    obs_df: pd.DataFrame,
    obs_var_name: str,
    decades: np.ndarray,
    title: str,
    year_col_name: str = "effective_dec_year",
    num_samples: int = 1000,
    figsize: tuple = (10, 5),
    bad_min: bool = True,
) -> None:
    """
    Estimate the return periods using ths obs days for a given decade.

    Parameters
    ==========

    obs_df : pd.DataFrame
        DataFrame of observed data.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    decades : np.ndarray
        Array of decades to use.
    title : str
        Title of the plot.
    year_col_name : str, optional
        Name of the column to use as the year axis in the observed DataFrame, by default "effective_dec_year".
    num_samples : int, optional
        Number of samples to use for the bootstrapping, by default 1000.
    figsize : tuple, optional
        Figure size, by default (10, 5).
    bad_min : bool, optional
        Whether to use the block minima or maxima, by default True.
        True for minima, False for maxima.

    Returns
    =======

    None

    """
    
    # Sort out the decade years
    decade_years = []

    # Set up the decade years
    for i, decade in enumerate(decades):
        if i == 0:
            decade_years.append(np.arange(decade, decade + 11))
        else:
            decade_years.append(np.arange(decade + 1, decade + 11))

    # flatten the list of arrays into a single list
    decade_years_flat = np.concatenate(decade_years)

    # Filter out the years not in the model data
    filtered_years = [year for year in decade_years_flat if year in obs_df[year_col_name].unique()]

    # Reshape the filtered list back into the original decade structure
    filtered_decade_years = []

    for i, decade in enumerate(decades):
        if i == 0:
            filtered_decade_years.append(np.array([year for year in filtered_years if 1960 <= year <= 1970]))
        else:
            filtered_decade_years.append(np.array([year for year in filtered_years if decade + 1 <= year <= decade + 10]))

    # Set up the min or max bad year
    if bad_min:
        bad_year = obs_df.loc[
            obs_df[obs_var_name].idxmin()
        ]
    else:
        bad_year = obs_df.loc[
            obs_df[obs_var_name].idxmax()
        ]

    # Print the bad year
    print(f"Bad year: {bad_year}")

    # Print the value on this bad year
    print(f"Value: {bad_year[obs_var_name]}")

    # Set up the params for the decade
    decade_params = np.zeros([
        len(filtered_decade_years),
        num_samples,
        3
    ])

    # Loop over the unique winter years
    for i, decade in tqdm(enumerate(filtered_decade_years)):
        # Subset the obs data to this decade
        obs_df_decade = obs_df[obs_df[year_col_name].isin(decade)]

        # Set up the params for this decade
        params_decade_this = np.zeros([num_samples, 3])

        # Loop over the number of samples
        for j in range(num_samples):
            params_decade_this[j, :] = gev.fit(
                np.random.choice(
                    obs_df_decade[obs_var_name],
                    size=len(obs_df_decade),
                    replace=True,
                )
            )

        # Append the params to the model year
        decade_params[i, :, :] = params_decade_this

    # Set up arrays for the mean, 2.5th and 97.5th percentiles
    period_decade_mean = np.zeros([len(filtered_decade_years)])
    period_decade_0025 = np.zeros([len(filtered_decade_years)])
    period_decade_0975 = np.zeros([len(filtered_decade_years)])

    # Find the value for the obs worst event
    worst_obs_event_value = bad_year[obs_var_name]

    # Loop over the unique years to fit the ppfs
    for i, decade in tqdm(enumerate(filtered_decade_years)):
        # Subset to the params for this decade
        params_this = decade_params[i, :, :]

        # Estimate the period for the mean
        period_decade_mean[i] = estimate_period(
            return_level=worst_obs_event_value,
            loc=np.mean(params_this[:, 1]),
            scale=np.mean(params_this[:, 2]),
            shape=np.mean(params_this[:, 0]),
        )

        # Estimate the period for the 2.5th percentile
        period_decade_0025[i] = estimate_period(
            return_level=worst_obs_event_value,
            loc=np.percentile(params_this[:, 1], 2.5),
            scale=np.percentile(params_this[:, 2], 2.5),
            shape=np.percentile(params_this[:, 0], 2.5),
        )

        # Estimate the period for the 97.5th percentile
        period_decade_0975[i] = estimate_period(
            return_level=worst_obs_event_value,
            loc=np.percentile(params_this[:, 1], 97.5),
            scale=np.percentile(params_this[:, 2], 97.5),
            shape=np.percentile(params_this[:, 0], 97.5),
        )

    # Put these into a dataframe with the decades
    obs_decades_rls = pd.DataFrame(
        {
            "decade": decades,
            "mean": period_decade_mean,
            "0025": period_decade_0025,
            "0975": period_decade_0975,
        }
    )

    # for each row of "mean", "025", "975" calculate the return period in years
    obs_decades_rls["mean_rp (years)"] = 1 / (obs_decades_rls["mean"] / 100)
    obs_decades_rls["0025_rp (years)"] = 1 / (obs_decades_rls["0025"] / 100)
    obs_decades_rls["0975_rp (years)"] = 1 / (obs_decades_rls["0975"] / 100)

    # save the obs decades rls
    save_idr = "/home/users/benhutch/unseen_multi_year/dfs"

    # # set up the current time
    # current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # save the obs decades rls
    obs_decades_rls.to_csv(f"{save_idr}/obs_decades_rls.csv", index=False)

    # Set up the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the mean as a scatter
    ax.scatter(
        obs_decades_rls["decade"],
        obs_decades_rls["mean_rp (years)"],
        color="blue",
        label="mean",
    )

    # Plot the 025 and 975 percentiles
    plt.errorbar(
        obs_decades_rls["decade"],
        obs_decades_rls["mean_rp (years)"],
        yerr=[
            obs_decades_rls["mean_rp (years)"] - obs_decades_rls["0975_rp (years)"], # lower errors
            obs_decades_rls["0025_rp (years)"] - obs_decades_rls["mean_rp (years)"], # upper errors
        ],
        fmt="none",
        ecolor="blue",
        capsize=5,
    )

    # Set up a log yscale
    plt.yscale("log")

    # Set up the y ticks
    plt.yticks([10, 100, 500, 1000], ["10", "100", "500", "1000"])

    # Set up the y axis labels
    plt.ylabel("Return period (years)")

    # Set up the title
    plt.title(title)

    return None


# Write a function to compare the lead time dependent trends
def lead_time_trends(
    model_df: pd.DataFrame,
    obs_df: pd.DataFrame,
    model_var_name: str,
    obs_var_name: str,
    lead_name: str,
    ylabel: str,
    suptitle: str,
    figsize: tuple = (10, 5),
) -> None:
    """
    Plots the lead time dependent trends for the model and observed data.

    Parameters
    ==========

    model_df : pd.DataFrame
        DataFrame of model data.
    obs_df : pd.DataFrame
        DataFrame of observed data.
    model_var_name : str
        Name of the column to use in the model DataFrame.
    obs_var_name : str
        Name of the column to use in the observed DataFrame.
    lead_name : str
        Name of the column to use as the lead time axis in the model DataFrame.
    ylabel : str
        Label for the y-axis.
    suptitle : str
        Title of the plot.
    figsize : tuple, optional
        Figure size, by default (10, 5).

    Returns
    =======

    None
    
    """

    # create copies of the model and obs dfs
    model_df_copy = model_df.copy()
    obs_df_copy = obs_df.copy()

    # Set up teh figure with three rows and 4 columns
    fig, axs = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=figsize,
        sharex=True,
        sharey=True,
    )

    # Flatten the axes
    axs_flat = axs.flatten()

    # Get the unique lead times
    unique_leads = sorted(model_df_copy[lead_name].unique())

    # Loop over the unique leads
    for i, lead in enumerate(unique_leads):
        # Subset the model data
        model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

        # Extract the unique effective dec years for the model
        unique_eff_dec_years_model = model_df_lead_this["effective_dec_year"].unique()

        # Extract the unique effective dec years for the obs
        unique_eff_dec_years_obs = obs_df_copy["effective_dec_year"].unique()

        obs_df_lead_this = None

        # Set up the effective dec years to pool over
        eff_dec_years = np.arange(1970, 2017 + 1)

        # Subset the model data to these efefctive dec years
        model_df_lead_this = model_df_lead_this[
            model_df_lead_this["effective_dec_year"].isin(eff_dec_years)
        ]

        # Same for the obs data
        obs_df_lead_this = obs_df_copy[
            obs_df_copy["effective_dec_year"].isin(eff_dec_years)
        ]

        # extract the unique effective dec years for the model
        unique_eff_dec_years_model = model_df_lead_this["effective_dec_year"].unique()

        # extract the unique effective dec years for the obs
        unique_eff_dec_years_obs = obs_df_lead_this["effective_dec_year"].unique()

        # if the unique eff dec years for the model and obs are not the same
        if not np.array_equal(unique_eff_dec_years_model, unique_eff_dec_years_obs):
            print("Unique effective decadal years for model and obs are not the same")
            # Subset the obs this to the unique eff dec years for the model
            obs_df_lead_this = obs_df_copy[
                obs_df_copy["effective_dec_year"].isin(unique_eff_dec_years_model)
            ]

        # Plot the model scatter points as red circles
        axs_flat[i].scatter(
            model_df_lead_this["effective_dec_year"],
            model_df_lead_this[model_var_name],
            color="red",
            label="model",
        )

        # Set up the unique number of members in the model data
        nmembers = model_df_lead_this["member"].nunique()

        # Calculate the slope and intercept for the model
        slopes_model_this = np.zeros([nmembers])
        intercepts_model_this = np.zeros([nmembers])

        # print the shape of sloes model this
        print(slopes_model_this.shape)
        print(intercepts_model_this.shape)

        # Loop over the members
        for j, member in enumerate(model_df_lead_this["member"].unique()):
            # Subset the data for this member
            data_this = model_df_lead_this[model_df_lead_this["member"] == member]

            # Calculate the linear trend
            slope_model_this, intercept_model_this, _, _, _ = linregress(
                data_this["effective_dec_year"], data_this[model_var_name]
            )

            # Store the slope and intercept
            slopes_model_this[j] = slope_model_this
            intercepts_model_this[j] = intercept_model_this

        # Calculate the mean slope and intercept
        slope_model_mean_this = np.mean(slopes_model_this)
        intercept_model_mean_this = np.mean(intercept_model_this)

        # calculate the actual intercept
        _, intercept_model_this, _, _, _ = linregress(
            model_df_lead_this["effective_dec_year"], model_df_lead_this[model_var_name]
        )

        # Calculate the 5th and 95th percentiles
        slope_model_5th_this = np.percentile(slopes_model_this, 5)
        slope_model_95th_this = np.percentile(slopes_model_this, 95)

        # Calculate the CI
        ci_model_this = (slope_model_95th_this - slope_model_5th_this) / 2

        # Plot the slope as a black dashed line
        axs_flat[i].plot(
            model_df_lead_this["effective_dec_year"].unique(),
            slope_model_mean_this * model_df_lead_this["effective_dec_year"].unique()
            + intercept_model_this,
            color="red",
            linestyle="--",
        )

        # if obs_df_lead_this is not None
        if obs_df_lead_this is not None:
            # Plot the observed values as black crosses
            axs_flat[i].scatter(
                obs_df_lead_this["effective_dec_year"],
                obs_df_lead_this[obs_var_name],
                color="black",
                marker="x",
                label="obs",
            )

            # Calculate the slope and intercept for the obs
            slope_obs_this, intercept_obs_this, _, _, _ = linregress(
                obs_df_lead_this["effective_dec_year"], obs_df_lead_this[obs_var_name]
            )
        else:
            # Plot the observed values as black crosses
            axs_flat[i].scatter(
                obs_df_copy["effective_dec_year"],
                obs_df_copy[obs_var_name],
                color="black",
                marker="x",
                label="obs",
            )

            # Calculate the slope and intercept
            slope_obs_this, intercept_obs_this, _, _, _ = linregress(
                obs_df_copy["effective_dec_year"], obs_df_copy[obs_var_name]
            )

        # plot the obs slope as a black dashed line
        axs_flat[i].plot(
            obs_df_copy["effective_dec_year"].unique(),
            slope_obs_this * obs_df_copy["effective_dec_year"].unique() + intercept_obs_this,
            color="black",
            linestyle="--",
        )

        # include the lead in a text box in the top left
        axs_flat[i].text(
            0.05,
            0.95,
            f"lead {lead}",
            transform=axs_flat[i].transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5),
        )

        # Set the title
        axs_flat[i].set_title(f"model: {slope_model_mean_this:.3f} (+/- {ci_model_this:.3f}), obs: {slope_obs_this:.3f}")

        # Remove the y-axis ticks
        # axs_flat[i].yaxis.set_ticks([])

        # if the axis is on the bottom

    # Set a tight layout
    plt.tight_layout()

    return None

# Define a function to apply the lead time dependent trend removal
def lead_time_trend_corr(
    model_df: pd.DataFrame,
    x_axis_name: str,
    y_axis_name: str,
    lead_name: str,
    suffix: str = "_dt",
    member_name: str = "member",
) -> pd.DataFrame:
    """
    Removes a linear trend from each lead time for each member.

    Parameters
    ==========

    model_df : pd.DataFrame
        DataFrame of model data.
    x_axis_name : str
        Name of the column to use as the x-axis.
    y_axis_name : str
        Name of the column to use as the y-axis.
    lead_name : str
        Name of the column to use as the lead time axis.
    suffix : str, optional
        Suffix to add to the column names, by default "_dt".
    member_name : str, optional
        Name of the column to use as the member identifier, by default "member".

    Returns
    =======

    pd.DataFrame
        DataFrame with the linear trends removed.
    
    """

    # Make a copy of the dataframe
    model_df_copy = model_df.copy()

    # Extract the unique lead times
    unique_leads = model_df_copy[lead_name].unique()

    # Set up a new column in the dataframe (y_axis_name + suffix)
    # full of NaNs
    model_df_copy[y_axis_name + suffix] = np.nan

    # create a new empty df to store the new data
    model_df_dt = pd.DataFrame()

    # Loop over the unique leads
    for lead in unique_leads:        
        # Subset the model data to this lead
        model_df_lead_this = model_df_copy[model_df_copy[lead_name] == lead]

        # print the head of model df lead this
        print(model_df_lead_this.head())

        # print the tail of model df lead this
        print(model_df_lead_this.tail())

        # Set up the slopes and intercepts
        slopes = np.zeros(model_df_lead_this[member_name].nunique())

        # Loop over the members
        for i, member in enumerate(model_df_lead_this[member_name].unique()):
            # Subset the data for this member
            data_this = model_df_lead_this[model_df_lead_this[member_name] == member]

            # Calculate the linear trend
            slope, intercept, _, _, _ = linregress(data_this[x_axis_name], data_this[y_axis_name])

            # Store the slope and intercept
            slopes[i] = slope

        # Calculate the mean slope and intercept
        slope_mean = np.mean(slopes)

        # print mean slope for lead
        print(f"Mean slope for lead {lead}: {slope_mean}")

        # Calculate the intercept using all of the data
        _, intercept_mean, _, _, _ = linregress(model_df_lead_this[x_axis_name], model_df_lead_this[y_axis_name])

        # Calculate the trend line
        trend_line = slope_mean * model_df_lead_this[x_axis_name] + intercept_mean

        # # print the shape of trend line
        # print(trend_line.shape)

        # work out the final point on the trend line
        final_point = trend_line.iloc[-1]

        # # print the shape of model df copy.loc
        # print(model_df_copy.loc[model_df_copy[lead_name] == lead, y_axis_name + suffix].shape)

        # # print the values of this
        # print(model_df_copy.loc[model_df_copy[lead_name] == lead, y_axis_name + suffix])

        # Add a column containing the new detrended values to the model df this
        model_df_lead_this[y_axis_name + suffix] = final_point - trend_line + model_df_lead_this[y_axis_name]

        # print the model df lead this head
        print(model_df_lead_this.head())

        # print the model df lead this tail
        print(model_df_lead_this.tail())

        # concatenate the model df lead this to the model df dt
        model_df_dt = pd.concat([model_df_dt, model_df_lead_this])

        # # print the head of model df corr
        # print(model_df_corr.head())

        # # print the tail of model df corr
        # print(model_df_corr.tail())

    return model_df_dt

# Set up a function for the dot plot
def dot_plot_subplots(
    obs_df_left: pd.DataFrame,
    model_df_left: pd.DataFrame,
    obs_df_right: pd.DataFrame,
    model_df_right: pd.DataFrame,
    obs_val_name_left: str,
    model_val_name_left: str,
    obs_val_name_right: str,
    model_val_name_right: str,
    model_time_name: str,
    ylabel_left: str,
    ylabel_right: str,
    title_left: str,
    title_right: str,
    obs_label: str = "Observed",
    model_label: str = "Modelled",
    very_bad_label: str = "unseen events",
    bad_label: str = "extreme events",
    normal_label: str = "events",
    ylims_left: tuple = (0, 120),
    ylims_right: tuple = (0, 120),
    dashed_quant: float = 0.8,
    solid_line: Callable[[np.ndarray], float] = np.max,
    figsize: tuple = (10, 5),
    save_prefix: str = "dot_plot",
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots",
    simple_detrend: bool = False,
) -> None:
    """
    Plots the dotplot for e.g. no. exceedance days.

    Parameters
    ==========

    obs_df: pd.DataFrame
        The DataFrame for the obs

    model_df: pd.DataFrame
        The DataFrame for the model

    obs_val_name: str
        The name of the obs value column

    model_val_name: str
        The name of the model value column

    model_time_name: str
        The name of the model time column

    ylabel: str
        The y-axis label for the figure

    title: str
        The title for the figure

    obs_label: str
        The label for the obs data. Default is "Observed".

    model_label: str
        The label for the model data. Default is "Modelled".

    very_bad_label: str
        The label for the very bad events. Default is "unseen events".

    bad_label: str
        The label for the bad events. Default is "extreme events".

    normal_label: str
        The label for the normal events. Default is "events".

    ylims: tuple
        The y-axis limits. Default is (0, 120).

    dashed_quant: float
        The quantile to use for the dashed line. Default is 0.8.

    solid_line: Callable[[np.ndarray], float]
        The function to use for the solid line. Default is np.max.

    figsize: tuple
        The figure size. Default is (10, 5).

    save_prefix: str
        The prefix to use when saving the plots. Default is "dot_plot".

    save_dir: str
        The directory to save the plots to. Default is "/gws/nopw/j04/canari/users/benhutch/plots".

    simple_detrend: bool
        Whether to use a simple detrending method. Default is False.

    Returns
    =======

    None

    """

    # # if simple detrend is True
    # if simple_detrend:
    #     print("Detrending the obs data")

    #     # Ensure the data is in the correct format and handle NaN values
    #     obs_data = obs_df[obs_val_name].values
    #     model_data = model_df[model_val_name].values

    #     if np.isnan(obs_data).any() or np.isnan(model_data).any():
    #         raise ValueError("Data contains NaN values. Please handle NaNs before detrending.")

    #     # Detrend the obs data
    #     obs_df[obs_val_name] = signal.detrend(obs_data)

    #     # Detrend the model data
    #     model_df[model_val_name] = signal.detrend(model_data)

    # # Assert that the index of the obs df is a datetime in years
    # assert isinstance(
    #     obs_df.index, pd.DatetimeIndex
    # ), "Index  of obs must be a datetime"

    # Set up the figure with 1 row and 4 columns
    fig, axs = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=(20, 5),  # Adjust the figsize as needed
        sharey=False,
    )

    # Set up the model dfs
    model_dfs_list = [model_df_left, model_df_right]
    obs_dfs_list = [obs_df_left, obs_df_right]

    # Set up the obs val names
    obs_val_names = [obs_val_name_left, obs_val_name_right]
    model_val_names = [model_val_name_left, model_val_name_right]

    # Set up the ylabels
    ylabels = [ylabel_left, ylabel_right]

    # Set up the titles
    titles = [title_left, title_right]

    # Set up the ylims
    ylims_list = [ylims_left, ylims_right]

    # Set up the axes
    axs_list = [[axs[0]], [axs[1]]]

    # loop over the axes
    for i, axes in enumerate(axs_list):
        # Extract the axes for this
        ax_big = axes[0]
        # ax_small = axes[1]
        
        # Set up the obs and model dfs
        obs_df = obs_dfs_list[i]
        model_df = model_dfs_list[i]

        # Set up the obs and model val names
        obs_val_name = obs_val_names[i]
        model_val_name = model_val_names[i]

        # Set up the ylabel
        ylabel = ylabels[i]

        # Set up the title
        title = titles[i]

        # Set up the ylims
        ylims = ylims_list[i]

        # add a horizontal line for the 0.8 quantile of the observations
        ax_big.axhline(
            np.quantile(obs_df[obs_val_name], dashed_quant),
            color="blue",
            linestyle="--",
        )

        # for the max value of the obs
        ax_big.axhline(
            solid_line(obs_df[obs_val_name]),
            color="blue",
            linestyle="-.",
        )

        # print the year of the worst observed value
        print("The worst event occurs in the year:", obs_df[obs_val_name].idxmax())
        print("The no. days for the worst event is:", solid_line(obs_df[obs_val_name]))

        # plot the scatter points for the obs
        ax_big.scatter(
            obs_df.index,
            obs_df[obs_val_name],
            color="blue",
            marker="x",
            label=obs_label,
            zorder=2,
        )

        # calculate the trend in the obs time series
        obs_trend = np.polyfit(
            obs_df.index.year,
            obs_df[obs_val_name],
            1,
        )

        # Plot the trend as a dashed blue line
        # ax_big.plot(
        #     obs_df.index,
        #     np.polyval(obs_trend, obs_df.index.year),
        #     color="blue",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f"Obs trend: {round(obs_trend[0], 3)} C/year",
        # )

        # if solid line is np.max and dahsed line is above 0.5
        if solid_line == np.max and dashed_quant > 0.5:
            print("Bad events have high values")

            # Separate model data by threshold
            very_bad_events = model_df[
                model_df[model_val_name] > solid_line(obs_df[obs_val_name])
            ]

            # Model data above 80th percentile
            bad_events = model_df[
                (model_df[model_val_name] > np.quantile(obs_df[obs_val_name], dashed_quant))
                & (model_df[model_val_name] < solid_line(obs_df[obs_val_name]))
            ]

            # Model data below 80th percentile
            events = model_df[
                model_df[model_val_name] < np.quantile(obs_df[obs_val_name], dashed_quant)
            ]

        else:
            print("Bad events have low values")

            # assert that solid_line is np.min
            assert solid_line == np.min, "Solid line must be np.min"

            # assert that dashed_quant is below 0.5
            assert dashed_quant < 0.5, "Dashed quantile must be below 0.5"

            # Separate model data by threshold
            very_bad_events = model_df[
                model_df[model_val_name] < solid_line(obs_df[obs_val_name])
            ]

            # Model data above 80th percentile
            bad_events = model_df[
                (model_df[model_val_name] < np.quantile(obs_df[obs_val_name], dashed_quant))
                & (model_df[model_val_name] > solid_line(obs_df[obs_val_name]))
            ]

            # Model data below 80th percentile
            events = model_df[
                model_df[model_val_name] > np.quantile(obs_df[obs_val_name], dashed_quant)
            ]

        # print the chance of a very bad event
        print(
            f"The chance of a very bad event is: {len(very_bad_events) / len(model_df) * 100}%"
        )

        # if len very bad events is not zero
        if len(very_bad_events) != 0:
            # print the change of the event in terms of 1 in x years
            print(
                f"The chance of a very bad event is: 1 in {round(len(model_df) / len(very_bad_events))} years"
            )

            # Plot the points below the minimum of the obs
            ax_big.scatter(
                very_bad_events[model_time_name],
                very_bad_events[model_val_name],
                color="red",
                alpha=0.8,
                label=very_bad_label,
            )
        else:
            print("no very bad events")

        # Plot the points below the 20th percentile
        ax_big.scatter(
            bad_events[model_time_name],
            bad_events[model_val_name],
            color="orange",
            alpha=0.8,
            label=bad_label,
        )

        # Plot the points above the 20th percentile
        ax_big.scatter(
            events[model_time_name],
            events[model_val_name],
            color="grey",
            alpha=0.8,
            label=normal_label,
        )

        # calculate the trend in all of the model data
        model_trend = np.polyfit(
            model_df[model_time_name].dt.year,
            model_df[model_val_name],
            1,
        )

        # plot the model trend as a dashed red line
        # ax_big.plot(
        #     model_df[model_time_name],
        #     np.polyval(model_trend, model_df[model_time_name].dt.year),
        #     color="green",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f"Model trend: {round(model_trend[0], 3)} C/year",
        # )

        # calculate the model trend in the events
        model_trend_events = np.polyfit(
            events[model_time_name].dt.year,
            events[model_val_name],
            1,
        )

        # plot the model trend for events in a dashed grey line
        # ax_big.plot(
        #     events[model_time_name],
        #     np.polyval(model_trend_events, events[model_time_name].dt.year),
        #     color="grey",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f"Model events: {round(model_trend_events[0], 3)} C/year",
        # )

        # calculate the model trend in the bad events
        model_trend_bad = np.polyfit(
            bad_events[model_time_name].dt.year,
            bad_events[model_val_name],
            1,
        )

        # # plot the model trend for bad events in a dashed orange line
        # ax_big.plot(
        #     bad_events[model_time_name],
        #     np.polyval(model_trend_bad, bad_events[model_time_name].dt.year),
        #     color="orange",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f"Model bad events: {round(model_trend_bad[0], 3)} C/year",
        # )

        # calculate the model trend in the very bad events
        model_trend_very_bad = np.polyfit(
            very_bad_events[model_time_name].dt.year,
            very_bad_events[model_val_name],
            1,
        )

        # # plot the model trend for very bad events in a dashed red line
        # ax_big.plot(
        #     very_bad_events[model_time_name],
        #     np.polyval(model_trend_very_bad, very_bad_events[model_time_name].dt.year),
        #     color="red",
        #     linestyle="--",
        #     linewidth=2,
        #     label=f"Model v. bad events: {round(model_trend_very_bad[0], 3)} C/year",
        # )
        
        # include the legend
        ax_big.legend(fontsize=10, ncol=3, loc="upper center")

        # label the y-axis
        ax_big.set_ylabel(ylabel, fontsize=14)

        # set up the x-axis

        # increase the size of the value labels
        ax_big.tick_params(axis="x", labelsize=12)

        # same for the y-axis
        ax_big.tick_params(axis="y", labelsize=12)

        # set up the ylims
        ax_big.set_ylim(ylims)

        # set the title
        ax_big.set_title(title, fontsize=14, fontweight="bold")

        # # do the events plots for the no. exceedance days on the second plot
        # # do the events plots for the no. exceedance days on the second plot
        # ax_small.boxplot(
        #     obs_df[obs_val_name],
        #     positions=[0.5],
        #     boxprops=dict(color="blue", facecolor="white"),
        #     whiskerprops=dict(color="blue"),
        #     capprops=dict(color="blue"),
        #     flierprops=dict(markerfacecolor="blue", markeredgecolor="blue"),
        #     medianprops=dict(color="blue"),
        #     patch_artist=True,
        #     vert=True,
        #     widths=0.5,
        #     labels=[obs_label],
        # )

        # # plot the model data
        # ax_small.boxplot(
        #     model_df[model_val_name],
        #     positions=[1.5],
        #     boxprops=dict(color="red", facecolor="white"),
        #     whiskerprops=dict(color="red"),
        #     capprops=dict(color="red"),
        #     flierprops=dict(markerfacecolor="red", markeredgecolor="red"),
        #     medianprops=dict(color="red"),
        #     patch_artist=True,
        #     vert=True,
        #     widths=0.5,
        #     labels=[model_label],
        # )

        # # # set up the xlabels for the second plot
        # # xlabels = ["Observed", "Model"]

        # # # add the xlabels to the second subplot
        # # axs[1].set_xticks(xlabels)

        # # # remove the xlabel
        # # axs[1].set_xlabel("")

        # # # remove the yticks and axis lin
        # # axs[1].set_yticks([])

        # # remove the xticks from the second plot
        # ax_small.set_xticks([])

    # specify a tight layout
    plt.tight_layout()

    # # set up a fname for the plot
    # fname = f"obs-{obs_val_name}_model-{model_val_name}_quantile-{dashed_quant}_solid-{solid_line.__name__}_{save_prefix}_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.pdf"

    # # form the savepath
    # savepath = os.path.join(save_dir, fname)

    # if not os.path.exists(savepath):
    #     print(f"Saving plot to {savepath}")
    #     # save the plot
    #     plt.savefig(savepath, bbox_inches="tight", dpi=800)

    #     # print that we have saved the plot
    #     print(f"Saved plot to {savepath}")

    return



if __name__ == "__main__":
    print("This script is not intended to be run directly.")
    print("Please import the functions and use them in your script.")
    sys.exit(1)
