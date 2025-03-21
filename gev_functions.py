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

from scipy.optimize import curve_fit
from scipy.stats import linregress, percentileofscore, gaussian_kde
from scipy.stats import genextreme as gev
from sklearn.metrics import mean_squared_error, r2_score
from iris.util import equalise_attributes

# # Suppress warnings
# warnings.filterwarnings('ignore')


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
    # Define the function to fit
    slope, intercept, r_value, p_value, std_err = linregress(
        df[x_axis_name], df[y_axis_name]
    )

    # Calculate the trend line
    trend = slope * df[x_axis_name] + intercept

    # Determine the final point on the trend line
    final_point = trend.iloc[-1]

    # Create a new column with the detrended values
    df[y_axis_name + suffix] = final_point - trend + df[y_axis_name]

    return df


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

    # Set up the n members
    members = df[member_name].unique()
    n_members = len(members)

    # Set up the slopes
    slopes = np.zeros(n_members)
    intercepts = np.zeros(n_members)

    # Loop over the members
    for i, member in enumerate(members):
        # Get the data for this member
        data = df[df[member_name] == member]

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
    trend = intercepts_mean + slopes_mean * df[x_axis_name]

    # Determine the final point on the trend line
    final_point = trend.iloc[-1]

    # Create a new column with the detrended values
    df[y_axis_name + suffix] = final_point - trend + df[y_axis_name]

    return df


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
            time_data = time_data[time_data[min_max_var_name] < np.percentile(time_data[min_max_var_name], percentile)]
        else:
            # Find the rows that are greater than the threshold
            time_data = time_data[time_data[min_max_var_name] > np.percentile(time_data[min_max_var_name], percentile)]

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
            bias = data_this[model_var_name].mean() - obs_df_lead_this[obs_var_name].mean()
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
    shape_obs, loc_obs, scale_obs = gev.fit(obs_df[obs_var_name])

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
                lead_this = np.random.choice(
                    data_this[model_lead_name].unique()
                )

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
        shape_model, loc_model, scale_model = gev.fit(pseudo_obs_this)

        # Store the model parameters
        gev_params["model_shape"][0][i] = shape_model
        gev_params["model_loc"][0][i] = loc_model
        gev_params["model_scale"][0][i] = scale_model

    return gev_params

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
        model_df[model_var_name], bins=20, color="red", alpha=0.5, label=model_label, density=True,
    )

    # Plot the distributions
    ax0.hist(obs_df[obs_var_name], bins=20, color="black", alpha=0.5, label=obs_label, density=True)

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
            model_df.groupby(model_time_name)[f"{model_var_name}{detrend_suffix}"].mean(),
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
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize, sharey=True, layout="compressed")

    # Plot the observed data scatter
    ax0 = axs[0]
    ax1 = axs[1]

    # Create the scatter plot
    sc0 = ax0.scatter(
        obs_df[obs_x_var_name],
        obs_df[obs_y_var_name],
        c=obs_df[obs_cmap_var_name],
        cmap=cmap,
        s=100,
    )

    # Include text for the 2010 point
    ax0.text(
        obs_df.loc[2010, obs_x_var_name] + 0.2,
        obs_df.loc[2010, obs_y_var_name] + 0.1,
        "2010",
        color="red",
        verticalalignment="top",
    )

    # Include a vertical dashed line for the mean x variable
    ax0.axvline(obs_df[obs_x_var_name].mean(), color="black", linestyle="--")

    # Include a horizontal dashed line for the mean y variable
    ax0.axhline(obs_df[obs_y_var_name].mean(), color="black", linestyle="--")

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

    # Perform kernel density estimate
    xy = np.vstack([x_var_model, y_var_model])
    kde = gaussian_kde(xy)
    xmin, xmax = x_var_model.min(), x_var_model.max()
    ymin, ymax = y_var_model.min(), y_var_model.max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions).T, X.shape)

    # Plot the scatter for the model data
    sc1 = ax1.scatter(
        x_var_model,
        y_var_model,
        c=model_df[model_cmap_var_name],
        cmap=cmap,
        s=10,
    )

    # Plot the density contours
    ax1.contour(X, Y, Z, levels=5, colors="black")

    # Include a vertical dashed line for the mean x variable
    ax1.axvline(x_var_model.mean(), color="black", linestyle="--")

    # Include a horizontal dashed line for the mean y variable
    ax1.axhline(y_var_model.mean(), color="black", linestyle="--")

    # Set the title
    ax1.set_title(model_title)

    # Set the x label
    ax1.set_xlabel(xlabel)

    # if ylims is not None
    if xlims is not None:
        # set the y-axis limits
        ax1.set_xlim(xlims)

    # # Set up a tight layout before adding the colorbar
    # fig.tight_layout()

    # Add the colorbar after setting up the tight layout
    cbar = fig.colorbar(sc0, ax=axs, orientation="vertical", pad=0.02)

    # Set the label for the colorbar
    cbar.set_label(cmap_label)

    # Set the super title
    fig.suptitle(sup_title, y=1.05)

    # Adjust the layout again if necessary
    plt.subplots_adjust(top=0.9)

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
            path_this = f"s{init_year}-r{member}i1p1f2/{freq}/{model_var_name}/gn/files/d*/"

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
            bias = model_df_lead_this[model_var_name].mean() - obs_df_lead_this[obs_var_name].mean()
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
            bias = model_df_lead_this[model_var_name].mean() - obs_df[obs_var_name].mean()

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

    # Plot the full field data for the observations
    ax0.plot(
        obs_df_full_field[obs_time_name],
        obs_df_full_field[obs_var_name_full_field],
        color="black",
        label="obs",
    )

    # quantify linear trend for the obs
    slope_obs_ff, intercept_obs_ff, _, _, _ = linregress(
        obs_df_full_field[obs_time_name], obs_df_full_field[obs_var_name_full_field]
    )

    # print the slope and intercept
    print(f"obs slope: {slope_obs_ff}, obs intercept: {intercept_obs_ff}")

    # calclate the trend line
    trend_line_obs_ff = slope_obs_ff * obs_df_full_field[obs_time_name] + intercept_obs_ff

    # # calculate the final point
    # final_point_obs_ff = trend_line_obs_ff.iloc[-1]

    # # pivot the trend line
    # trend_line_obs_ff_pivot = final_point_obs_ff - trend_line_obs_ff + obs_df_full_field[obs_var_name_full_field]

    # plot this line as a dashed black line
    ax0.plot(obs_df_full_field[obs_time_name], trend_line_obs_ff, color="black", linestyle="--")

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
        slope_model_ff_mean * model_df_full_field[model_time_name].unique() + intercept_model_ff_mean,
        color="red",
        linestyle="--",
    )

    # Set the title for ax0
    ax0.set_title(f"Full field, obs slope: {slope_obs_ff:.3f}, model slope: {slope_model_ff_mean:.3f} (+/- {ci_model_ff:.3f})", fontweight="bold")

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
    trend_line_obs_block = slope_obs_block * obs_df_block[obs_time_name] + intercept_obs_block

    # plot the trend line
    ax1.plot(obs_df_block[obs_time_name], trend_line_obs_block, color="black", linestyle="--")

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
        slope_model_block_mean * model_df_block[model_time_name].unique() + intercept_model_block_mean,
        color="red",
        linestyle="--",
    )

    # Set the title for ax1
    ax1.set_title(f"Block minima/maxima, obs slope: {slope_obs_block:.3f}, model slope: {slope_model_block_mean:.3f} (+/- {ci_model_block:.3f})", fontweight="bold")

    # Set the ylabel
    ax0.set_ylabel(ylabel)

    # Set a tight layout
    plt.tight_layout()

    return None



if __name__ == "__main__":
    print("This script is not intended to be run directly.")
    print("Please import the functions and use them in your script.")
    sys.exit(1)
