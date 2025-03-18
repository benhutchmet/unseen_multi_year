"""
gev_functions.py

Functions for chapter 2 daily extremes analysis.

Author: Ben Hutchins, 2025
"""

# Local imports
import os
import sys
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

# Specific imports
from tqdm import tqdm
from matplotlib import gridspec
from datetime import datetime, timedelta

from scipy.optimize import curve_fit
from scipy.stats import linregress, percentileofscore, gaussian_kde
from scipy.stats import genextreme as gev
from sklearn.metrics import mean_squared_error, r2_score

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
            # plot the data detrended in grey with a label
            ax.plot(
                data_this[model_time_name],
                data_this[f"{model_var_name}{detrend_suffix}"],
                color="grey",
                alpha=0.2,
                label="Model ens dtr",
            )
        else:
            # plot the data detrended in grey
            ax.plot(
                data_this[model_time_name],
                data_this[f"{model_var_name}{detrend_suffix}"],
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
    ax.plot(
        obs_df[obs_time_name],
        obs_df[f"{obs_var_name}{detrend_suffix}"],
        color="black",
        label="Obs dtr",
    )

    if plot_min:
        # Include a solid black line for the min value of the observed data (no dt)
        ax.axhline(obs_df[obs_var_name].min(), color="black", linestyle="--")

        # Include a solid black line for the min value of the observed data (dt)
        ax.axhline(obs_df[f"{obs_var_name}{detrend_suffix}"].min(), color="black")
    else:
        # Include a solid black line for the max value of the observed data (dt)
        ax.axhline(obs_df[f"{obs_var_name}]"].max(), color="black", linestyle="--")

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

if __name__ == "__main__":
    print("This script is not intended to be run directly.")
    print("Please import the functions and use them in your script.")
    sys.exit(1)
