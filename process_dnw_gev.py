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
from process_temp_gev import model_drift_corr_plot

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import sigmoid, dot_plot

# Silence warnings
warnings.filterwarnings("ignore")


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

    # Fit the sigmoid curve to the observed data
    popt, pcov = curve_fit(
        sigmoid, obs_df_copy[obs_ws_col], ch_df["total_gen"], p0=p0, method="dogbox"
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

    return obs_df_copy, model_df_copy


# Define the main function
def main():
    # Start the timer
    start = time.time()

    # Set up the directory in which the dfs are stored
    dfs_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs/"

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

    # Load the obs wind data
    df_obs_sfcWind = pd.read_csv(
        os.path.join(dfs_dir, "ERA5_sfcWind_UK_wind_box_1960-2018_daily_2025-02-26.csv")
    )

    # Convert the 'time' column to datetime, assuming it represents days since "1950-01-01 00:00:00"
    df_obs_sfcWind["time"] = pd.to_datetime(
        df_obs_sfcWind["time"], origin="1952-01-01", unit="D"
    )

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
    common_wyears = np.arange(1961, 2017 + 1)

    # Subset the obs data to the common_wyears
    df_obs = df_obs[df_obs["effective_dec_year"].isin(common_wyears)]

    # Subset the model data to the common_wyears
    df_model_djf = df_model_djf[df_model_djf["effective_dec_year"].isin(common_wyears)]

    # Create a new column for data_tas_c in df_model_full_djf
    df_model_djf["data_tas_c"] = df_model_djf["data_tas"] - 273.15

    # Plot the lead pdfs to visualise the biases/drifts
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_tas_c",
        obs_var_name="data_c",
        lead_name="winter_year",
        xlabel="Temperature (°C)",
        suptitle="Lead dependent temperature PDFs, DJF all days, 1961-2017",
        figsize=(10, 5),
    )

    # Plot the lead pdfs to visualise the biases/drifts
    # but for wind speed
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_sfcWind",
        obs_var_name="data_sfcWind",
        lead_name="winter_year",
        xlabel="10m Wind Speed (m/s)",
        suptitle="Lead dependent wind speed PDFs, DJF all days, 1961-2017",
        figsize=(10, 5),
    )

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
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_tas_c_drift_bc",
        obs_var_name="data_c",
        lead_name="winter_year",
        xlabel="Temperature (°C)",
        suptitle="Lead dependent temperature PDFs, DJF all days, 1961-2017 (model drift + bias corrected)",
        figsize=(10, 5),
    )

    # Plot the lead pdfs to visualise the biases/drifts
    # but for wind speed
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_sfcWind_drift_bc",
        obs_var_name="data_sfcWind",
        lead_name="winter_year",
        xlabel="10m Wind Speed (m/s)",
        suptitle="Lead dependent wind speed PDFs, DJF all days, 1961-2017 (model drift + bias corrected)",
        figsize=(10, 5),
    )

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
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_sfcWind_dt_sigmoid_total_wind_gen",
        obs_var_name="data_sfcWind_dt_sigmoid_total_wind_gen",
        lead_name="winter_year",
        xlabel="Wind Power Generation (GW)",
        suptitle="Lead dependent wind power generation PDFs, DJF all days, 1961-2017 (detrended, no BC wind)",
        figsize=(10, 5),
    )

    # plot the lead pdfs to visualise the biases/drifts
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_sfcWind_drift_bc_dt_sigmoid_total_wind_gen",
        obs_var_name="data_sfcWind_dt_sigmoid_total_wind_gen",
        lead_name="winter_year",
        xlabel="Wind Power Generation (GW)",
        suptitle="Lead dependent wind power generation PDFs, DJF all days, 1961-2017 (detrended, BC wind)",
        figsize=(10, 5),
    )

    # sys.exit()

    # convert the temperature tyo demand for bias corrected T data
    df_obs, df_model_djf = temp_to_demand(
        obs_df=df_obs,
        model_df=df_model_djf,
        obs_temp_col="data_c_dt",
        model_temp_col="data_tas_c_drift_bc_dt",
    )

    # convert the temperature tyo demand for non bias corrected T data
    df_obs, df_model_djf = temp_to_demand(
        obs_df=df_obs,
        model_df=df_model_djf,
        obs_temp_col="data_c_dt",
        model_temp_col="data_tas_c_dt",
    )

    # do the same for temperature
    # convert the temperature tyo demand for bias corrected T data
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_tas_c_dt_UK_demand",
        obs_var_name="data_c_dt_UK_demand",
        lead_name="winter_year",
        xlabel="Demand (GW)",
        suptitle="Lead dependent demand PDFs, DJF all days, 1961-2017 (detrended, no BC T)",
        figsize=(10, 5),
    )

    # do the same for temperature
    # convert the temperature tyo demand for non bias corrected T data
    gev_funcs.plot_lead_pdfs(
        model_df=df_model_djf,
        obs_df=df_obs,
        model_var_name="data_tas_c_drift_bc_dt_UK_demand",
        obs_var_name="data_c_dt_UK_demand",
        lead_name="winter_year",
        xlabel="Demand (GW)",
        suptitle="Lead dependent demand PDFs, DJF all days, 1961-2017 (detrended, BC T)",
        figsize=(10, 5),
    )

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
    gev_funcs.plot_multi_var_dist(
        obs_df=df_obs,
        model_df=df_model_djf,
        model_df_bc=df_model_djf,
        obs_var_names=obs_var_names,
        model_var_names=model_var_names,
        model_var_names_bc=model_var_names_bc,
        row_titles=[
            "Temp (°C)",
            "Demand (GW)",
            "10m wind speed (m/s)",
            "Wind power gen. (GW)",
            "Demand net wind (GW)",
        ],
        subplot_titles=subplot_titles,
        figsize=(15, 15),
    )

    # now plot the relationships between variables here
    gev_funcs.plot_rel_var(
        obs_df=df_obs,
        model_df=df_model_djf,
        model_df_bc=df_model_djf,
        obs_var_names=("data_c_dt", "data_sfcWind_dt"),
        model_var_names=("data_tas_c_dt", "data_sfcWind_dt"),
        model_var_names_bc=("data_tas_c_drift_bc_dt", "data_sfcWind_drift_bc_dt"),
        row_title="T vs sfcWind",
        figsize=(15, 5),
    )

    # now quantify the seasonal block maxima for demand net wind
    # first for the observations
    block_max_obs_dnw = gev_funcs.obs_block_min_max(
        df=df_obs,
        time_name="effective_dec_year",
        min_max_var_name="demand_net_wind",
        new_df_cols=["data_sfcWind_dt_sigmoid_total_wind_gen", "data_c_dt_UK_demand", "time"],
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
        ],
        winter_year="winter_year",
        process_min=False,
    )

    # make sure effective dec year is in the block max obs data
    block_max_model_dnw["effective_dec_year"] = block_max_model_dnw["init_year"] + (
        block_max_model_dnw["winter_year"] - 1
    )

    # Plot the biases in these
    gev_funcs.plot_lead_pdfs(
        model_df=block_max_model_dnw,
        obs_df=block_max_obs_dnw,
        model_var_name="demand_net_wind_bc_max",
        obs_var_name="demand_net_wind_max",
        lead_name="winter_year",
        xlabel="Demand net wind (GW)",
        suptitle="Lead dependent demand net wind PDFs, DJF all days, 1961-2017 (detrended, BC T + sfcWind)",
        figsize=(10, 5),
    )

    # apply a uniform bias correction to the block maxima from the model
    bias = block_max_model_dnw["demand_net_wind_bc_max"].mean() - block_max_obs_dnw[
        "demand_net_wind_max"
    ].mean()

    # print the bias
    print(f"Bias: {bias}")

    # apply the bias correction
    block_max_model_dnw["demand_net_wind_bc_max_bc"] = (
        block_max_model_dnw["demand_net_wind_bc_max"] - bias
    )

    # Plot the biases in these
    gev_funcs.plot_lead_pdfs(
        model_df=block_max_model_dnw,
        obs_df=block_max_obs_dnw,
        model_var_name="demand_net_wind_bc_max_bc",
        obs_var_name="demand_net_wind_max",
        lead_name="winter_year",
        xlabel="Demand net wind (GW)",
        suptitle="Lead dependent demand net wind PDFs, DJF all days, 1961-2017 (detrended, BC T + sfcWind + BC)",
        figsize=(10, 5),
    )

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
        suptitle="Lead dependent demand net wind PDFs, DJF all days, 1961-2017 (detrended, BC T + sfcWind + BC)",
        figsize=(15, 5),
    )

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
