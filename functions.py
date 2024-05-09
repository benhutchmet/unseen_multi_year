# Functions for UNSEEN work

# Local imports
import os
import sys
import glob
import random

# Third party imports
import numpy as np
import xarray as xr
import pandas as pd
from tqdm import tqdm
from scipy import stats, signal
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
import xesmf as xe

# Import types
from typing import Any, Callable, Union

# Path to modules
import dictionaries as dicts


# Function for loading each of the ensemble members for a given model
def load_model_data(
    model_variable: str,
    model: str,
    experiment: str,
    start_year: int,
    end_year: int,
    avg_period: int,
    grid: dict,
):
    """
    Function for loading each of the ensemble members for a given model

    Parameters
    ----------

    model_variable: str
        The variable to load from the model data
        E.g. 'pr' for precipitation

    model: str
        The model to load the data from
        E.g. 'HadGEM3-GC31-MM'

    experiment: str
        The experiment to load the data from
        E.g. 'historical' or 'dcppA-hindcast'

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    Returns
    -------

    model_data dict[str, xr.DataArray]
        A dictionary of the model data for each ensemble member
        E.g. model_data['r1i1p1f1'] = xr.DataArray
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    # Set the n years
    n_years = len(years)

    # Extract the lon and lat bounds
    lon1, lon2 = grid["lon1"], grid["lon2"]
    lat1, lat2 = grid["lat1"], grid["lat2"]

    # Set up the directory where the csv files are stored
    csv_dir = "/home/users/benhutch/multi_year_unseen/paths"

    # Assert that the folder exists
    assert os.path.exists(csv_dir), "The csv directory does not exist"

    # Assert that the folder is not empty
    assert os.listdir(csv_dir), "The csv directory is empty"

    # Extract the csv file for the model and experiment
    csv_file = glob.glob(f"{csv_dir}/*.csv")[0]

    # Verify that the csv file exists
    assert csv_file, "The csv file does not exist"

    # Load the csv file
    csv_data = pd.read_csv(csv_file)

    # Extract the path for the model and experiment and variable
    model_path = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == model_variable),
        "path",
    ].values[0]

    print(model_path)

    # Assert that the model path exists
    assert os.path.exists(model_path), "The model path does not exist"

    # Assert that the model path is not empty
    assert os.listdir(model_path), "The model path is empty"

    # Create an empty list of files
    model_file_list = []

    no_members = 0

    # BADC pattern
    # /badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/tas/g?/files/d????????/*.nc"

    # Extract the first part of the model_path
    model_path_root = model_path.split("/")[1]

    # If the model path root is gws
    if model_path_root == "gws":
        print("The model path root is gws")
        # List the files in the model path
        model_files = os.listdir(model_path)

        # Loop over the years
        for year in years:
            # Find all of the files for the given year
            year_files = [file for file in model_files if f"s{year}" in file]

            # Find the filenames for the given year
            # After the final '/' in the path
            year_files_split = [file.split("/")[-1] for file in year_files]

            # Split by _ and extract the 4th element
            year_files_split = [file.split("_")[4] for file in year_files_split]

            # Split by - and extract the 1st element
            year_files_split = [file.split("-")[1] for file in year_files_split]

            # Find the unique members
            unique_combinations = np.unique(year_files_split)

            # # Print the year and the number of files
            # print(year, len(year_files))
            if year == years[0]:
                # Set the no members
                no_members = len(year_files)

            # # Print no
            # print("Number of members", no_members)
            # print("Number of unique combinations", len(unique_combinations))
            # print("Unique combinations", unique_combinations)

            # Assert that the len unique combinations is the same as the no members
            assert (
                len(unique_combinations) == no_members
            ), "The number of unique combinations is not the same as the number of members"

            # Assert that the number of files is the same as the number of members
            assert (
                len(year_files) == no_members
            ), "The number of files is not the same as the number of members"

            # Append the year files to the model file list
            model_file_list.append(year_files)
    elif model_path_root == "badc":
        print("The model path root is badc")

        # Loop over the years
        for year in years:
            # Form the path to the files for this year
            year_path = f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"

            # Find the files for the given year
            year_files = glob.glob(year_path)

            # Extract the number of members
            # as the number of unique combinations of r*i*p?f?
            # here f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"
            # List the directories in model_path
            dirs = os.listdir(model_path)

            # Split these by the delimiter '-'
            dirs_split = [dir.split("-") for dir in dirs]

            # Find the unique combinations of r*i*p?f?
            unique_combinations = np.unique(dirs_split)

            # Set the no members
            no_members = len(unique_combinations)

            # Assert that the number of files is the same as the number of members
            assert (
                len(year_files) == no_members
            ), "The number of files is not the same as the number of members"

            # Append the year files to the model file list
            model_file_list.append(year_files)

    # Flatten the model file list
    model_file_list = [file for sublist in model_file_list for file in sublist]

    # Print the number of files
    print("Number of files:", len(model_file_list))

    # Print
    print(f"opening {model_path}/{model_file_list[0]}")

    # From the first file extract the number of lats and lons
    ds = xr.open_dataset(f"{model_path}/{model_file_list[0]}")

    # Extract the time series for the gridbox
    ds = ds.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(dim=("lat", "lon"))

    # Print the first time of the first file
    print("First time:", ds["time"][0].values)

    # Extract the first year from the first file
    first_year = int(str(ds["time"][0].values)[:4])

    # Print the first year
    print("First year:", first_year)

    # Assert that the first year is the same as the start year
    assert first_year == start_year, "The first year is not the same as the start year"

    # Print the window over which we are slicing the time
    print("Slicing over:", f"{first_year}-12-01", f"{first_year + avg_period}-12-01")

    # Extract the time slice between
    # First december to second march
    ds_slice = ds.sel(
        time=slice(f"{first_year}-12-01", f"{first_year + avg_period}-12-01")
    )

    # Extract the nmonths
    n_months = len(ds_slice["time"])

    # Print the number of months
    print("Number of months:", n_months)

    # Form the empty array to store the data
    model_data = np.zeros([n_years, no_members, n_months])

    # Print the shape of the model data
    print("Shape of model data:", model_data.shape)

    # Loop over the years
    for year in tqdm(years, desc="Processing years"):
        for member in tqdm(
            (unique_combinations),
            desc=f"Processing members for year {year}",
            leave=False,
        ):
            # Find the file for the given year and member
            file = [
                file
                for file in model_file_list
                if f"s{year}" in file and member in file
            ][0]

            # set the member index
            member_index = np.where(unique_combinations == member)[0][0]

            # Load the file
            ds = xr.open_dataset(f"{model_path}/{file}")

            # Extract the time series for the gridbox
            ds = ds.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(
                dim=("lat", "lon")
            )

            # Extract the time slice between
            ds_slice = ds.sel(time=slice(f"{year}-12-01", f"{year + avg_period}-12-01"))

            # Extract the data
            model_data[year - start_year, member_index, :] = ds_slice[
                model_variable
            ].values

    # p[rint the shape of the model data
    print("Shape of model data:", model_data.shape)

    # Return the model data
    return model_data


# Define a function for preprocessing the model data
def preprocess(
    ds: xr.Dataset,
):
    """
    Preprocess the model data using xarray

    Parameters

    ds: xr.Dataset
        The dataset to preprocess
    """

    # Return the dataset
    return ds


# Write a new function for loading the model data using xarray
def load_model_data_xarray(
    model_variable: str,
    model: str,
    experiment: str,
    start_year: int,
    end_year: int,
    grid: dict,
    first_fcst_year: int,
    last_fcst_year: int,
    months: list,
    engine: str = "netcdf4",
    parallel: bool = True,
):
    """
    Function for loading each of the ensemble members for a given model using xarray

    Parameters
    ----------

    model_variable: str
        The variable to load from the model data
        E.g. 'pr' for precipitation

    model: str
        The model to load the data from
        E.g. 'HadGEM3-GC31-MM'

    experiment: str
        The experiment to load the data from
        E.g. 'historical' or 'dcppA-hindcast'

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    first_fcst_year: int
        The first forecast year for taking the time average
        E.g. 1960

    last_fcst_year: int
        The last forecast year for taking the time average
        E.g. 1962

    months: list
        The months to take the time average over
        E.g. [10, 11, 12, 1, 2, 3] for October to March

    engine: str
        The engine to use for opening the dataset
        Passed to xarray.open_mfdataset
        Defaults to 'netcdf4'

    parallel: bool
        Whether to use parallel processing
        Passed to xarray.open_mfdataset
        Defaults to True

    Returns
    -------

    model_data dict[str, xr.DataArray]
        A dictionary of the model data for each ensemble member
        E.g. model_data['r1i1p1f1'] = xr.DataArray
    """

    # Extract the lat and lon bounds
    lon1, lon2, lat1, lat2 = grid["lon1"], grid["lon2"], grid["lat1"], grid["lat2"]

    # Set up the path to the csv file
    csv_path = "paths/*.csv"

    # Find the csv file
    csv_file = glob.glob(csv_path)[0]

    # Load the csv file
    csv_data = pd.read_csv(csv_file)

    # Extract the path for the given model and experiment and variable
    model_path = csv_data.loc[
        (csv_data["model"] == model)
        & (csv_data["experiment"] == experiment)
        & (csv_data["variable"] == model_variable),
        "path",
    ].values[0]

    # Assert that the model path exists
    assert os.path.exists(model_path), "The model path does not exist"

    # Assert that the model path is not empty
    assert os.listdir(model_path), "The model path is empty"

    # Extract the first part of the model_path
    model_path_root = model_path.split("/")[1]

    # If the model path root is gws
    if model_path_root == "gws":
        print("The model path root is gws")

        # List the files in the model path
        model_files = os.listdir(model_path)

        # Loop over the years
        for year in range(start_year, end_year + 1):
            # Find all of the files for the given year
            year_files = [file for file in model_files if f"s{year}" in file]

            # Split the year files by '/'
            year_files_split = [file.split("/")[-1] for file in year_files]

            # Split the year files by '_'
            year_files_split = [file.split("_")[4] for file in year_files_split]

            # Split the year files by '-'
            year_files_split = [file.split("-")[1] for file in year_files_split]

            # Find the unique combinations
            unique_combinations = np.unique(year_files_split)

            # Assert that the len unique combinations is the same as the no members
            assert len(unique_combinations) == len(
                year_files
            ), "The number of unique combinations is not the same as the number of members"

    elif model_path_root == "badc":
        print("The model path root is badc")

        # Loop over the years
        for year in range(start_year, end_year + 1):
            # Form the path to the files for this year
            year_path = f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"

            # Find the files for the given year
            year_files = glob.glob(year_path)

            # Extract the number of members
            # as the number of unique combinations of r*i*p?f?
            # here f"{model_path}/s{year}-r*i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"
            # List the directories in model_path
            dirs = os.listdir(model_path)

            # Split these by the delimiter '-'
            dirs_split = [dir.split("-") for dir in dirs]

            # Find the unique combinations of r*i*p?f?
            unique_combinations = np.unique(dirs_split)

            # Assert that the number of files is the same as the number of members
            assert len(year_files) == len(
                unique_combinations
            ), "The number of files is not the same as the number of members"
    else:
        print("The model path root is neither gws nor badc")
        ValueError("The model path root is neither gws nor badc")

    # Set up unique variant labels
    unique_variant_labels = np.unique(unique_combinations)

    # Print the number of unique variant labels
    print("Number of unique variant labels:", len(unique_variant_labels))
    print("For model:", model)

    # print the unique variant labels
    print("Unique variant labels:", unique_variant_labels)

    # Create an empty list for forming the list of files for each ensemble member
    member_files = []

    # If the model path root is gws
    if model_path_root == "gws":
        print("Forming the list of files for each ensemble member for gws")

        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Initialize the member files
            variant_label_files = []

            for year in range(start_year, end_year + 1):
                # Find the file for the given year and member
                file = [
                    file
                    for file in model_files
                    if f"s{year}" in file and variant_label in file
                ][0]

                # Append the model path to the file
                file = f"{model_path}/{file}"

                # Append the file to the member files
                variant_label_files.append(file)

            # Append the member files to the member files
            member_files.append(variant_label_files)
    elif model_path_root == "badc":
        print("Forming the list of files for each ensemble member for badc")

        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Initialize the member files
            variant_label_files = []

            for year in range(start_year, end_year + 1):
                # Form the path to the files for this year
                path = f"{model_path}/s{year}-r{variant_label}i?p?f?/Amon/{model_variable}/g?/files/d????????/*.nc"

                # Find the files which match the path
                year_files = glob.glob(path)

                # Assert that the number of files is 1
                assert len(year_files) == 1, "The number of files is not 1"

                # Append the file to the variant label files
                variant_label_files.append(year_files[0])

            # Append the variant label files to the member files
            member_files.append(variant_label_files)
    else:
        print("The model path root is neither gws nor badc")
        ValueError("The model path root is neither gws nor badc")

    # Assert that member files is a list withiin a list
    assert isinstance(member_files, list), "member_files is not a list"

    # Assert that member files is a list of lists
    assert isinstance(member_files[0], list), "member_files is not a list of lists"

    # Assert that the length of member files is the same as the number of unique variant labels
    assert len(member_files) == len(
        unique_variant_labels
    ), "The length of member_files is not the same as the number of unique variant labels"

    # Initialize the model data
    dss = []

    # Will depend on the model here
    # for s1961 - CanESM5 and IPSL-CM6A-LR both initialized in January 1962
    # So 1962 will be their first year
    if model not in ["CanESM5", "IPSL-CM6A-LR"]:
        # Find the index of the forecast first year
        first_fcst_year_idx = first_fcst_year - start_year
        last_fcst_year_idx = (last_fcst_year - first_fcst_year) + 1
    else:
        # Find the index of the forecast first year
        # First should be index 0 normally
        first_fcst_year_idx = (first_fcst_year - start_year) - 1
        last_fcst_year_idx = last_fcst_year - first_fcst_year

    # Flatten the member files list
    member_files = [file for sublist in member_files for file in sublist]

    init_year_list = []
    # Loop over init_years
    for init_year in tqdm(
        range(start_year, end_year + 1), desc="Processing init years"
    ):
        print(f"processing init year {init_year}")
        # Set up the member list
        member_list = []
        # Loop over the unique variant labels
        for variant_label in unique_variant_labels:
            # Find the matching path for the given year and member
            # e.g file containing f"s{init_year}-{variant_label}
            file = [
                file for file in member_files if f"s{init_year}-{variant_label}" in file
            ][0]

            # Open all leads for specified variant label
            # and init_year
            member_ds = xr.open_mfdataset(
                file,
                combine="nested",
                concat_dim="time",
                preprocess=lambda ds: preprocess(ds, grid),
                parallel=parallel,
                engine=engine,
                coords="minimal",  # expecting identical coords
                data_vars="minimal",  # expecting identical vars
                compat="override",  # speed up
            ).squeeze()

            # init_year = start_year and variant_label is unique_variant_labels[0]
            if init_year == start_year and variant_label == unique_variant_labels[0]:
                # Set new int time
                member_ds = set_integer_time_axis(xro=member_ds, first_month_attr=True)
            else:
                # Set new integer time
                member_ds = set_integer_time_axis(member_ds)

            # Append the member dataset to the member list
            member_list.append(member_ds)
        # Concatenate the member list along the ensemble_member dimension
        member_ds = xr.concat(member_list, "member")
        # Append the member dataset to the init_year list
        init_year_list.append(member_ds)
    # Concatenate the init_year list along the init dimension
    # and rename as lead time
    ds = xr.concat(init_year_list, "init").rename({"time": "lead"})

    # Set up the members
    ds["member"] = unique_variant_labels
    ds["init"] = np.arange(start_year, end_year + 1)

    # Return ds
    return ds


# # # Loop over the member files
# # for member_file in tqdm(member_files, desc="Processing members"):
# #     # print("Processing member:", member_file)

# #     # Open the files
# #     ds = xr.open_mfdataset(member_file,
# #                            preprocess=lambda ds: preprocess(ds, first_fcst_year_idx, last_fcst_year_idx, lat1, lat2, lon1, lon2, months),
# #                            combine='nested',
# #                            concat_dim='time',
# #                            join='override',
# #                            coords='minimal',
# #                            engine='netcdf4',
# #                            parallel=True)

# #     # Append the dataset to the model data
# #     dss.append(ds)

# # # Concatenate the datasets
# # ds = xr.concat(dss, dim='ensemble_member')

# # Return the model data
# return member_files, unique_variant_labels


def set_integer_time_axis(
    xro: Union[xr.DataArray, xr.Dataset],
    offset: int = 1,
    time_dim: str = "time",
    first_month_attr: bool = False,
) -> Union[xr.DataArray, xr.Dataset]:
    """
    Set time axis to integers starting from `offset`.

    Used in hindcast preprocessing before the concatenation of `intake-esm` happens.

    Inputs:
    xro: xr.DataArray or xr.Dataset
        The input xarray DataArray or Dataset whose time axis is to be modified.

    offset: int, optional
        The starting point for the new integer time axis. Default is 1.

    time_dim: str, optional
        The name of the time dimension in the input xarray object. Default is "time".

    first_month_attr: bool, optional
        Whether to include the first month as an attribute in the dataset.
        Default is False.

    Returns:
    xr.DataArray or xr.Dataset
        The input xarray object with the time axis set to integers starting from `offset`.
    """

    if first_month_attr:
        # Extract the first forecast year-month pair
        first_month = xro[time_dim].values[0]

        # Add the first month as an attribute to the dataset
        xro.attrs["first_month"] = str(first_month)

    xro[time_dim] = np.arange(offset, offset + xro[time_dim].size)
    return xro


# Function for loading the observations
def load_obs_data(
    obs_variable: str,
    regrid_obs_path: str,
    start_year: int,
    end_year: int,
    avg_period: int,
    grid: dict,
):
    """
    Function for loading the observations

    Parameters
    ----------

    obs_variable: str
        The variable to load from the model data
        E.g. 'si10' for sfcWind

    regrid_obs_path: str
        The path to the regridded observations

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    avg_period: int
        The number of years to average over
        E.g. 1 for 1-year, 5 for 5-year, etc.

    grid: dict
        The grid to load the data over

    Returns

    obs_data: np.array
        The observations
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    # Set up the new years
    new_years = []

    # Set the n years
    n_years = len(years)

    # Extract the lon and lat bounds
    lon1, lon2 = grid["lon1"], grid["lon2"]
    lat1, lat2 = grid["lat1"], grid["lat2"]

    # Open the obs
    obs = xr.open_mfdataset(regrid_obs_path, combine="by_coords", parallel=True)[
        obs_variable
    ]

    # Combine the first two expver variables
    obs = obs.sel(expver=1).combine_first(obs.sel(expver=5))

    # Extract the time series for the gridbox
    obs = obs.sel(lat=slice(lat1, lat2), lon=slice(lon1, lon2)).mean(dim=("lat", "lon"))

    # Convert numpy.datetime64 to datetime
    final_time = obs["time"][-1].values.astype(str)

    # Extract the year and month
    final_year = int(final_time[:4])
    final_month = int(final_time[5:7])

    # If the final time is not november or december
    if not (final_month == 11 or final_month == 12):
        # Check that the final year - avg_period is not less than the end year
        if (final_year - 1) - avg_period < end_year:
            # Set the end year to the final year - avg_period
            end_year = (final_year - 1) - avg_period
    else:
        print("The final year has november or december")

    # Set the new years
    new_years = np.arange(start_year, end_year + 1)

    # Print the first time of the new years
    print("First time:", new_years[0])
    print("Last time:", new_years[-1])

    # Print the years we are slicing over
    print("Slicing over:", f"{start_year}-12-01", f"{start_year + avg_period}-03-30")

    # Extract the time slice between
    obs_slice = obs.sel(
        time=slice(f"{start_year}-12-01", f"{start_year + avg_period}-11-30")
    )

    # Extract the nmonths
    n_months = len(obs_slice["time"])

    # Print the number of months
    print("Number of months:", n_months)

    # Form the empty array to store the data
    obs_data = np.zeros([len(new_years), n_months])

    # Print the shape of the obs data
    print("Shape of obs data:", obs_data.shape)

    # Loop over the years
    for year in tqdm(new_years, desc="Processing years"):
        # We only have obs upt to jjuly 2023

        # Extract the time slice between
        obs_slice = obs.sel(time=slice(f"{year}-12-01", f"{year + avg_period}-11-30"))

        # Extract the data
        obs_data[year - start_year, :] = obs_slice.values

    # Print the shape of the obs data
    print("Shape of obs data:", obs_data.shape)

    # Set up the obs years
    obs_years = np.arange(new_years[0], new_years[-1] + 1)

    # Return the obs data
    return obs_data, obs_years


# Function for calculating the obs_stats
def calculate_obs_stats(
    obs_data: np.ndarray, start_year: int, end_year: int, avg_period: int, grid: dict
):
    """
    Calculate the observations stats

    Parameters
    ----------

        obs_data: np.ndarray
            The observations data
            With shape (nyears, nmonths)

        start_year: int
            The start year for the data
            E.g. 1961

        end_year: int
            The end year for the data
            E.g. 1990

        avg_period: int
            The number of years to average over
            E.g. 1 for 1-year, 5 for 5-year, etc.

        grid: dict
            The grid to load the data over

    Returns
    -------

        obs_stats: dict
            A dictionary containing the obs stats

    """

    # Define the mdi
    mdi = -9999.0

    # Define the obs stats
    obs_stats = {
        "avg_period_mean": [],
        "mean": mdi,
        "sigma": mdi,
        "skew": mdi,
        "kurt": mdi,
        "start_year": mdi,
        "end_year": mdi,
        "avg_period": mdi,
        "grid": mdi,
        "min_20": mdi,
        "max_20": mdi,
        "min_10": mdi,
        "max_10": mdi,
        "min_5": mdi,
        "max_5": mdi,
        "min": mdi,
        "max": mdi,
        "sample_size": mdi,
    }

    # Set the start year
    obs_stats["start_year"] = start_year

    # Set the end year
    obs_stats["end_year"] = end_year

    # Set the avg period
    obs_stats["avg_period"] = avg_period

    # Set the grid
    obs_stats["grid"] = grid

    # Process the obs
    obs_copy = obs_data.copy()

    # Take the mean over the 1th axis (i.e. over the 12 months)
    obs_year = np.mean(obs_copy, axis=1)

    # Set the average period mean
    obs_stats["avg_period_mean"] = obs_year

    # Get the sample size
    obs_stats["sample_size"] = len(obs_year)

    # Take the mean over the 0th axis (i.e. over the years)
    obs_stats["mean"] = np.mean(obs_year)

    # Take the standard deviation over the 0th axis (i.e. over the years)
    obs_stats["sigma"] = np.std(obs_year)

    # Take the skewness over the 0th axis (i.e. over the years)
    obs_stats["skew"] = stats.skew(obs_year)

    # Take the kurtosis over the 0th axis (i.e. over the years)
    obs_stats["kurt"] = stats.kurtosis(obs_year)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats["min"] = np.min(obs_year)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats["max"] = np.max(obs_year)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats["min_5"] = np.percentile(obs_year, 5)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats["max_5"] = np.percentile(obs_year, 95)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats["min_10"] = np.percentile(obs_year, 10)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats["max_10"] = np.percentile(obs_year, 90)

    # Take the min over the 0th axis (i.e. over the years)
    obs_stats["min_20"] = np.percentile(obs_year, 20)

    # Take the max over the 0th axis (i.e. over the years)
    obs_stats["max_20"] = np.percentile(obs_year, 80)

    # Return the obs stats
    return obs_stats


# Write a function which does the plotting
def plot_events(
    model_data: np.ndarray,
    obs_data: np.ndarray,
    obs_stats: dict,
    start_year: int,
    end_year: int,
    bias_adjust: bool = True,
    figsize_x: int = 10,
    figsize_y: int = 10,
    do_detrend: bool = False,
):
    """
    Plots the events on the same axis.

    Parameters
    ----------

    model_data: np.ndarray
        The model data
        With shape (nyears, nmembers, nmonths)

    obs_data: np.ndarray
        The observations data
        With shape (nyears, nmonths)

    obs_stats: dict
        A dictionary containing the obs stats

    start_year: int
        The start year for the data
        E.g. 1961

    end_year: int
        The end year for the data
        E.g. 1990

    bias_adjust: bool
        Whether to bias adjust the model data
        Default is True

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    do_detrend: bool
        Whether to detrend the data
        Default is False

    Returns
    -------
    None
    """

    # Set up the years
    years = np.arange(start_year, end_year + 1)

    if len(model_data.shape) == 3:
        # Take the mean over the 2th axis (i.e. over the months)
        # For the model data
        model_year = np.mean(model_data, axis=2)
    else:
        # For the model data
        model_year = model_data

    if len(obs_data.shape) == 2:
        # Take the mean over the 1th axis (i.e. over the members)
        # For the obs data
        obs_year = np.mean(obs_data, axis=1)
    else:
        # For the obs data
        obs_year = obs_data

    # if the bias adjust is True
    if bias_adjust:
        print("Bias adjusting the model data")

        # Flatten the model data
        model_flat = model_year.flatten()

        # Find the difference between the model and obs
        bias = np.mean(model_flat) - np.mean(obs_year)

        # Add the bias to the model data
        model_year = model_year - bias

    # If the detrend is True
    if do_detrend:
        print("Detrending the data")

        # Use the scipy detrend function
        model_year = signal.detrend(model_year, axis=0)

        # Use the scipy detrend function
        obs_year = signal.detrend(obs_year, axis=0)

        # Calculate the new minimum for the obs
        obs_stats["min"] = np.min(obs_year)

        # Calculate the 20th percentile for the obs
        obs_stats["min_20"] = np.percentile(obs_year, 20)

    # Set the figure size
    plt.figure(figsize=(figsize_x, figsize_y))

    # Plot the model data
    for i in range(model_year.shape[1]):

        # Separate data into two groups based on the condition
        below_20th = model_year[:, i] < obs_stats["min_20"]
        above_20th = ~below_20th

        # Plot points below the 20th percentile with a label
        plt.scatter(
            years[below_20th],
            model_year[below_20th, i],
            color="blue",
            alpha=0.8,
            label="model wind drought" if i == 0 else None,
        )

        # Plot points above the 20th percentile without a label
        plt.scatter(
            years[above_20th],
            model_year[above_20th, i],
            color="grey",
            alpha=0.8,
            label="HadGEM3-GC31-MM" if i == 0 else None,
        )

    # Plot the obs
    plt.scatter(years, obs_year, color="k", label="ERA5")

    # Plot the 20th percentile
    plt.axhline(obs_stats["min_20"], color="black", linestyle="-")

    # Plot the min
    plt.axhline(obs_stats["min"], color="black", linestyle="--")

    # Add a legend in the upper left
    plt.legend(loc="upper left")

    # Add the axis labels
    plt.xlabel("Year")

    # Add the axis labels
    plt.ylabel("Average Wind speed (m/s)")

    # Show the plot
    plt.show()


# Write a function which does the bootstrapping to calculate the statistics
def model_stats_bs(model: np.ndarray, nboot: int = 10000) -> dict:
    """
    Repeatedly samples the model data with replacement across its members to
    produce many samples equal in length to the reanalysis time series. This
    gives a single pseudo-time series from which the moments of the distribution
    can be calculated. The process is repeated to give a distribution of the
    moments.

    Parameters
    ----------

    model: np.ndarray
        The model data
        With shape (nyears, nmembers, nmonths)

    nboot: int
        The number of bootstrap samples to take
        Default is 10000

    Returns
    -------

    model_stats: dict
        A dictionary containing the model stats with the following keys:
        'mean', 'sigma', 'skew', 'kurt'
    """

    # Set up the model stats
    model_stats = {"mean": [], "sigma": [], "skew": [], "kurt": []}

    # Set up the number of years
    n_years = model.shape[0]

    # Set up the number of members
    n_members = model.shape[1]

    # TODO: Does autocorrelation need to be accounted for?
    # If so, use a block bootstrap

    # Set up the arrays
    mean_boot = np.zeros(nboot)
    sigma_boot = np.zeros(nboot)

    skew_boot = np.zeros(nboot)
    kurt_boot = np.zeros(nboot)

    # Create the indexes for the ensemble members
    index_ens = range(n_members)

    # Loop over the number of bootstraps
    for iboot in tqdm(np.arange(nboot)):
        # print(f"Bootstrapping {iboot + 1} of {nboot}")

        # Create the index for time
        ind_time_this = range(0, n_years)

        # Create an empty array to store the data
        model_boot = np.zeros([n_years])

        # Set the year index
        year_index = 0

        # Loop over the years
        for itime in ind_time_this:

            # Select a random ensemble member
            ind_ens_this = random.choices(index_ens)

            # Logging
            # print(f"itime is {itime} of {n_years}")
            # print(f"year_index is {year_index} of {n_years} "
            #       f"iboot is {iboot} of {nboot} "
            #       f"ind_ens_this is {ind_ens_this}")

            # Extract the data
            model_boot[year_index] = model[itime, ind_ens_this]

            # Increment the year index
            year_index += 1

        # Calculate the mean
        mean_boot[iboot] = np.mean(model_boot)

        # Calculate the sigma
        sigma_boot[iboot] = np.std(model_boot)

        # Calculate the skew
        skew_boot[iboot] = stats.skew(model_boot)

        # Calculate the kurtosis
        kurt_boot[iboot] = stats.kurtosis(model_boot)

    # Append the mean to the model stats
    model_stats["mean"] = mean_boot

    # Append the sigma to the model stats
    model_stats["sigma"] = sigma_boot

    # Append the skew to the model stats
    model_stats["skew"] = skew_boot

    # Append the kurt to the model stats
    model_stats["kurt"] = kurt_boot

    # Return the model stats
    return model_stats


# Write a function which plots the four moments
def plot_moments(
    model_stats: dict,
    obs_stats: dict,
    figsize_x: int = 10,
    figsize_y: int = 10,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the four moments of the distribution of the model data and the
    observations.

    Parameters
    ----------

    model_stats: dict
        A dictionary containing the model stats with the following keys:
        'mean', 'sigma', 'skew', 'kurt'

    obs_stats: dict
        A dictionary containing the obs stats

    figsize_x: int
        The figure size in the x direction
        Default is 10

    figsize_y: int
        The figure size in the y direction
        Default is 10

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Output
    ------

    None
    """

    # Set up the figure as a 2x2
    fig, axs = plt.subplots(2, 2, figsize=(figsize_x, figsize_y))

    ax1, ax2, ax3, ax4 = axs.ravel()

    # Plot the mean
    ax1.hist(model_stats["mean"], bins=100, density=True, color="red", label="model")

    # Plot the mean of the obs
    ax1.axvline(obs_stats["mean"], color="black", linestyle="-", label="ERA5")

    # Calculate the position of the obs mean in the distribution
    obs_mean_pos = stats.percentileofscore(model_stats["mean"], obs_stats["mean"])

    # Add a title
    ax1.set_title(f"Mean, {obs_mean_pos:.2f}%")

    # Include a textbox in the top right corner
    ax1.text(
        0.95,
        0.95,
        "a)",
        transform=ax1.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
        zorder=100,
    )

    # Plot the skewness
    ax2.hist(model_stats["skew"], bins=100, density=True, color="red", label="model")

    # Plot the skewness of the obs
    ax2.axvline(obs_stats["skew"], color="black", linestyle="-", label="ERA5")

    # Calculate the position of the obs skewness in the distribution
    obs_skew_pos = stats.percentileofscore(model_stats["skew"], obs_stats["skew"])

    # Add a title
    ax2.set_title(f"Skewness, {obs_skew_pos:.2f}%")

    # Include a textbox in the top right corner
    ax2.text(
        0.95,
        0.95,
        "b)",
        transform=ax2.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
        zorder=100,
    )

    # Plot the kurtosis
    ax3.hist(model_stats["kurt"], bins=100, density=True, color="red", label="model")

    # Plot the kurtosis of the obs
    ax3.axvline(obs_stats["kurt"], color="black", linestyle="-", label="ERA5")

    # Calculate the position of the obs kurtosis in the distribution
    obs_kurt_pos = stats.percentileofscore(model_stats["kurt"], obs_stats["kurt"])

    # Add a title
    ax3.set_title(f"Kurtosis, {obs_kurt_pos:.2f}%")

    # Include a textbox in the top right corner
    ax3.text(
        0.95,
        0.95,
        "c)",
        transform=ax3.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
        zorder=100,
    )

    # Plot the sigma
    ax4.hist(model_stats["sigma"], bins=100, density=True, color="red", label="model")

    # Plot the sigma of the obs
    ax4.axvline(obs_stats["sigma"], color="black", linestyle="-", label="ERA5")

    # Calculate the position of the obs sigma in the distribution
    obs_sigma_pos = stats.percentileofscore(model_stats["sigma"], obs_stats["sigma"])

    # Add a title
    ax4.set_title(f"Standard deviation, {obs_sigma_pos:.2f}%")

    # Include a textbox in the top right corner
    ax4.text(
        0.95,
        0.95,
        "d)",
        transform=ax4.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="square", facecolor="white", alpha=0.5),
        zorder=100,
    )

    return


# Write a function to plot the distribution of the model and obs data
def plot_distribution(
    model_data: dict,
    obs_data: dict,
    save_dir: str = "/gws/nopw/j04/canari/users/benhutch/plots/",
) -> None:
    """
    Plot the distribution of the model and obs data

    Parameters
    ----------

    model_data: dict
        A dictionary of the model data

    obs_data: dict
        A dictionary containing the obs data

    save_dir: str
        The directory to save the plots to
        Default is "/gws/nopw/j04/canari/users/benhutch/plots/"

    Returns
    -------

    None
    """

    # Assemble the model data into a continuous time series
    model_data_flat = model_data.flatten()

    # plot the model data
    sns.distplot(model_data_flat, label="model", color="red")

    # Plot the obs data
    sns.distplot(obs_data.mean(axis=1), label="obs", color="black")

    # Include a textbox with the sample size
    plt.text(
        0.05,
        0.90,
        f"model N = {model_data_flat.shape[0]}\n" f"obs N = {obs_data.shape[0]}",
        transform=plt.gca().transAxes,
        bbox=dict(facecolor="white", alpha=0.5),
    )

    # Add a legend
    plt.legend()

    # Add a title
    # TODO: hard coded title
    plt.title("Distribution of 10m wind speed")

    return
