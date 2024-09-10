#!/usr/bin/env python

"""
process_UNSEEN.py
=================

This script takes as input the variable name (e.g. "tas"), the country (e.g.
"United Kingdom"), the season (e.g. "ONDJFM"), the first year (e.g. 1960) and
the last year (e.g. 2014) and then loads the model and observed data for the
period, variable and region specified. It then performs the fidelity testing
(mean, sigma, skewness, kurtosis) for the model and observed data and saves the
resulting plot to the output directory.

Usage:
------

    $ python process_UNSEEN.py --variable tas --country "United Kingdom" --season ONDJFM --first_year 1960 --last_year 2014 --model_fcst_year 1

Arguments:
----------

    --variable: str
        The variable name (e.g. "tas").

    --country: str
        The country name (e.g. "United Kingdom").

    --season: str
        The season name (e.g. "ONDJFM").

    --first_year: int
        The first year of the period (e.g. 1960).

    --last_year: int
        The last year of the period (e.g. 2014).

    --model_fcst_year: int
        The forecast year of the model data to extract the season from (e.g.
        for first NDJFM from s1960 initialization, set to 0). First complete 
        ONDJFM season starts in 1961, so set to 1.

Returns:
--------

    A plot of the fidelity testing results for the model and observed data.
"""

# Local imports
import os
import sys
import time
import argparse

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
from datetime import datetime, timedelta

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
import functions as funcs


# Define the main function
def main():
    # Start the timer
    start = time.time()

    # Set up the argument parser
    parser = argparse.ArgumentParser(description="Process UNSEEN data.")
    parser.add_argument("--variable", type=str, help="The variable name (e.g. tas).")
    parser.add_argument("--country", type=str, help="The country name (e.g. United Kingdom).")
    parser.add_argument("--season", type=str, help="The season name (e.g. ONDJFM).")
    parser.add_argument("--first_year", type=int, help="The first year of the period (e.g. 1960).")
    parser.add_argument("--last_year", type=int, help="The last year of the period (e.g. 2014).")
    parser.add_argument("--model_fcst_year", type=int, help="The forecast year of the model data to extract the season from (e.g. for first NDJFM from s1960 initialization, set to 0). First complete ONDJFM season starts in 1961, so set to 1.")

    # Parse the arguments
    args = parser.parse_args()

    # print the arguments
    print(f"Variable: {args.variable}")
    print(f"Country: {args.country}")
    print(f"Season: {args.season}")
    print(f"First year: {args.first_year}")
    print(f"Last year: {args.last_year}")
    print(f"Model forecast year: {args.model_fcst_year}")

    # if country contains a _
    # e.g. United_Kingdom
    # replace with a space
    if "_" in args.country:
        args.country = args.country.replace("_", " ")

    # set up the obs variable depending on the variable
    if args.variable == "tas":
        obs_var = "t2m"
    elif args.variable == "sfcWind":
        obs_var = "si10"
    else:
        raise ValueError("Variable not recognised")
    
    # Set up the months depending on the season
    if args.season == "DJF":
        months = [12, 1, 2]
    elif args.season == "MAM":
        months = [3, 4, 5]
    elif args.season == "JJA":
        months = [6, 7, 8]
    elif args.season == "SON":
        months = [9, 10, 11]
    elif args.season == "ONDJFM":
        months = [10, 11, 12, 1, 2, 3]
    elif args.season == "NDJFM":
        months = [11, 12, 1, 2, 3]
    else:
        raise ValueError("Season not recognised")

    # set up the hard coded args
    model = "HadGEM3-GC31-MM"
    experiment = "dcppA-hindcast"
    freq = "Amon"

    # Depending on the model forecast year
    # set the leads to extract from the model
    if args.model_fcst_year == 0 and args.season == "NDJFM":
        leads = [1, 2, 3, 4, 5]
    elif args.model_fcst_year == 1 and args.season == "ONDJFM":
        leads = [12, 13, 14, 15, 16, 17]
    else:
        raise ValueError("Model forecast year and season combination not recognised")

    # Set up the output directory for the dfs
    output_dir_dfs = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs"

    # Set up the name for the obs df
    obs_df_name = f"ERA5_obs_{args.variable}_{args.country}_{args.season}_{args.first_year}_{args.last_year}.csv"

    # Set up the name for the model df
    model_df_name = f"{model}_{args.variable}_{args.country}_{args.season}_{args.first_year}_{args.last_year}_{experiment}_{freq}.csv"

    # form the full paths for the dfs
    obs_df_path = os.path.join(output_dir_dfs, obs_df_name)
    model_df_path = os.path.join(output_dir_dfs, model_df_name)

    # if the obs df exists and the model df exists
    if os.path.exists(obs_df_path) and os.path.exists(model_df_path):
        print("Loading the observed and model dfs")
        
        # load the dfs
        obs_df = pd.read_csv(obs_df_path)
        model_df = pd.read_csv(model_df_path)
        
        # print("Loaded the dfs")
        # print("----------------")
        # print("Script complete")
    else:
        # Set up the path to the ERA5 data
        # if the variable is tas
        if args.variable == "tas":
            # already regridded!
            obs_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/t2m_ERA5_regrid_HadGEM.nc"
        # if the variable is sfcWind
        elif args.variable == "sfcWind":
            # needs regridding
            obs_path = "/gws/nopw/j04/canari/users/benhutch/ERA5/surface_wind_ERA5.nc"
        else:
            raise ValueError("Variable not recognised") 

        # Load the model ensemble
        model_ds = funcs.load_model_data_xarray(
            model_variable=args.variable,
            model=model,
            experiment=experiment,
            start_year=args.first_year,
            end_year=args.last_year,
            first_fcst_year=int(args.first_year) + 1,
            last_fcst_year=int(args.first_year) + 2,
            months=months,
            frequency=freq,
            parallel=False,
        )

        # print that we have loaded the model data
        print("Loaded the model data")

        # # Get the size of the model data in bytes
        # size_in_bytes = model_ds[args.variable].size * model_ds[args.variable].dtype.itemsize

        # # Convert bytes to gigabytes
        # size_in_gb = size_in_bytes / (1024 ** 3)

        # # Print the size
        # print(f"Model data size: {size_in_gb} GB")

        # Modify member coordiante before conbersion to iris
        model_ds['member'] = model_ds['member'].str[1:-6].astype(int)

        # convert to an iris cube
        model_cube = model_ds[args.variable].squeeze().to_iris()

        # Load the observed data
        obs_ds = xr.open_mfdataset(
            obs_path,
            combine="by_coords",
            parallel=False,
            engine="netcdf4",
        )

        # Restrict the time to the region we are interested in
        obs_ds = obs_ds.sel(time=slice(f"{int(args.first_year)}-{months[0]}-01", f"{int(args.last_year) + 1}-{months[-1]}-31"))

        # If expver is present in the observations
        if "expver" in obs_ds.coords:
            # Combine the first two expver variables
            obs_ds = obs_ds.sel(expver=1).combine_first(obs_ds.sel(expver=5))

        # # Get the size of the observed data in bytes
        # size_in_bytes = obs_ds[obs_var].size * obs_ds[obs_var].dtype.itemsize

        # # Convert bytes to gigabytes
        # size_in_gb = size_in_bytes / (1024 ** 3)

        # # Print the size
        # print(f"Observed data size: {size_in_gb} GB")

        # convert to an iris cube
        obs_cube = obs_ds[obs_var].squeeze().to_iris()

        # if the lats and lons are not the same
        if not model_cube.coord("latitude").shape == obs_cube.coord("latitude").shape or not model_cube.coord("longitude").shape == obs_cube.coord("longitude").shape:
            print("Regridding model data")
            # regrid the obs cube to the model cube
            obs_cube = obs_cube.regrid(model_cube, iris.analysis.Linear())

        # make sure the cubes are correct in -180 to 180 lons
        obs_cube = obs_cube.intersection(longitude=(-180, 180))
        model_cube = model_cube.intersection(longitude=(-180, 180))

        # create the mask
        MASK_MATRIX = funcs.create_masked_matrix(
            country=args.country,
            cube=model_cube,
        )

        # print the shape of the mask matrix
        print(f"Mask matrix shape: {MASK_MATRIX.shape}")

        # print the sum of the mask matrix
        print(f"Mask matrix sum: {np.sum(MASK_MATRIX)}")

        # Apply the mask to the observed data
        obs_values = obs_cube.data * MASK_MATRIX
        model_values = model_cube.data * MASK_MATRIX

        # Where there are zeros in the mask we want to set these to Nans
        obs_values_masked = np.where(MASK_MATRIX == 0, np.nan, obs_values)
        model_values_masked = np.where(MASK_MATRIX == 0, np.nan, model_values)

        # Take the Nanmean of the data
        obs_values = np.nanmean(obs_values_masked, axis=(1, 2))
        model_values = np.nanmean(model_values_masked, axis=(3, 4))

        # Set up the ref time for the observations
        ref_time_obs = datetime(1900, 1, 1)

        # Extract the obs time points
        obs_time_points = obs_cube.coord("time").points

        # convert to obs datetimes
        obs_datetimes = [ref_time_obs + timedelta(hours=int(tp)) for tp in obs_time_points]

        # Set up a dataframe for the observations
        obs_df = pd.DataFrame(
            {
                "time": obs_datetimes,
                "obs": obs_values,
            }
        )

        # set up an empty df for the model data
        model_df = pd.DataFrame()

        # extract the init, member and lead time points
        init_years = model_cube.coord("init").points
        members = model_cube.coord("member").points
        lead_times = model_cube.coord("lead").points

        # loop through the inits, members and leadtimes
        for i, init_year in enumerate(init_years):
            for m, member in enumerate(members):
                for l, lead_time in enumerate(lead_times):
                    # get the model data
                    model_data = model_values[i, m, l]
                    
                    # set up the model df this
                    model_df_this = pd.DataFrame(
                        {
                            'init_year': [init_year],
                            'member': [member],
                            'lead': [lead_time],
                            'data': [model_data],
                        },
                    )

                    # concat to the model df
                    model_df = pd.concat([model_df, model_df_this])

        # print the head of the obs df
        print(obs_df.head())

        # print the head of the model df
        print(model_df.head())

        # save the dfs
        if not os.path.exists(output_dir_dfs):
            os.makedirs(output_dir_dfs)

        # save the obs df
        if not os.path.exists(obs_df_path):
            print("Saving the observed df")
            obs_df.to_csv(obs_df_path, index=False)

        # save the model df
        if not os.path.exists(model_df_path):
            print("Saving the model df")
            model_df.to_csv(model_df_path, index=False)

    # constrain the obs df to only months 10, 11, 12, 1, 2, 3
    # esnure that the time is a datetime
    obs_df["time"] = pd.to_datetime(obs_df["time"])
    
    # set the time as the index for the obs df
    obs_df.set_index("time", inplace=True)

    # # remove the name of the index
    # obs_df.index.name = None

    # print the head of the obs df
    print(obs_df.head())

    # constrain to the months
    obs_df = obs_df[obs_df.index.month.isin(months)]

    # if months contains 12, 1 in sequence
    if 12 in months and 1 in months:
        # shift back by months and take the annual mean
        obs_df = obs_df.shift(-int(months[-1])).resample("A").mean()

    # if there are any Nans in the obs df, drop them
    obs_df.dropna(inplace=True)

    # set up time as a column
    obs_df.reset_index(inplace=True)

    # print the head of the obs df
    print(obs_df.head())

    # print the tail of the obs df
    print(obs_df.tail())

    # create a new model df for subsetting to first ONDJFM
    model_df_ondjfm = pd.DataFrame()

    # loop over the unique init years and members in model_df
    for init_year in model_df['init_year'].unique():
        for member in model_df['member'].unique():
            # extract the model data
            model_data = model_df[(model_df['init_year'] == init_year) & (model_df['member'] == member)]

            # subset to lead values [12, 13, 14, 15, 16, 17] and take the mean
            # first complete ONDJFM season
            # FIXME: Hardcoded for now
            model_data = model_data[model_data['lead'].isin(leads)]

            mean_data = model_data['data'].mean()

            # create a dataframe this
            model_data_this = pd.DataFrame(
                {
                    'init_year': [init_year],
                    'member': [member],
                    'data': [mean_data]
                }
            )

            model_df_ondjfm = pd.concat([model_df_ondjfm, model_data_this])

    # # print the head of the model df
    # print(model_df_ondjfm.head())

    # # print the tail of the model df
    # print(model_df_ondjfm.tail())

    # # print the shape of the model df
    # print(model_df_ondjfm.shape)

    # plot the fidelity testing
    funcs.plot_fidelity(
        obs_df=obs_df,
        model_df=model_df_ondjfm,
        obs_val_name="obs",
        model_val_name="data",
        obs_time_name="time",
        model_time_name="init_year",
        model_member_name="member",
        model_lead_name=None,
        nboot=10000,
        figsize=(10, 8),
        save_dir="/gws/nopw/j04/canari/users/benhutch/plots/",
        fname_root=f"fidelity_{args.variable}_{args.country}_{args.season}_{args.first_year}_{args.last_year}_{model}_{experiment}_{freq}_fcst_year_{args.model_fcst_year}",
    )

    print("----------------")
    # print the amount of time taken
    print(f"Time taken to load model data: {time.time() - start}")
    print("----------------")
    print("Script complete")

# Run the main function
if __name__ == "__main__":
    main()