"""
process_spatial_fid_testing.py
=======================

This script processes obs and model data into arrays suitable for performing 
the spatial fidelity test. These arrays are then saved to a directory on the GWS
before later being processed in the spatial fidelity test script.

Note: this case considers only the first winter (DJF, or ONDJFM) for the 
initialized decadal hindcasts.

Usage:
------

    $ python process_spatial_fid_testing.py \
        --variable "sfcWind" \
        --region "UK" \
        --init_year "1960" \
        --season "DJF" \
        --winter "1"

Arguments:
----------

    --variable : str : variable name (e.g. tas, pr, psl)
    --region : str : region name (e.g. UK, France, Germany)
    --init_year : int : initialisation year (e.g. 1960)
    --season : str : season name (e.g. DJF, MAM, JJA, SON)
    --winter : int : winter number (e.g. 1, 2, 3)
    
Returns:
--------

    obs_data_array : np.array : processed obs data array for the given variable, region, and season
    model_data_array : np.array : processed model data array for the given variable, region, and season

    Both saved to the /gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs /obs and /model folders.

Author:
-------

    Ben W. Hutchins, University of Reading, February 2025
    
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
import iris

# Specific imports
from tqdm import tqdm
from datetime import datetime, timedelta

# # Specific imports
# from ncdata.iris_xarray import cubes_to_xarray, cubes_from_xarray

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
import functions as funcs


# Define the main function
def main():

    # Set up the start time
    start_time = time.time()

    # Hard-code the test path
    test_obs_wind_path = (
        "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_wind_daily_1960_1965.nc"
    )
    output_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs"
    meta_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/metadata"
    members_list = [
        "r10i1p1f2",
        "r1i1p1f2",
        "r2i1p1f2",
        "r3i1p1f2",
        "r4i1p1f2",
        "r5i1p1f2",
        "r6i1p1f2",
        "r7i1p1f2",
        "r8i1p1f2",
        "r9i1p1f2",
    ]

    # Define the parser
    parser = argparse.ArgumentParser(
        description="Process obs and model data into arrays suitable for performing the spatial fidelity test."
    )

    # Define the arguments
    parser.add_argument(
        "--variable", type=str, help="variable name (e.g. tas, pr, psl)"
    )
    parser.add_argument(
        "--region", type=str, help="region name (e.g. UK, France, Germany)"
    )
    parser.add_argument("--init_year", type=int, help="initialisation year (e.g. 1960)")
    parser.add_argument(
        "--season", type=str, help="season name (e.g. DJF, MAM, JJA, SON)"
    )
    parser.add_argument("--winter", type=int, help="winter number (e.g. 1, 2, 3)")

    # Parse the arguments
    args = parser.parse_args()

    # Print the args
    print("========================================")
    print(f"Variable: {args.variable}")
    print(f"Region: {args.region}")
    print(f"Init Year: {args.init_year}")
    print(f"Season: {args.season}")
    print(f"Winter: {args.winter}")
    print("========================================")

    # if the variable is not sfcWind exit with an error
    if args.variable != "sfcWind":
        print("Error: variable must be sfcWind")
        sys.exit()

    # if init year is not 1960, 1961, 1962, 1963, 1964, or 1965, exit with an error
    if args.init_year not in [1960, 1961, 1962, 1963, 1964, 1965]:
        print("Error: init year must be 1960, 1961, 1962, 1963, 1964, or 1965")
        sys.exit()

    # Set up the months depending on the season
    if args.season == "DJF":
        months = [12, 1, 2]
    elif args.season == "ONDJFM":
        months = [10, 11, 12, 1, 2, 3]
    else:
        raise ValueError("Season must be DJF or ONDJFM")

    # Set up the region
    # if the region is UK
    if args.region == "UK":
        gridbox = {
            "lat1": 50,
            "lat2": 60,
            "lon1": -7,
            "lon2": 5,
        }
    else:
        raise ValueError("Region not recognised")

    # set up the array fnames
    obs_array_fname = (
        f"ERA5_{args.variable}_{args.region}_{args.init_year}_{args.season}_day.npy"
    )
    model_array_fname = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_{args.init_year}_{args.season}_day.npy"

    # set up the full obs and model atrray paths
    obs_array_path = os.path.join(output_dir, "obs", obs_array_fname)
    model_array_path = os.path.join(output_dir, "model", model_array_fname)

    # if the obs array already exists, exit with an error
    if os.path.exists(obs_array_path):
        print(f"Error: {obs_array_path} already exists")
        sys.exit()

    # if the model array already exists, exit with an error
    if os.path.exists(model_array_path):
        print(f"Error: {model_array_path} already exists")
        sys.exit()

    # Load the obs data
    # FIXME: Hardcoded for now
    obs_data = iris.load_cube(test_obs_wind_path, "si10")

    # Loop over the members list
    # Set up an empty list to store the data
    model_ds_list = []

    # loop over the members
    for m, member_this in tqdm(enumerate(members_list)):

        # print member this
        print(member_this)

        # Load in the data
        model_ds_this = funcs.load_model_data_xarray(
            model_variable=args.variable,
            model="HadGEM3-GC31-MM",
            experiment="dcppA-hindcast",
            start_year=args.init_year,
            end_year=args.init_year,
            first_fcst_year=args.init_year + 1,
            last_fcst_year=args.init_year + 10,
            months=months,
            member=member_this,
            frequency="day",
            parallel=False,
        )

        # Append to the list
        model_ds_list.append(model_ds_this)

    # Combine the list by members
    model_ds = xr.concat(model_ds_list, dim="member")

    # Turn into a cube
    model_cube = model_ds[args.variable].to_iris()

    # Make sure this is on a -180 to 180 grid
    model_cube = model_cube.intersection(longitude=(-180, 180))

    # print the model cube
    print(model_cube)

    # print the obs data
    print(obs_data)

    # Constrain the obs to the winter
    # FIXME: hardcoded as DJF for now
    obs_data = obs_data.extract(
        iris.Constraint(
            time=lambda cell: datetime(int(args.init_year), 12, 1)
            <= cell.point
            < datetime(int(args.init_year) + 1, 3, 1)
        )
    )

    # print the obs data
    print(obs_data)

    # Set up the leads to extract from the model data
    leads_djf_model = np.arange(
        31 + ((args.winter - 1) * 360), 31 + 90 + ((args.winter - 1) * 360)
    )

    # print the min lead we are extracting
    print("=================================")
    print(f"Min lead: {leads_djf_model[0]}")
    print(f"Max lead: {leads_djf_model[-1]}")
    print("=================================")

    # Extract the relevant leads
    model_cube = model_cube.extract(iris.Constraint(lead=leads_djf_model))

    # Extract the data for the gridbox
    obs_data_box = obs_data.intersection(
        longitude=(gridbox["lon1"], gridbox["lon2"]),
        latitude=(gridbox["lat1"], gridbox["lat2"]),
    )

    # Extract the data for the gridbox
    model_cube_box = model_cube.intersection(
        longitude=(gridbox["lon1"], gridbox["lon2"]),
        latitude=(gridbox["lat1"], gridbox["lat2"]),
    )

    # Regrid the obs data to the model data
    obs_data_box_regrid = obs_data_box.regrid(model_cube_box, iris.analysis.Linear())

    # Set up the name for the lats array
    lats_array_fname = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_{args.init_year}_{args.season}_day_lats.npy"
    lons_array_fname = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_{args.init_year}_{args.season}_day_lons.npy"
    members = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_{args.init_year}_{args.season}_day_members.npy"
    obs_times = f"ERA5_{args.variable}_{args.region}_{args.init_year}_{args.season}_day_times.npy"

    # Set up the paths for the lats and lons
    lats_array_path = os.path.join(meta_dir, lats_array_fname)
    lons_array_path = os.path.join(meta_dir, lons_array_fname)

    # Set up the path for the members
    members_path = os.path.join(meta_dir, members)

    # Set up the path for the obs times
    obs_times_path = os.path.join(meta_dir, obs_times)

    # if the lats array already exists, exit with an error
    if os.path.exists(lats_array_path):
        print(f"{lats_array_path} already exists")
    else:
        # save the lats array
        np.save(lats_array_path, model_cube_box.coord("latitude").points)

    # if the lons array already exists, exit with an error
    if os.path.exists(lons_array_path):
        print(f"{lons_array_path} already exists")
    else:
        # save the lons array
        np.save(lons_array_path, model_cube_box.coord("longitude").points)

    # if the members array already exists, exit with an error
    if os.path.exists(members_path):
        print(f"{members_path} already exists")
    else:
        # save the members array
        np.save(members_path, members_list)

    # if the obs times array already exists, exit with an error
    if os.path.exists(obs_times_path):
        print(f"{obs_times_path} already exists")
    else:
        # save the obs times array
        np.save(obs_times_path, obs_data_box_regrid.coord("time").points)

    # Set up the obs data array
    obs_data_array = obs_data_box_regrid.data
    obs_data_array = obs_data_array.filled(np.nan)

    # Set up the model data array
    model_data_array = model_cube_box.data
    model_data_array = model_data_array.filled(np.nan)

    # # print the shape of the data
    # print("=================================")
    # print("Shape of the obs data array:")
    # print(obs_data_array.shape)
    # print("Shape of model data array:")
    # print(model_data_array.shape)
    # print("Values of the obs data array:")
    # print(obs_data_array)
    # # print the values of the model data array
    # print("Values of the model data array:")
    # print(model_data_array)
    # print("=================================")

    # Save the obs data array
    np.save(obs_array_path, obs_data_array)

    # Save the model data array
    np.save(model_array_path, model_data_array)

    # Set up the end time
    end_time = time.time()

    # Print the time taken
    print(f"Time taken: {end_time - start_time} seconds")

    return None


if __name__ == "__main__":
    main()
