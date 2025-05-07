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
        --winter "1" \
        --frequency "day"

Arguments:
----------

    --variable : str : variable name (e.g. tas, pr, psl)
    --region : str : region name (e.g. UK, France, Germany)
    --init_year : int : initialisation year (e.g. 1960)
    --season : str : season name (e.g. DJF, MAM, JJA, SON)
    --winter : int : winter number (e.g. 1, 2, 3)
    --frequency : str : frequency of the data (e.g. day, month, year)
    
Returns:
--------

    obs_data_array : np.array : processed obs data array for the given variable, region, and season
    model_data_array : np.array : processed model data array for the given variable, region, and season

    Both saved to the /gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs /obs and /model folders.

Author:
-------

    Ben W. Hutchins, University of Reading, February 2025
    
"""
#%%
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
import datetime
import cf_units

# Specific imports
from tqdm import tqdm
from datetime import datetime, timedelta
from iris.util import equalise_attributes, equalise_cubes, describe_diff, unify_time_units

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
    # test_obs_wind_path = (
    #     "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_wind_daily_1952_2020.nc"
    # )

    # test obs tas path
    test_obs_tas_path = (
        "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_t2m_daily_1950_2020.nc"
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
    # print("=================================")
    # print("TESTING: Using only one member")
    # print("=================================")
    # # --------------------------
    # # NOTE: For testing purposes
    # # --------------------------
    members_list_test = ["r1i1p1f2"]

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
    parser.add_argument(
        "--frequency", type=str, help="frequency of the data (e.g. day, month, year)"
    )

    # Check if running in IPython
    if "ipykernel_launcher" in sys.argv[0]:
        # Manually set arguments for IPython
        args = parser.parse_args(["--variable", "uas", "--region", "Europe", "--init_year", "1960", "--season", "DJF", "--winter", "1", "--frequency", "day"])
    else:
        # Parse arguments normally
        args = parser.parse_args()

    # Print the args
    print("========================================")
    print(f"Variable: {args.variable}")
    print(f"Region: {args.region}")
    print(f"Init Year: {args.init_year}")
    print(f"Season: {args.season}")
    print(f"Winter: {args.winter}")
    print(f"Frequency: {args.frequency}")
    print("========================================")

    # if the args.frequency is Amon
    if args.frequency == "Amon":
        print("Setting different output dir")

        # Set up the new amon output dir
        output_dir = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_arrs/Amon"

        # if the output dir does not exist, create it
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    # # if the variable is not sfcWind exit with an error
    # if args.variable != "sfcWind":
    #     print("Error: variable must be sfcWind")
    #     sys.exit()

    # # if init year is not 1960, 1961, 1962, 1963, 1964, or 1965, exit with an error
    # if args.init_year not in [1960, 1961, 1962, 1963, 1964, 1965]:
    #     print("Error: init year must be 1960, 1961, 1962, 1963, 1964, or 1965")
    #     sys.exit()

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
    elif args.region == "Europe":
        gridbox = {
            "lon1": -11,  # degrees east
            "lon2": 30,
            "lat1": 35,  # degrees north
            "lat2": 70,
        }
    elif args.region == "NA":
        gridbox = {
            "lon1": -50,  # degrees east
            "lon2": 30,
            "lat1": 30,  # degrees north
            "lat2": 80,
        }
    elif args.region == "global":
        print("Global region selected")
        gridbox = None
    else:
        raise ValueError("Region not recognised")

    # set up the current time
    current_time = datetime.now()

    # format the current time
    # as YYYYMMDD_HHMMSS
    current_time = current_time.strftime("%Y%m%d_%H%M%S")

    # set up the array fnames
    obs_array_fname = (
        f"ERA5_{args.variable}_{args.region}_{args.init_year}_{args.season}_day_{current_time}.npy"
    )
    model_array_fname = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_{args.init_year}_{args.season}_{args.frequency}_{current_time}.npy"

    # form the directory
    obs_dir = os.path.join(output_dir, "obs")
    model_dir = os.path.join(output_dir, "model")

    # if the obs dir does not exist, create it
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir)

    # if the model dir does not exist, create it
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # set up the full obs and model atrray paths
    obs_array_path = os.path.join(obs_dir, obs_array_fname)
    model_array_path = os.path.join(model_dir, model_array_fname)

    # if the obs array already exists, exit with an error
    if os.path.exists(obs_array_path):
        print(f"Error: {obs_array_path} already exists")

    # if the model array already exists, exit with an error
    if os.path.exists(model_array_path):
        print(f"Error: {model_array_path} already exists")

    # # Load the obs data
    # # FIXME: Hardcoded for now
    # # obs_data = iris.load_cube(test_obs_wind_path, "si10")
    # obs_data = iris.load_cube(test_obs_tas_path, "t2m")

    if args.variable == "tas":
        # Set up the obs_path
        obs_path = (
            "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_t2m_daily_1950_2020.nc"
        )

        # Load the obs data
        obs_data = iris.load_cube(obs_path, "t2m")
    elif args.variable == "sfcWind":
        # Set up the obs_path
        obs_path = (
            "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_wind_daily_1952_2020.nc"
        )

        # Load the obs data
        obs_data = iris.load_cube(obs_path, "si10")
    elif args.variable == "psl":
        # Set up the obs path
        obs_path = (
            "/gws/nopw/j04/canari/users/benhutch/ERA5/ERA5_msl_daily_1960_2020_daymean.nc"
        )

        # Load the data
        obs_data = iris.load_cube(obs_path, "msl")
    elif args.variable in ["uas", "vas"]:
        print("Loading U10 and V10 data for model only")
    else:
        raise ValueError("Variable not recognised")

    # Set up the remaining years
    remaining_years = [str(year) for year in range(2021, 2025 + 1)]

    # Set up the path to the observed data
    remaining_files_dir = os.path.join("/gws/nopw/j04/canari/users/benhutch/ERA5/", "year_month")

    # Set up an empty cubelist
    obs_cubelist_first = []
    obs_cubelist_u10 = []
    obs_cubelist_v10 = []

    # Loop over the remaining years
    # for year in tqdm(remaining_years):
    #     for month in ["01", "02", "12"]:
    #         # if the year is 2025 and the month is 12, then skip
    #         if year == "2025" and month == "12":
    #             continue
            
    #         # Set up the fname this
    #         fname_this = f"ERA5_EU_T_U10_V10_msl{year}_{month}_daymean.nc"

    #         # if the variable is tas
    #         if args.variable == "tas":
    #             # Set up the path to the observed data
    #             obs_path_this = os.path.join(remaining_files_dir, fname_this)

    #             # Load the observed data
    #             obs_cube_this = iris.load_cube(obs_path_this, "t2m")

    #             # Append to the cubelist
    #             obs_cubelist_first.append(obs_cube_this)
    #         elif args.variable == "sfcWind":
    #             # Set up the path to the observed data
    #             obs_path_this = os.path.join(remaining_files_dir, fname_this)

    #             # Load the observed data
    #             obs_cube_u10 = iris.load_cube(obs_path_this, "u10")
    #             obs_cube_v10 = iris.load_cube(obs_path_this, "v10")

    #             # Append to the cubelist
    #             obs_cubelist_u10.append(obs_cube_u10)
    #             obs_cubelist_v10.append(obs_cube_v10)
    #         elif args.variable == "psl":
    #             # Set up the path to the observed data
    #             obs_path_this = os.path.join(remaining_files_dir, fname_this)

    #             # Load the observed data
    #             obs_cube_this = iris.load_cube(obs_path_this, "msl")

    #             # Append to the cubelist
    #             obs_cubelist_first.append(obs_cube_this)
    #         else:
    #             raise ValueError("Variable not recognised")

    # convert the list to a cube list
    # obs_cubelist_first = iris.cube.CubeList(obs_cubelist_first)
    # obs_cubelist_u10 = iris.cube.CubeList(obs_cubelist_u10)
    # obs_cubelist_v10 = iris.cube.CubeList(obs_cubelist_v10)

    # # Concatenate the cubelist
    # if args.variable in ["tas", "psl"]:
    #     print("obs cubelist:", obs_cubelist_first)

    #     removed_attrs = equalise_attributes(obs_cubelist_first)

    #     obs_cube = obs_cubelist_first.concatenate_cube()
    # elif args.variable == "sfcWind":
    #     # removed the attributes
    #     removed_attrs_u10 = equalise_attributes(obs_cubelist_u10)
    #     removed_attrs_v10 = equalise_attributes(obs_cubelist_v10)

    #     obs_cube_u10 = obs_cubelist_u10.concatenate_cube()
    #     obs_cube_v10 = obs_cubelist_v10.concatenate_cube()

    #     # Calculate the wind speed from the data
    #     # Calculate wind speed
    #     windspeed_10m = (obs_cube_u10 ** 2 + obs_cube_v10 ** 2) ** 0.5
    #     windspeed_10m.rename("si10")

    #     # rename as obs cube
    #     obs_cube = windspeed_10m

    # # print the obs cube
    # print(obs_cube)

    # Loop over the members list
    # Set up an empty list to store the data
    model_ds_list = []

    # loop over the members
    print("=================================")
    print("Looping over the members")
    # print("TESTING: Using only one member")
    print("=================================")
    for m, member_this in tqdm(enumerate(members_list)):

        # print member this
        print(member_this)

        # if init year is greater than 2018,
        # just set init year to 2018
        if args.init_year > 2018:
            # Load in the data
            model_ds_this = funcs.load_model_data_xarray(
                model_variable=args.variable,
                model="HadGEM3-GC31-MM",
                experiment="dcppA-hindcast",
                start_year=2018,
                end_year=2018,
                first_fcst_year=args.init_year + 1,
                last_fcst_year=args.init_year + 10,
                months=months,
                member=member_this,
                frequency=args.frequency,
                parallel=False,
            )
        else:
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
                frequency=args.frequency,
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

    # # Set up the leads to extract from the model data
    # leads_djf_model = np.arange(
    #     31 + ((args.winter - 1) * 360), 31 + 90 + ((args.winter - 1) * 360)
    # )

    # # print the min lead we are extracting
    # print("=================================")
    # print(f"Min lead: {leads_djf_model[0]}")
    # print(f"Max lead: {leads_djf_model[-1]}")
    # print("=================================")

    # # # Extract the relevant leads
    # model_cube = model_cube.extract(iris.Constraint(lead=leads_djf_model))

    # # Extract the data for the gridbox
    # obs_data_box = obs_data.intersection(
    #     longitude=(gridbox["lon1"], gridbox["lon2"]),
    #     latitude=(gridbox["lat1"], gridbox["lat2"]),
    # )

    # # do the same for the obs cube
    # obs_cube_box = obs_cube.intersection(
    #     longitude=(gridbox["lon1"], gridbox["lon2"]),
    #     latitude=(gridbox["lat1"], gridbox["lat2"]),
    # )

    if args.region == "global":
        model_cube_box = model_cube
    else:
        # Extract the data for the gridbox
        model_cube_box = model_cube.intersection(
            longitude=(gridbox["lon1"], gridbox["lon2"]),
            latitude=(gridbox["lat1"], gridbox["lat2"]),
        )

    # # Regrid the obs data to the model data
    # obs_data_box_regrid = obs_data_box.regrid(model_cube_box, iris.analysis.Linear())

    # # regrid the obs cube box
    # obs_cube_box_regrid = obs_cube_box.regrid(
    #     model_cube_box, iris.analysis.Linear()
    # )

    # # Set up a cube list for the obs cubes
    # obs_cubelist = iris.cube.CubeList([obs_data_box_regrid, obs_cube_box_regrid])

    # loop over the cube and set the units
    # for cube in obs_cubelist:
    #     if args.variable == "sfcWind":
    #         cube.units = "m/s"
    #     elif args.variable == "tas":
    #         cube.units = "K"
    #     elif args.variable == "psl":
    #         cube.units = "Pa"
    #     else:
    #         raise ValueError("Variable not recognised")
        
    #     if cube.coord("time").var_name != "time":
    #         cube.coord("time").var_name = "time"

    #     if cube.cell_methods is not None:
    #                 cube.cell_methods = None
    #     if cube.attributes is not None:
    #         cube.attributes = None

    #     if cube.long_name != None:
    #         # set the long name to the variable name
    #         cube.long_name = None
    #     elif cube.standard_name != None:
    #         # set the standard name to the variable name
    #         cube.standard_name = None
    #     elif cube.var_name != None:
    #         # set the variable name to the variable name
    #         cube.var_name = None
    #     else:
    #         print("No long name, standard name or variable name found.")

    #     time_coord = cube.coord("time")
        
    #     # Convert time points to a consistent format (e.g., midnight)
    #     new_time_points = np.floor(time_coord.points)  # Round down to the nearest day
    #     time_coord.points = new_time_points

    #     # Remove bounds to avoid mismatches
    #     if time_coord.has_bounds():
    #         time_coord.bounds = None

    # # remove the attributes
    # removed_attrs = equalise_attributes(obs_cubelist)

    # # equalise the attributes
    # equalise_cubes(obs_cubelist, apply_all=True)

    # # unify the time coordinate
    # unify_time_units(obs_cubelist)

    # # Describe the difference between the two cubes
    # describe_diff(obs_cubelist[0], obs_cubelist[1])

    # # if the variable is psl, then fix the time coordinate
    # if args.variable == "psl":
    #     time_coord_first = obs_cubelist[0].coord("time")

    #     # extract the units for this
    #     time_units_first = time_coord_first.units

    #     # print the time units
    #     print("=================================")
    #     print("Time units:")
    #     print(time_units_first)
    #     print("=================================")

    #     # print the time coord
    #     print("=================================")
    #     print("Time coord:")
    #     print(time_coord_first)
    #     print("=================================")

    #     new_time_unit = cf_units.Unit(time_units_first, calendar=cf_units.CALENDAR_STANDARD)

    #     # print the new time unit
    #     print("=================================")
    #     print("New time unit:")
    #     print(new_time_unit)
    #     print("=================================")

    #     # loop over the cubes
    #     for cube in obs_cubelist:
    #         # set the time coord to the new time unit
    #         tcoord = cube.coord("time")
    #         tcoord.units = cf_units.Unit(tcoord.units.origin, calendar='standard')

    #     # sys.exit()



    #     # # Define a common time unit and calendar
    #     # common_time_unit = obs_cubelist[0].coord("time").units

    #     # common_calendar = obs_cubelist[0].coord("time").calendar

    #     # # Convert the time coordinates to the common unit and calendar
    #     # for cube in obs_cubelist:
    #     #     cube.coord("time").convert_units(common_time_unit)
    #     #     cube.coord("time").calendar = common_calendar

    # # unify the time coordinates
    # unify_time_units(obs_cubelist)

    # # loop over and print the cubes
    # for cube in obs_cubelist:
    #     print(cube)
    #     print(cube.attributes)
    #     print(cube.cell_methods)
    #     print(cube.long_name)
    #     print(cube.standard_name)
    #     print(cube.var_name)
    #     print(cube.coord("time"))

    # # concatenate the cubes
    # obs_cube_full = obs_cubelist.concatenate_cube()

    # # print the obs cube
    # print(obs_cube_full)

    # # print the time coordinates of the obs cube full
    # print(obs_cube_full.coord("time"))

    # # # print the time taken
    # # print("=================================")
    # # print("Time taken to load the data:")
    # # print(f"{time.time() - start_time} seconds")
    # # print("=================================")

    # # sys.exit()

    # obs_data = obs_cube_full

    # # Constrain the obs to the winter
    # # FIXME: hardcoded as DJF for now
    # obs_data = obs_data.extract(
    #     iris.Constraint(
    #         time=lambda cell: datetime(int(args.init_year), 12, 1)
    #         <= cell.point
    #         < datetime(int(args.init_year) + 1, 3, 1)
    #     )
    # )

    # Set up the name for the lats array
    lats_array_fname = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_{args.init_year}_{args.season}_{args.frequency}_lats.npy"
    lons_array_fname = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_{args.init_year}_{args.season}_{args.frequency}_lons.npy"
    members = f"HadGEM3-GC31-MM_{args.variable}_{args.region}_{args.init_year}_{args.season}_{args.frequency}_members.npy"
    obs_times = f"ERA5_{args.variable}_{args.region}_{args.init_year}_{args.season}_{args.frequency}_times_{current_time}.npy"

    # Set up the paths for the lats and lons
    lats_array_path = os.path.join(meta_dir, lats_array_fname)
    lons_array_path = os.path.join(meta_dir, lons_array_fname)

    # Set up the path for the members
    members_path = os.path.join(meta_dir, members)

    # # Set up the path for the obs times
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

    # # if the obs times array already exists, exit with an error
    # if os.path.exists(obs_times_path):
    #     print(f"{obs_times_path} already exists")
    # else:
    #     # save the obs times array
    #     np.save(obs_times_path, obs_data.coord("time").points)

    # # Set up the obs data array
    # obs_data_array = obs_data.data
    # obs_data_array = obs_data_array.filled(np.nan)

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

    # # # Save the obs data array
    # np.save(obs_array_path, obs_data_array)

    # Save the model data array
    np.save(model_array_path, model_data_array)

    # Set up the end time
    end_time = time.time()

    # Print the time taken
    print(f"Time taken: {end_time - start_time} seconds")

    return None


if __name__ == "__main__":
    main()

# %%
