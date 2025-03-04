#!/usr/bin/env python

"""
process_daily_canari.py
=======================

This script processes daily canari data into dataframes for a given period, year, variable, and country combination.

Usage:
------

    $ python process_daily_canari.py --variable tas --country "United Kingdom" --year 1960 --member 1 --period HIST2

Arguments:
----------

    --variable : str : variable name (e.g. tas, pr, psl)
    --country : str : country name (e.g. United Kingdom, France, Germany)
    --year : int : year (e.g. 1960)
    --member : int : ensemble member number (e.g. 1-40)
    --period : str : period (e.g. HIST2 or HIST1)

Returns:
--------

    dataframes : pd.DataFrame : processed dataframes for the given variable, country, and initialisation year which are saved to the /gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs folder.

Author:
-------

    Ben W. Hutchins, University of Reading, 2025
    
"""

# Local imports
import os
import sys
import glob
import time
import argparse

# Third-party imports
import numpy as np
import pandas as pd
import iris
import cftime

# Specific imports
from tqdm import tqdm
from iris.util import equalise_attributes

# Import the dictionaries
import dictionaries as dic

# Load my specific functions
sys.path.append("/home/users/benhutch/unseen_functions")
from functions import create_masked_matrix


# Function for adding realization metadata
# for r1i1p1f1, r2i1p1f1, etc.
def realization_metadata(cube, field, fpath):
    """Modify the cube's metadata to add a "realization" coordinate.

    A function which modifies the cube's metadata to add a "realization"
    (ensemble member) coordinate from the filename if one doesn't already exist
    in the cube.

    """
    # Add an ensemble member coordinate if one doesn't already exist.
    if not cube.coords("realization"):
        # The ensemble member is encoded in the filename as *_???.pp where ???
        # is the ensemble member.

        # # Regular expression pattern for the desired format
        # pattern = re.compile(r"(r\d+i\d+p\d+f\d+)")

        # Split the fpath by /
        fpath_split = fpath.split("/")

        # extract the member as the 9th element
        realization_number = fpath_split[9]

        # print the realization number
        print("loading member: ", realization_number)

        realization_coord = iris.coords.AuxCoord(
            np.int32(realization_number), "realization", units="1"
        )

        cube.add_aux_coord(realization_coord)

    return cube


# Define a function to convert 360-day calendar time to datetime
def convert_360_day_calendar(time_value):
    return cftime.num2date(
        time_value, units="seconds since 1950-01-01 00:00:00", calendar="360_day"
    )


# Define the main function
def main():
    """
    Main function for processing canari data into dataframes.
    """

    # Set up the start timer
    start_time = time.time()

    # Set up the hard-coded paths
    base_canari_dir = "/gws/nopw/j04/canari/shared/large-ensemble/priority/"
    output_dir_dfs = "/gws/nopw/j04/canari/users/benhutch/unseen/saved_dfs"

    # Set up tyhe hard codeed args
    domain = "ATM"  # atmospheric variables

    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Process daily canari data into dataframes."
    )

    # Add the arguments
    parser.add_argument(
        "--variable", type=str, help="variable name (e.g. tas, pr, psl)"
    )
    parser.add_argument(
        "--country",
        type=str,
        help="country name (e.g. United Kingdom, France, Germany)",
    )
    parser.add_argument("--year", type=int, help="year (e.g. 1960)")
    parser.add_argument("--member", type=int, help="ensemble member number (e.g. 1-40)")
    parser.add_argument(
        "--period", type=str, help="period (e.g. HIST2 or HIST1 or SSP370)"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("=====================================================================")
    print("Processing canari data for the following arguments:")
    print("=====================================================================")
    print(f"Variable: {args.variable}")
    print(f"Country: {args.country}")
    print(f"Year: {args.year}")
    print(f"Member: {args.member}")
    print(f"Period: {args.period}")
    print("=====================================================================")

    # if country has a space, replace with _
    country = args.country.replace(" ", "_")

    # Set up the dfname
    df_name = f"canari-le-{args.variable}-{country}-{args.year}-member-{args.member}-{args.period}.csv"

    # Print the output path
    print("=====================================================================")
    print(f"Output path: {os.path.join(output_dir_dfs, df_name)}")

    # if the df already exists, raise an error
    if os.path.exists(os.path.join(output_dir_dfs, df_name)):
        raise ValueError(f"Dataframe {df_name} already exists!")

    # If the variable is tas or t2m
    if args.variable in ["tas", "t2m"]:
        # Set up the hardcoded var name
        var_name = "m01s03i236_7"

        # Check that file exist for the specified variable combination
        fpaths = f"{base_canari_dir}/{args.period}/{str(args.member)}/{domain}/yearly/{str(args.year)}/*{var_name}.nc"

        # glob the fpaths
        fpaths = glob.glob(fpaths)

        # if the fpaths is empty
        if not fpaths:
            raise ValueError(
                f"No files found for {args.variable} in {args.year} for {args.period}."
            )

        # print the length of the fpaths
        print("=====================================================================")
        print(f"Number of files found: {len(fpaths)}")
        print("=====================================================================")

        # Set up the iris constrain
        constraint = iris.Constraint(
            var_name,
            realization=lambda value: True,
        )

        # Load the cubes
        cubes = iris.load(
            fpaths,
            constraint,
            callback=realization_metadata,
        )

        # remove attrs
        removed_attrs = equalise_attributes(cubes)

        # Merge the cubes
        model_cube = cubes.merge_cube()

        # Make sure cube is on the correct grid system
        model_cube = model_cube.intersection(longitude=(-180, 180))

        # if the args.country contains a _
        # replace with a space
        if "_" in args.country:
            args.country = args.country.replace("_", " ")

        # set up the mask matrix
        MASK_MATRIX = create_masked_matrix(
            country=args.country,
            cube=model_cube,
        )

        # Extract the model data
        model_data = model_cube.data

        # Apply the mask to the model cube
        model_values = model_data * MASK_MATRIX

        # Where there are zeros in the mask we want to set these to Nans
        model_values_masked = np.where(MASK_MATRIX == 0, np.nan, model_values)

        # Take the Nanmean of the data
        model_values = np.nanmean(model_values_masked, axis=(2, 3))

        # Extract the ini years, member and lead times
        realizations = model_cube.coord("realization").points
        model_times = model_cube.coord("time").points

        # set up the dataframe
        model_df = pd.DataFrame(
            {
                "member": realizations,
                "time": model_times,
                f"{args.variable}": model_data,
            }
        )
    elif args.variable in ["sfcWind", "si10"]:
        # Set up the hardcoded var name
        var_name_u = "m01s03i225_2"
        var_name_v = "m01s03i226_2"

        # Check that file exist for the specified variable combination
        fpaths_u = f"{base_canari_dir}/{args.period}/{str(args.member)}/{domain}/yearly/{str(args.year)}/*{var_name_u}.nc"
        fpaths_v = f"{base_canari_dir}/{args.period}/{str(args.member)}/{domain}/yearly/{str(args.year)}/*{var_name_v}.nc"

        # glob the fpaths
        fpaths_u = glob.glob(fpaths_u)
        fpaths_v = glob.glob(fpaths_v)

        # if the fpaths is empty
        if not fpaths_u:
            raise ValueError(
                f"No files found for {args.variable} in {args.year} for {args.period}."
            )

        if not fpaths_v:
            raise ValueError(
                f"No files found for {args.variable} in {args.year} for {args.period}."
            )

        # print the length of the fpaths
        print("=====================================================================")
        print(f"Number of files found u: {len(fpaths_u)}")
        print(f"Number of files found v: {len(fpaths_v)}")
        print("=====================================================================")

        # Set up the iris constraint for u
        constraint_u = iris.Constraint(
            var_name_u,
            realization=lambda value: True,
        )

        # Set up the iris constraint for v
        constraint_v = iris.Constraint(
            var_name_v,
            realization=lambda value: True,
        )

        # Load the cubes u
        cubes_u = iris.load(
            fpaths_u,
            constraint_u,
            callback=realization_metadata,
        )

        # Load the cubes v
        cubes_v = iris.load(
            fpaths_v,
            constraint_v,
            callback=realization_metadata,
        )

        # remove attrs
        removed_attrs_u = equalise_attributes(cubes_u)
        removed_attrs_v = equalise_attributes(cubes_v)

        # Merge the cubes u
        model_cube_u = cubes_u.merge_cube()
        model_cube_v = cubes_v.merge_cube()

        # Make sure cube is on the correct grid system
        model_cube_u = model_cube_u.intersection(longitude=(-180, 180))
        model_cube_v = model_cube_v.intersection(longitude=(-180, 180))

        # if the args.country contains a _
        # replace with a space
        if "_" in args.country:
            args.country = args.country.replace("_", " ")

        # if the country is United Kingdom
        if args.country == "United Kingdom":
            # Set up the mask matrix
            MASK_MATRIX = create_masked_matrix(
                country=args.country,
                cube=model_cube_u,
            )

            # Extract the model data
            model_data_u = model_cube_u.data
            model_data_v = model_cube_v.data

            # Apply the mask to the model cube
            model_values_u = model_data_u * MASK_MATRIX
            model_values_v = model_data_v * MASK_MATRIX

            # Where there are zeros in the mask we want to set these to Nans
            model_values_masked_u = np.where(MASK_MATRIX == 0, np.nan, model_values_u)
            model_values_masked_v = np.where(MASK_MATRIX == 0, np.nan, model_values_v)

            # Take the Nanmean of the data
            model_values_u = np.nanmean(model_values_masked_u, axis=(1, 2))
            model_values_v = np.nanmean(model_values_masked_v, axis=(1, 2))
        elif args.country == "North Sea":
            print("Taking gridbox average for the North Sea")

            # Set up the gridbox
            gridbox = dic.north_sea_kay

            # Subset to the north sea region
            model_cube_u = model_cube_u.intersection(
                longitude=(gridbox["lon1"], gridbox["lon2"]),
                latitude=(gridbox["lat1"], gridbox["lat2"]),
            )

            model_cube_v = model_cube_v.intersection(
                longitude=(gridbox["lon1"], gridbox["lon2"]),
                latitude=(gridbox["lat1"], gridbox["lat2"]),
            )

            # print the dimensions of modeul cube u
            print(model_cube_u.shape)

            # Extract the model data
            model_data_u = model_cube_u.data
            model_data_v = model_cube_v.data

            # Take the Nanmean of the data
            model_values_u = np.nanmean(model_data_u, axis=(1, 2))
            model_values_v = np.nanmean(model_data_v, axis=(1, 2))
        elif args.country == "UK wind box":
            print("Taking gridbox average for the UK wind box")

            # Set up the gridbox
            gridbox = dic.wind_gridbox

            # Subset to the north sea region
            model_cube_u = model_cube_u.intersection(
                longitude=(gridbox["lon1"], gridbox["lon2"]),
                latitude=(gridbox["lat1"], gridbox["lat2"]),
            )

            model_cube_v = model_cube_v.intersection(
                longitude=(gridbox["lon1"], gridbox["lon2"]),
                latitude=(gridbox["lat1"], gridbox["lat2"]),
            )

            # print the dimensions of modeul cube u
            print(model_cube_u.shape)

            # Extract the model data
            model_data_u = model_cube_u.data
            model_data_v = model_cube_v.data

            # Take the Nanmean of the data
            model_values_u = np.nanmean(model_data_u, axis=(1, 2))
            model_values_v = np.nanmean(model_data_v, axis=(1, 2))
        else:
            raise ValueError("Country not recognised")

        # Extract the ini years, member and lead times
        realizations = model_cube_u.coord("realization").points
        model_times = model_cube_u.coord("time").points

        # repeat the value of realizations
        # the same length as the model_times
        realizations = np.repeat(realizations, len(model_times))

        # print the shape of the components
        print("=====================================================================")
        print(f"Realizations shape: {realizations.shape}")
        print(f"Model times shape: {model_times.shape}")
        print(f"Model values u shape: {model_values_u.shape}")
        print(f"Model values v shape: {model_values_v.shape}")
        print("=====================================================================")

        model_df = pd.DataFrame(
            {
                "member": realizations,
                "time": model_times,
                f"{args.variable}_u": model_values_u,
                f"{args.variable}_v": model_values_v,
            }
        )
    else:
        raise NotImplementedError(f"Variable {args.variable} not implemented yet!")

    # Apply the conversion function to the 'time' column
    model_df["time"] = model_df["time"].apply(convert_360_day_calendar)

    # Save the dataframe
    model_df.to_csv(
        os.path.join(output_dir_dfs, df_name),
        index=False,
    )

    # print where we have saved the dataframe to
    print("=====================================================================")
    print(f"Dataframe saved to {os.path.join(output_dir_dfs, df_name)}")
    print("=====================================================================")

    # set the end time
    end_time = time.time()

    # print the time taken
    print("=====================================================================")
    print(f"Time taken: {end_time - start_time} seconds.")
    print("=====================================================================")

    return

if __name__ == "__main__":
    main()
