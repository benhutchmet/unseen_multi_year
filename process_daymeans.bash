#!/bin/bash

# Load the jaspy module
module load jaspy

# # load my python environment
# source activate bens-conda-env2

# Change directory to the correct one
cd /gws/nopw/j04/canari/users/benhutch/ERA5/year_month/

# Set up the fnames
fnames=(
    "ERA5_EU_T_U10_V10_msl2021_02.nc"
    "ERA5_EU_T_U10_V10_msl2021_12.nc"
    "ERA5_EU_T_U10_V10_msl2022_01.nc"
    "ERA5_EU_T_U10_V10_msl2022_02.nc"
    "ERA5_EU_T_U10_V10_msl2022_12.nc"
    "ERA5_EU_T_U10_V10_msl2023_01.nc"
    "ERA5_EU_T_U10_V10_msl2023_02.nc"
    "ERA5_EU_T_U10_V10_msl2023_12.nc"
    "ERA5_EU_T_U10_V10_msl2024_01.nc"
    "ERA5_EU_T_U10_V10_msl2024_02.nc"
    "ERA5_EU_T_U10_V10_msl2024_12.nc"
    "ERA5_EU_T_U10_V10_msl2025_01.nc"
    "ERA5_EU_T_U10_V10_msl2025_02.nc"
)

# Loop over the fnames
for fname in "${fnames[@]}"; do
    # Echo the file name
    echo "Processing file: ${fname}"

    # Run the script
    cdo daymean "${fname}" "${fname%.nc}_daymean.nc"
done