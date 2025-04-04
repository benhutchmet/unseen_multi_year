#!/bin/bash

# Load the jaspy module
module load jaspy

# load my python environment
source activate bens-conda-env2

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_spatial_fid_testing.py"

# hard code the args
variable="tas"
region="global"
season="DJF"
winter_year="1"
frequency="Amon"

# Echo the hard coded args
echo "Variable: ${variable}"
echo "Region: ${region}"
echo "Season: ${season}"
echo "Winter year: ${winter_year}"
echo "Frequency: ${frequency}"

# loop over years between 1960 and 2018
for year in {1960..2018}
do
    # echo the year
    echo "Processing year: ${year}"

    # Run the script
    python ${process_script} \
        --variable ${variable} \
        --region ${region} \
        --init_year ${year} \
        --season ${season} \
        --winter ${winter_year} \
        --frequency ${frequency}

    # echo the year
    echo "Finished processing year: ${year}"
done

# End of file
echo "End of file"