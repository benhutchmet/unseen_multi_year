#!/bin/bash

# Load the jaspy module
module load jaspy

# load my python environment
source activate bens-conda-env2

# set up the members
# specific to CanARI
years=()
for i in {1960..2018}; do
  years+=("$i")
done

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_monthly_HadUK_grid.py"

# loop over the years
for year in "${years[@]}"; do
    # Echo the year
    echo "Year: ${year}"

    # Run the script
    python ${process_script} \
        --year ${year} --variable "tas"
done