#!/bin/bash
#SBATCH --job-name="process_fidelity_testing"
#SBATCH --time=10:00:00
#SBATCH --mem=80000
#SBATCH --account=scenario
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o /home/users/benhutch/unseen_multi_year/logs/submit_process_fid_testing-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_multi_year/logs/submit_process_fid_testing-%A_%a.err

# Set up the usage messages
usage="Usage: sbatch /home/users/benhutch/unseen_multi_year/submit_process_fid_testing.bash <variable> <region> <season> <winter_year> <year>"

# Check the number of CLI arguments
if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Load the jaspy module
module load jaspy

# load my python environment
source activate bens-conda-env2

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_spatial_fid_testing.py"

# Extract the arguments
variable=$1
region=$2
season=$3
winter_year=$4
year=$5
frequency="day"

# Echo the args used
echo "Variable: ${variable}"
echo "Region: ${region}"
echo "Season: ${season}"
echo "Winter year: ${winter_year}"
echo "Year: ${year}"

# Run the script
python ${process_script} \
    --variable ${variable} \
    --region ${region} \
    --init_year ${year} \
    --season ${season} \
    --winter ${winter_year} \
    --frequency ${frequency}

# End of file
echo "End of file"