#!/bin/bash
#SBATCH --job-name="process_model_climatology"
#SBATCH --time=05:00:00
#SBATCH --mem=80000
#SBATCH --account=scenario
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o /home/users/benhutch/unseen_multi_year/logs/submit_process_clim-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_multi_year/logs/submit_process_clim-%A_%a.err

# Set up the usage message
usage="Usage: sbatch submit_process_clim.bash <variable> <season> <region> <start_year> <end_year>"

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
process_script="/home/users/benhutch/unseen_multi_year/process_model_climatologies.py"

# Extract the arguments
variable=$1
season=$2
region=$3
start_year=$4
end_year=$5

# Echo the args used
echo "Variable: ${variable}"
echo "Region: ${region}"
echo "Season: ${season}"
echo "Start year: ${start_year}"
echo "End year: ${end_year}"

# Run the script
python ${process_script} \
    --variable ${variable} \
    --season ${season} \
    --region ${region} \
    --start_year ${start_year} \
    --end_year ${end_year}

# End of file
echo "End of file"

