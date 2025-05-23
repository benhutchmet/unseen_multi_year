#!/bin/bash
#SBATCH --job-name="process_model_block"
#SBATCH --time=08:00:00
#SBATCH --mem=100000
#SBATCH --account=scenario
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o /home/users/benhutch/unseen_multi_year/logs/submit_process_block-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_multi_year/logs/submit_process_block-%A_%a.err

# Set up the usage message
usage="Usage: sbatch submit_process_model_block.bash <variable> <season> <region>"

# Check the number of CLI arguments
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Load the jaspy module
module load jaspy

# load my python environment
source activate bens-conda-env2

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_model_block_min_max.py"

# Extract the arguments
variable=$1
season=$2
region=$3

# Echo the args used
echo "Variable: ${variable}"
echo "Region: ${region}"
echo "Season: ${season}"

# Run the script
python ${process_script} \
    --variable ${variable} \
    --season ${season} \
    --region ${region} \

# End of file
echo "End of file"