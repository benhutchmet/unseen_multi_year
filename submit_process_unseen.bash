#!/bin/bash
#SBATCH --job-name=sub-proc-unseen
#SBATCH --partition=high-mem
#SBATCH --mem=200000
#SBATCH --time=1000:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/sub-proc-unseen-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/sub-proc-unseen-%A_%a.err
# sbatch ~/unseen_multi_year/submit_process_unseen.bash "sfcWind" "United Kingdom" "ONDJFM" "1960" "1970"

# Set up the usage message
usage_msg = "Usage: sbatch submit_process_unseen.bash <variable> <country> <season> <start_year> <end_year>"

# Check the number of arguments
if [ "$#" -ne 5 ]; then
    echo "Illegal number of parameters"
    echo $usage_msg
    exit 1
fi

# Set up the CLI args
variable=$1
country=$2
season=$3
start_year=$4
end_year=$5

# Load the required modules
module load jaspy

# Echo the CLI args
echo "Variable: $variable"
echo "Country: $country"
echo "Season: $season"
echo "Start year: $start_year"
echo "End year: $end_year"

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_UNSEEN.py"

# Run the process script
python ${process_script} \
    --variable ${variable} \
    --country ${country} \
    --season ${season} \
    --first_year ${start_year} \
    --last_year ${end_year}