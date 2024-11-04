#!/bin/bash
#SBATCH --job-name=sub-proc-unseen
#SBATCH --partition=high-mem
#SBATCH --mem=800000
#SBATCH --time=1000:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/sub-proc-unseen-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/sub-proc-unseen-%A_%a.err
# sbatch ~/unseen_multi_year/submit_process_unseen.bash "CanESM5" "tas" "United_Kingdom" "ONDJFM" 1960 2018 "1-9" "True" "None" 10
# sbatch ~/unseen_multi_year/submit_process_unseen.bash "MPI-ESM1-2-HR" "tas" "United_Kingdom" "ONDJFM" 1960 2018 "1-10" "True" "None" 10
# sbatch ~/unseen_multi_year/submit_process_unseen.bash "MPI-ESM1-2-HR" "sfcWind" "United_Kingdom" "ONDJFM" 1960 2018 "1-10" "False" "None" 10

# Set up the usage message
usage_msg = "Usage: sbatch submit_process_unseen.bash <model> <variable> <country> <season> <start_year> <end_year> <lead_year> <detrend> <bias_corr> <percentile>"

# Check the number of arguments
if [ "$#" -ne 10 ]; then
    echo "Illegal number of parameters"
    echo $usage_msg
    exit 1
fi

# Set up the CLI args
model=$1
variable=$2
country=$3
season=$4
start_year=$5
end_year=$6
lead_year=$7
detrend=$8
bias_corr=$9
percentile=${10}

# Load the required modules
module load jaspy

# set model fcst year as 1
export model_fcst_year=1

# Echo the CLI args
echo "Model: $model"
echo "Variable: $variable"
echo "Country: $country"
echo "Season: $season"
echo "Start year: $start_year"
echo "End year: $end_year"
echo "Lead year: $lead_year"
echo "Detrend: $detrend"
echo "Bias correction: $bias_corr"
echo "Percentile: $percentile"

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_UNSEEN.py"

# Run the process script
python ${process_script} \
    --model ${model} \
    --variable ${variable} \
    --country ${country} \
    --season ${season} \
    --first_year ${start_year} \
    --last_year ${end_year} \
    --model_fcst_year ${model_fcst_year} \
    --lead_year ${lead_year} \
    --detrend ${detrend} \
    --bias_correct ${bias_corr} \
    --percentile ${percentile}