#!/bin/bash
#SBATCH --job-name=sub-process-daily-obs
#SBATCH --partition=high-mem
#SBATCH --mem=200000
#SBATCH --time=500:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/sub-process-daily-obs-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/sub-process-daily-obs-%A_%a.err

# Set up the usage messages
usage="Usage: sbatch submit_process_daily_obs.bash <variable> <country>"

# Check the number of CLI arguments
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

module load jaspy

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_obs_daily.py"

# Run the script
python ${process_script} \
    --variable $1 \
    --country $2

# End of file
echo "End of file"