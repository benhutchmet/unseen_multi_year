#!/bin/bash
#SBATCH --job-name="process_canari"
#SBATCH --time=500:00
#SBATCH --mem=150000
#SBATCH --cpus=1
#SBATCH --account=canari
#SBATCH --partition=highres
#SBATCH --qos=highres
#SBATCH -o /home/users/benhutch/unseen_functions/logs/submit_process_canari-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/submit_process_canari-%A_%a.err
#SBATCH --array=1950-2014

# Set up the usage messages
usage="Usage: sbatch submit_process_canari.bash <variable> <country>"

# Check the number of CLI arguments
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Load the jaspy module
module load jaspy

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_daily_canari.py"

# Extract the arguments
variable=$1
country=$2
year=${SLURM_ARRAY_TASK_ID}
period="HIST2"

# Echo the args used
echo "Variable: ${variable}"
echo "Country: ${country}"
echo "Year: ${year}"
echo "Period: ${period}"

# Run the script
python ${process_script} \
    --variable ${variable} \
    --country ${country} \
    --year ${year} \
    --period ${period}

# End of file
echo "End of file"