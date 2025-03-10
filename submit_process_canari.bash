#!/bin/bash
#SBATCH --job-name="process_canari"
#SBATCH --time=500:00
#SBATCH --mem=50000
#SBATCH --account=canari
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o /home/users/benhutch/unseen_functions/logs/submit_process_canari-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/submit_process_canari-%A_%a.err

# Set up the usage messages
usage="Usage: sbatch submit_process_canari.bash <variable> <country> <year>"

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

# set up the members
# specific to CanARI
members=()
for i in {1..40}; do
  members+=("$i")
done

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_daily_canari.py"

# Extract the arguments
variable=$1
country=$2
year=$3
period="HIST2"

# Echo the args used
echo "Variable: ${variable}"
echo "Country: ${country}"
echo "Year: ${year}"
echo "Period: ${period}"

# loop over the months
for member in "${members[@]}"; do
    # Echo the member number
    echo "Member no: ${member}"

    # Run the script
    python ${process_script} \
        --variable ${variable} \
        --country ${country} \
        --year ${year} \
        --member ${member} \
        --period ${period}
done

# End of file
echo "End of file"