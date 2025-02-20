#!/bin/bash
#SBATCH --job-name="submit_download_ERA5"
#SBATCH --time=20:00:00
#SBATCH --mem=10000M
#SBATCH --account=canari
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o /home/users/benhutch/unseen_multi_year/logs/submit_download_ERA5-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_multi_year/logs/submit_download_ERA5-%A_%a.err
#SBATCH --array=1940-2024

# Set up the usage message
usage="Usage: sbatch submit_download_ERA5_jasmin.bashs"

# Check the number of CLI arguments
if [ "$#" -ne 0 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Load the jaspy module
module load jaspy

# Load my python environment
source activate bens-conda-env2

# Set up the months
months=()
for i in {1..12}; do
  months+=("$i")
done

# Set up the year
year=${SLURM_ARRAY_TASK_ID}

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/downloading_data/download_ERA5_jasmin.py"

# Echo the year arg
echo "Year: ${year}"

# Loop over the months
for month in "${months[@]}"; do
    # Echo the month number
    echo "Month no: ${month}"

    # Run the script
    python ${process_script} \
        --year ${year} \
        --month ${month}
done

# End of file
echo "End of file"