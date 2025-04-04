#!/bin/bash
#SBATCH --job-name="submit_process_k_means"
#SBATCH --time=02:00:00
#SBATCH --mem=100000
#SBATCH --account=canari
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH -o /home/users/benhutch/unseen_multi_year/logs/submit_process_k_means-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_multi_year/logs/submit_process_k_means-%A_%a.err

# Set up the usage messages
usage="Usage: sbatch submit_process_k_means.bash"

# load the jaspy module
module load jaspy

# load my python environment
source activate bens-conda-env2

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_obs_k_means.py"

# run the script
python ${process_script}

# End of file
echo "End of file"