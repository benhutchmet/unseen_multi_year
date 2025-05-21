#!/bin/bash
#SBATCH --job-name="sub-process-daily"
#SBATCH --time=03:00:00
#SBATCH --mem=10000M
#SBATCH --account=canari
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH -o /home/users/benhutch/unseen_functions/logs/submit_process_daily_DePreSys-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/submit_process_daily_DePreSys-%A_%a.err

# Set up the usage messages
usage="Usage: sbatch submit_process_analogs.bash <variable> <country> <init_year>"

# Check the number of CLI arguments
if [ "$#" -ne 3 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up a list of members
# Specific to HadGEM3-GC31-MM
members=("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")

# set up members as ints 1-10
members=($(seq 1 10))

# Load jaspy
module load jaspy

# load my python environment
source activate bens-conda-env2

# Set up the process script
process_script="/home/users/benhutch/unseen_multi_year/process_daily_unseen.py"

# loop over the months
for member in "${members[@]}"; do
    
    # Echo the CLI arguments
    echo "Init year; ${SLURM_ARRAY_TASK_ID}"
    echo "Member no: ${member}"

    # Run the script
    #!/bin/bash
    python ${process_script} \
        --variable $1 \
        --country $2 \
        --init_year $3 \
        --member ${member}
done

# End of file
echo "End of file"