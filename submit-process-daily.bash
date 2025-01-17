#!/bin/bash
#SBATCH --job-name=sub-process-daily
#SBATCH --partition=high-mem
#SBATCH --mem=100000
#SBATCH --time=500:00
#SBATCH -o /home/users/benhutch/unseen_functions/logs/sub-process-daily-%A_%a.out
#SBATCH -e /home/users/benhutch/unseen_functions/logs/sub-process-daily-%A_%a.err
#SBATCH --array=1960-2018

# Set up the usage messages
usage="Usage: sbatch submit_process_analogs.bash <variable> <country>"

# Check the number of CLI arguments
if [ "$#" -ne 2 ]; then
    echo "Illegal number of parameters"
    echo $usage
    exit 1
fi

# Set up a list of members
# Specific to HadGEM3-GC31-MM
members=("1", "2", "3", "4", "5", "6", "7", "8", "9", "10")

# set up members as ints 1-10
members=($(seq 1 10))
    
module load jaspy

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
        --init_year ${SLURM_ARRAY_TASK_ID} \
        --member ${member}
done

# End of file
echo "End of file"