#!/bin/bash

#SBATCH -A grp_spanias   # account to charge for the job one of grp_gpedriel or grp_spanias
#SBATCH -N 1             # number of nodes
#SBATCH --mem=12G        # amount of memory requested
#SBATCH -c 20           # number of cores 
#SBATCH -t 0-03:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition general or htc
#SBATCH -q public       # QOS
#SBATCH -o jobs/slurm.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e jobs/slurm.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

#Activate our enviornment
source activate CBO

#Change to the directory of our script
# cd ~/home/mmalu/cbo_code_011525

#Run the software/python script
python cbo_fin_main_rw.py