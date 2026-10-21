#!/bin/bash
#SBATCH -A EUHPC_B38_106
#SBATCH -p boost_usr_prod
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=120000
#SBATCH -J o5_newtasks_seed42_motion
#SBATCH -o /leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/slurm/%j_%x.out
#SBATCH -e /leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/slurm/%j_%x.err
set -euo pipefail
source /leonardo_work/EUHPC_B38_106/rlfv/repo/leonardo/env.sh
LEDGER=$RLFV_WORK/repo/code/multiview_spatial_o5/newtasks_seed42_jobs.json
singularity exec -B "$RLFV_BINDS" --env OMP_NUM_THREADS=8,LD_LIBRARY_PATH="$RLFV_LD" "$IMAGE" \
 /bin/python "$RLFV_WORK/vpro_paper_plots/experiments/fit_motion_o5.py" --ledger "$LEDGER" --arm A0
