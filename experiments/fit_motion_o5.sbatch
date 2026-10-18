#!/bin/bash
#SBATCH -A EUHPC_B38_106
#SBATCH -p boost_usr_prod
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=120000
#SBATCH -J spatial_o5_motion
#SBATCH -o /leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/slurm/%j_%x.out
#SBATCH -e /leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/slurm/%j_%x.err
set -euo pipefail
source /leonardo_work/EUHPC_B38_106/rlfv/repo/leonardo/env.sh
PLOT_ROOT=$RLFV_WORK/vpro_paper_plots
LEDGER=$RLFV_WORK/repo/code/multiview_spatial_o5/posttrain_jobs.json
singularity exec -B "$RLFV_BINDS" --env OMP_NUM_THREADS=8 "$IMAGE" \
 /bin/python "$PLOT_ROOT/experiments/fit_motion_o5.py" --ledger "$LEDGER"
singularity exec -B "$RLFV_BINDS" --env PYTHONPATH="$RLFV_WORK/vpro_plot_pydeps",MPLBACKEND=Agg "$IMAGE" \
 /bin/python "$PLOT_ROOT/plot_motion_o5.py" --ledger "$LEDGER"
flock "$RUN/slurm/o5_posttrain_record.lock" python3 "$RLFV_WORK/repo/code/multiview_spatial_o5/posttrain.py" --collect
