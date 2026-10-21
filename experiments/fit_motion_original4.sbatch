#!/bin/bash
#SBATCH -A EUHPC_B38_106
#SBATCH -p boost_usr_prod
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=120000
#SBATCH -J spatial_o5_original4_motion
#SBATCH -o /leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/slurm/%j_%x.out
#SBATCH -e /leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/slurm/%j_%x.err
set -euo pipefail
source /leonardo_work/EUHPC_B38_106/rlfv/repo/leonardo/env.sh
PLOT_ROOT=$RLFV_WORK/vpro_paper_plots
LEDGER=${ORIGINAL4_LEDGER:-$RLFV_WORK/repo/code/multiview_spatial_o5/original4_jobs.json}
# Slurm afterok on all four original4 Policy Probes; no automatic retries.
singularity exec -B "$RLFV_BINDS" --env OMP_NUM_THREADS=8 "$IMAGE" \
 /bin/python "$PLOT_ROOT/experiments/fit_motion_o5.py" --ledger "$LEDGER"
# Rendering is reached only after validated COMPLETE measured Motion results.
singularity exec -B "$RLFV_BINDS" --env PYTHONPATH="$RLFV_WORK/vpro_plot_pydeps",MPLBACKEND=Agg "$IMAGE" \
 /bin/python "$PLOT_ROOT/plot_motion_original4.py" --ledger "$LEDGER"
python3 "$RLFV_WORK/repo/code/multiview_spatial_o5/original4_pipeline.py" --collect
