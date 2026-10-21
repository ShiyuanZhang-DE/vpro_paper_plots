#!/bin/bash
#SBATCH -A EUHPC_B38_106
#SBATCH -p boost_usr_prod
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=80000
#SBATCH -J lapa_fig7_probe
#SBATCH -o /leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/slurm/%j_%x.out
#SBATCH -e /leonardo_scratch/fast/EUHPC_B38_106/rlfv/runs/slurm/%j_%x.err

set -euo pipefail

source /leonardo_work/EUHPC_B38_106/rlfv/repo/leonardo/env.sh

PLOT_ROOT=$RLFV_WORK/vpro_paper_plots

INIT_LABEL=$RUN/latent_labels/dk1_original4_lapaft_sideonly_lapainit_59093801_ckpt030000_idx1
SCR_LABEL=$RUN/latent_labels/dk1_original4_lapaft_sideonly_scratch_59094662_ckpt030000_idx1

OUT=$PLOT_ROOT/results/probe_realworld_lapa_local.csv

export OMP_NUM_THREADS=8

echo "START $(date -u +%FT%TZ)"
echo "host=$(hostname)"
echo "INIT_LABEL=$INIT_LABEL"
echo "SCR_LABEL=$SCR_LABEL"
echo "OUT=$OUT"

singularity exec \
  -B "$RLFV_BINDS" \
  --env OMP_NUM_THREADS=8 \
  "$IMAGE" \
  /bin/python \
  "$PLOT_ROOT/experiments/fit_decode_realworld_lapa.py" \
  --init-root "$INIT_LABEL" \
  --scratch-root "$SCR_LABEL" \
  --output "$OUT"

echo
echo "DONE $(date -u +%FT%TZ)"
echo "OUT=$OUT"
