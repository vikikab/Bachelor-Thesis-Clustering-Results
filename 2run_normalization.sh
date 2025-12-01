#!/bin/bash
#SBATCH --job-name=normalization
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/slurm-output/slurm-%j.out"

set -e

home_dir="/mnt/nfs/homedirs/$USER"
env_name="clusteringspectct"
python_bin="$home_dir/miniconda3/envs/$env_name/bin/python"

mkdir -p "$home_dir/slurm-output"

# Go to the directory where you run `sbatch normalization.sh`
cd "${SLURM_SUBMIT_DIR}"

echo "Starting job ${SLURM_JOBID}"
echo "SLURM assigned me these nodes:"
squeue -j "${SLURM_JOBID}" -O nodelist | tail -n +2

echo "Using Python interpreter: $python_bin"
"$python_bin" -V

# -------- Input & output paths (match your existing files) --------
CT_IN="${SLURM_SUBMIT_DIR}/output_ct.nii.gz"
SPECT_IN="${SLURM_SUBMIT_DIR}/output_spect.nii.gz"
MASK_IN="${SLURM_SUBMIT_DIR}/combined_mask.nii.gz"

OUT_CT="${SLURM_SUBMIT_DIR}/ct_norm.nii.gz"
OUT_SPECT="${SLURM_SUBMIT_DIR}/spect_norm.nii.gz"

# Sanity checks for inputs
if [ ! -f "$CT_IN" ]; then
    echo "ERROR: CT input not found: $CT_IN"
    exit 1
fi

if [ ! -f "$SPECT_IN" ]; then
    echo "ERROR: SPECT input not found: $SPECT_IN"
    exit 1
fi

if [ ! -f "$MASK_IN" ]; then
    echo "ERROR: Mask not found: $MASK_IN"
    exit 1
fi

echo "Using CT:     $CT_IN"
echo "Using SPECT:  $SPECT_IN"
echo "Using mask:   $MASK_IN"
echo "Output CT:    $OUT_CT"
echo "Output SPECT: $OUT_SPECT"

# -------- Run normalization --------
"$python_bin" normalize_ct_spect.py \
    --ct "$CT_IN" \
    --spect "$SPECT_IN" \
    --mask "$MASK_IN" \
    --out_ct "$OUT_CT" \
    --out_spect "$OUT_SPECT" \
    --spect_log
