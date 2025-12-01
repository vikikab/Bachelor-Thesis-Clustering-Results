#!/bin/bash
#SBATCH --job-name=feature_extraction
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/slurm-output/slurm-%j.out"

set -e

home_dir="/mnt/nfs/homedirs/$USER"
env_name="clusteringspectct"
python_bin="$home_dir/miniconda3/envs/$env_name/bin/python"

mkdir -p "$home_dir/slurm-output"

# Go to directory where sbatch was executed
cd "${SLURM_SUBMIT_DIR}"

echo "Starting job ${SLURM_JOBID}"
echo "SLURM assigned me these nodes:"
squeue -j "${SLURM_JOBID}" -O nodelist | tail -n +2

echo "Using Python interpreter: $python_bin"
"$python_bin" -V

# -------- Input & output paths --------
CT_IN="${SLURM_SUBMIT_DIR}/ct_norm.nii.gz"
SPECT_IN="${SLURM_SUBMIT_DIR}/spect_norm.nii.gz"
MASK_IN="${SLURM_SUBMIT_DIR}/combined_mask.nii.gz"

OUT_FEATURES="${SLURM_SUBMIT_DIR}/features_ct_spect.npz"

# ---- Sanity checks for inputs ----
if [ ! -f "$CT_IN" ]; then
    echo "ERROR: CT input not found: $CT_IN"
    exit 1
fi

if [ ! -f "$SPECT_IN" ]; then
    echo "ERROR: SPECT input not found: $SPECT_IN"
    exit 1
fi

if [ ! -f "$MASK_IN" ]; then
    echo "ERROR: Mask input not found: $MASK_IN"
    exit 1
fi

echo "Using CT:       $CT_IN"
echo "Using SPECT:    $SPECT_IN"
echo "Using mask:     $MASK_IN"
echo "Output file:    $OUT_FEATURES"

# -------- Run feature extraction --------
"$python_bin" extract_features_ct_spect.py \
    --ct "$CT_IN" \
    --spect "$SPECT_IN" \
    --mask "$MASK_IN" \
    --out_features "$OUT_FEATURES" \
    --include_gradients \
    --debug_print
