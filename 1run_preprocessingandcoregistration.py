#!/bin/bash
#SBATCH --job-name=my_job
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/slurm-output/slurm-%j.out"

set -e

home_dir="/mnt/nfs/homedirs/$USER"
env_name="clusteringspectct"
python_bin="$home_dir/miniconda3/envs/$env_name/bin/python"

mkdir -p "$home_dir/slurm-output"

# Go to the directory where you ran `sbatch` (your project dir)
cd "${SLURM_SUBMIT_DIR}"

echo "Starting job ${SLURM_JOBID}"
echo "SLURM assigned me these nodes:"
squeue -j "${SLURM_JOBID}" -O nodelist | tail -n +2

echo "Using Python interpreter: $python_bin"
"$python_bin" -V

# -------- Input file paths (in current directory) --------
CT_PATH="${SLURM_SUBMIT_DIR}/ct.nii.gz"
SPECT_PATH="${SLURM_SUBMIT_DIR}/spect.nii.gz"

# Optional: sanity checks so errors are clearer
if [ ! -f "$CT_PATH" ]; then
    echo "ERROR: CT file not found: $CT_PATH"
    exit 1
fi

if [ ! -f "$SPECT_PATH" ]; then
    echo "ERROR: SPECT file not found: $SPECT_PATH"
    exit 1
fi

echo "Using CT: $CT_PATH"
echo "Using SPECT: $SPECT_PATH"

# -------- Run your script with arguments --------
"$python_bin" preprocessingandcoregistration.py \
    --ct "$CT_PATH" \
    --spect "$SPECT_PATH" \
    --out_ct output_ct.nii.gz \
    --out_spect output_spect.nii.gz \
    --out_bone_mask bone_mask.nii.gz \
    --out_active_mask active_mask.nii.gz \
    --out_combined_mask combined_mask.nii.gz \
    --bone_thresh 200 \
    --active_fraction 0.2 \
    --morph_radius 3
