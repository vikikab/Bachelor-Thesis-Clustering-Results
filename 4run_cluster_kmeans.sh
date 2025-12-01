#!/bin/bash
#SBATCH --job-name=kmeans_clustering
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output="/mnt/nfs/homedirs/%u/slurm-output/slurm-%j.out"

set -e

home_dir="/mnt/nfs/homedirs/$USER"
env_name="clusteringspectct"
python_bin="$home_dir/miniconda3/envs/$env_name/bin/python"

mkdir -p "$home_dir/slurm-output"

# Go to directory where the user ran: sbatch cluster_kmeans.sh
cd "${SLURM_SUBMIT_DIR}"

echo "Starting job ${SLURM_JOBID}"
echo "SLURM assigned me these nodes:"
squeue -j "${SLURM_JOBID}" -O nodelist | tail -n +2

echo "Using Python interpreter: $python_bin"
"$python_bin" -V


# -------- Input & output paths --------
FEATURES_IN="${SLURM_SUBMIT_DIR}/features_ct_spect.npz"
REF_NIFTI="${SLURM_SUBMIT_DIR}/ct_norm.nii.gz"

OUT_LABELS="${SLURM_SUBMIT_DIR}/kmeans_k4.nii.gz"
OUT_CENTERS="${SLURM_SUBMIT_DIR}/kmeans_k4_centers.npz"


# ---- Sanity checks ----
if [ ! -f "$FEATURES_IN" ]; then
    echo "ERROR: Feature file not found: $FEATURES_IN"
    exit 1
fi

if [ ! -f "$REF_NIFTI" ]; then
    echo "ERROR: Reference NIfTI not found: $REF_NIFTI"
    exit 1
fi

echo "Using features: $FEATURES_IN"
echo "Using reference NIfTI: $REF_NIFTI"
echo "Output labels: $OUT_LABELS"
echo "Output centers: $OUT_CENTERS"


# -------- Run K-Means clustering --------
"$python_bin" cluster_kmeans_ct_spect.py \
    --features "$FEATURES_IN" \
    --ref_nifti "$REF_NIFTI" \
    --n_clusters 4 \
    --out_labels "$OUT_LABELS" \
    --out_centers "$OUT_CENTERS" \
    --standardize \
    --debug_print
