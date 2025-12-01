#!/usr/bin/env python3
"""
Extract per-voxel + neighborhood features from normalized CT/SPECT for clustering.

Inputs:
    - Normalized CT (NIfTI)
    - Normalized SPECT (NIfTI)
    - Mask (NIfTI) – e.g. combined bone & active mask

Features (per voxel inside mask):
    - ct_norm
    - spect_norm
    - ct_mean_3, ct_std_3   (3x3x3)
    - ct_mean_5, ct_std_5   (5x5x5)
    - sp_mean_3, sp_std_3   (3x3x3)
    - sp_mean_5, sp_std_5   (5x5x5)
    - (optional) ct_grad_mag, sp_grad_mag

Outputs:
    - .npz file with:
        X              : (N_voxels, D_features) feature matrix
        indices        : (N_voxels, 3) voxel indices (z, y, x)
        feature_names  : list of feature names (length D_features)

Usage:
    python extract_features_ct_spect.py \
        --ct ct_norm.nii.gz \
        --spect spect_norm.nii.gz \
        --mask combined_mask.nii.gz \
        --out_features features_ct_spect.npz \
        --include_gradients
"""

import argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import uniform_filter


def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # shape: (z, y, x)
    return img, arr


def masked_local_stats(arr, mask, window_size):
    """
    Compute masked local mean and std using a 3D uniform filter.

    arr:  (Z,Y,X) float array
    mask: (Z,Y,X) boolean array
    window_size: int (e.g. 3 or 5)

    Returns:
        mean, std  (same shape as arr)
    """
    arr = arr.astype(np.float32)
    mask_f = mask.astype(np.float32)

    # Multiply by mask so outside-mask voxels don't contribute.
    arr_masked = arr * mask_f

    # Uniform filter gives local mean.
    # mean_num = mean(arr * mask, window)
    # mean_den = mean(mask, window)
    mean_num = uniform_filter(arr_masked, size=window_size, mode='constant', cval=0.0)
    mean_den = uniform_filter(mask_f, size=window_size, mode='constant', cval=0.0)

    # Safely compute masked mean = sum(arr*mask) / sum(mask)
    eps = 1e-6
    mean = mean_num / (mean_den + eps)

    # For variance: E[X^2] - (E[X])^2, using masked E[X^2]
    arr2_masked = (arr ** 2) * mask_f
    mean_num2 = uniform_filter(arr2_masked, size=window_size, mode='constant', cval=0.0)
    ex2 = mean_num2 / (mean_den + eps)
    var = ex2 - mean ** 2
    var = np.clip(var, a_min=0.0, a_max=None)
    std = np.sqrt(var)

    # Outside mask where mean_den ~ 0, set stats to 0 for cleanliness
    mean[mean_den < eps] = 0.0
    std[mean_den < eps] = 0.0

    return mean, std


def gradient_magnitude(arr, spacing=(1.0, 1.0, 1.0)):
    """
    Approximate 3D gradient magnitude using numpy.gradient.
    spacing: (sz, sy, sx) physical spacing (from NIfTI if you want).
    """
    sz, sy, sx = spacing
    # np.gradient expects spacing in order of axes, so (z, y, x)
    gz, gy, gx = np.gradient(arr.astype(np.float32), sz, sy, sx, edge_order=1)
    grad_mag = np.sqrt(gz**2 + gy**2 + gx**2)
    return grad_mag


def main():
    parser = argparse.ArgumentParser(
        description="Extract voxelwise + neighborhood CT/SPECT features for clustering."
    )
    parser.add_argument("--ct", required=True, help="Path to normalized CT NIfTI.")
    parser.add_argument("--spect", required=True, help="Path to normalized SPECT NIfTI.")
    parser.add_argument("--mask", required=True, help="Path to mask NIfTI (voxels to use).")
    parser.add_argument("--out_features", default="features_ct_spect.npz",
                        help="Output .npz file with features.")

    parser.add_argument("--win_small", type=int, default=3,
                        help="Small neighborhood size (odd, e.g., 3).")
    parser.add_argument("--win_large", type=int, default=5,
                        help="Large neighborhood size (odd, e.g., 5).")
    parser.add_argument("--include_gradients", action="store_true",
                        help="Include gradient magnitude features.")
    parser.add_argument("--debug_print", action="store_true")

    args = parser.parse_args()

    # --- Load images ---
    if args.debug_print:
        print("Loading CT, SPECT, and mask...")

    ct_img, ct = load_nifti(args.ct)
    sp_img, sp = load_nifti(args.spect)
    mask_img, mask = load_nifti(args.mask)

    if ct.shape != sp.shape or ct.shape != mask.shape:
        raise ValueError(f"Shape mismatch: CT {ct.shape}, SPECT {sp.shape}, mask {mask.shape}")

    mask_bool = mask.astype(bool)

    if args.debug_print:
        print("CT shape:", ct.shape)
        print("SPECT shape:", sp.shape)
        print("Mask voxels (True):", int(mask_bool.sum()))

    # --- Base features ---
    ct_f = ct.astype(np.float32)
    sp_f = sp.astype(np.float32)

    feature_volumes = []
    feature_names = []

    # 1) Raw normalized intensities
    feature_volumes.append(ct_f)
    feature_names.append("ct_norm")

    feature_volumes.append(sp_f)
    feature_names.append("sp_norm")

    # 2) Neighborhood stats for CT
    if args.debug_print:
        print(f"Computing CT neighborhood stats (win={args.win_small})...")
    ct_mean_3, ct_std_3 = masked_local_stats(ct_f, mask_bool, args.win_small)
    feature_volumes.append(ct_mean_3)
    feature_names.append(f"ct_mean_{args.win_small}")
    feature_volumes.append(ct_std_3)
    feature_names.append(f"ct_std_{args.win_small}")

    if args.debug_print:
        print(f"Computing CT neighborhood stats (win={args.win_large})...")
    ct_mean_5, ct_std_5 = masked_local_stats(ct_f, mask_bool, args.win_large)
    feature_volumes.append(ct_mean_5)
    feature_names.append(f"ct_mean_{args.win_large}")
    feature_volumes.append(ct_std_5)
    feature_names.append(f"ct_std_{args.win_large}")

    # 3) Neighborhood stats for SPECT
    if args.debug_print:
        print(f"Computing SPECT neighborhood stats (win={args.win_small})...")
    sp_mean_3, sp_std_3 = masked_local_stats(sp_f, mask_bool, args.win_small)
    feature_volumes.append(sp_mean_3)
    feature_names.append(f"sp_mean_{args.win_small}")
    feature_volumes.append(sp_std_3)
    feature_names.append(f"sp_std_{args.win_small}")

    if args.debug_print:
        print(f"Computing SPECT neighborhood stats (win={args.win_large})...")
    sp_mean_5, sp_std_5 = masked_local_stats(sp_f, mask_bool, args.win_large)
    feature_volumes.append(sp_mean_5)
    feature_names.append(f"sp_mean_{args.win_large}")
    feature_volumes.append(sp_std_5)
    feature_names.append(f"sp_std_{args.win_large}")

    # 4) Optional gradients
    if args.include_gradients:
        if args.debug_print:
            print("Computing gradient magnitudes...")

        # Infer spacing from SimpleITK (z,y,x)
        spacing = ct_img.GetSpacing()
        # SimpleITK spacing is (x,y,z); numpy array is (z,y,x) so reorder:
        sx, sy, sz = spacing
        spacing_zyx = (sz, sy, sx)

        ct_grad = gradient_magnitude(ct_f, spacing_zyx)
        sp_grad = gradient_magnitude(sp_f, spacing_zyx)

        feature_volumes.append(ct_grad)
        feature_names.append("ct_grad_mag")
        feature_volumes.append(sp_grad)
        feature_names.append("sp_grad_mag")

    # --- Stack features for masked voxels ---
    if args.debug_print:
        print("Stacking features for masked voxels...")

    feature_volumes = [fv.astype(np.float32) for fv in feature_volumes]
    num_features = len(feature_volumes)

    # Get indices of voxels inside mask (z, y, x)
    indices = np.array(np.where(mask_bool)).T  # shape (N_voxels, 3)

    # For each feature volume, flatten at masked voxels
    features_list = []
    for fv in feature_volumes:
        features_list.append(fv[mask_bool])

    # Shape: (num_features, N_voxels) -> transpose to (N_voxels, num_features)
    X = np.vstack(features_list).T

    if args.debug_print:
        print("Feature matrix shape:", X.shape)
        print("Number of features:", num_features)
        print("Feature names:", feature_names)

    # Save to npz
    np.savez(
        args.out_features,
        X=X,
        indices=indices,
        feature_names=np.array(feature_names, dtype=object),
    )

    print("Saved features to:", args.out_features)
    print("Shape X:", X.shape)
    print("Number of voxels:", X.shape[0])
    print("Number of features:", X.shape[1])


if __name__ == "__main__":
    main()

