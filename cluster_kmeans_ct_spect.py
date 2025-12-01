#!/usr/bin/env python3
"""
Run K-means clustering on precomputed voxel features (CT+SPECT) and save
cluster labels as a NIfTI volume aligned with the CT.

Inputs:
    - features_ct_spect.npz (from extract_features_ct_spect.py)
        Contains:
            X             : (N_voxels, D_features) feature matrix
            indices       : (N_voxels, 3) voxel indices (z, y, x) in FULL volume
            feature_names : (D_features,) list/array of feature names
    - Reference NIfTI (e.g. ct_norm.nii.gz) to get image geometry & shape.

Outputs:
    - NIfTI label volume (e.g. kmeans_labels.nii.gz) with:
        - shape == reference volume shape
        - int labels in [0, n_clusters-1] on masked voxels
        - -1 (or 0, configurable) outside the mask

Usage:
    python cluster_kmeans_ct_spect.py \
        --features features_ct_spect.npz \
        --ref_nifti ct_norm.nii.gz \
        --n_clusters 4 \
        --out_labels kmeans_labels.nii.gz \
        --standardize \
        --debug_print
"""

import argparse
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans


def load_features(npz_path, debug=False):
    data = np.load(npz_path, allow_pickle=True)
    X = data["X"]                    # (N_voxels, D_features)
    indices = data["indices"]        # (N_voxels, 3) int (z,y,x)
    feature_names = data["feature_names"]

    if debug:
        print("Loaded features from:", npz_path)
        print("  X shape:", X.shape)
        print("  indices shape:", indices.shape)
        print("  feature_names:", feature_names)

    return X, indices, feature_names


def standardize_features(X, debug=False):
    """
    Z-score standardization per feature: (x - mean) / std.
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-6
    X_std = (X - mean) / std

    if debug:
        print("Standardized features:")
        print("  mean (approx):", X_std.mean(axis=0)[:5])
        print("  std (approx):", X_std.std(axis=0)[:5])

    return X_std, mean, std


def run_kmeans(X, n_clusters=4, random_state=0, max_iter=300, n_init=10, debug=False):
    if debug:
        print("Running K-means with:")
        print(f"  n_clusters={n_clusters}, random_state={random_state}, "
              f"max_iter={max_iter}, n_init={n_init}")
        print("  Feature matrix shape:", X.shape)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init,
        max_iter=max_iter,
        verbose=0
    )
    kmeans.fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    if debug:
        print("K-means finished.")
        unique, counts = np.unique(labels, return_counts=True)
        print("Cluster label distribution:")
        for u, c in zip(unique, counts):
            print(f"  label {u}: {c} voxels")

    return labels, centers


def save_labels_nifti(labels, indices, ref_nifti_path, out_path,
                      outside_value=-1, debug=False):
    """
    Create a 3D volume of labels with same geometry as ref_nifti_path.
    - labels: (N_voxels,) int array
    - indices: (N_voxels, 3) z,y,x positions in full volume
    """
    ref_img = sitk.ReadImage(ref_nifti_path)
    shape = sitk.GetArrayFromImage(ref_img).shape  # (z,y,x)

    if debug:
        print("Reference NIfTI:", ref_nifti_path)
        print("  shape:", shape)

    # Initialize with outside_value (e.g. -1)
    label_vol = np.full(shape, fill_value=outside_value, dtype=np.int16)

    z = indices[:, 0].astype(int)
    y = indices[:, 1].astype(int)
    x = indices[:, 2].astype(int)

    label_vol[z, y, x] = labels.astype(np.int16)

    out_img = sitk.GetImageFromArray(label_vol)
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, out_path)

    if debug:
        print("Saved label volume to:", out_path)


def main():
    parser = argparse.ArgumentParser(
        description="K-means clustering on CT+SPECT voxel features and save label NIfTI."
    )

    parser.add_argument("--features", required=True,
                        help="Path to features_ct_spect.npz.")
    parser.add_argument("--ref_nifti", required=True,
                        help="Reference NIfTI (e.g. ct_norm.nii.gz).")
    parser.add_argument("--out_labels", default="kmeans_labels.nii.gz",
                        help="Output NIfTI with K-means labels.")
    parser.add_argument("--out_centers", default=None,
                        help="Optional .npy or .npz path to save cluster centers.")

    parser.add_argument("--n_clusters", type=int, default=4,
                        help="Number of clusters for K-means.")
    parser.add_argument("--random_state", type=int, default=0,
                        help="Random seed for K-means.")
    parser.add_argument("--max_iter", type=int, default=300,
                        help="Max iterations for K-means.")
    parser.add_argument("--n_init", type=int, default=10,
                        help="Number of initializations for K-means.")

    parser.add_argument("--standardize", action="store_true",
                        help="Standardize features before clustering.")
    parser.add_argument("--outside_label_value", type=int, default=-1,
                        help="Label value outside the mask (default: -1).")
    parser.add_argument("--debug_print", action="store_true")

    args = parser.parse_args()

    # --- Load features ---
    X, indices, feature_names = load_features(args.features, debug=args.debug_print)

    # --- Optional feature standardization ---
    if args.standardize:
        X, mean, std = standardize_features(X, debug=args.debug_print)
    else:
        if args.debug_print:
            print("Skipping feature standardization (using raw features).")

    # --- Run K-means ---
    labels, centers = run_kmeans(
        X,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        max_iter=args.max_iter,
        n_init=args.n_init,
        debug=args.debug_print
    )

    # --- Save label NIfTI ---
    save_labels_nifti(
        labels=labels,
        indices=indices,
        ref_nifti_path=args.ref_nifti,
        out_path=args.out_labels,
        outside_value=args.outside_label_value,
        debug=args.debug_print
    )

    # --- Optionally save centers ---
    if args.out_centers is not None:
        # Save as npz with feature_names for interpretability
        np.savez(
            args.out_centers,
            centers=centers,
            feature_names=feature_names
        )
        if args.debug_print:
            print("Saved cluster centers to:", args.out_centers)

    print("Done.")
    print("  Labels NIfTI:", args.out_labels)
    if args.out_centers is not None:
        print("  Centers:", args.out_centers)


if __name__ == "__main__":
    main()
