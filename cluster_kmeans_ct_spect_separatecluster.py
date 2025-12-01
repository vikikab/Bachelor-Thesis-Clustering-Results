#!/usr/bin/env python3
"""
Run K-means clustering on precomputed voxel features (CT+SPECT) and save:
    1) a full 3D label NIfTI (all clusters)
    2) one 3D NIfTI per cluster (binary masks)

Inputs:
    - features_ct_spect.npz (from extract_features_ct_spect.py)
        Contains:
            X             : (N_voxels, D_features) feature matrix
            indices       : (N_voxels, 3) voxel indices (z, y, x) in FULL volume
            feature_names : (D_features,) list/array of feature names
    - Reference NIfTI (e.g. ct_norm.nii.gz) to get image geometry & shape.

Outputs:
    - NIfTI label volume (e.g. kmeans_labels.nii.gz)
    - NIfTI binary volumes for each cluster:
          <cluster_prefix>_0.nii.gz
          <cluster_prefix>_1.nii.gz
          ...

Usage:
    python cluster_kmeans_ct_spect.py \
        --features features_ct_spect.npz \
        --ref_nifti ct_norm.nii.gz \
        --n_clusters 4 \
        --out_labels kmeans_labels.nii.gz \
        --cluster_prefix cluster \
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


def create_label_volume(labels, indices, ref_img, outside_value=-1, debug=False):
    """
    Create a 3D label volume with same geometry as ref_img.
    - labels: (N_voxels,) int array
    - indices: (N_voxels, 3) z,y,x positions in full volume
    """
    shape = sitk.GetArrayFromImage(ref_img).shape  # (z,y,x)

    if debug:
        print("Reference shape:", shape)

    label_vol = np.full(shape, fill_value=outside_value, dtype=np.int16)

    z = indices[:, 0].astype(int)
    y = indices[:, 1].astype(int)
    x = indices[:, 2].astype(int)

    label_vol[z, y, x] = labels.astype(np.int16)
    return label_vol


def save_nifti_from_array(arr, ref_img, out_path, debug=False):
    img = sitk.GetImageFromArray(arr)
    img.CopyInformation(ref_img)
    sitk.WriteImage(img, out_path)
    if debug:
        print("Saved:", out_path)


def save_per_cluster_volumes(label_vol, ref_img, n_clusters, prefix, debug=False):
    """
    For each cluster k, create a binary volume:
        prefix_k.nii.gz
    where:
        voxel = 1 if label==k
              = 0 otherwise
    """
    for k in range(n_clusters):
        vol_k = (label_vol == k).astype(np.uint8)
        out_path = f"{prefix}_{k}.nii.gz"
        save_nifti_from_array(vol_k, ref_img, out_path, debug=debug)
        if debug:
            print(f"  Cluster {k}: {int(vol_k.sum())} voxels")


def main():
    parser = argparse.ArgumentParser(
        description="K-means clustering on CT+SPECT voxel features and save label NIfTI + per-cluster maps."
    )

    parser.add_argument("--features", required=True,
                        help="Path to features_ct_spect.npz.")
    parser.add_argument("--ref_nifti", required=True,
                        help="Reference NIfTI (e.g. ct_norm.nii.gz).")

    parser.add_argument("--out_labels", default="kmeans_labels.nii.gz",
                        help="Output NIfTI with K-means labels (all clusters).")
    parser.add_argument("--cluster_prefix", default="cluster",
                        help="Prefix for saving per-cluster 3D volumes, e.g. 'cluster' -> cluster_0.nii.gz, ...")
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

    # --- Create full 3D label volume ---
    ref_img = sitk.ReadImage(args.ref_nifti)
    label_vol = create_label_volume(
        labels=labels,
        indices=indices,
        ref_img=ref_img,
        outside_value=args.outside_label_value,
        debug=args.debug_print
    )

    # --- Save combined label volume ---
    save_nifti_from_array(label_vol, ref_img, args.out_labels, debug=args.debug_print)

    # --- Save per-cluster volumes (3D binary maps) ---
    save_per_cluster_volumes(
        label_vol=label_vol,
        ref_img=ref_img,
        n_clusters=args.n_clusters,
        prefix=args.cluster_prefix,
        debug=args.debug_print
    )

    # --- Optionally save centers ---
    if args.out_centers is not None:
        np.savez(
            args.out_centers,
            centers=centers,
            feature_names=feature_names
        )
        if args.debug_print:
            print("Saved cluster centers to:", args.out_centers)

    print("Done.")
    print("  Labels NIfTI:", args.out_labels)
    print("  Per-cluster volumes prefix:", args.cluster_prefix)
    if args.out_centers is not None:
        print("  Centers:", args.out_centers)


if __name__ == "__main__":
    main()
