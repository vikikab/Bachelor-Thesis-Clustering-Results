#!/usr/bin/env python3

import argparse
import SimpleITK as sitk
import numpy as np


def load_nifti(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return img, arr


def save_nifti(arr, ref_img, out_path):
    out_img = sitk.GetImageFromArray(arr)
    out_img.CopyInformation(ref_img)
    sitk.WriteImage(out_img, out_path)


def zscore_normalize(arr, mask):
    vals = arr[mask]
    mean = vals.mean()
    std = vals.std() + 1e-6
    return (arr - mean) / std


def minmax_normalize(arr, mask):
    vals = arr[mask]
    mn = vals.min()
    mx = vals.max()
    return (arr - mn) / (mx - mn + 1e-6)


def log_normalize(arr, mask):
    # log1p safe for zero intensities
    arr_log = np.log1p(arr)
    return zscore_normalize(arr_log, mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct", required=True)
    parser.add_argument("--spect", required=True)
    parser.add_argument("--mask", required=True)

    parser.add_argument("--out_ct", default="ct_norm.nii.gz")
    parser.add_argument("--out_spect", default="spect_norm.nii.gz")

    parser.add_argument("--ct_clip_min", type=float, default=-1000)
    parser.add_argument("--ct_clip_max", type=float, default=2000)
    parser.add_argument("--spect_log", action="store_true",
                        help="Use log-normalization for SPECT.")

    args = parser.parse_args()

    # Load images
    ct_img, ct = load_nifti(args.ct)
    spect_img, spect = load_nifti(args.spect)
    mask_img, mask = load_nifti(args.mask)

    mask_bool = mask.astype(bool)

    # --- Normalize CT ---
    ct = np.clip(ct, args.ct_clip_min, args.ct_clip_max)
    ct_norm = zscore_normalize(ct, mask_bool)

    # --- Normalize SPECT ---
    if args.spect_log:
        spect_norm = log_normalize(spect, mask_bool)
    else:
        spect_norm = zscore_normalize(spect, mask_bool)

    # Save outputs
    save_nifti(ct_norm, ct_img, args.out_ct)
    save_nifti(spect_norm, spect_img, args.out_spect)

    print("Saved normalized CT to:", args.out_ct)
    print("Saved normalized SPECT to:", args.out_spect)


if __name__ == "__main__":
    main()
