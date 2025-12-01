#!/usr/bin/env python3
"""
Preprocess & co-register CT and SPECT NIfTI volumes for per-voxel analysis.

- Rigid registration (SPECT -> CT)
- Resample SPECT into CT space
- Bone mask from CT (HU threshold) with morphological cleanup & hole filling
- Active region mask from SPECT (percentage of max uptake)
- Combined mask = bone & active

Usage:
    python preprocess_ct_spect.py \
        --ct ct.nii.gz \
        --spect spect.nii.gz \
        --out_ct ct_resampled.nii.gz \
        --out_spect spect_registered.nii.gz \
        --out_bone_mask bone_mask.nii.gz \
        --out_active_mask active_mask.nii.gz \
        --out_combined_mask combined_mask.nii.gz \
        --bone_thresh 150 \
        --active_fraction 0.3
"""

import argparse
import SimpleITK as sitk
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and co-register CT and SPECT NIfTI volumes."
    )
    parser.add_argument("--ct", type=str, required=True, help="Path to CT NIfTI file.")
    parser.add_argument("--spect", type=str, required=True, help="Path to SPECT NIfTI file.")

    parser.add_argument("--out_ct", type=str, default="ct_resampled.nii.gz",
                        help="Output CT path (resampled / just copied).")
    parser.add_argument("--out_spect", type=str, default="spect_registered.nii.gz",
                        help="Output registered SPECT path.")
    parser.add_argument("--out_bone_mask", type=str, default="bone_mask.nii.gz",
                        help="Output bone mask path.")
    parser.add_argument("--out_active_mask", type=str, default="active_mask.nii.gz",
                        help="Output active SPECT mask path.")
    parser.add_argument("--out_combined_mask", type=str, default="combined_mask.nii.gz",
                        help="Output combined (bone & active) mask path.")

    parser.add_argument("--bone_thresh", type=float, default=150.0,
                        help="HU threshold for bone mask (default: 150).")
    parser.add_argument("--active_fraction", type=float, default=0.3,
                        help="Fraction of SPECT max for active mask threshold (0–1).")

    parser.add_argument("--morph_radius", type=int, default=1,
                        help="Radius for morphological operations (in voxels).")
    parser.add_argument("--debug_print", action="store_true",
                        help="Print some extra debug info.")

    return parser.parse_args()


def register_spect_to_ct(ct_img, spect_img, debug_print=False):
    """
    Rigid registration (Euler3D) of SPECT (moving) to CT (fixed).
    Returns the registered SPECT image in CT space.
    """
    if debug_print:
        print("CT spacing:", ct_img.GetSpacing())
        print("SPECT spacing:", spect_img.GetSpacing())
        print("CT size:", ct_img.GetSize())
        print("SPECT size:", spect_img.GetSize())
        print("CT direction:", ct_img.GetDirection())
        print("SPECT direction:", spect_img.GetDirection())

    fixed = sitk.Cast(ct_img, sitk.sitkFloat32)
    moving = sitk.Cast(spect_img, sitk.sitkFloat32)

    # Initialize with geometry centers
    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method = sitk.ImageRegistrationMethod()

    # Multimodal metric
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
    registration_method.SetMetricSamplingPercentage(0.01)

    # Optimizer
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    # Multi-resolution
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)

    if debug_print:
        print("Starting registration...")

    final_transform = registration_method.Execute(fixed, moving)

    if debug_print:
        print("Registration completed.")
        print("Final metric value:", registration_method.GetMetricValue())
        print("Optimizer stop condition:",
              registration_method.GetOptimizerStopConditionDescription())

    # Resample moving (SPECT) to CT space using the final transform
    spect_registered = sitk.Resample(
        spect_img,
        ct_img,
        final_transform,
        sitk.sitkLinear,
        0.0,
        sitk.sitkFloat32
    )

    return spect_registered


def create_bone_mask(ct_img, hu_threshold=150.0, morph_radius=1, debug_print=False):
    """
    Create a bone mask from CT based on a HU threshold, with morphological
    opening/closing and hole filling.
    """
    if debug_print:
        print(f"Creating bone mask with HU threshold {hu_threshold}...")

    # Threshold CT to get initial bone mask
    mask = ct_img > hu_threshold
    mask = sitk.Cast(mask, sitk.sitkUInt8)


    #in case we don't get it running we can omit the with morphological
    #opening/closing and hole filling which is an optimization

    # Build radius tuple (e.g. (1,1,1) for 3D) for older SimpleITK APIs
    dim = ct_img.GetDimension()  # should be 3 for 3D
    radius_tuple = (morph_radius,) * dim

    if morph_radius > 0:
        if debug_print:
            print(f"Applying morphological opening with radius {radius_tuple}...")
        mask = sitk.BinaryMorphologicalOpening(
            mask,
            kernelRadius=radius_tuple,
            kernelType=sitk.sitkBall
        )

        if debug_print:
            print(f"Applying morphological closing with radius {radius_tuple}...")
        mask = sitk.BinaryMorphologicalClosing(
            mask,
            kernelRadius=radius_tuple,
            kernelType=sitk.sitkBall
        )

     # Fill holes in 3D (works in recent SimpleITK; if yours complains, we can swap to a filter)
    if debug_print:
        print("Filling holes in bone mask...")
    mask = sitk.BinaryFillhole(mask)

    return mask


#optional

def create_active_mask(spect_img, active_fraction=0.3, debug_print=False):
    """
    Create an active-region mask from SPECT based on a fraction of the max intensity.
    """
    spect_array = sitk.GetArrayFromImage(spect_img).astype(np.float32)
    sp_max = float(np.max(spect_array))

    if sp_max <= 0:
        raise ValueError("SPECT maximum intensity is non-positive; cannot threshold.")

    threshold = active_fraction * sp_max

    if debug_print:
        print(f"SPECT max intensity: {sp_max:.3f}")
        print(f"Active fraction: {active_fraction} -> threshold: {threshold:.3f}")

    mask = spect_img > threshold
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    return mask


def main():
    args = parse_args()

    if args.debug_print:
        print("Loading images...")

    #load CT + SPECT
    ct_img = sitk.ReadImage(args.ct)
    spect_img = sitk.ReadImage(args.spect)

    # Register SPECT to CT (rigid)
    spect_registered = register_spect_to_ct(
        ct_img,
        spect_img,
        debug_print=args.debug_print
    )

    # Bone mask from CT
    bone_mask = create_bone_mask(
        ct_img,
        hu_threshold=args.bone_thresh,
        morph_radius=args.morph_radius,
        debug_print=args.debug_print #problematisch
    )

    # Active mask from registered SPECT
    active_mask = create_active_mask(
        spect_registered,
        active_fraction=args.active_fraction,
        debug_print=args.debug_print
    )

    # Combined mask: bone & active (logical AND because 0/1)
    if args.debug_print:
        print("Combining bone & active masks...")

    combined_mask = bone_mask * active_mask  # both are uint8 (0 or 1)

    # Save outputs
    if args.debug_print:
        print("Saving outputs...")

    sitk.WriteImage(ct_img, args.out_ct)  # CT is reference, just save/copy
    sitk.WriteImage(spect_registered, args.out_spect)
    sitk.WriteImage(bone_mask, args.out_bone_mask)
    sitk.WriteImage(active_mask, args.out_active_mask)
    sitk.WriteImage(combined_mask, args.out_combined_mask)

    if args.debug_print:
        # Some basic stats about masks
        bone_arr = sitk.GetArrayFromImage(bone_mask)
        active_arr = sitk.GetArrayFromImage(active_mask)
        combined_arr = sitk.GetArrayFromImage(combined_mask)

        print(f"Bone mask voxels: {int(np.sum(bone_arr))}")
        print(f"Active mask voxels: {int(np.sum(active_arr))}")
        print(f"Combined mask voxels: {int(np.sum(combined_arr))}")

    print("Done.")


if __name__ == "__main__":
    main()
