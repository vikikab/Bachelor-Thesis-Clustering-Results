import SimpleITK as sitk

ct_img = sitk.ReadImage("ct.nii.gz")        # Fixed image (reference)
spect_img = sitk.ReadImage("spect.nii.gz")  # Moving image (will be aligned to CT)
print("CT size:", ct_img.GetSize(), "spacing:", ct_img.GetSpacing())
print("SPECT size:", spect_img.GetSize(), "spacing:", spect_img.GetSpacing())
