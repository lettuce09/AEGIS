"""Optional geometry check for the NIfTI preprocessing extra."""
import csv

import numpy as np
import pytest

from aegis.prepare import volumes


def test_nifti_geometry_and_label_pair(tmp_path):
    sitk = pytest.importorskip("SimpleITK")
    image = sitk.GetImageFromArray(np.arange(3 * 16 * 16, dtype=np.float32).reshape(3, 16, 16))
    label_array = np.zeros((3, 16, 16), dtype=np.uint8)
    label_array[1, 4:12, 5:10] = 1
    mask = sitk.GetImageFromArray(label_array)
    image.SetSpacing((1.2, 1.3, 2.0))
    mask.CopyInformation(image)
    sitk.WriteImage(image, str(tmp_path / "image.nii.gz"))
    sitk.WriteImage(mask, str(tmp_path / "mask.nii.gz"))
    manifest = tmp_path / "volumes.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["volume_id", "patient_id", "image", "mask", "split"])
        writer.writeheader()
        writer.writerow(dict(volume_id="synthetic", patient_id="synthetic", image="image.nii.gz", mask="mask.nii.gz", split="train"))
    result = volumes(manifest, tmp_path / "valid", image_size=32)
    assert result["input_slices"] == 3
    assert result["saved_slices"] == 1
    mask.SetOrigin((5.0, 0.0, 0.0))
    sitk.WriteImage(mask, str(tmp_path / "mask.nii.gz"))
    with pytest.raises(ValueError, match="physical geometry differs"):
        volumes(manifest, tmp_path / "mismatched", image_size=32)
