"""Generate a tractogram from the downloaded iflow example dataset."""

import argparse
from pathlib import Path
from typing import cast

import nibabel as nib
from nibabel.nifti1 import Nifti1Image
from nibabel.spatialimages import SpatialImage
import numpy as np
import tractography as tg


def main(output_path: Path = Path("iflow-tractogram.tck")) -> Path:
    """Download iflow, track from its FOD image, and save a TCK tractogram."""
    
    # download() caches the archive between runs and returns the directory where
    # the dataset files were extracted, without pCloud's outer iflow folder.
    data_dir = tg.data.download()
    fod_path = data_dir / "derivatives" / "sub-001" / "mrtrix3" / "sub-001_fod.nii.gz"

    # This NIfTI image stores a spherical-harmonic (SH) coefficient vector at
    # each voxel. The cast tells the type checker that this file is a NIfTI
    # image; nibabel's generic loader return type is broader.
    fod = cast(Nifti1Image, nib.loadsave.load(fod_path))

    # FreeSurfer's wm.mgz contains nonzero values in white-matter voxels. Make
    # these voxels a binary mask and retain the file's affine so the domain is
    # expressed in the anatomical image's world coordinates.
    wm_path = data_dir / "derivatives" / "sub-001" / "freesurfer" / "mri" / "wm.mgz"
    wm = cast(SpatialImage, nib.loadsave.load(wm_path))
    mask_data = (wm.get_fdata() > 0).astype(np.uint8)
    mask = Nifti1Image(mask_data, wm.affine)
    domain = tg.domain.Image(mask)

    # Sample 100 000 points on the domain boundary. Each seed includes an inward-
    # pointing orientation. A fixed random seed makes this example repeatable.
    seeds = tg.domain.seed(domain, n_seeds=100000, rng_seed=42)

    # The low-level tracking function accepts an (n_seeds, 6) NumPy array:
    # each row is the seed's x, y, z location followed by dx, dy, dz direction.
    seed_array = np.asarray(
        [np.concatenate((seed.location, seed.orientation), axis=0) for seed in seeds],
        dtype=np.float32,
    ).reshape((-1, 6))

    # The SH local model interprets the FOD coefficients and supplies the
    # orientation values and derivatives used by transport tracking.
    model = tg.local.RealSymmetricSphericalHarmonics(fod.shape[-1])
    algorithm = tg.algorithm.Transport(model, step_size=0.001)

    # Wrap the 4D NIfTI image as a field that the OpenCL tracking kernel can
    # evaluate at each point along a streamline.
    field = tg.field.Image(fod)

    # Track up to 300 mm per seed, saving points every 1 mm. step_size controls
    # the smaller internal integration steps used by the algorithm. This
    # low-level function requires an available OpenCL runtime.
    result = tg.tracking.tractogram(
        algorithm,
        domain,
        field,
        seed_array,
        save_at=1.0,
        maximum_streamline_length=300.0,
    )

    # Create the output folder if needed, then save the streamlines as a TCK
    # file, a format supported by common neuroimaging tools.
    output_path = output_path.expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.streamlines.save(result, output_path)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("iflow-tractogram.tck"),
        help="path for the output TCK file (default: %(default)s)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(main(_parse_args().output))
