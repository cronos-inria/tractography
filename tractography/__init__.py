from typing import Optional

from . import algorithms, configuration, connectivity, core, nifti, seeds, utils
from .algorithms.configuration import Algorithm, BaseConfiguration

import nibabel as nib
import numpy as np
import numpy.typing as npt


def connectome(
    fod: nib.Nifti1Image,
    segmentation: nib.Nifti1Image,
    n_seeds: int = 100000,
    config: Optional[BaseConfiguration] = None,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Generate a structural connectivity matrix from ODFs

    This function performs tractography using fODF data within and returns a
    connectivity matrix representing the structural connections between
    labeled brain regions.

    Args:
        fod: The FOD used to generate the streamlines.
        segmentation: 3D labeled image indicating different brain regions.
        n_seeds: Total number of seeds to generate for tractography.
        config: Configuration object specifying processing parameters and the
            algorithm to use. If None, a default configuration for the
            transport algorithm is loaded. See tg.configuration.load.

    Returns:
        connectome: A symmetric connectivity matrix representing the number
            of streamlines connecting each pair of brain regions.
        labels: The labels associated with each row and column of the matrix.
    """

    # Generate the seeds in the segmented areas.
    mask = nifti.threshold(segmentation, 0.1)
    seed_fod = nifti.multiply(fod, mask, order=0)
    s = seeds.from_fod(seed_fod.get_fdata(), seed_fod.affine, n_seeds)

    # Perform tractography.
    streamlines = tractogram(fod, s, config, endpoints_only=True).streamlines

    # Transform the segmentation into vertices.
    vertices, labels = connectivity.convert_segmentation(
        segmentation.get_fdata(), segmentation.affine
    )

    # Map vertices to streamlines.
    mapping = connectivity.map_vertices(vertices, streamlines, labels)
    symmetric_mapping = connectivity.symmetrize_mapping(mapping)

    # Compile the final matrix.
    return connectivity.compile_connectivity_matrix(symmetric_mapping)


def histogram(
    fod: nib.Nifti1Image,
    seed_mask: nib.Nifti1Image,
    n_seeds: int = 1000000,
    config: BaseConfiguration | None = None,
) -> nib.Nifti1Image:
    """Generates the streamlines histogram

    The histogram correponds to the FOD associated to a particular tracgogram. That is,
    the distribution of streamline orientations, for each voxel. This function
    generates the histogram directly, without saving the intermediate streamlines and
    therefore allows a much larger number of seeds to be used.

    Args:
        fod: The FOD used to generate the streamlines.
        seed_mask: The mask where seeds are generated. They are uniforly distributed
            in the non-zero voxels of the mask and the orientation distribution follows
            the local FOD (see tg.seed.from_fod).
        n_seeds: The number of seeds (streamlines) to generate.
        config: The configuration. If None, the default configuration for the
            diffusion algorithm is used.

    Return:
        The histogram in the form of FOD.

    """

    if config is None:
        config = configuration.load(Algorithm.DIFFUSION)

    # Apply the mask to the FOD to get the seeding FOD.
    seed_fod = nifti.multiply(fod, seed_mask)

    # Load the implementation based on the config file.
    implementation = getattr(algorithms, config.algorithm.value).histogram
    histogram = implementation(
        fod.get_fdata(),
        fod.affine,
        seed_fod.get_fdata(),
        seed_fod.affine,
        n_seeds,
        config,
    )

    return nib.Nifti1Image(histogram, fod.affine)


def tractogram(
    fod: nib.Nifti1Image,
    seeds: list[seeds.Seed],
    config: BaseConfiguration | None = None,
    endpoints_only: bool = False,
) -> nib.streamlines.Tractogram:
    """Generate a tractogram from dMRI data

    The tractogram, which is simply a list of streamlines, is generated
    using the specified algorithm.

    Args:
        fod: The FOD used to generate the streamlines.
        seeds: The seeds used for tractography. See tg.seeds.
        config: Configuration object specifying processing parameters and the
            algorithm to use. If None, a default configuration for the
            transport algorithm is loaded. See tg.configuration.load.
        endpoints_only: Only the start and end points of the streamlines will
            be returned. Greatly reduces the memory footprint.

    Return:
        The generated tractogram, i.e. a list of streamlines in RAS+ millimeter
        space.

    """

    if config is None:
        config = configuration.load(Algorithm.TRANSPORT)

    implementation = config.implementation(
        fod.get_fdata(), fod.affine, config.batch_size, config
    )

    # Perform tractography in batches.
    all_streamlines = []
    for s in np.array_split(seeds, len(seeds) // config.batch_size):
        streamlines = implementation.run(s)

        # Clean a bit.
        streamlines = [s for s in streamlines if len(s) > config.min_steps]

        if endpoints_only:
            streamlines = [s[[0, -1]] for s in streamlines]

        all_streamlines.extend(streamlines)

    return nib.streamlines.Tractogram(all_streamlines, affine_to_rasmm=np.eye(4))
