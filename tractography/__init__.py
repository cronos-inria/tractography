from . import algorithms, configuration, connectivity, core, seeds, utils
from .algorithms.configuration import Algorithm, BaseConfiguration

import numpy as np


def connectome(
    odf,
    odf_affine,
    segmentation,
    segmentation_affine,
    n_seeds: int = 100000,
    config: BaseConfiguration | None = None,
):
    """Generate a structural connectivity matrix from ODFs

    This function performs tractography using fODF data within and returns a
    connectivity matrix representing the structural connections between
    labeled brain regions.

    Args:
        odf: 4D array of fODF data.
        odf_affine: Affine transformation matrix for the fODF image space.
        segmentation: 3D labeled image indicating different brain regions.
        segmentation_affine: Affine transformation matrix for the
            segmentation image.
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
    seed_odf = core.apply_mask(odf, odf_affine, segmentation > 0, segmentation_affine)
    s = seeds.from_odf(seed_odf, odf_affine, n_seeds)

    # Perform tractography.
    streamlines = tractogram(odf, odf_affine, s, config, endpoints_only=True)

    # Transform the segmentation into vertices.
    vertices, labels = connectivity.convert_segmentation(
        segmentation, segmentation_affine
    )

    # Map vertices to streamlines.
    mapping = connectivity.map_vertices(vertices, streamlines, labels)
    symmetric_mapping = connectivity.symmetrize_mapping(mapping)

    # Compile the final matrix.
    return connectivity.compile_connectivity_matrix(symmetric_mapping)


def histogram(
    data,
    affine,
    seeds: list[seeds.Seed],
    config: BaseConfiguration | None = None,
):

    if config is None:
        config = configuration.load(Algorithm.TRANSPORT)

    implementation = config.implementation(data, affine, config.batch_size, config)

    # Construct histogram in batches.
    histogram = 0
    for s in np.array_split(seeds, len(seeds) // config.batch_size):
        batch_histogram, bin_centers = implementation.histogram(s)
        histogram = histogram + batch_histogram

    return histogram, bin_centers


def tractogram(
    data,
    affine,
    seeds: list[seeds.Seed],
    config: BaseConfiguration | None = None,
    endpoints_only: bool = False,
):
    """Generate a tractogram from dMRI data

    The tractogram, which is simply a list of streamlines, is generated
    using the specified algorithm.

    Args:
        data: The image data used to perform tractography. It must represent
            fiber orientation distributions in spherical harmonics form.
        affine: The affine transformation of the data.
        seeds: The seeds used for tractography. See tg.seeds.
        config: Configuration object specifying processing parameters and the
            algorithm to use. If None, a default configuration for the
            transport algorithm is loaded. See tg.configuration.load.
        endpoints_only: Only the start and end points of the streamlines will
            be returned. Greatly reduces the memory footprint.

    Return:
        The generated tractogram, i.e. a list of streamlines.

    """

    if config is None:
        config = configuration.load(Algorithm.TRANSPORT)

    implementation = config.implementation(data, affine, config.batch_size, config)

    # Perform tractography in batches.
    all_streamlines = []
    for s in np.array_split(seeds, len(seeds) // config.batch_size):
        streamlines = implementation.run(s)

        # Clean a bit.
        streamlines = [s for s in streamlines if len(s) > config.min_steps]
        streamlines = [
            s
            for s in streamlines
            if not np.any(np.isnan(s)) and not np.any(np.isinf(s))
        ]
        if endpoints_only:
            streamlines = [s[[0, -1]] for s in streamlines]

        all_streamlines.extend(streamlines)

    return all_streamlines
