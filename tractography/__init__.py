from . import algorithms, configuration, connectivity, core, seeds, utils
from .core import Algorithm

import numpy as np


def connectome(
    odf,
    odf_affine,
    segmentation,
    segmentation_affine,
    n_seeds: int = 100000,
    algorithm: Algorithm = Algorithm.DIFFUSION,
    config: configuration.Configuration | None = None,
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
        algorithm: The algorithm used to perform tractography.
        config: Configuration object specifying processing parameters.
            If None, a default configuration is loaded.

    Returns:
        connectome: A symmetric connectivity matrix representing the number
            of streamlines connecting each pair of brain regions.
        labels: The labels associated with each row and column of the matrix.
    """

    # Generate the seeds in the segmented areas.
    seed_odf = core.apply_mask(odf, odf_affine, segmentation > 0, segmentation_affine)
    s = seeds.from_odf(seed_odf, odf_affine, n_seeds)

    # Perform tractography.
    streamlines = tractogram(odf, odf_affine, s, algorithm, config, endpoints_only=True)

    # Transform the segmentation into vertices.
    vertices, labels = connectivity.convert_segmentation(
        segmentation, segmentation_affine
    )

    # Map vertices to streamlines.
    mapping = connectivity.map_vertices(vertices, streamlines, labels)
    symmetric_mapping = connectivity.symmetrize_mapping(mapping)

    # Compile the final matrix.
    return connectivity.compile_connectivity_matrix(symmetric_mapping)


def tractogram(
    data,
    affine,
    seeds: list[seeds.Seed],
    algorithm: Algorithm = Algorithm.DIFFUSION,
    config: configuration.Configuration | None = None,
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
        algorithm: The algorithm used to perform tractography.
        config: Configuration object specifying processing parameters.
            If None, a default configuration is loaded.

    Return:
        The generated tractogram, i.e. a list of streamlines.

    """

    if config is None:
        config = configuration.load()

    # Normalize the ODFs if needed.
    if algorithm != Algorithm.FACT:
        data = utils.normalize_odf(data)

    if algorithm == Algorithm.DETERMINISTIC:
        tractography = algorithms.Deterministic(data, affine, config.batch_size, config)
    elif algorithm == Algorithm.PROBABILISTIC:
        tractography = algorithms.Probabilistic(data, affine, config.batch_size, config)
    elif algorithm == Algorithm.BOLTZMANN:
        tractography = algorithms.Boltzmann(data, affine, config.batch_size, config)
    elif algorithm == Algorithm.FACT:
        tractography = algorithms.FACT(data, affine, config.batch_size, config)
    elif algorithm == Algorithm.DIFFUSION:
        tractography = algorithms.Diffusion(data, affine, config.batch_size, config)
    else:
        raise ValueError(f"No algorithm associated with {algorithm}.")

    # Perform tractography in batches.
    all_streamlines = []
    for s in np.array_split(seeds, len(seeds) // config.batch_size):
        streamlines = tractography.run(s)

        # Clean a bit.
        streamlines = [s for s in streamlines if len(s) > config.min_steps]
        streamlines = [s for s in streamlines if not np.any(np.isnan(s)) and not np.any(np.isinf(s))]
        if endpoints_only:
            streamlines = [s[[0, -1]] for s in streamlines]

        all_streamlines.extend(streamlines)

    return all_streamlines
