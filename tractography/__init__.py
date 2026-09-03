from typing import Optional

from . import algorithms, configuration, connectivity, core, domain, mesh, nifti, seeds, utils
from .algorithms.core import Algorithm, BaseConfiguration

from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import numpy.typing as npt


def connectome(
    fod: Nifti1Image,
    segmentation: Nifti1Image,
    n_seeds: int = 1000000,
    config: BaseConfiguration | None = None,
    distance_upper_bound: float = 4.0,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Generates a structural connectivity matrix directly without storing streamlines

    Similar to how ``histogram`` bypasses intermediate streamline storage to
    compute orientation distributions, this function tracks streamlines on the
    GPU and immediately records which pairs of labeled regions are connected,
    producing a connectivity matrix in a single pass.

    Args:
        fod: The FOD used to generate the streamlines.
        segmentation: 3D labeled image indicating different brain regions.
            Integer labels; 0 is treated as background.
        n_seeds: The number of seeds (streamlines) to generate.
        config: The configuration. If None, the default configuration for the
            deterministic algorithm is used.
        distance_upper_bound: Maximum distance between a streamline endpoint
            and a labelled vertex for the endpoint to be assigned that label.

    Returns:
        matrix: A symmetric (n_labels × n_labels) connectivity matrix counting
            the number of streamlines connecting each pair of regions.
        labels: A 1D array of the unique non-zero labels from the segmentation.

    """

    if config is None:
        config = configuration.load(Algorithm.DETERMINISTIC)

    # Build the seed FOD by masking the FOD with the segmentation.
    mask = nifti.threshold(segmentation, 0.1)
    seed_fod = nifti.multiply(fod, mask, order=0)

    # Convert the segmentation to labelled vertices (world-space points).
    seg_data = segmentation.get_fdata().astype(np.int32)
    vertices, vertex_labels = connectivity.convert_segmentation(
        seg_data, segmentation.affine
    )

    # Extract unique labels (excluding background 0).
    labels = np.unique(seg_data)
    labels = labels[labels > 0]
    n_labels = len(labels)

    # Reindex the labels to be continuous.
    label_map = {l:i for i, l in enumerate(labels)}
    vertex_relabels = np.array([label_map[v] for v in vertex_labels])

    # Resolve the algorithm-specific implementation.
    implementation = algorithms.resolve(config.algorithm).connectome
    if implementation is None:
        raise ValueError(f"No connectome implementation registered for algorithm: {config.algorithm}")

    matrix = implementation(
        fod.get_fdata(),
        fod.affine,
        seed_fod.get_fdata(),
        seed_fod.affine,
        vertices,
        vertex_relabels,
        n_labels,
        n_seeds,
        config,
        distance_upper_bound,
    )

    # Symmetrize: count (A→B) and (B→A) together.
    matrix = matrix + matrix.T

    return matrix, labels


def histogram(
    fod: Nifti1Image,
    seed_mask: Nifti1Image,
    n_seeds: int = 1000000,
    config: BaseConfiguration | None = None,
) -> Nifti1Image:
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
    implementation = algorithms.resolve(config.algorithm).histogram
    histogram = implementation(
        fod.get_fdata(),
        fod.affine,
        seed_fod.get_fdata(),
        seed_fod.affine,
        n_seeds,
        config,
    )

    return Nifti1Image(histogram, fod.affine)


def tractogram(
    fod: Nifti1Image,
    seeds: list[seeds.Seed],
    config: BaseConfiguration | None = None,
) -> Tractogram:
    """Generate a tractogram from dMRI data

    The tractogram, which is simply a list of streamlines, is generated
    using the specified algorithm.

    Args:
        fod: The FOD used to generate the streamlines.
        seeds: The seeds used for tractography. See tg.seeds.
        config: Configuration object specifying processing parameters and the
            algorithm to use. If None, a default configuration for the
            transport algorithm is loaded. See tg.configuration.load.

    Return:
        The generated tractogram, i.e. a list of streamlines in RAS+ millimeter
        space.

    """

    if config is None:
        config = configuration.load(Algorithm.TRANSPORT)

    implementation = algorithms.resolve(config.algorithm).tractogram

    # No seeds means no streamlines.
    n_seeds = len(seeds)
    if n_seeds == 0:
        return Tractogram([], affine_to_rasmm=np.eye(4))

    # Pad the seeds to have a multiple of the batch size.
    padded_seeds = list(seeds)
    extra_seeds = n_seeds % config.batch_size
    if extra_seeds > 0:
        padded_seeds.extend([padded_seeds[-1]] * (config.batch_size - extra_seeds))

    # Perform tractography in batches.
    all_streamlines = []
    cache = None
    for i in range(0, len(padded_seeds), config.batch_size):
        s = padded_seeds[i : i + config.batch_size]
        tracto, cache = implementation(fod, s, config, cache)
        streamlines = tracto.streamlines

        # Clean a bit.
        streamlines = [s for s in streamlines if len(s) > config.min_n_points]

        all_streamlines.extend(streamlines)

    all_streamlines = all_streamlines[:n_seeds]  # Ignore extra seeds.
    return Tractogram(all_streamlines, affine_to_rasmm=np.eye(4))
