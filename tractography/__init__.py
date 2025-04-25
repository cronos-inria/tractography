from . import algorithms, configuration, core, seeds, utils
from .core import Algorithm


def tractogram(
    data,
    affine,
    seeds: list[seeds.Seed],
    algorithm: Algorithm,
    config: configuration.Configuration | None = None,
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
        step_size: The time interval between two steps.
        n_steps: The maximum number of tractography steps to perform for each
            streamline.
        max_angle: The maximum angle between two steps.

    Return:
        The generated tractogram, i.e. a list of streamlines.

    """

    if config is None:
        config = configuration.load()

    # Normalize the ODFs if needed.
    if algorithm != Algorithm.FACT:
        data = utils.normalize_odf(data)

    if algorithm == Algorithm.DETERMINISTIC:
        streamlines = algorithms.deterministic(data, affine, seeds, config)
    elif algorithm == Algorithm.PROBABILISTIC:
        probabilistic = algorithms.Probabilistic(data, affine, len(seeds), config)
        streamlines = probabilistic.run(seeds)
    elif algorithm == Algorithm.BOLTZMANN:
        streamlines = algorithms.boltzmann(data, affine, seeds, config)
    elif algorithm == Algorithm.FACT:
        fact = algorithms.FACT(data, affine, len(seeds), config)
        streamlines = fact.run(seeds)
    else:
        raise ValueError(f"No algorithm associated with {algorithm}.")

    # Clean a bit.
    if algorithm != Algorithm.FACT and algorithm != Algorithm.PROBABILISTIC:
        streamlines = core.remove_duplicate_endpoints(streamlines, config.step_size)
    streamlines = [s for s in streamlines if len(s) > config.min_steps]

    return streamlines
