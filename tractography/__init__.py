from . import algorithm as algorithm_module
from . import algorithms, configuration, connectivity, core, data, domain as domain_module
from . import field as field_module
from . import local, nifti, seeds, tracking, utils
from .algorithms.core import Algorithm, BaseConfiguration
from .algorithm import BaseAlgorithm

from nibabel.affines import apply_affine
from nibabel.nifti1 import Nifti1Image
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

seeds_module = seeds


def _tracking_components(
    fod: Nifti1Image,
    tracking_algorithm: BaseAlgorithm,
    domain_image: Nifti1Image | None = None,
):
    """Construct low-level domain and field objects for an algorithm instance."""
    _validate_algorithm(tracking_algorithm)
    if tracking_algorithm.local_model.n_parameters != fod.shape[-1]:
        raise ValueError(
            "Algorithm local model parameter count does not match the FOD: "
            f"{tracking_algorithm.local_model.n_parameters} != {fod.shape[-1]}"
        )

    if domain_image is None:
        fod_values = np.asanyarray(fod.dataobj)
        mask = np.any(fod_values != 0, axis=-1).astype(np.uint8)
        domain_image = Nifti1Image(mask, fod.affine)

    return tracking_algorithm, domain_module.Image(domain_image), field_module.Image(fod)


def _algorithm_from_config(
    fod: Nifti1Image,
    config: BaseConfiguration,
) -> BaseAlgorithm:
    """Create a low-level algorithm instance for CLI configuration values."""
    model_type = algorithms.core.LocalModel.from_shape(fod.shape)
    if model_type == algorithms.core.LocalModel.DTI:
        local_model = local.DiffusionTensorImaging()
    else:
        local_model = local.RealSymmetricSphericalHarmonics(fod.shape[-1])

    step_size = float(config.step_size)
    algorithm_type = config.algorithm
    if algorithm_type == Algorithm.TRANSPORT:
        tracking_algorithm = algorithm_module.Transport(
            local_model,
            step_size=step_size,
            gamma=float(getattr(config, "inverse_curvature", 1.0)),
        )
    elif algorithm_type == Algorithm.DETERMINISTIC:
        max_angle = np.deg2rad(getattr(config, "maximum_angle", 30.0))
        tracking_algorithm = algorithm_module.Deterministic(
            local_model, step_size=step_size, max_step_size=max_angle
        )
    elif algorithm_type == Algorithm.DIFFUSION:
        tracking_algorithm = algorithm_module.Diffusion(
            local_model,
            step_size=step_size,
            gamma=float(getattr(config, "inverse_curvature", 1.0)),
            noise_variance=float(getattr(config, "noise_variance", 0.1)),
        )
    elif algorithm_type == Algorithm.PROBABILISTIC:
        max_angle = np.deg2rad(getattr(config, "maximum_angle", 20.0))
        tracking_algorithm = algorithm_module.Probabilistic(
            local_model, step_size=step_size, max_angle=max_angle
        )
    else:
        raise ValueError(f"Unsupported tracking algorithm: {algorithm_type}")

    return tracking_algorithm


def _validate_algorithm(tracking_algorithm: BaseAlgorithm) -> None:
    if not isinstance(tracking_algorithm, BaseAlgorithm):
        raise TypeError("algorithm must be a BaseAlgorithm instance.")


def _seeds_to_tracking_array(seed_list: list[seeds_module.Seed]) -> np.ndarray:
    """Convert public Seed objects to the low-level ``(n, 6)`` layout."""
    if not seed_list:
        return np.empty((0, 6), dtype=np.float32)
    return np.asarray(
        [np.concatenate((seed.location, seed.orientation)) for seed in seed_list],
        dtype=np.float32,
    )


def _local_model_from_parameter_count(n_parameters: int):
    model_type = algorithms.core.LocalModel.from_shape((n_parameters,))
    if model_type == algorithms.core.LocalModel.DTI:
        return local.DiffusionTensorImaging()
    return local.RealSymmetricSphericalHarmonics(n_parameters)


def connectome(
    field: Nifti1Image | field_module.BaseField,
    n_seeds: int = 1000000,
    domain: domain_module.BaseDomain | Nifti1Image | None = None,
    algorithm: BaseAlgorithm | None = None,
    segmentation: Nifti1Image | None = None,
    *,
    distance_upper_bound: float = 4.0,
    maximum_streamline_length: float = 300.0,
    minimum_streamline_length: float = 0.0,
) -> tuple[csr_matrix | npt.NDArray, npt.NDArray]:
    """Generate a boundary or region connectome.

    ``field`` may be a supported four-dimensional NIfTI image or an existing
    spatial field. For an image, the corresponding local model is inferred
    from its coefficient count. If ``algorithm`` is omitted, transport is
    selected. The tracking domain defaults to nonzero image-field voxels.

    Without ``segmentation``, the raw directed boundary-vertex connectome is
    returned as a sparse matrix, with vertex indices in ``labels``. When a
    segmentation is supplied, boundary vertices are assigned to their nearest
    positive segmentation voxel within ``distance_upper_bound``. The directed
    boundary matrix is aggregated by region without symmetrizing it.

    Args:
        field: A supported 4D NIfTI image or a prepared spatial field.
        n_seeds: Number of tracks to launch from the FOD-domain boundary.
        domain: Optional tracking domain, either a domain object or a 3D
            NIfTI image. If omitted, nonzero image-field voxels are used.
        algorithm: Orientation-update algorithm. Defaults to transport using
            the local model inferred from the field's parameter count.
        segmentation: Optional 3D image of positive integer region labels.
        distance_upper_bound: Maximum distance for assigning a boundary vertex
            to a nonzero segmentation voxel.
        maximum_streamline_length: Maximum propagation distance.
        minimum_streamline_length: Minimum propagated distance for a track to
            contribute to the connectome.
    """

    if not isinstance(n_seeds, (int, np.integer)) or n_seeds < 0:
        raise ValueError("n_seeds must be a non-negative integer.")
    if not np.isfinite(distance_upper_bound) or distance_upper_bound < 0:
        raise ValueError("distance_upper_bound must be finite and non-negative.")
    if not np.isfinite(maximum_streamline_length) or maximum_streamline_length < 0:
        raise ValueError("maximum_streamline_length must be finite and non-negative.")
    if not np.isfinite(minimum_streamline_length) or minimum_streamline_length < 0:
        raise ValueError("minimum_streamline_length must be finite and non-negative.")
    if minimum_streamline_length > maximum_streamline_length:
        raise ValueError(
            "minimum_streamline_length must not exceed maximum_streamline_length."
        )

    seg_data = None
    labels = None
    if segmentation is not None:
        if not isinstance(segmentation, Nifti1Image):
            raise TypeError("segmentation must be a Nifti1Image or None.")
        seg_data = np.asanyarray(segmentation.dataobj)
        if seg_data.ndim != 3:
            raise ValueError("Segmentation image must be three-dimensional.")
        labels = np.unique(seg_data[seg_data > 0])
        labels = labels[labels > 0]
        if len(labels) == 0:
            return np.zeros((0, 0), dtype=np.float32), labels

    if isinstance(field, Nifti1Image):
        if field.ndim != 4:
            raise ValueError("The tracking image must be four-dimensional.")
        tracking_field = field_module.Image(field)
        domain_image = Nifti1Image(
            np.any(np.asanyarray(field.dataobj) != 0, axis=-1).astype(np.uint8),
            field.affine,
        )
        local_model = _local_model_from_parameter_count(field.shape[-1])
    elif isinstance(field, field_module.BaseField):
        tracking_field = field
        local_model = (
            _local_model_from_parameter_count(field.n_parameters)
            if algorithm is None
            else None
        )
        if isinstance(field, field_module.Image):
            domain_image = Nifti1Image(
                np.any(field.values != 0, axis=-1).astype(np.uint8),
                np.linalg.inv(field.inverse_affine),
            )
        else:
            domain_image = None
    else:
        raise TypeError("field must be a Nifti1Image or a BaseField instance.")

    if algorithm is None:
        assert local_model is not None
        tracking_algorithm = algorithm_module.Transport(local_model)
    else:
        _validate_algorithm(algorithm)
        tracking_algorithm = algorithm
    if tracking_algorithm.local_model.n_parameters != tracking_field.n_parameters:
        raise ValueError(
            "Algorithm local model parameter count does not match the field: "
            f"{tracking_algorithm.local_model.n_parameters} != "
            f"{tracking_field.n_parameters}"
        )

    if domain is not None and not isinstance(
        domain, (Nifti1Image, domain_module.BaseDomain)
    ):
        raise TypeError("domain must be a BaseDomain, Nifti1Image, or None.")
    if isinstance(domain, Nifti1Image):
        tracking_domain = domain_module.Image(domain)
    elif domain is not None:
        tracking_domain = domain
    elif domain_image is not None:
        if not np.any(np.asanyarray(domain_image.dataobj) != 0):
            if labels is None:
                return csr_matrix((0, 0), dtype=np.float32), np.empty(0, dtype=np.intp)
            return np.zeros((len(labels), len(labels)), dtype=np.float32), labels
        tracking_domain = domain_module.Image(domain_image)
    else:
        raise ValueError(
            "A domain is required when the field is not an image field."
        )

    boundary_matrix = tracking.connectome(
        tracking_algorithm,
        tracking_domain,
        tracking_field,
        n_seeds,
        maximum_streamline_length=maximum_streamline_length,
        minimum_streamline_length=minimum_streamline_length,
    )

    if labels is None:
        return boundary_matrix, np.arange(boundary_matrix.shape[0], dtype=np.intp)

    assert seg_data is not None and labels is not None and segmentation is not None
    voxel_indices = np.argwhere(seg_data > 0)
    affine = segmentation.affine if segmentation.affine is not None else np.eye(4)
    voxel_centers = apply_affine(affine, voxel_indices)
    distances, nearest = cKDTree(voxel_centers).query(
        tracking_domain.boundary.vertices,
        distance_upper_bound=distance_upper_bound,
    )
    distances = np.asarray(distances).reshape(-1)
    nearest = np.asarray(nearest, dtype=np.intp).reshape(-1)
    assigned = np.isfinite(distances)
    label_indices = {label: index for index, label in enumerate(labels)}
    vertex_labels = np.zeros(len(tracking_domain.boundary.vertices), dtype=np.intp)
    assigned_voxels = voxel_indices[nearest[assigned]]
    assigned_labels = seg_data[
        assigned_voxels[:, 0], assigned_voxels[:, 1], assigned_voxels[:, 2]
    ]
    vertex_labels[assigned] = np.fromiter(
        (label_indices[label] for label in assigned_labels),
        dtype=np.intp,
        count=np.count_nonzero(assigned),
    )
    selected = np.flatnonzero(assigned)
    assignment = csr_matrix(
        (
            np.ones(len(selected), dtype=np.float32),
            (vertex_labels[selected], selected),
        ),
        shape=(len(labels), len(vertex_labels)),
    )
    matrix = (assignment @ boundary_matrix @ assignment.T).tocsr()
    return matrix.toarray(), labels


def tod(
    field: Nifti1Image | field_module.BaseField,
    n_seeds: int = 1000000,
    domain: domain_module.BaseDomain | Nifti1Image | None = None,
    algorithm: BaseAlgorithm | None = None,
    *,
    maximum_streamline_length: float = 300.0,
) -> Nifti1Image:
    """Estimate a track-orientation distribution (TOD) from a field.

    ``field`` may be a supported four-dimensional NIfTI image or an existing
    spatial field. For an image, the local model is inferred from its
    coefficient count. If ``algorithm`` is omitted, transport is selected.
    Seeds are sampled from the boundary of ``domain``. When no domain is
    supplied, nonzero image-field voxels define it. The TOD is accumulated on
    the field's image grid. For a custom field, an image-backed domain is
    required to define the output grid.

    Args:
        field: A supported 4D NIfTI image or a prepared spatial field.
        n_seeds: Number of boundary-seeded tracks.
        domain: Optional tracking domain, either a domain object or a 3D
            NIfTI image. If omitted, nonzero image-field voxels are used.
        algorithm: Orientation-update algorithm. Defaults to transport using
            the local model inferred from the field's parameter count.
        maximum_streamline_length: Maximum propagation distance.

    Returns:
        A NIfTI image with 45 real, symmetric spherical-harmonic TOD
        coefficients per voxel.

    Raises:
        TypeError: If the field, domain, or algorithm has an unsupported type.
        ValueError: If the image/model is unsupported, the field and algorithm
            are incompatible, or a numeric option is invalid.
    """

    if not isinstance(n_seeds, (int, np.integer)) or n_seeds < 0:
        raise ValueError("n_seeds must be a non-negative integer.")
    if not np.isfinite(maximum_streamline_length) or maximum_streamline_length < 0:
        raise ValueError("maximum_streamline_length must be finite and non-negative.")
    if isinstance(field, Nifti1Image):
        if field.ndim != 4:
            raise ValueError("The tracking image must be four-dimensional.")
        tracking_field = field_module.Image(field)
        domain_image = Nifti1Image(
            np.any(np.asanyarray(field.dataobj) != 0, axis=-1).astype(np.uint8),
            field.affine,
        )
        local_model = _local_model_from_parameter_count(field.shape[-1])
        grid_shape = field.shape[:3]
        grid_affine = field.affine
        grid_header = field.header.copy()
    elif isinstance(field, field_module.BaseField):
        tracking_field = field
        local_model = (
            _local_model_from_parameter_count(field.n_parameters)
            if algorithm is None
            else None
        )
        if isinstance(field, field_module.Image):
            domain_image = Nifti1Image(
                np.any(field.values != 0, axis=-1).astype(np.uint8),
                np.linalg.inv(field.inverse_affine),
            )
            grid_shape = tuple(int(size) for size in field.dims[:3])
            grid_affine = np.linalg.inv(field.inverse_affine)
            grid_header = None
        else:
            domain_image = None
            grid_shape = None
            grid_affine = None
            grid_header = None
    else:
        raise TypeError("field must be a Nifti1Image or a BaseField instance.")

    if algorithm is None:
        assert local_model is not None
        tracking_algorithm = algorithm_module.Transport(local_model)
    else:
        _validate_algorithm(algorithm)
        tracking_algorithm = algorithm
    if tracking_algorithm.local_model.n_parameters != tracking_field.n_parameters:
        raise ValueError(
            "Algorithm local model parameter count does not match the field: "
            f"{tracking_algorithm.local_model.n_parameters} != "
            f"{tracking_field.n_parameters}"
        )

    if domain is not None and not isinstance(
        domain, (Nifti1Image, domain_module.BaseDomain)
    ):
        raise TypeError("domain must be a BaseDomain, Nifti1Image, or None.")
    if isinstance(domain, Nifti1Image):
        if domain.ndim != 3:
            raise ValueError("Domain image must be three-dimensional.")
        domain_image = domain
        if grid_shape is None:
            grid_shape = domain.shape
            grid_affine = domain.affine
            grid_header = domain.header.copy()
    elif domain is not None:
        tracking_domain = domain
        if grid_shape is None and isinstance(domain, domain_module.Image):
            grid_shape = tuple(int(size) for size in domain.dims[:3])
            grid_affine = np.linalg.inv(domain.inverse_affine)
    elif domain_image is not None:
        domain = domain_image
    else:
        raise ValueError(
            "A domain is required when the field is not an image field."
        )

    if grid_shape is None or grid_affine is None:
        raise ValueError(
            "A field image or image-based domain is required to define the TOD grid."
        )

    grid_header = grid_header.copy() if grid_header is not None else None
    if grid_header is not None:
        grid_header.set_data_shape(grid_shape)
        grid_header.set_data_dtype(np.float32)
    image_grid = Nifti1Image(
        np.zeros(grid_shape, dtype=np.float32), grid_affine, header=grid_header
    )

    if isinstance(domain, Nifti1Image):
        if not np.any(np.asanyarray(domain.dataobj) != 0):
            output_header = image_grid.header.copy()
            output_header.set_data_shape(tuple(grid_shape) + (45,))
            return Nifti1Image(
                np.zeros(tuple(grid_shape) + (45,), dtype=np.float32),
                grid_affine,
                header=output_header,
            )
        tracking_domain = domain_module.Image(domain)
    elif domain is None:
        raise ValueError("Could not infer a tracking domain from the field.")

    return tracking.tod(
        tracking_algorithm,
        tracking_domain,
        tracking_field,
        image_grid,
        n_seeds,
        maximum_streamline_length=maximum_streamline_length,
    )


def tractogram(
    field: Nifti1Image | field_module.BaseField,
    seeds: list[seeds_module.Seed] | int,
    domain: domain_module.BaseDomain | Nifti1Image | None = None,
    algorithm: BaseAlgorithm | None = None,
    *,
    save_at: float = 1.0,
    maximum_streamline_length: float = 300.0,
    minimum_streamline_length: float = 0.0,
    batch_size: int = 50000,
) -> Tractogram:
    """Generate streamlines from a NIfTI image or an existing field.

    This is the main entry point for tractography. Pass a four-dimensional
    NIfTI image containing either six diffusion-tensor coefficients per voxel
    or 45 real, symmetric spherical-harmonic coefficients per voxel. The
    appropriate local model and image field are created automatically. A
    prepared :class:`tractography.field.BaseField` can also be passed directly.

    If ``algorithm`` is omitted, transport tracking is used. If ``domain`` is
    omitted, the nonzero voxels of the image field define the tracking domain.
    For a custom field, provide a domain unless the field is an image field
    created by this package.

    ``seeds`` may be a list of :class:`tractography.seeds.Seed` objects or an
    integer count. An integer samples seed locations uniformly on the domain
    boundary, nudges them slightly inside the domain, and points each seed
    inward. A seed list gives explicit starting positions and orientations.

    Example:
        ``tractogram(fod_image, seeds=10000)``

    The returned NiBabel tractogram is held in memory. For very large runs,
    use the command-line interface, which can stream TRX output in batches.

    Args:
        field: A supported 4D NIfTI image or a prepared spatial field.
        seeds: Explicit seed objects, or a non-negative number of boundary
            seeds to generate.
        domain: Optional tracking domain, either a domain object or a 3D
            NIfTI image. If omitted, nonzero image-field voxels are used.
        algorithm: Orientation-update algorithm. Defaults to transport using
            the local model inferred from the field's parameter count.
        save_at: Distance interval between saved points, in field coordinates.
        maximum_streamline_length: Maximum propagation distance per track.
        minimum_streamline_length: Minimum saved-point distance for retaining
            a streamline.
        batch_size: Maximum seeds passed to each low-level tracking call.

    Returns:
        A NiBabel tractogram whose points are in the field/domain world
        coordinate system.

    Raises:
        TypeError: If the field, seeds, domain, or algorithm has an unsupported
            type.
        ValueError: If the image/model is unsupported, the field and algorithm
            are incompatible, or a numeric tracking option is invalid.
    """

    if not np.isfinite(save_at) or save_at <= 0:
        raise ValueError("save_at must be finite and positive.")
    if not np.isfinite(maximum_streamline_length) or maximum_streamline_length < 0:
        raise ValueError("maximum_streamline_length must be finite and non-negative.")
    if not np.isfinite(minimum_streamline_length) or minimum_streamline_length < 0:
        raise ValueError("minimum_streamline_length must be finite and non-negative.")
    if minimum_streamline_length > maximum_streamline_length:
        raise ValueError(
            "minimum_streamline_length must not exceed maximum_streamline_length."
        )
    if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if isinstance(field, Nifti1Image):
        if field.ndim != 4:
            raise ValueError("The tracking image must be four-dimensional.")
        tracking_field = field_module.Image(field)
        domain_image = Nifti1Image(
            np.any(np.asanyarray(field.dataobj) != 0, axis=-1).astype(np.uint8),
            field.affine,
        )
        local_model = _local_model_from_parameter_count(field.shape[-1])
    elif isinstance(field, field_module.BaseField):
        tracking_field = field
        local_model = (
            _local_model_from_parameter_count(field.n_parameters)
            if algorithm is None
            else None
        )
        if isinstance(field, field_module.Image):
            affine = np.linalg.inv(field.inverse_affine)
            domain_image = Nifti1Image(
                np.any(field.values != 0, axis=-1).astype(np.uint8), affine
            )
        else:
            domain_image = None
    else:
        raise TypeError("field must be a Nifti1Image or a BaseField instance.")

    if algorithm is None:
        assert local_model is not None
        tracking_algorithm = algorithm_module.Transport(local_model)
    else:
        _validate_algorithm(algorithm)
        tracking_algorithm = algorithm
    if tracking_algorithm.local_model.n_parameters != tracking_field.n_parameters:
        raise ValueError(
            "Algorithm local model parameter count does not match the field: "
            f"{tracking_algorithm.local_model.n_parameters} != "
            f"{tracking_field.n_parameters}"
        )

    if isinstance(seeds, (int, np.integer)):
        if seeds < 0:
            raise ValueError("seeds must be a non-negative integer or a list of Seed objects.")
        n_seeds = int(seeds)
    elif isinstance(seeds, list) and all(
        isinstance(seed, seeds_module.Seed) for seed in seeds
    ):
        n_seeds = len(seeds)
    else:
        raise TypeError("seeds must be a non-negative integer or a list of Seed objects.")

    if domain is not None and not isinstance(
        domain, (Nifti1Image, domain_module.BaseDomain)
    ):
        raise TypeError("domain must be a BaseDomain, Nifti1Image, or None.")
    if n_seeds == 0:
        return Tractogram([], affine_to_rasmm=np.eye(4))

    if isinstance(domain, Nifti1Image):
        tracking_domain = domain_module.Image(domain)
    elif domain is not None:
        tracking_domain = domain
    elif domain_image is not None:
        tracking_domain = domain_module.Image(domain_image)
    else:
        raise ValueError(
            "A domain is required when the field is not an image field and "
            "does not provide an image-derived nonzero domain."
        )

    if isinstance(seeds, (int, np.integer)):
        seeds = domain_module.seed(tracking_domain, n_seeds)
    if not seeds:
        return Tractogram([], affine_to_rasmm=np.eye(4))

    streamlines = []
    for start in range(0, len(seeds), batch_size):
        result = tracking.tractogram(
            tracking_algorithm,
            tracking_domain,
            tracking_field,
            _seeds_to_tracking_array(seeds[start : start + batch_size]),
            save_at=save_at,
            maximum_streamline_length=maximum_streamline_length,
        )
        streamlines.extend(
            streamline
            for streamline in result.streamlines
            if len(streamline) > int(np.ceil(minimum_streamline_length / save_at))
        )
    return Tractogram(streamlines, affine_to_rasmm=np.eye(4))
