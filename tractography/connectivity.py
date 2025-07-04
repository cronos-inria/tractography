from collections import defaultdict
from itertools import product
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

import nibabel as nib
import numpy as np
import scipy.spatial


def compile_length_connectivity_matrix(
    mapping: defaultdict,
    streamlines: Sequence[np.ndarray],
    labels: Optional[Iterable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compiles a length connectivity matrix from a mapping and streamlines

    Args:
        mapping: A map from labels to streamline ids generated using
            the map_labels function.
        streamlines: The streamlines used to build the mapping.
        labels: The labels of the connectivity matrix.

    Returns:
        matrix: The connectivity matrix in the form of a square 2D array.
        labels: A 1D array that contains the labels for each row/column.

    """

    def length(line):
        return np.sum(np.sqrt(np.sum(np.power(np.diff(line, axis=0), 2), axis=1)))

    def mean_length(indices):
        return np.mean([length(streamlines[i]) for i in indices])

    matrix, labels = compile_connectivity_matrix(mapping, labels, mean_length, np.Inf)

    # The distance between a label and itself is 0.
    np.fill_diagonal(matrix, 0)

    return matrix, labels


def compile_connectivity_matrix(
    mapping: defaultdict,
    labels: Optional[Iterable] = None,
    compile_fun: Optional[Callable] = None,
    default_value: Any = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compiles a connectivity matrix from a mapping

    This function can compile many types of connectivity matrices, depending on
    the compile_fun that is provided. By default, if no function is provided,
    a streamline count matrix is returned.

    Args:
        mapping: A map from labels to streamline ids generated using
            the map_labels function.
        labels (optional): The labels of the connectivity matrix.
        compile_fun (optional): The function that compiles a list of
            streamlines into a single connectivity value.
        default_value (optional): The default value used in the connectivity
            matrix when a label pair does not appear in the mapping.

    Returns:
        matrix: The connectivity matrix in the form of a square 2D array.
        labels: A 1D array that contains the labels for each row/column.

    """

    if compile_fun is None:

        def compile_fun(streamline_indices):
            return len(streamline_indices)

    # If the labels are not provided, we use the unique labels from the
    # mapping.
    if labels is None:
        keys = [k for k, v in mapping.items() if len(v) != 0]
        labels = np.unique([label for key in keys for label in key])

    matrix = np.full((len(labels),) * 2, default_value)

    # Compute the value of the connectivity matrix for each label pair. Here we
    # use the product of labels instead of just combinations because the
    # mapping is not necessarily symmetric.
    for row, column in product(range(len(labels)), repeat=2):
        indices = mapping[(labels[row], labels[column])]
        if len(indices) != 0:
            matrix[row, column] = compile_fun(indices)

    return matrix, labels


def compile_metric_connectivity_matrix(
    mapping: defaultdict,
    streamlines: Sequence[np.ndarray],
    metric: np.ndarray,
    labels: Optional[Iterable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compiles a metric connectivity matrix from a mapping and a metric

    The metric is a scalar valued metric, for example FA, MD, etc. The
    connectivity between two regions will be the mean of the metric along
    the streamlines connecting the two regions.

    Args:
        mapping: A map from labels to streamline ids generated using
            the map_labels function.
        streamlines: The streamlines used to build the mapping. The streamlines
            should be in voxel space.
        metric: A metric (FA, MD, etc) image volume.
        labels: The labels of the connectivity matrix.

    Returns:
        matrix: The connectivity matrix in the form of a square 2D array.
        labels: A 1D array that contains the labels for each row/column.

    """

    def mean_metric(indices):

        # Find all of the voxels intersected by the streamlines.
        mask = np.zeros(metric.shape, dtype=bool)
        for index in indices:
            voxels = streamlines[index].astype(int)
            for voxel in voxels:
                mask[voxel[0], voxel[1], voxel[2]] = True

        return np.mean(metric[mask])

    matrix, labels = compile_connectivity_matrix(mapping, labels, mean_metric, 0.0)

    return matrix, labels


def convert_segmentation(segmentation, affine):
    """Converts a segmentation to a list of labelled vertices

    The zero label is ignored. Nibabel convention is assumed meaning the
    center of the first voxel is (0, 0, 0).

    Args:
        segmentation: The image containing the region label for each voxel.
        affine: The affine transformation from voxels to world space.

    Returns:
        vertices: The vertices corresponding to the center of the voxels.
        labels: The label of each vertex.

    """

    indices = np.nonzero(segmentation)
    labels = segmentation[indices]
    vertices = np.transpose(indices).astype(float)
    vertices = nib.affines.apply_affine(affine, vertices)

    return vertices, labels


def filter_mapping(mapping: defaultdict, streamline_indices: Iterable) -> defaultdict:
    """Filters a mapping to keep a subset of streamlines

    Args:
        mapping: The mapping to filter.
        streamline_indices: The streamline indices to keep in the output
            mapping.

    Returns:
        filtered_mapping: The new mapping with only the requested streamlines.

    """

    indices_set = set(streamline_indices)

    filtered_mapping = defaultdict(lambda: [])
    for key, value in mapping.items():
        new_value = list(set(value) & indices_set)
        if len(new_value) != 0:
            filtered_mapping[key] = new_value

    return filtered_mapping


def compile_weighted_connectivity_matrix(
    mapping: defaultdict, weights: np.ndarray, labels: Optional[Iterable] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compiles a weighted connectivity matrix from a mapping and a vector of
    weights containing one weighting coefficient for each streamline. The
    connectivity between two regions will be the sum of the weights associated
    to the streamlines connecting the two regions.

    Args:
        mapping: A map from labels to streamline ids generated using
            the map_labels function.
        weights: The weights associated to each streamline.
        labels: The labels of the connectivity matrix.

    Returns:
        matrix: The connectivity matrix in the form of a square 2D array.
        labels: A 1D array that contains the labels for each row/column.

    """

    def weighting(indices):
        return np.sum(weights[indices])

    matrix, labels = compile_connectivity_matrix(mapping, labels, weighting, 0.0)

    return matrix, labels


def extract_upper_mapping(
    mapping: defaultdict, offset: Optional[int] = 0
) -> defaultdict:
    """Extract the upper triangular part from a mapping object

    Args:
        mapping: The mapping from which the upper triangular part will be
            extracted.
        offset (optional): Determines the offset from the principal diagonal
            that will be kept. (Default: 0)
    Returns:
         upper: Upper triangular part of the input mapping with given offset
            distance.
    """
    upper = defaultdict(lambda: [])
    for key, value in mapping.items():
        i, j = sorted(list(key))
        if j - i >= offset:
            upper[(i, j)] = value
    return upper


def map_vertices(
    vertices: np.ndarray,
    streamlines: Iterable[np.ndarray],
    vertex_labels: Union[np.ndarray, None] = None,
    distance_upper_bound: float = 4.0,
) -> defaultdict:
    """Returns a mapping from pairs of vertices to streamline ids

    The returned mapping is a defaultdict that uses pairs of vertices as keys
    and a lists of streamline ids as values. It is not symmetric i.e.
    mapping[(0, 1)] is not the same as mapping[(1, 0)]. When no streamlines
    are associated with a given vertex pair, the mapping contains an empty
    list.

    Args:
        vertices: A numpy array with a shape of (N, 3) where N is the number
            of vertices.
        streamlines: The tractogram used to build the connectivity matrix.
        vertex_labels (optional): The labels associated with each vertex.
            The vertex number is used by default.
        distance_upper_bound (optional): The maximum acceptable distance
            between the streamlines end points and the cortex. Streamlines with
            a distance greater than this value are excluded from the mapping.

    """

    if vertex_labels is None:
        vertex_labels = np.arange(len(vertices))

    # Find the neighbors of the streamline end points on the surface.
    vertices_tree = scipy.spatial.cKDTree(vertices)
    streamline_starts = np.array([s[0, :] for s in streamlines])
    start_distances, start_vertices = vertices_tree.query(
        streamline_starts, distance_upper_bound=distance_upper_bound
    )
    streamline_ends = np.array([s[-1, :] for s in streamlines])
    end_distances, end_vertices = vertices_tree.query(
        streamline_ends, distance_upper_bound=distance_upper_bound
    )

    # Prune streamlines that are too far from points.
    to_keep = np.logical_and(
        start_distances < distance_upper_bound, end_distances < distance_upper_bound
    )

    # Add them to the mapping.
    mapping = defaultdict(lambda: [])
    for i, (start, end) in enumerate(zip(start_vertices, end_vertices)):
        if to_keep[i]:
            mapping[(vertex_labels[start], vertex_labels[end])].append(i)

    return mapping


def symmetrize_mapping(mapping: defaultdict) -> defaultdict:
    """Take a mapping and make it symmetric with respect to the labels

    Args:
        mapping: Mapping to make symmetric.
    Returns:
        sym_map: The symmetric version of the input mapping.
    """
    sym_map = defaultdict(lambda: [])
    for key, value in mapping.items():
        sym_map[key].extend(value)
        sym_map[key[::-1]].extend(value)

    return sym_map
