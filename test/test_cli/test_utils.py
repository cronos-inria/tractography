import numpy as np
import pytest
from nibabel.nifti1 import Nifti1Image

import tractography as tg
from tractography.cli.utils import make_tracking_algorithm


@pytest.mark.parametrize(
    ("algorithm", "algorithm_type"),
    [
        (tg.Algorithm.TRANSPORT, tg.algorithm.Transport),
        (tg.Algorithm.DETERMINISTIC, tg.algorithm.Deterministic),
        (tg.Algorithm.DIFFUSION, tg.algorithm.Diffusion),
        (tg.Algorithm.PROBABILISTIC, tg.algorithm.Probabilistic),
    ],
)
def test_make_tracking_algorithm_infers_model_and_constructs_type(
    algorithm, algorithm_type
):
    field = Nifti1Image(np.zeros((1, 1, 1, 6), dtype=np.float32), np.eye(4))

    result = make_tracking_algorithm(field, algorithm)

    assert isinstance(result, algorithm_type)
    assert isinstance(result.local_model, tg.local.DiffusionTensorImaging)
