import numpy as np

import tractography as tg


def test_local_model_evaluate():
    """Test the basic evaluation of a local model."""

    # Some very basic image domain.
    parameters = np.ones((45,))

    n_angles = 100
    phi = np.linspace(0, 2 * np.pi, n_angles, dtype=np.float32)
    theta = np.linspace(0, np.pi, n_angles, dtype=np.float32)
    orientations = np.stack(tg.core.sph2cart(phi, theta, 1.0), axis=-1)

    model = tg.local.RealSymmetricSphericalHarmonics(len(parameters))
    reference = model.evaluate(parameters, orientations)
    result = tg.local.evaluate(model, parameters, orientations)

    # The OpenCL harmonic formulas use float32 constants and arithmetic,
    # while the NumPy reference uses SciPy's float64 spherical harmonics.
    assert np.allclose(result, reference, rtol=1e-4, atol=1e-4)