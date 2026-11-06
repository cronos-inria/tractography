import numpy as np

import tractography as tg


def test_local_model_evaluate():
    """Test the basic evaluation of a local model."""

    # Some very basic image domain.
    n_angles = 100
    parameters = np.ones((n_angles, 45))
    phi = np.linspace(0, 2 * np.pi, n_angles, dtype=np.float32)
    theta = np.linspace(0, np.pi, n_angles, dtype=np.float32)
    orientations = np.stack(tg.core.sph2cart(phi, theta, 1.0), axis=-1)

    model = tg.local.RealSymmetricSphericalHarmonics(parameters.shape[1])
    reference = model.evaluate(parameters, orientations)
    result = tg.local.evaluate(model, parameters, orientations)

    # The OpenCL harmonic formulas use float32 constants and arithmetic,
    # while the NumPy reference uses SciPy's float64 spherical harmonics.
    assert np.allclose(result, reference, rtol=1e-4, atol=1e-4)


def test_local_model_evaluate_per_orientation_parameters():
    """Evaluate each orientation with its corresponding coefficient row."""
    n_orientations = 12
    n_parameters = 45
    phi = np.linspace(0, 2 * np.pi, n_orientations, dtype=np.float32)
    theta = np.linspace(0, np.pi, n_orientations, dtype=np.float32)
    orientations = np.stack(tg.core.sph2cart(phi, theta, 1.0), axis=-1)
    parameters = np.random.default_rng(42).normal(
        size=(n_orientations, n_parameters)
    )
    model = tg.local.RealSymmetricSphericalHarmonics(n_parameters)

    reference = np.stack(
        [
            model.evaluate(parameters[i : i + 1], orientations[i : i + 1])[0]
            for i in range(n_orientations)
        ]
    )
    numpy_result = model.evaluate(parameters, orientations)
    opencl_result = tg.local.evaluate(model, parameters, orientations)

    assert np.allclose(numpy_result, reference, rtol=1e-6, atol=1e-6)
    assert np.allclose(opencl_result, reference, rtol=1e-4, atol=1e-4)


def test_diffusion_tensor_model_evaluate_and_opencl_match():
    """Evaluate DTI coefficients in diagonal and off-diagonal terms."""
    parameters = np.array(
        [
            [2.0, 1.5, 0.7, 0.3, -0.4, 0.2],
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    phi = np.array([0.7, 1.2], dtype=np.float32)
    theta = np.array([1.1, 0.8], dtype=np.float32)
    orientations = np.stack(tg.core.sph2cart(phi, theta, 1.0), axis=-1)
    model = tg.local.DiffusionTensorImaging()

    expected = np.empty((2, 3), dtype=np.float64)
    for i, (coefficients, orientation) in enumerate(zip(parameters, orientations)):
        tensor = np.array(
            [
                [coefficients[0], coefficients[3], coefficients[4]],
                [coefficients[3], coefficients[1], coefficients[5]],
                [coefficients[4], coefficients[5], coefficients[2]],
            ]
        )
        phi_i, theta_i, _ = tg.core.cart2sph(*orientation)
        e_theta = np.array(
            [
                np.cos(theta_i) * np.cos(phi_i),
                np.cos(theta_i) * np.sin(phi_i),
                -np.sin(theta_i),
            ]
        )
        e_phi = np.array([-np.sin(phi_i), np.cos(phi_i), 0.0])
        tensor_dot_u = tensor @ orientation
        expected[i] = [
            orientation @ tensor_dot_u,
            2.0 * tensor_dot_u @ e_theta,
            2.0 * tensor_dot_u @ e_phi,
        ]

    numpy_result = model.evaluate(parameters, orientations)
    opencl_result = tg.local.evaluate(model, parameters, orientations)

    assert model.n_parameters == 6
    np.testing.assert_allclose(numpy_result, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(opencl_result, expected, rtol=1e-5, atol=1e-5)


def test_diffusion_tensor_model_empty_evaluation():
    """Return the standard empty local-model result shape."""
    model = tg.local.DiffusionTensorImaging()

    result = model.evaluate(
        np.empty((0, model.n_parameters)), np.empty((0, 3))
    )

    assert result.shape == (0, 3)
    assert result.dtype == np.float32