import numpy as np
from nibabel.nifti1 import Nifti1Image

import tractography as tg


def _quadratic_fod_coefficients(baseline, epsilon, n_coefficients=15):
    """Fit coefficients for baseline + epsilon * x * z."""
    directions = tg.core.fibonacci_sphere(1000)
    phi, theta, _ = tg.core.cart2sph(*directions.T)
    harmonics, _ = tg.core.ishtmtx(phi, theta, n_coefficients)
    values = baseline + epsilon * directions[:, 0] * directions[:, 2]
    return np.linalg.lstsq(harmonics, values, rcond=None)[0].astype(np.float32)


def _expected_quadratic_update(orientation, baseline, epsilon, step_size, gamma):
    """Compute one spherical exponential step from the analytic FOD gradient."""
    raw_value = baseline + epsilon * orientation[0] * orientation[2]
    value = np.logaddexp(0.0, 100.0 * raw_value) / 100.0
    softmax_derivative = 1.0 / (1.0 + np.exp(-100.0 * raw_value))
    gradient = epsilon * np.array([orientation[2], 0.0, orientation[0]])
    tangent_gradient = gradient - orientation * np.dot(orientation, gradient)
    tangent = step_size * gamma * softmax_derivative / value * tangent_gradient
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm == 0:
        return orientation.copy()
    return (
        np.cos(tangent_norm) * orientation
        + np.sin(tangent_norm) / tangent_norm * tangent
    )


def test_transport_update_orientation():
    """Test basic orientation update."""

    n_coefficients = 45
    data = np.zeros((3, 3, 3, n_coefficients))
    data[:, :, :, 0] = 1
    field = tg.field.Image(Nifti1Image(data, np.eye(4)))
    local_model = tg.local.RealSymmetricSphericalHarmonics(n_coefficients)
    algorithm = tg.algorithm.Transport(local_model)
    assert isinstance(algorithm, tg.algorithm.BaseAlgorithm)

    points = np.zeros((1, 3))
    orientations = np.zeros((1, 3))
    orientations[:, -1] = 1
    local_models = field.evaluate(points)
    reference_orientations = algorithm.update_orientation(local_models, orientations)
    updated_orientations = tg.algorithm.update_orientation(algorithm, local_models, orientations)

    assert np.allclose(reference_orientations, updated_orientations)


def test_transport_update_orientation_matches_analytic_step():
    """Check a nonzero update against the analytic gradient and sphere step."""
    baseline = 0.1
    epsilon = 0.3
    step_size = 0.04
    gamma = 1.7
    parameters = _quadratic_fod_coefficients(baseline, epsilon)[None, :]
    orientation = np.array([tg.core.sph2cart(0.5, 0.8, 1.0)]).reshape(1, 3)
    model = tg.local.RealSymmetricSphericalHarmonics(parameters.shape[1])
    algorithm = tg.algorithm.Transport(model, step_size=step_size, gamma=gamma)

    numpy_result = algorithm.update_orientation(parameters, orientation)
    opencl_result = tg.algorithm.update_orientation(algorithm, parameters, orientation)
    expected = _expected_quadratic_update(
        orientation[0], baseline, epsilon, step_size, gamma
    )

    np.testing.assert_allclose(numpy_result[0], expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(opencl_result[0], expected, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(np.linalg.norm(numpy_result, axis=1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(opencl_result, axis=1), 1.0, atol=1e-5)


def test_transport_update_orientation_batches_distinct_models():
    """Keep per-orientation model coefficients aligned with their inputs."""
    fod_parameters = [(0.1, 0.2), (0.25, 0.4), (0.05, 0.55)]
    local_models = np.stack(
        [_quadratic_fod_coefficients(base, epsilon) for base, epsilon in fod_parameters]
    )
    phi = np.array([0.3, 1.2, 2.4])
    theta = np.array([0.6, 1.1, 2.0])
    orientations = np.stack(tg.core.sph2cart(phi, theta, 1.0), axis=-1)
    step_size = 0.025
    gamma = 1.3
    model = tg.local.RealSymmetricSphericalHarmonics(local_models.shape[1])
    algorithm = tg.algorithm.Transport(model, step_size=step_size, gamma=gamma)

    numpy_result = algorithm.update_orientation(local_models, orientations)
    opencl_result = tg.algorithm.update_orientation(algorithm, local_models, orientations)
    expected = np.stack(
        [
            _expected_quadratic_update(
                orientation, base, epsilon, step_size, gamma
            )
            for orientation, (base, epsilon) in zip(orientations, fod_parameters)
        ]
    )

    np.testing.assert_allclose(numpy_result, expected, rtol=3e-4, atol=3e-4)
    np.testing.assert_allclose(opencl_result, expected, rtol=3e-4, atol=3e-4)


def test_transport_update_orientation_empty_input():
    """Return the expected empty result without starting OpenCL."""
    model = tg.local.RealSymmetricSphericalHarmonics(15)
    algorithm = tg.algorithm.Transport(model)
    local_models = np.empty((0, model.n_parameters), dtype=np.float32)
    orientations = np.empty((0, 3), dtype=np.float32)

    result = tg.algorithm.update_orientation(algorithm, local_models, orientations)

    assert result.shape == (0, 3)
    assert result.dtype == np.float32


def test_exps2_zero_and_tangent_steps():
    """Handle a zero tangent and follow the spherical exponential otherwise."""
    orientations = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    tangents = np.array([[0.0, 0.0, 0.0], [0.0, 0.3, 0.0]])

    result = tg.algorithm.exps2(orientations, tangents, 1.0)

    np.testing.assert_allclose(result[0], orientations[0])
    np.testing.assert_allclose(
        result[1], [np.cos(0.3), np.sin(0.3), 0.0], atol=1e-7
    )
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-7)


def test_transport_update_orientation_at_and_near_poles():
    """Keep updates finite and normalized at both coordinate singularities."""
    baseline = 0.1
    epsilon = 0.3
    parameters = _quadratic_fod_coefficients(baseline, epsilon)[None, :]
    phi = np.array([0.0, 1.1, 0.7, 2.0])
    theta = np.array([0.0, np.pi, 1e-4, np.pi - 1e-4])
    orientations = np.stack(tg.core.sph2cart(phi, theta, 1.0), axis=-1)
    model = tg.local.RealSymmetricSphericalHarmonics(parameters.shape[1])
    algorithm = tg.algorithm.Transport(model, step_size=0.01)

    result = algorithm.update_orientation(
        np.repeat(parameters, len(orientations), axis=0), orientations
    )

    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-6)
    assert np.all(np.linalg.norm(result - orientations, axis=1) > 0.0)


def test_deterministic_update_orientation_finds_local_maximum():
    """Use Newton ascent to find the nearby maximum of an SH model."""
    parameters = _quadratic_fod_coefficients(0.1, 0.3)[None, :]
    orientation = np.array([[0.2, 0.0, np.sqrt(1.0 - 0.2**2)]], dtype=np.float32)
    model = tg.local.RealSymmetricSphericalHarmonics(parameters.shape[1])
    algorithm = tg.algorithm.Deterministic(model)

    numpy_result = algorithm.update_orientation(parameters, orientation)
    opencl_result = tg.algorithm.update_orientation(algorithm, parameters, orientation)
    initial_value = model.evaluate(parameters, orientation)[0, 0]
    maximum_value = model.evaluate(parameters, numpy_result)[0, 0]

    assert isinstance(algorithm, tg.algorithm.BaseAlgorithm)
    assert maximum_value > initial_value
    np.testing.assert_allclose(numpy_result[0], [1 / np.sqrt(2), 0.0, 1 / np.sqrt(2)], atol=2e-3)
    np.testing.assert_allclose(opencl_result, numpy_result, atol=2e-3)
    np.testing.assert_allclose(np.linalg.norm(numpy_result, axis=1), 1.0, atol=1e-6)


def test_deterministic_update_orientation_empty_input():
    """Return an empty result without starting OpenCL."""
    model = tg.local.RealSymmetricSphericalHarmonics(15)
    algorithm = tg.algorithm.Deterministic(model)
    local_models = np.empty((0, model.n_parameters), dtype=np.float32)
    orientations = np.empty((0, 3), dtype=np.float32)

    result = tg.algorithm.update_orientation(algorithm, local_models, orientations)

    assert result.shape == (0, 3)
    assert result.dtype == np.float32


def test_diffusion_update_orientation_is_seeded_and_normalized():
    """Initialize device RNG states reproducibly for each orientation."""
    parameters = np.zeros((3, 15), dtype=np.float32)
    parameters[:, 0] = 1.0
    orientations = np.array(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    model = tg.local.RealSymmetricSphericalHarmonics(15)
    first = tg.algorithm.Diffusion(model, noise_variance=1.0, seed=1234)
    second = tg.algorithm.Diffusion(model, noise_variance=1.0, seed=1234)

    first_result = tg.algorithm.update_orientation(first, parameters, orientations)
    second_result = tg.algorithm.update_orientation(second, parameters, orientations)

    np.testing.assert_allclose(first_result, second_result)
    np.testing.assert_allclose(np.linalg.norm(first_result, axis=1), 1.0, atol=1e-5)
    assert np.any(np.linalg.norm(first_result - orientations, axis=1) > 0.0)