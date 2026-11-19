import numpy as np

import tractography as tg


def u0u3_coefficients(epsilon: float = 0.3, n_coefficients: int = 45) -> np.ndarray:
    """Fit SH coefficients to the analytic FOD ``1 + epsilon * u_x * u_z``."""
    directions = tg.core.fibonacci_sphere(5000)
    values = 1.0 + epsilon * directions[:, 0] * directions[:, 2]
    azimuths, colatitudes, _ = tg.core.cart2sph(*directions.T)
    basis, _ = tg.core.ishtmtx(azimuths, colatitudes, n_coefficients)
    return np.asarray(np.linalg.pinv(basis) @ values)


def u0u3_reference(
    step_size: float,
    gamma: float,
    theta_0: float,
    phi_0: float,
    epsilon: float,
    n_steps: int,
) -> np.ndarray:
    """Integrate the analytic ``1 + epsilon * u_x * u_z`` drift by Euler steps."""
    orientation = np.asarray(tg.core.sph2cart(phi_0, theta_0, 1.0))
    orientations = [orientation.copy()]
    identity = np.eye(3)

    for _ in range(n_steps - 1):
        value = 1.0 + epsilon * orientation[0] * orientation[2]
        gradient = np.array([orientation[2], 0.0, orientation[0]])
        tangent = (identity - np.outer(orientation, orientation)) @ gradient
        tangent *= gamma * epsilon / value
        orientation += step_size * tangent
        orientation /= np.linalg.norm(orientation)
        orientations.append(orientation.copy())

    return np.asarray(orientations)
