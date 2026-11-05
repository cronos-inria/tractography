# Real Symmetric Spherical Harmonics

`tg.local.RealSymmetricSphericalHarmonics` evaluates a real symmetric spherical-harmonic representation of a fiber orientation distribution (FOD).
It is the 45-coefficient, `lmax=8` model used by the tracking kernels and follows the real spherical-harmonic convention used by MRtrix and DIPY.
For the general interface and supported tracking coefficient layouts, see [Local Models](local-models.md).

## Evaluate a model

Create a model with the number of coefficients in the FOD, then provide those coefficients and an array of Cartesian orientation vectors.
The orientation array has shape `(n_orientations, 3)`, with one `(x, y, z)` direction per row.
Use nonzero vectors. Their lengths do not affect the angular spherical-harmonic evaluation.

```python
import numpy as np
import tractography as tg

# One real symmetric SH coefficient vector with lmax=8.
parameters = np.zeros(45, dtype=np.float32)
parameters[0] = 1.0

phi = np.array([0.0, np.pi / 2], dtype=np.float32)
theta = np.array([np.pi / 2, np.pi / 2], dtype=np.float32)
orientations = np.stack(tg.core.sph2cart(phi, theta, 1.0), axis=-1)
model = tg.local.RealSymmetricSphericalHarmonics(len(parameters))

# NumPy reference implementation.
reference = model.evaluate(parameters, orientations)

# OpenCL implementation, requiring a working OpenCL runtime.
result = tg.local.evaluate(model, parameters, orientations)
print(result.shape)  # (2, 3)
print(np.allclose(result, reference))  # True
```

The returned array has one row per orientation, with columns in this order:

1. The FOD value after a smooth positive softplus transform.
2. The derivative with respect to the polar angle `theta`.
3. The derivative with respect to azimuth `phi`, pre-divided by `sin(theta)`.

The derivative convention in the third column is required by the orientation-coordinate calculations used by tractography.
The NumPy and OpenCL implementations are intended to agree within floating-point tolerance.
