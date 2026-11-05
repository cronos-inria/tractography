# Transport Orientation Algorithm

`tg.algorithm.Transport` is the built-in `tg.algorithm.BaseAlgorithm` implementation for updating orientations using the value and angular derivatives from a local model.
It is an orientation-update component, not the `tg.Algorithm.TRANSPORT` enum value or the complete tractography workflow selected by that value.
For the shared subclass interface, see [Base Algorithm](base-algorithm.md).

## Configure and update orientations

Construct a `Transport` instance with a local model and optional update parameters:

```python
import numpy as np
import tractography as tg

model = tg.local.RealSymmetricSphericalHarmonics(n_parameters=45)
algorithm = tg.algorithm.Transport(model, step_size=0.01, gamma=1.0)

# One row of 45 model coefficients and one Cartesian orientation.
local_models = np.zeros((1, model.n_parameters), dtype=np.float32)
local_models[0, 0] = 1.0
orientations = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

# NumPy implementation.
updated = algorithm.update_orientation(local_models, orientations)

# OpenCL implementation. Requires a working OpenCL runtime.
updated_opencl = tg.algorithm.update_orientation(
    algorithm, local_models, orientations
)
```

`step_size` scales the orientation step and defaults to `0.01`.
`gamma` scales the drift and defaults to `1.0`.
Both values are used directly as multipliers, and the implementation does not validate their ranges.

## Update rule

The local model provides a value $f$ and angular derivatives for each input direction.
`Transport` combines these derivatives in a tangent basis at the direction, divides by `max(f, 10^{-6})`, and scales the resulting drift by `step_size * gamma`.
The updated direction is computed with the spherical exponential map, which moves along the sphere while preserving unit length for unit-length inputs.

The `update_orientation` method expects `local_models` with shape `(n_orientations, model.n_parameters)` and `orientations` with shape `(n_orientations, 3)`.
It returns one updated Cartesian direction per input row.
The NumPy method uses `model.evaluate`; the module-level function uses OpenCL and provides a corresponding device implementation.
