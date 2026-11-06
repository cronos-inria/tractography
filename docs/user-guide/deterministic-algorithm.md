# Deterministic Orientation Algorithm

`tg.algorithm.Deterministic` is a `tg.algorithm.BaseAlgorithm` implementation that iteratively seeks a nearby maximum of a local model on the unit sphere.
It is an orientation-update component, not the registered `tg.Algorithm.DETERMINISTIC` tracking workflow.
For the shared subclass interface, see [Base Algorithm](base-algorithm.md).

## Configure and update orientations

Construct a `Deterministic` instance with a local model:

```python
import numpy as np
import tractography as tg

model = tg.local.RealSymmetricSphericalHarmonics(n_parameters=45)
algorithm = tg.algorithm.Deterministic(model)

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

The default iteration limit is 10 and the tolerance is `1e-5`.
The OpenCL update angle is capped at 0.5 radians by default.
Set these options with `max_iterations`, `tolerance`, and `max_step_size`, respectively.
The NumPy implementation uses `max_iterations` and `tolerance`; `max_step_size` only affects OpenCL.
The inherited `step_size` attribute is used by tracking to advance the streamline and is not used by either orientation update.

## Update rule

The NumPy reference minimizes the negative model value with SciPy's SLSQP optimizer.
It constrains directions to unit length and supplies the analytic gradient from the local model.
The OpenCL update instead builds a tangent frame at the current orientation and uses the model-provided gradient directly.
The model's azimuthal derivative is already divided by $\sin(\theta)$, so it multiplies the unit azimuth tangent directly.
OpenCL estimates the Hessian by finite differences of the gradient at nearby directions.
When the Hessian is negative definite, it uses the Newton-ascent direction. Otherwise it uses a fixed small gradient-ascent step.
It limits the angular update using `max_step_size`, applies the spherical exponential map, and backtracks to accept only a step that increases the model value.
The two implementations are algorithmically distinct, but both seek a nearby local maximum for each corresponding model and input orientation.

The update is performed independently for each matching row of `local_models` and `orientations`.
The NumPy method uses `local_model.evaluate`; the module-level function uses the corresponding OpenCL implementation.
