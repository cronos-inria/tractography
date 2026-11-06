# Probabilistic Orientation Algorithm

`tg.algorithm.Probabilistic` is a `tg.algorithm.BaseAlgorithm` implementation that samples a nearby orientation according to the local-model density.
It is an orientation-update component, not the registered `tg.Algorithm.PROBABILISTIC` tracking workflow.
For the shared subclass interface, see [Base Algorithm](base-algorithm.md).

## Configure and update orientations

Construct a `Probabilistic` instance with a local model and optional sampling and random-number parameters:

```python
import numpy as np
import tractography as tg

model = tg.local.RealSymmetricSphericalHarmonics(n_parameters=45)
algorithm = tg.algorithm.Probabilistic(
    model,
    n_samples=64,
    max_angle=np.pi / 12,
    seed=1234,
)

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

`n_samples` defaults to 64 candidates per input direction.
`max_angle` defaults to $\pi/12$ radians (15 degrees) and defines the radius of the spherical cap around the current orientation.
`seed` is optional.
The NumPy implementation uses a NumPy random generator, while OpenCL derives a device-side random state from a seed drawn by the algorithm instance.
A supplied seed makes newly constructed instances reproducible when they make the same sequence of calls on the same backend.
NumPy and OpenCL use different random-number generators, so their samples are not expected to match exactly.

## Sampling rule

For each input orientation, the algorithm draws `n_samples` candidate directions uniformly by area from the spherical cap whose angular radius is `max_angle`.
It evaluates the local model at each candidate and selects one with probability proportional to the candidate's nonnegative model value.
Non-finite and negative values contribute zero weight.
If all candidate weights are zero, the normalized input orientation is returned.
The result is a finite-candidate approximation to sampling from the local orientation density restricted to that cap.

The `update_orientation` methods take one local-model parameter row for each orientation, with shapes `(n_orientations, model.n_parameters)` and `(n_orientations, 3)`, respectively.
The output contains one normalized Cartesian direction per input row.
The inherited `step_size` parameter defaults to `0.25` and is used by tracking to advance the streamline, not by orientation sampling.
