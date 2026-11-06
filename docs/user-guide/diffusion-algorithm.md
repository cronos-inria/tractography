# Diffusion Orientation Algorithm

`tg.algorithm.Diffusion` is a `tg.algorithm.BaseAlgorithm` implementation that updates orientations using local-model drift and Gaussian noise in the tangent plane of the unit sphere.
It is an orientation-update component that can be used with `tg.tracking.tractogram`, not the registered `tg.Algorithm.DIFFUSION` tractography workflow.
For the shared subclass interface, see [Base Algorithm](base-algorithm.md).

## Configure and update orientations

Construct a `Diffusion` instance with a local model and optional integration and random-number parameters:

```python
import numpy as np
import tractography as tg

model = tg.local.RealSymmetricSphericalHarmonics(n_parameters=45)
algorithm = tg.algorithm.Diffusion(
    model,
    step_size=0.001,
    gamma=1.0,
    noise_variance=0.1,
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

`step_size`, `gamma`, and `noise_variance` default to `0.001`, `1.0`, and `0.1`, respectively.
`seed` is optional.
The NumPy implementation uses a NumPy random generator, while the OpenCL implementation derives one MWC64X state per work item on the device from a seed drawn by the algorithm instance.
A supplied seed makes newly constructed instances reproducible when they make the same sequence of calls on the same backend.
NumPy and OpenCL use different random-number generators, so their sampled updates are not expected to match exactly.

## Update rule

At each orientation, the local model supplies a value $f$ and tangent-frame derivatives $d_\theta$ and $d_\phi$.
The drift is

$$
\mathbf{b} = \frac{d_\theta\mathbf{e}_\theta + d_\phi\mathbf{e}_\phi}{\max(f, 10^{-6})}.
$$

Two independent standard normal samples provide the tangent-plane noise
$\boldsymbol{\eta} = \xi_\theta\mathbf{e}_\theta + \xi_\phi\mathbf{e}_\phi$.
The update applies the spherical exponential map to the tangent increment

$$
\mathbf{x} = \gamma\,\Delta t\,\mathbf{b}
+ \sqrt{\sigma^2\gamma\,\Delta t}\,\boldsymbol{\eta},
$$

where $\Delta t$ is `step_size` and $\sigma^2$ is `noise_variance`.
When `noise_variance` is zero or negative, the implementation omits the random draws and uses a zero noise term.

The `update_orientation` methods expect `local_models` with shape `(n_orientations, model.n_parameters)` and `orientations` with shape `(n_orientations, 3)`.
For tracking, each OpenCL work item initializes a private random state for its streamline and reuses it through that streamline's updates.
