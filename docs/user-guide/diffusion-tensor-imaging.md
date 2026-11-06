# Diffusion Tensor Imaging

`tg.local.DiffusionTensorImaging` evaluates a symmetric diffusion tensor along Cartesian orientation vectors.
For the general local-model interface and supported coefficient layouts, see [Local Models](local-models.md).

## Tensor coefficient layout

Each parameter row contains the six independent entries of a symmetric $3\times3$ tensor in this order:

$$
(D_{xx}, D_{yy}, D_{zz}, D_{xy}, D_{xz}, D_{yz}).
$$

The tensor is reconstructed as

$$
D = \begin{bmatrix}
D_{xx} & D_{xy} & D_{xz} \\
D_{xy} & D_{yy} & D_{yz} \\
D_{xz} & D_{yz} & D_{zz}
\end{bmatrix},
$$

and the model value for orientation $u$ is the quadratic form $u^\mathsf{T} D u$.
The model does not normalize orientations, so provide unit vectors when interpreting this as directional diffusivity.

## Evaluate a tensor model

The parameter array has shape `(n_orientations, 6)`, with one tensor per orientation row.
The orientation array has shape `(n_orientations, 3)`.
The result has shape `(n_orientations, 3)`, with columns for the tensor value, its derivative along the polar-angle tangent, and its derivative along the unit azimuthal tangent.
The final derivative corresponds to the azimuth derivative pre-divided by `sin(theta)`, as required by the tracking algorithms.

```python
import numpy as np
import tractography as tg

# One diagonal tensor: diag(2.0, 1.0, 0.5).
parameters = np.array([[2.0, 1.0, 0.5, 0.0, 0.0, 0.0]], dtype=np.float32)
orientations = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
model = tg.local.DiffusionTensorImaging()

# NumPy reference implementation.
reference = model.evaluate(parameters, orientations)
print(reference)  # [[2. 0. 0.]]

# OpenCL implementation, requiring a working OpenCL runtime.
result = tg.local.evaluate(model, parameters, orientations)
print(np.allclose(result, reference))  # True
```

The model returns the raw quadratic-form value and angular derivatives without a positivity transform.
For tracking use, the six tensor coefficients can be stored in a field whose last dimension is six.
