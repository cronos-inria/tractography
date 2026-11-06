# Local Models

A local model maps the coefficients at a location to a value for a given fiber orientation.
Tracking uses that value and its orientation derivatives to choose and update streamline directions.
The model coefficients are typically stored in a [field](fields.md), such as the 4D image described in [Image Fields](image-fields.md).

## Models used during tracking

The tracking kernels currently recognize two coefficient layouts, selected from the last dimension of the FOD data:

- Six coefficients represent a diffusion tensor, ordered as `Dxx`, `Dyy`, `Dzz`, `Dxy`, `Dxz`, `Dyz`.
  The directional value is the quadratic form `uᵀ D u`, where `u` is a unit orientation vector.
- Forty-five coefficients represent a real symmetric spherical-harmonic FOD with maximum even degree `lmax=8`.
  The coefficients are evaluated against the real spherical-harmonic basis and transformed to a positive value.

Other coefficient counts are not currently supported by tracking and raise `ValueError` during model selection.

## Evaluation interface

`tg.local.BaseLocalModel` describes a local model that can be evaluated at Cartesian orientations.
The built-in `tg.local.RealSymmetricSphericalHarmonics` implementation is documented in [Real Symmetric Spherical Harmonics](local-spherical-harmonics.md).
The built-in `tg.local.DiffusionTensorImaging` implementation is documented in [Diffusion Tensor Imaging](diffusion-tensor-imaging.md).

The module-level `tg.local.evaluate(model, parameters, orientations)` evaluates a model with OpenCL.
A model also provides `model.evaluate(parameters, orientations)`, which is the NumPy reference implementation for the built-in spherical-harmonic model.
`orientations` is an array of shape `(n_orientations, 3)`, with one Cartesian `(x, y, z)` direction per row.
For spherical-harmonic evaluation, each direction should be a nonzero vector. Its azimuth and polar angle are derived from its Cartesian coordinates.
`parameters` is the model's one-dimensional coefficient array.

Both evaluation methods return a float32 array of shape `(n_orientations, 3)`.
Its columns are the model value, the derivative with respect to the polar angle `theta`, and the derivative with respect to azimuth `phi` divided by `sin(theta)`.
For an empty orientation array, the result has shape `(0, 3)`.
The OpenCL method requires an available OpenCL runtime.

A custom `BaseLocalModel` implementation supplies OpenCL source defining `local_model_evaluate(model, orientation)`, which receives a `float4` Cartesian orientation and returns a `ModelValue` containing the value and angular derivatives.
The `phi` derivative must be pre-divided by `sin(theta)` to match the tracking kernels' orientation-coordinate convention.
