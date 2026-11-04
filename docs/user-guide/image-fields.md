# Image Fields

`tg.field.Image` is an implementation of the field interface described in [Fields](fields.md).
It stores a vector of local parameters at each voxel and evaluates those vectors at world-space points.

## Image field example

`tg.field.Image` accepts a 4D NiBabel spatial image, such as a NIfTI image.
The first three dimensions are spatial, and the fourth dimension contains the parameters at each voxel.
Images with any number of dimensions other than four raise `ValueError`.
The image affine maps voxel coordinates to world coordinates and is used when looking up values.

```python
import nibabel as nib
import numpy as np
import tractography as tg

# Two parameters at each voxel in a small 3D image.
data = np.zeros((4, 4, 4, 2), dtype=np.float32)
data[..., 0] = 1.0
data[..., 1] = 2.0

# The image affine places voxel (1, 1, 1) at world coordinate (11, 21, 31).
affine = np.eye(4)
affine[:3, 3] = (10, 20, 30)
image = nib.Nifti1Image(data, affine)
field = tg.field.Image(image)
```

## Evaluate the field

Input points are mapped from world coordinates to voxel coordinates using the inverse image affine.
The nearest voxel is selected by rounding each coordinate to the nearest integer, with halfway values rounded to even.
Points outside the image bounds return a zero vector.
The result has shape `(n_points, n_parameters)`.

```python
points = np.array(
    [
        [11, 21, 31],  # voxel (1, 1, 1)
        [14, 20, 30],  # outside the image bounds
    ],
    dtype=np.float32,
)

# NumPy implementation for direct evaluation.
reference_result = field.evaluate(points)
print(reference_result)  # [[1. 2.]
                         #  [0. 0.]]

# OpenCL implementation, which requires an available OpenCL runtime.
result = tg.field.evaluate(field, points)
print(result)  # [[1. 2.]
               #  [0. 0.]]
```

`field.evaluate(points)` is the NumPy reference implementation.
`tg.field.evaluate(field, points)` evaluates the field with OpenCL and returns the same values.
Both methods take world-space points with shape `(n_points, 3)` and return a float32 array with shape `(n_points, n_parameters)`.
