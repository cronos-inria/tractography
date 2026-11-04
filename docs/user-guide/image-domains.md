# Image Domains

`tg.domain.Image` is one implementation of the general domain interface described in [Domains](domains.md).

## Image domain example

`tg.domain.Image` accepts a 3D NiBabel spatial image, such as a NIfTI image.
Every nonzero voxel is considered inside the domain, and the image affine is used to relate voxel coordinates to world coordinates.
Images with any number of dimensions other than three raise `ValueError`.

```python
import nibabel as nib
import numpy as np
import tractography as tg

# A small mask with a translated voxel-to-world affine.
mask_data = np.zeros((4, 4, 4), dtype=np.uint8)
mask_data[1:3, 1:3, 1:3] = 1
affine = np.eye(4)
affine[:3, 3] = (10, 20, 30)
image = nib.Nifti1Image(mask_data, affine)

domain = tg.domain.Image(image)
```

## Test points for membership

Every nonzero voxel is inside the domain.
Points are mapped to the nearest voxel using the inverse image affine, and points outside the image bounds or in a zero-valued voxel are outside the domain.

```python
points = np.array(
    [
        [11, 21, 31],  # voxel (1, 1, 1), which is nonzero
        [10, 20, 30],  # voxel (0, 0, 0), which is zero
        [14, 20, 30],  # outside the image bounds
    ],
    dtype=np.float32,
)

# NumPy reference implementation, useful for direct checks.
reference_result = domain.contains(points)
print(reference_result)  # [ True False False]

# OpenCL implementation for checking points on the available compute device.
result = tg.domain.contains(domain, points)
print(result)  # [ True False False]
```

`domain.contains(points)` is the NumPy reference implementation.
`tg.domain.contains(domain, points)` evaluates the same membership test with OpenCL and requires a working OpenCL runtime.
Both methods take world-space points with shape `(n_points, 3)` and return one Boolean result per point.