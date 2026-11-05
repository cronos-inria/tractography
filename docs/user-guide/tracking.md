# Tracking

Tracking turns seed positions and directions into streamlines by repeatedly evaluating a local model and updating each direction.
For most workflows, use the public `tg.tractogram` function, which selects a registered implementation from the configuration.
For the configuration and algorithm options, see [Base Algorithm](base-algorithm.md) and [Transport Orientation Algorithm](transport-algorithm.md).

## Generate a tractogram

The public function accepts an FOD image, a collection of seed objects, and an optional configuration.
If no configuration is provided, it loads the transport algorithm's defaults.
See [Getting Started](../getting_started.md) for a complete example that creates seeds, runs tracking, and saves the result.

```python
import tractography as tg

fod, white_matter, _ = tg.data.sh.cross()
seeds = tg.seeds.from_mask(white_matter.get_fdata(), white_matter.affine, 100)
fod = tg.nifti.multiply(fod, white_matter)

config = tg.configuration.load(tg.Algorithm.TRANSPORT)
config.batch_size = len(seeds)
tractogram = tg.tractogram(fod, seeds, config)
```

The returned NiBabel `Tractogram` contains streamlines in RAS+ millimeter coordinates.
The algorithm configuration controls tracking behavior, including the integration step, streamline stopping criteria, and batching.

## Low-level tracking function

`tractography.tracking.tractogram` is a lower-level OpenCL function for combining a `BaseAlgorithm`, `BaseDomain`, `BaseField`, and seed array directly.
Unlike `tg.tractogram`, it does not load a configuration or resolve a registered algorithm implementation.
Import it explicitly with `from tractography.tracking import tractogram`.

The `seeds` argument has shape `(n_seeds, 6)`. Each row contains an initial position `(x, y, z)` followed by a direction `(dx, dy, dz)`.
Positions, directions, the domain, and the field must use compatible coordinates.
The field's parameter count must match the algorithm's local model parameter count.

The kernel advances each seed using the algorithm's OpenCL update and stops when the point leaves the domain or the field cannot be evaluated.
It includes the initial seed position and saves subsequent points at intervals of `save_at` (default `1.0`).
`maximum_streamline_length` (default `300.0`) sets the per-streamline storage capacity, computed as `floor(maximum_streamline_length / save_at)` points.
Streamlines can terminate earlier when tracking leaves the domain or field.

For example, after constructing compatible `algorithm`, `domain`, and `field` objects, and arranging seed positions and directions in the six-column format:

```python
from tractography.tracking import tractogram as track

result = track(algorithm, domain, field, seeds, save_at=1.0)
```

This lower-level function requires a working OpenCL runtime.