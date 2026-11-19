# Tracking

Tracking turns seed positions and directions into streamlines by repeatedly evaluating a local model and updating each direction.
For most workflows, use the public `tg.tractogram` function, which adapts image inputs and a `BaseAlgorithm` instance to the shared low-level tracker in `tractography.tracking`.
For algorithm classes and update parameters, see [Base Algorithm](base-algorithm.md) and [Transport Orientation Algorithm](transport-algorithm.md).

## Generate a tractogram

The public function accepts a field or FOD image, an optional domain, an optional algorithm, and a seed list or seed count.
If the domain is omitted, nonzero image-field voxels define it.
If the algorithm is omitted, transport is used.
See [Getting Started](../getting_started.md) for a complete example that creates seeds, runs tracking, and saves the result.

```python
import tractography as tg

fod, white_matter, _ = tg.data.sh.cross()
seeds = tg.seeds.from_mask(white_matter, 100)
model = tg.local.RealSymmetricSphericalHarmonics(fod.shape[-1])
algorithm = tg.algorithm.Transport(model, step_size=0.001, gamma=1.0)
tractogram = tg.tractogram(
	fod,
	seeds,
	white_matter,
	algorithm,
	save_at=0.25,
	maximum_streamline_length=300.0,
	batch_size=len(seeds),
)
```

The returned NiBabel `Tractogram` contains streamlines in RAS+ millimeter coordinates.
The `BaseAlgorithm` instance controls orientation updates and the integration step.
`save_at`, `maximum_streamline_length`, `minimum_streamline_length`, and `batch_size` are explicit function options.
`batch_size` bounds each low-level tracking call, while the returned tractogram is accumulated in memory.
The optional `domain` argument supplies the tracking domain; otherwise, the nonzero FOD voxels define it.
The Python function returns an in-memory tractogram.
The CLI can write `.trx` output incrementally in configured seed batches, without collecting the entire tractogram in RAM.

## Low-level tracking function

`tractography.tracking.tractogram` is a lower-level OpenCL function for combining a `BaseAlgorithm`, `BaseDomain`, `BaseField`, and seed array directly.
Unlike `tg.tractogram`, it does not adapt image/configuration inputs or convert the public `Seed` objects.
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