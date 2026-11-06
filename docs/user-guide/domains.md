# Domains

Domains define where tractography is allowed to proceed.
`tg.domain.BaseDomain` describes the interface that domain implementations provide.
`tg.domain.Image` is one built-in implementation, using the nonzero voxels of a 3D image, and other implementations can represent different kinds of regions.

## Domain interface

A domain implementation provides `get_source()` and `get_args(context)` to make its membership test available to an OpenCL kernel, and `contains(points)` to check points against the domain.
The module-level `tg.domain.contains(domain, points)` runs the domain's OpenCL membership test and can be used with any implementation that provides the required OpenCL source and arguments.
The point-query interface takes world-space coordinates with shape `(n_points, 3)` and returns one Boolean result per input point.

Domains also provide a boundary mesh to `BaseDomain`. The shared boundary sampler chooses triangles proportional to surface area, samples uniformly within each selected triangle, and points seed orientations inward. Boundary triangle winding must therefore produce outward-facing normals.
Use `tg.domain.seed(domain, n_seeds, rng_seed=None)` to generate `Seed` objects from that boundary.
Sampling runs on OpenCL and requires a working OpenCL runtime.
The optional `rng_seed` makes the sample reproducible, and a count of zero returns an empty list.

For an example using marching cubes to construct the image boundary, see [Image Domains](image-domains.md).