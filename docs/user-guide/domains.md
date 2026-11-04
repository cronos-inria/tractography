# Domains

Domains define where tractography is allowed to proceed.
`tg.domain.BaseDomain` describes the interface that domain implementations provide.
`tg.domain.Image` is one built-in implementation, using the nonzero voxels of a 3D image, and other implementations can represent different kinds of regions.

## Domain interface

A domain implementation provides `get_source()` and `get_args(context)` to make its membership test available to an OpenCL kernel, and `contains(points)` to check points against the domain.
The module-level `tg.domain.contains(domain, points)` runs the domain's OpenCL membership test and can be used with any implementation that provides the required OpenCL source and arguments.
The point-query interface takes world-space coordinates with shape `(n_points, 3)` and returns one Boolean result per input point.

For a complete example, see [Image Domains](image-domains.md), which demonstrates both point-query methods with `tg.domain.Image`.