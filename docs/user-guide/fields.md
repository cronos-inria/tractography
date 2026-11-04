# Fields

Fields provide local model parameters at world-space points.
`tg.field.BaseField` defines the interface that field implementations provide.
The built-in `tg.field.Image` implementation reads parameter vectors from a 4D image.

## Field interface

A field implementation provides `get_source()` and `get_args(context)` to make its data available to an OpenCL kernel, and `evaluate(points)` as a direct evaluation method.
The OpenCL source must define the `FIELD_ARGS` and `FIELD_BUILD` macros, the `Field` struct, and the `field_evaluate` function.
The point-query interface takes world-space coordinates with shape `(n_points, 3)` and returns one row of field values per point.

For the built-in image implementation and a complete example, see [Image Fields](image-fields.md).
