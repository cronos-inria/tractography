# Base Algorithm

`tg.algorithm.BaseAlgorithm` is the abstract interface for classes that update fiber orientations from local-model data.
It is distinct from `tg.Algorithm`, the enum used to select a complete tractography implementation and its configuration.
For the built-in orientation updaters, see [Transport](transport-algorithm.md), [Diffusion](diffusion-algorithm.md), and [Deterministic](deterministic-algorithm.md).

## Interface

A concrete subclass initializes the base class with a `tg.local.BaseLocalModel` and implements three methods:

- `get_source()` returns OpenCL source for the algorithm-specific orientation update.
- `get_args()` returns the scalar OpenCL arguments, in the order expected by the source.
- `update_orientation(local_models, orientations)` provides a NumPy implementation for the same update.
- `get_rng_seed()` provides a host seed for device-side per-work-item random-state initialization. Stateless algorithms may use the default implementation.

The `local_models` array contains one row of model parameters per orientation and has shape `(n_orientations, local_model.n_parameters)`.
The `orientations` array has shape `(n_orientations, 3)`, with Cartesian directions in each row.
The method returns updated directions with the same shape.

## Implementing a subclass

The OpenCL source returned by `get_source()` is combined with a generated kernel that expects these definitions:

- `ALGORITHM_ARGS` declares the scalar kernel arguments.
- `ALGORITHM_BUILD(a)` constructs the algorithm state named `a` from those arguments.
- `Algorithm` is the state type used by the update function.
- `algorithm_update_orientation(Algorithm, float*, float4, uint2*)` returns the updated `float4` orientation for one local model. The pointer refers to that work item's private RNG state and may be unused by stateless algorithms.

The order and types returned by `get_args()` must match `ALGORITHM_ARGS`.
The generated kernels append a `uint` RNG seed and initialize a private state for each work item before calling the update function.
The update function receives a pointer to one row of local-model parameters.
The associated model source must provide the `ModelValue` type and `local_model_evaluate` function, as described in [Local Models](local-models.md).

The class-level NumPy method and the module-level `tg.algorithm.update_orientation(algorithm, local_models, orientations)` function should implement equivalent updates.
The module-level function executes the OpenCL implementation, validates the local-model array shape, and returns an empty `(0, 3)` float32 array without starting OpenCL when no orientations are supplied.
A working OpenCL runtime is required for non-empty inputs to that function.
