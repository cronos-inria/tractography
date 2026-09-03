#include <boost/multiprecision/gmp.hpp>

#include <CGAL/Bbox_3.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Labeled_mesh_domain_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>
#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/make_mesh_3.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace py = pybind11;
namespace parameters = CGAL::parameters;

using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point = Kernel::Point_3;
using MeshDomain = CGAL::Labeled_mesh_domain_3<Kernel>;
using ConcurrencyTag = CGAL::Sequential_tag;
using Triangulation = CGAL::Mesh_triangulation_3<
    MeshDomain,
    CGAL::Default,
    ConcurrencyTag>::type;
using C3t3 = CGAL::Mesh_complex_3_in_triangulation_3<Triangulation>;
using MeshCriteria = CGAL::Mesh_criteria_3<Triangulation>;

struct MaskView {
    const std::uint8_t* data;
    std::array<std::size_t, 3> shape;
    std::array<double, 3> spacing;

    int operator()(const Point& point) const {
        const std::array<double, 3> coordinates = {
            CGAL::to_double(point.x()),
            CGAL::to_double(point.y()),
            CGAL::to_double(point.z()),
        };
        std::array<std::size_t, 3> indices;

        for (std::size_t axis = 0; axis < 3; ++axis) {
            const double lower = -0.5 * spacing[axis];
            const double upper =
                (static_cast<double>(shape[axis]) - 0.5) * spacing[axis];

            // The support is closed at its lower face and open at its upper
			// face.
            if (coordinates[axis] < lower || coordinates[axis] >= upper) {
                return 0;
            }

            const double index =
                std::floor(coordinates[axis] / spacing[axis] + 0.5);
            if (index < 0.0 || index >= static_cast<double>(shape[axis])) {
                return 0;
            }
            indices[axis] = static_cast<std::size_t>(index);
        }

        const std::size_t offset =
            (indices[0] * shape[1] + indices[1]) * shape[2] + indices[2];
        return data[offset] == 0 ? 0 : 1;
    }
};

struct Input {
    py::array_t<std::uint8_t, py::array::c_style> mask;
    MaskView view;
    CGAL::Bbox_3 bounds;
};

void validate_positive(double value, const char* name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw py::value_error(std::string(name) + " must be positive and finite.");
    }
}

Input prepare_input(
    py::array_t<std::uint8_t, py::array::c_style> mask,
    double spacing_x,
    double spacing_y,
    double spacing_z) {
    validate_positive(spacing_x, "spacing_x");
    validate_positive(spacing_y, "spacing_y");
    validate_positive(spacing_z, "spacing_z");

    const py::buffer_info buffer = mask.request();
    if (buffer.ndim != 3) {
        throw py::value_error("mask must be a three-dimensional array.");
    }

    std::array<std::size_t, 3> shape;
    for (std::size_t axis = 0; axis < 3; ++axis) {
        if (buffer.shape[axis] <= 0) {
            throw py::value_error("mask dimensions must be non-empty.");
        }
        shape[axis] = static_cast<std::size_t>(buffer.shape[axis]);
    }

    const std::array<double, 3> spacing = {
        spacing_x,
        spacing_y,
        spacing_z,
    };
    MaskView view{
        static_cast<const std::uint8_t*>(buffer.ptr),
        shape,
        spacing,
    };
    const CGAL::Bbox_3 bounds(
        -0.5 * spacing_x,
        -0.5 * spacing_y,
        -0.5 * spacing_z,
        (static_cast<double>(shape[0]) - 0.5) * spacing_x,
        (static_cast<double>(shape[1]) - 0.5) * spacing_y,
        (static_cast<double>(shape[2]) - 0.5) * spacing_z);
    return Input{std::move(mask), view, bounds};
}

py::tuple extract_mesh(const C3t3& mesh) {
    using VertexHandle = Triangulation::Vertex_handle;

    const auto begin = mesh.cells_in_complex_begin();
    const auto end = mesh.cells_in_complex_end();
    if (begin == end) {
        throw std::runtime_error("CGAL generated an empty tetrahedral mesh.");
    }

    std::map<VertexHandle, std::size_t> vertex_indices;
    std::size_t tetrahedron_count = 0;
    for (auto cell = begin; cell != end; ++cell) {
        ++tetrahedron_count;
        for (int local_index = 0; local_index < 4; ++local_index) {
            const VertexHandle vertex = cell->vertex(local_index);
            vertex_indices.emplace(vertex, vertex_indices.size());
        }
    }

    if (vertex_indices.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("The mesh has too many vertices for uint32 indices.");
    }
    if (vertex_indices.size() >
            static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max()) ||
        tetrahedron_count >
            static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max())) {
        throw std::overflow_error("The mesh is too large for a NumPy array.");
    }

    py::array_t<float> vertices({
        static_cast<py::ssize_t>(vertex_indices.size()),
        py::ssize_t{3},
    });
    py::array_t<std::uint32_t> tetrahedra({
        static_cast<py::ssize_t>(tetrahedron_count),
        py::ssize_t{4},
    });
    auto vertex_output = vertices.mutable_unchecked<2>();
    auto tetrahedron_output = tetrahedra.mutable_unchecked<2>();

    for (const auto& [vertex, index] : vertex_indices) {
        const Point& point = vertex->point().point();
        vertex_output(index, 0) = static_cast<float>(CGAL::to_double(point.x()));
        vertex_output(index, 1) = static_cast<float>(CGAL::to_double(point.y()));
        vertex_output(index, 2) = static_cast<float>(CGAL::to_double(point.z()));
    }

    std::size_t tetrahedron_index = 0;
    for (auto cell = begin; cell != end; ++cell, ++tetrahedron_index) {
        for (int local_index = 0; local_index < 4; ++local_index) {
            tetrahedron_output(tetrahedron_index, local_index) =
                static_cast<std::uint32_t>(
                    vertex_indices.at(cell->vertex(local_index)));
        }
    }

    return py::make_tuple(std::move(vertices), std::move(tetrahedra));
}

py::tuple create(
    py::array_t<std::uint8_t, py::array::c_style> mask,
    double spacing_x,
    double spacing_y,
    double spacing_z,
    double tetrahedra_size,
    double distance) {
    validate_positive(tetrahedra_size, "tetrahedra_size");
    validate_positive(distance, "distance");
    Input input = prepare_input(
        std::move(mask), spacing_x, spacing_y, spacing_z);
    MeshDomain domain(input.view, input.bounds);
    MeshCriteria criteria(
        parameters::cell_size(tetrahedra_size)
            .facet_distance(distance));

    C3t3 mesh;
    {
        py::gil_scoped_release release;
        mesh = CGAL::make_mesh_3<C3t3>(domain, criteria);
    }
    return extract_mesh(mesh);
}

PYBIND11_MODULE(_mesh, module) {
    module.doc() = "Private CGAL Mesh_3 bindings for binary NIfTI masks.";
    module.def(
        "create",
        &create,
        py::arg("mask").noconvert(),
        py::arg("spacing_x"),
        py::arg("spacing_y"),
        py::arg("spacing_z"),
        py::arg("tetrahedra_size"),
        py::arg("distance"),
        "Generate a uniform tetrahedral mesh from a C-contiguous uint8 mask.");
}
