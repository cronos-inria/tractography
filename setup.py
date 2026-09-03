from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup


extension = Pybind11Extension(
    "tractography._mesh",
    ["src/mesh.cpp"],
    cxx_std=17,
    libraries=["gmp", "mpfr"],
)

setup(
    ext_modules=[extension],
    cmdclass={"build_ext": build_ext},
)
