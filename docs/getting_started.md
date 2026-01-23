# Getting Started

<!-- optional text here -->

---

## Installation

If you have a GPU and have the drivers installed and up-to-date, the install should be as simple as cloning the repository and using pip. We recommend installing in editable mode with

```bash

   git clone https://github.com/cronos-inria/tractography.git
   cd tractography
   pip install -e .
```

If you would like to use the CPU version, then we recommend using anaconda for the installation

```bash

    conda create -n tractography python=3
    conda activate tractography
    conda install anaconda::binutils
    conda install numpy
    conda install -c conda-forge pocl
    conda install -c conda-forge pyopencl
    git clone https://github.com/cronos-inria/tractography.git
    cd tractography
    pip install -e .
```