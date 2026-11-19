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

## Quickstart (Python)

The example below downloads the iflow dataset, uses its FreeSurfer white-matter mask to define the tracking domain, and generates a tractogram from its FOD image.
The first run downloads the data and requires internet access and a working OpenCL runtime.
The dataset is cached, so subsequent runs reuse the downloaded files.
The documented example is also covered by the opt-in integration test `pixi run -e cpu test-examples`.

```python
--8<-- "examples/iflow_tractogram.py"
```

Run the example with `python examples/iflow_tractogram.py`.
By default, the script saves `iflow-tractogram.tck` in the current directory.
Pass `--output path/to/file.tck` to choose another output path.