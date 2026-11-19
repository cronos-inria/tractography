Welcome to tractography's homepage!
===================================
  
`tractography`, abbreviated `tg`, is a Python package that implements different tractography algorithms.

The software provides the ``tractography`` Python module, which includes **functions and tools to produce streamlines from diffusion MRI**.

Quickstart
----------

Install
^^^^^^^

If you have a GPU and have the drivers installed and up-to-date, the install should be as simple as cloning the repository and using pip. We recommend installing in editable mode with

.. code-block:: bash

   git clone https://github.com/cronos-inria/tractography.git
   cd tractography
   pip install -e .

If you would like to use the CPU version, then we recommend using anaconda for the installation

.. code-block:: bash

    conda create -n tractography python=3
    conda activate tractography
    conda install anaconda::binutils
    conda install numpy
    conda install -c conda-forge pocl
    conda install -c conda-forge pyopencl
    git clone https://github.com/cronos-inria/tractography.git
    cd tractography
    pip install -e .

Testing
^^^^^^^

To verify the installation, make sure you are in the tractography directory and run

.. code-block:: bash

   python -m unittest

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

Many feature of the package are available via the command-line interface.
You can see the different subcommands using :code:`tractography --help`.
A simple streamline generation pipeline follows these steps:

1. estimate fibre orientation distributions (FOD);
2. generate seeds from the FOD;
3. run the tractography algorithm to generate streamlines.

This package supports steps 2 and 3, step 1 has to be performed with other software.

For example, to perform tractography using the transport (deterministic) algorithm, the commands are

.. code-block:: bash

   tractography seeds from-fod path-to-fod.nii.gz 100000 seeds.txt
   tractography tractogram transport path-to-fod.nii.gz seeds.txt tractogram.tck

Python
^^^^^^

To generate streamlines from the Python package using the diffusion (probabilistic) algorithm

.. code-block:: python

   import tractography as tg
   import nibabel as nib

      # Load the FOD and white-matter mask as images.
      fod = nib.load("fod.nii.gz")
      wm = nib.load("wm.nii.gz")

      # Generate seeds and configure a tracking algorithm.
      seeds = tg.seeds.from_mask(wm, 50000)
      model = tg.local.RealSymmetricSphericalHarmonics(fod.shape[-1])
      algorithm = tg.algorithm.Diffusion(
         model, step_size=0.001, gamma=1.0, noise_variance=0.1
      )

      # Run tractography. The algorithm is passed directly, with no config object.
      streamlines = tg.tractogram(
         fod,
         seeds,
         wm,
         algorithm,
         save_at=0.25,
         maximum_streamline_length=300.0,
         batch_size=50000,
      )
