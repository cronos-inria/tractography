Welcome to tractography's homepage!
===================================
  
`tractography`, abbreviated `tg`, is a Python package that implements different tractography algorithms.

The software provides the ``tractography`` Python module, which includes **functions and tools to produce streamlines from diffusion MRI**.

Quickstart
----------

Install
^^^^^^^

To install the package, clone the repository and install the package and its dependencies in editable mode

.. code-block:: bash

   git clone https://gitlab.inria.fr/cronos/software/tractography.git
   cd tractography
   pip install -e .

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

Many feature of the package are available via the command-line interface.
You can see the different subcommands using :code:`tractography --help`.
The general pipeline follow these steps:

1. Estimate fibre orientation distributions (FOD);
2. Generate seeds from the FOD;
3. Run the tractography algorithm to generate streamlines.

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

   # Load the data using nibabel.
   fod_nii = nib.load("fod.nii.gz")
   fod = fod_nii.get_fdata()
   wm_nii = nib.load("wm.nii.gz")
   wm = wm_nii.get_fdata()

   # Generate the seeds by sampling the FOD.
   fod = tg.core.apply_mask(fod, fod_nii.affine, wm, wm_nii.affine)
   seeds = tg.seeds.from_fod(fod, fod_nii.affine, 50000)

   # Run the tractography algorithm to generate streamlines.
   streamlines = tg.tractogram(fod, fod_nii.affine, seeds)  # The default algorithm is diffusion

