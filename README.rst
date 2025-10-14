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

Many feature of the package are available via the CLI.
You can see the different subcommands using :code:`tractography --help`.
For example, to perform tractography using the transport (deterministic) algorithm, the simplest command is

.. code-block:: bash

   tractography tractogram transport my_fod.nii.gz tractogram.tck

Python
^^^^^^

To generete streamlines from the Python pacakge

.. code-block:: python

   import tractography as tg
   import nibabel as nib

   fod_nii = nib.load("fod.nii.gz")
   fod = fod_nii.get_fdata()

   wm_nii = nib.load("wm.nii.gz")
   wm = wm_nii.get_fdata()
 
   fod = tg.core.apply_mask(fod, fod_nii.affine, wm, wm_nii.affine)

   seeds = tg.seeds.from_odf(fod, fod_nii.affine, 50000)
   streamlines = tg.tractogram(fod, fod_nii.affine, seeds)  # The default algorithm is diffusion

