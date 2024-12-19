Welcome to tractography's homepage!
===================================
  
`tractography`, abbreviated `tg`, is a Python package that implements different tractography algorithms.

The software provides the ``tractography`` Python module, which includes **functions and tools to produce streamlines from diffusion MRI**.

Quickstart
----------

Install
^^^^^^^

To install the package, clone the repository and install the package and its dependencies in editable mode

.. code-block::

   git clone https://gitlab.inria.fr/cronos/software/tractography.git
   cd tractography
   pip install -e .

Command-line interface
^^^^^^^^^^^^^^^^^^^^^^

Many feature of the package are available via the CLI.
You can see the different subcommands using :code:`tractography --help`.
For example, to perform tractography using a deterministic algorithm, the simplest command is

.. code-block::

   tractography tractogram det my_fod.nii.gz tractogram.tck
