Getting Started
===============

If you want to help develop sunkit-image, or just want to try out the package, you will need to install it from GitHub.
The best way to do this is to create a new python virtual environment (either with pipenv or conda):

If you’re using the Anaconda Python distribution (recommended), first create a new environment for sunkit-dem development,

.. code-block:: shell

    conda create -n skdem-dev python
    conda activate skdem-dev

and clone the Github repo and install the dependencies:

.. code-block:: shell

    git clone https://github.com/sunpy/sunkit-image.git
    cd sunkit-dem
    pip install -e .

If you are looking to develop sunkit-dem, please see the :ref:`dev-guide`.
