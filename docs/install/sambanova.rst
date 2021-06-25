SambaNova (ALCF)
****************

`SambaNova <https://www.alcf.anl.gov/user-guides/sambanova>`_ is a wafer-scale, deep learning accelerator at Argonne Leadership Computing Facility (ALCF).


User installation
=================

Before installing DeepHyper, go to your project folder::

    cd ~/projects/PROJECTNAME
    mkdir sambanova && cd sambanova/

DeepHyper can be installed on SambaNova by following these commands::

    git clone https://github.com/deephyper/deephyper.git --depth 1
    ./deephyper/install/sambanova.sh

Then, restart your session.

.. warning::
    You will note that a new file ``~/.bashrc_sambanova`` was created and sourced in the ``~/.bashrc``. This is to avoid conflicting installations between the different systems available at the ALCF.

.. note::
    To test you installation run::

        ./deephyper/tests/system/test_sambanova.sh


A manual installation can also be performed with the following set of commands::

    # Install Miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh -O miniconda.sh
    bash $PWD/miniconda.sh -b -p $PWD/miniconda
    rm -f miniconda.sh

    # Install Postgresql
    wget http://get.enterprisedb.com/postgresql/postgresql-9.6.13-4-linux-x64-binaries.tar.gz -O postgresql.tar.gz
    tar -xf postgresql.tar.gz
    rm -f postgresql.tar.gz

    # adding Cuda
    echo "+cuda-10.2" >> ~/.soft.sambanova
    resoft

    source $PWD/miniconda/bin/activate

    # Create conda env for DeepHyper
    conda create -p dh-sambanova python=3.8 -y
    conda activate dh-sambanova/
    conda install gxx_linux-64 gcc_linux-64 -y
    # DeepHyper + Analytics Tools (Parsing logs, Plots, Notebooks)
    pip install deephyper[analytics,balsam]
    conda install tensorflow-gpu

.. warning::
    The same ``.bashrc`` is used both on Theta and Cooley. Hence adding a ``module load`` instruction to the ``.bashrc`` will not work on Cooley. In order to solve this issue you can add a specific statement to your ``.bashrc`` file and create separate *bashrc* files for Theta and Cooley and use them as follows.
    ::

        # SambaNova Specific
        if [[ $HOSTNAME = *"sambanova"* ]];
        then
            source ~/.bashrc_sambanova
        # Cooley Specific
        else
            source ~/.bashrc_cooley
        fi
