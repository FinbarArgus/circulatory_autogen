# Getting Started

## Initialising and Startup

The project must be run on a Linux operating system.

!!! tip
    If you're on Windows, you can install a virtual Linux machine. One of these virtual linux machines is VirtualBox Oracle, which can be downloaded from [here](https://www.virtualbox.org/).

    Also, you can download the latest version of Ubuntu using this [link](https://ubuntu.com/download/desktop).

!!! info
    Running on Windows is in development. It can work with some caveats. See [Running on Windows](#running-on-windows) for more information.

**1. Install OpenCOR**

Download and install OpenCOR software from this [link](https://opencor.ws/downloads/index.html).

**2. OpenCOR Tutorial**

Download the OpenCOR tutorial, which is a comprehensive tutorial including many examples: [OpenCOR Tutorial](https://tutorial-on-cellml-opencor-and-pmr.readthedocs.io/en/latest/_downloads/d271cfcef7e288704c61320e64d77e2d/OpenCOR-Tutorial-v17.pdf).

**3. Clone the project**

Clone the project from the [GitHub repository](https://github.com/FinbarArgus/circulatory_autogen).

!!! note
    If you have not worked with git and GitHub, firstly download and install git, and then open the terminal and navigate to a directory where you want the repository to be. Then write these commands to clone the project on your pc:

    - `git clone https://github.com/FinbarArgus/circulatory_autogen`

    - `git remote add origin https://github.com/FinbarArgus/circulatory_autogen`

    If you want to develop the code, then create a fork of the above repo in GitHub, then do the following lines instead of the above:

    - `git clone https://github.com/<YourUsername>/circulatory_autogen`

    - `git remote add origin https://github.com/<YourUsername>/circulatory_autogen`

    - `git remote add upstream https://github.com/FinbarArgus/circulatory_autogen`

## Directory Definition

In this tutorial, we use one particular directory for our project, but it can be different on every computer. So the base directory is defined as **main_dir** in all parts. For example, on our computer, this directory is as below:

`main_dir: Home/…/Desktop/`

The project directory **project_dir** is the directory where the GitHub Circulatory_autogen project is cloned to our computer. For example, the directory may be:

`[project_dir]: Home/…/Desktop/Project/Circulatory_autogen`

Also, OpenCOR files directory is needed for opening the project and installing python and pythonshell, and we show with **OpenCOR_dir**, which is below on our pc:

`OpenCOR_dir: Home/…/Desktop/OpenCOR`

!!! info
    If running on the ABI HPC, you can use the OpenCOR at the path: **/hpc/farg967/OpenCOR-2022-05-23-Linux/**. 
    
    Ignore the below installation steps.

## Python and Libraries Installation

To run openCOR, you need to use the Python version with openCOR. 

To install required python packages, navigate to `[OpenCOR_dir]` directory and run the below command.

```
./pip install <packagename>
```

!!! note
    **Required packages for autogeneration**:
    pandas pyyaml rdflib

    **Recommended but nor required packages for autogeneration (allows for better error checking)**:
    libcellml

    **Required packages for parameter identification**:
    mpi4py sympy

    **Required packages for mcmc bayesian identification**:
    emcee corner schwimmbad tqdm statsmodels

    **Required for some utilities**:
    ruamel.yaml

!!! warning
    Intalling **mpi4py** requires mpi to be available. Therefore, the following lines may be required to install the mpi software on your computer.

    === "Linux"
        ```
        sudo apt install libopenmpi-dev
        sudo apt install libffi7
        ```

    === "Mac"
        ```
        brew install openmpi
        ```

!!! warning 
    In versions of **OpenCOR < 0.8** you needed to nagivate to the `[OpenCOR_dir]/python/bin` directory and run the below command instead.

    ```
    ./python -m pip install <packagename>
    ```

!!! warning
    if you get an SSL error you must do the following before the pip install:

        cd [OpenCOR_dir]/python/bin
        export LD_LIBRARY_PATH=[OpenCOR_dir]/lib

    This would let the system know where to look for libcrypto.so.3 when loading the ssl module.
    This should only be a problem in **OpenCOR < 0.8**


## Running on Windows

You have to run the scripts in `src/scripts` explicitly since you cannot use the .sh files to run scripts. 

E.g. For running param id, navigate to to `C:\path\to\opencor\dir` and run the below command.

```
.\pythonshell.bat C:\path\to\circulatory\autogen\src\scripts\param_id_run_script.py
```

To be able to importing mpi4py, you may have to do the following:

1. Download MS MPI, install both .mis and SDK.

2. Set up environmental variables.

    !!! Tip
        Open `Control Panel` and select `Advanced System Settings`. Then select `Environmental Variables` and add the following.

            C:\Program Files\Microsoft MPI\
            C:\Program Files (x86)\Microsoft SDKs\MPI\

!!! info "Changes to be made"
    Currently **vessels** is used interchangeabley with **modules**. This will be changed to use **modules** in all instances, as the project now allows all types of modules, not just vessels.

    The connections between terminals and the venous system is hardcoded, as a terminal_venous_connection has to be made to sum up the flows and get an avergae concentration. This needs to be improved.
