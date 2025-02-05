# Welcome to Circulatory Autogen

Circulatory Autogen aims to automate the process of combining CellML modules into a system model, then doing parameter identification to calibrate to clinical data. 

Common use cases include generating coupled biochemical cell models, generating neuron to cardiomyocyte models, generating patient specific 0D blood flow networks and more.

## Why Circulatory Autogen

Circulatory Autogen provides the following benefits.

- Allows module reusability in system models:
    
    This software allows reusability and easy coupling of CellML modules. A wide range of openly available modules can be accessed and easily coupled to whatever modules you want to develop. Therefore, instead of having one huge model that will rarely be reused, you can easily integrate modules to create a system model and the modules can be used by others. 

- Provides the ability to calibrate model parameters to clinical or experimental data:

    This software supports parameter identification algorithms which you can use to calibrate the model parameters to clinical or experimental data.

- Code generation

    By generating models in CellML, the user can then re-generate their model in whatever language they want (C++, Fortran, Matlab) with libcellml. This allows for easy coupling with other types of models (PDE), C generation for embedded systems, and much more. 

- Open Source

    Circulatory Autogen is completely open source. Unlike Software like Simulink, you will never need to pay to use Circulatory Autogen and you can continue using it outside of academia. This further promotes reproducible science by encouraging anyone to use and check the modules or models that are created.

## What Circulatory Autogen can do

Circulatory Autogen provides the following two main functionalities.

1. **Model Autogeneration**

    This generates a system model by combining CellML modules in a user-defined network/arrangement. For more information, refer section [Model Generation and Simulation](model-generation-simulation.md).

2. **Parameter Identification**

    This allows calibrating model parameters to data. See section [Parameter Identification](parameter-identification.md) for more information.
