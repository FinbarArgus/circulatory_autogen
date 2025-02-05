# Welcome to Circulatory Autogen

Circulatory Autogen aims to automate the process of combining CellML modules into a system model, then doing parameter identification to calibrate to clinical data. 

Common use cases include generating coupled neuron models, generating neuron to cardiomyocyte models, generating patient specific 0D blood flow networks and more.

## Why Circulatory Autogen

Circulatory Autogen provides the following main two advantages.

- Allows module reusability in system models:
    
    This project provides a defined set of modules which can be used to connect with your module. Therefore, instead of having one huge model, you can easily integrate these modules to create a system model. 

- Provides the ability to calibrate model parameters to clinical data:

    This supports a couple of parameter identification algorithms which you can use to calibrate the model parameters to clinical data.

## What Circulatory Autogen do

Circulatory Autogen provides the following two main functionalities.

1. **Model Autogeneration**

    This generates a system model by combining CellML modules. For more information, refer section [Model Generation and Simulation](model-generation-simulation.md).

2. **Parameter Identification**

    This allows calibrating model parameters to real data. See section [Parameter Identification](parameter-identification.md) for more information.
