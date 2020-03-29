# Agent Based Modeling for Land Use Change

This reporsitory contains the code and results used for the paper: 

*Land Change in Latin American Urban Peripheries: An Agent-Based Model of Land Transaction with Informal mechanisms* by Diego Silva Ardila, Felipe González-Casabianca and  Alejandro Feged-Rivadeneira


## Code

All scripts can by found inside the folder: *scripts/*

The Agent Based Model simulations where done using Python and MESA (*https://mesa.readthedocs.io/en/master/*). The following scripts where implemented as part of the MESA framework for the project:
* **agents.py**: code for the different agents that interact in the model.
* **model.py**: code for the model.
* **city_generator.py**: code designed to create the different layouts that where needed (random, from raster etc..).


We also include the ratsers that were extracted using Google Earth Engine along with the corresponding scripts for the engine, inside the folder: *scripts/rasters*

## Experiment Results

The output of each experiment can be found inside the folder: *results/*

Each experiment is compressed inside a single folder that includes:
* **batch_run.py**: version of the script used to excecute the experiment (so it can be replicated)
* **city_generator**: version of the script used to generate the city grid for the experiment. 
* **output.xlsx**: the corresponding results for each of the prediction variables of the scenario.
* **tda.csv:** file with the final property values for each cell (in JSON format). This file can be used to further analyse the output of each experiment, particularly to excecute Topological Data Analysis.

Please refer to the paper for the details of each experiment. 


