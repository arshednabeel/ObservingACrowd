# Observing and Inferring A Collective

This repo contains the supporting code for the manuscript _'Observing a Collective to Infer the Characteristics of Agents'_

The code is organized as two subdirectories: `Simulation`, which contains code (MATLAB) to generate simulated data, and `Analysis`, which contains code (Python) that performs classification analysis. The general workflow is as follows:

- Use simulation scripts to generate simulated data (MAT files).
- Use the functions in `batch_processing.py` script to generate summary data.
- Use the functions in `figures.py` to reproduce figures in the manuscript.

## Simulation

`Runner_OandI_delv_rho_Nr.m` : Runs the 2D simulations for a variety of parameters: _delv_ (which is the intrinsic speed, s0), _Nr_ (which is the number ratio) and _rho_ which is the packing density.

`ABM_bidispese_delv_rho_Nr.m` : Code for the Agent based model for circular agents in 2D periodic domain for a given set of parameters.
`agents_Expmemory_per2D.m` : Contains the forces on the agents (self-propulsion, inter-agent short ranged interaction, brownian noise (turned off in the default))

`RandomizationOfAgents_InitialConditions.m` and `agents_Expmemory_per2D_Randomization.m` : These functions are used to create randomly packed arrangement of agents for the initial conditions to be used later in `ABM_bidispese_delv_rho_Nr.m`.

`parameters.m` and `parameters_additional.m` the required parameters for the ABM simulations. 

**NOTE: To reproduce the results in the paper (ArXiv link), run `Runner_OandI_delv_rho_Nr.m`. The system size N can be varied in the above m file. The parameters corresponding to the forces between the agents can be changed using the `parameters.m` file.**

## Analysis

Most of the heavy-lifting is done by the classes `AgentDynamics` (see `agent_dynamics.py`) and `DataClassifier` (see `classify.py`). `AgentDynamics` represents one simulation realization, while `DataClassifier` aggregates multiple realizations for a given set of parameters. See the methods of each class for more details, most of the methods are documented.

Once we have simulated data from the simulation scripts, `batch_processing.py` script contains functions to process and summarize classification results.
    
- `cache_all_data` and `cache_all_data_parallel` collects and summarizes simulation data (MAT files) into summary representations.
- `compute_classification_metrics` performs classification analysis (with both observers -- see paper for details) on the summary representations, and saves the confusion matrices.
- The functions in `figures.py` uses the confusion matrices and summary representations to generate figures from the manuscript.
