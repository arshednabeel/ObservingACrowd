# Observing and Inferring A Collective

This repo contains the supporting code for the manuscript _'Observing a Collective to Infer the Characteristics of Agents'_

The code is organized as two subdirectories: `Simulation`, which contains code (MATLAB) to generate simulated data, and `Analysis`, which contains code (Python) that performs classification analysis. The general workflow is as follows:

- Use simulation scripts to generate simulated data (MAT files).
- Use the functions in `batch_processing.py` script to generate summary data.
- Use the functions in `figures.py` to reproduce figures in the manuscript.

## Simulation

TODO

## Analysis

Most of the heavy-lifting is done by the classes `AgentDynamics` (see `agent_dynamics.py`) and `DataClassifier` (see `classify.py`). `AgentDynamics` represents one simulation realization, while `DataClassifier` aggregates multiple realizations for a given set of parameters. See the methods of each class for more details, most of the methods are documented.

Once we have simulated data from the simulation scripts, `batch_processing.py` script contains functions to process and summarize classification results.
    
- `cache_all_data` and `cache_all_data_parallel` collects and summarizes simulation data (MAT files) into summary representations.
- `compute_classification_metrics` performs classification analysis (with both observers -- see paper for details) on the summary representations, and saves the confusion matrices.
- The functions in `figures.py` uses the confusion matrices and summary representations to generate figures from the manuscript.
