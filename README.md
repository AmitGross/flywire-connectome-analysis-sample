# FlyWire Connectome Analysis – Sample Code

This repository contains a small subset of code from a larger connectomics analysis pipeline developed for analyzing neuronal structure in the FlyWire *Drosophila* connectome.

The project focuses on transforming neuron skeletons and synapse data into simplified graph representations that enable structural and computational analysis of neuronal organization.

## Pipeline Overview


Core operations include:
- Attaching synapses to neuron skeleton nodes
- Splitting neurons into axon and dendrite compartments
- Pruning and truncating neuronal skeleton graphs into simplified tree representations
- Aggregating synapses along compressed paths
- Computing Segregation Index (SI) structural features

These steps allow large neuron skeletons to be converted into compact graph structures suitable for downstream computational analysis.

## Files

**main_process.py**  
Example main workflow for processing neurons across a dataset.  
Handles neuron loading, synapse attachment, compartment splitting, and extraction of structural features.

**simple_tree_methods.py**  
Core methods for simplifying neuron skeleton graphs.  
Includes tree pruning, path compression, synapse aggregation along edges, and construction of simplified neuron tree representations.

**example_pipeline.py**  
Minimal example demonstrating how neuron skeletons are processed into simplified trees using the methods in this repository.

## Notes

This repository contains only a representative subset of the full project code.  
The complete analysis pipeline will be released through the lab repository alongside an accompanying preprint.
