# FlyWire Connectome Analysis – Sample Code

This repository contains a small subset of code from a larger connectomics analysis pipeline developed for studying neuronal structure in the FlyWire Drosophila connectome.

The code demonstrates core steps used in the project, including:
- Attaching synapses to neuron skeleton nodes
- Splitting neurons into axon and dendrite compartments
- Pruning and truncating neuronal skeleton graphs into simplified tree representations
- Aggregating synapses along paths
- Computing Segregation Index (SI) structural features

## Files

**main_process.py**  
Example main workflow for processing neurons.  
Runs the pipeline steps including synapse attachment, neuron compartment splitting, and simplified tree construction.

**simple_tree_methods.py**  
Core methods for processing neuron skeleton graphs, including pruning, path compression, synapse aggregation along edges, and simplified tree generation.

## Notes

This repository contains only a small subset of the full project code.  
The complete analysis pipeline will be released through the lab repository and an accompanying preprint.
