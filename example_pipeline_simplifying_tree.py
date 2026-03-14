"""
Example pipeline demonstrating how neurons are processed
into simplified tree representations.

Steps:
1. Load neuron skeleton (SWC)
2. Attach synapses
3. Split axon/dendrite compartments
4. Convert skeleton to simplified tree structure
"""

import time
import pickle
from pathlib import Path

from simple_tree_methods import swc_to_simple_tree


def process_neurons(neuron_ids, synapse_table):

    trees = []

    for neuron_id in neuron_ids:

        try:
            swc, synapses = upload_swc(neuron_id, synapse_table)

            healed_neuron = heal_attach(swc, synapses)
            split = get_split(healed_neuron)

            tree = swc_to_simple_tree(
                healed_neuron,
                split,
                synapses
            )

            trees.append(tree)

        except Exception as e:
            print(f"Error processing neuron {neuron_id}: {e}")

    return trees


if __name__ == "__main__":

    neuron_ids = [123456789, 987654321]  # example neurons
    synapse_table = None  # placeholder

    start = time.time()

    trees = process_neurons(neuron_ids, synapse_table)

    print(f"Processed {len(trees)} neurons")

    with open("trees.pkl", "wb") as f:
        pickle.dump(trees, f)