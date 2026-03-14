"""
FlyWire Connectome Processing Pipeline

This script processes neuron skeletons and synapse tables from the FlyWire
FAFB dataset. The pipeline performs skeleton healing, synapse attachment,
axon/dendrite splitting using synapse flow centrality, and extracts
connectivity features for downstream analysis.

This file represents a simplified extract from a larger research pipeline
used in my computational neuroscience work on the FlyWire connectome.

Datasets and full project infrastructure will be released through the lab
repository.
"""

import os
import sys
import time
import pickle
import random

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import networkx as nx

import navis
from fafbseg import flywire
from sklearn.cluster import AgglomerativeClustering

from nglui.statebuilder import ChainedStateBuilder
from caveclient import CAVEclient


# -------------------------------------------------------------------
# NOTE
# -------------------------------------------------------------------
# In the original research project this script relies on several
# internal utility modules (analysis methods, Neuroglancer helpers,
# feature extraction utilities, etc.). Those modules are not included
# here because they are part of a larger research codebase.
#
# Example imports from the full project:
#
# from altsi_methods_v2 import *
# from ng_methods_v2 import *
# from methods_all import *
#
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def get_split(item, flow_thresh=1):
    """
    Split neuron skeleton into axon/dendrite compartments
    using synapse flow centrality.
    """
    try:
        split = navis.split_axon_dendrite(
            item,
            metric="synapse_flow_centrality",
            reroot_soma=True,
            flow_thresh=flow_thresh,
        )
        return split
    except Exception:
        return "split issue"


def heal_attach_princeton(item, synapses):
    """
    Heal neuron skeleton and attach synapses from Princeton dataset.
    """
    try:
        healed_neurons = navis.heal_skeleton(item)

        # attach_synapses_princeton comes from the full project
        healed_neurons_att_syn = attach_synapses_princeton(
            healed_neurons, synapses
        )

        return healed_neurons_att_syn

    except Exception:
        return "heal or attach issue"


# -------------------------------------------------------------------
# Main processing
# -------------------------------------------------------------------

def main():

    # ---------------------------------------------------------------
    # Connect to FlyWire
    # ---------------------------------------------------------------

    client = CAVEclient("flywire_fafb_production")

    # ---------------------------------------------------------------
    # Paths (replace with your own dataset locations)
    # ---------------------------------------------------------------

    synapse_table_path = "path_to_synapse_table.csv"

    input_base_dir = "path_to_swc_folders"
    output_base_dir = "path_to_output_folder"

    # ---------------------------------------------------------------
    # Load synapse table
    # ---------------------------------------------------------------

    print("Loading synapse table...")

    allsynapses = pd.read_csv(synapse_table_path)

    allsynapses = allsynapses.reset_index(drop=False)

    allsynapses = allsynapses.drop(columns=["ctr_x", "ctr_y", "ctr_z"])

    # fix root id format
    allsynapses["pre_root_id_720575940"] = (
        720575940
        * 10 ** allsynapses["pre_root_id_720575940"].astype(str).str.len()
        + allsynapses["pre_root_id_720575940"].astype(int)
    ).astype(np.int64)

    allsynapses["post_root_id_720575940"] = (
        720575940
        * 10 ** allsynapses["post_root_id_720575940"].astype(str).str.len()
        + allsynapses["post_root_id_720575940"].astype(int)
    ).astype(np.int64)

    allsynapses.columns = [
        "synapse_id",
        "pre_x",
        "pre_y",
        "pre_z",
        "post_x",
        "post_y",
        "post_z",
        "size",
        "pre",
        "post",
        "neuropil",
    ]

    # remove autapses
    allsynapses = allsynapses[allsynapses["pre"] != allsynapses["post"]]

    # ---------------------------------------------------------------
    # Folder discovery
    # ---------------------------------------------------------------

    input_subfolders = {f.name for f in os.scandir(input_base_dir) if f.is_dir()}
    output_subfolders = {f.name for f in os.scandir(output_base_dir) if f.is_dir()}

    folders_to_process = input_subfolders - output_subfolders

    start_time_global = time.time()

    # ---------------------------------------------------------------
    # Processing loop
    # ---------------------------------------------------------------

    for folder_name in folders_to_process:

        print(f"\nProcessing folder: {folder_name}")

        issues_in_comp = []
        linker_neurons = []
        all_SI = []

        all_connectors = pd.DataFrame(
            columns=[
                "connector_id",
                "x",
                "y",
                "z",
                "partner_id",
                "type",
                "node_id",
                "compartment",
                "neuron",
                "synapse_id",
                "size",
            ]
        )

        input_folder = os.path.join(input_base_dir, folder_name)
        output_folder = os.path.join(output_base_dir, folder_name)

        os.makedirs(output_folder, exist_ok=True)

        # -----------------------------------------------------------
        # Load SWC skeletons
        # -----------------------------------------------------------

        swc_list = load_swc(input_folder)

        swc_list_id = [np.int64(i.id) for i in swc_list]

        allsynapses2 = allsynapses[
            (allsynapses["pre"].isin(swc_list_id))
            | (allsynapses["post"].isin(swc_list_id))
        ]

        swc_list_to_analyse = navis.NeuronList(swc_list[:])

        healed_attached_neurons_list = heal_attach_princeton(
            swc_list_to_analyse, allsynapses2
        )

        num_items = len(healed_attached_neurons_list)

        start_time = time.time()

        # -----------------------------------------------------------
        # Iterate neurons
        # -----------------------------------------------------------

        for i in range(num_items):

            print(f"-------- neuron {i} --------")

            n = healed_attached_neurons_list[i]

            if len(n.nodes) > 80000:
                issues_in_comp.append([n.id, len(n.nodes)])
                continue

            split = get_split(n)

            if isinstance(split, str):
                issues_in_comp.append([n.id, "split failed"])
                continue

            if len(split) not in [2, 3]:
                issues_in_comp.append([n.id, "unexpected split"])
                continue

            # compartment assignment
            axon_swc = None
            dend_swc = None
            linker_swc = None
            linker = False

            if len(split) == 3:

                if split[0].compartment == "dendrite":
                    dend_swc = split[0]
                    linker_swc = split[1]
                    axon_swc = split[2]

                else:
                    axon_swc = split[0]
                    linker_swc = split[1]
                    dend_swc = split[2]

                linker = True
                linker_neurons.append(n.id)

            elif len(split) == 2:

                if split[0].compartment == "dendrite":
                    dend_swc = split[0]
                    axon_swc = split[1]
                else:
                    axon_swc = split[0]
                    dend_swc = split[1]

            # -------------------------------------------------------
            # Segregation Index
            # -------------------------------------------------------

            SI, IG = SI_calc(
                [
                    "",
                    (
                        len(axon_swc.presynapses),
                        len(axon_swc.postsynapses),
                    ),
                    (
                        len(dend_swc.presynapses),
                        len(dend_swc.postsynapses),
                    ),
                ]
            )

            all_SI.append([n.id, SI])

            # -------------------------------------------------------
            # Connector extraction
            # -------------------------------------------------------

            axon_connectors = axon_swc.connectors
            axon_connectors["compartment"] = "A"

            dend_connectors = dend_swc.connectors
            dend_connectors["compartment"] = "D"

            if linker:

                linker_connectors = linker_swc.connectors
                linker_connectors["compartment"] = "L"

                neuron_connectors = pd.concat(
                    [axon_connectors, dend_connectors, linker_connectors]
                )

            else:

                neuron_connectors = pd.concat(
                    [axon_connectors, dend_connectors]
                )

            neuron_connectors["neuron"] = n.id

            neuron_connectors = neuron_connectors.reset_index(drop=True)

            all_connectors = pd.concat(
                [all_connectors, neuron_connectors]
            )

        # -----------------------------------------------------------
        # Save outputs
        # -----------------------------------------------------------

        with open(os.path.join(output_folder, "connectors.pkl"), "wb") as f:
            pickle.dump(all_connectors, f)

        with open(os.path.join(output_folder, "all_SI.pkl"), "wb") as f:
            pickle.dump(all_SI, f)

        with open(os.path.join(output_folder, "issues.pkl"), "wb") as f:
            pickle.dump(issues_in_comp, f)

        with open(os.path.join(output_folder, "linker.pkl"), "wb") as f:
            pickle.dump(linker_neurons, f)

        elapsed_time = time.time() - start_time

        print(f"Finished folder {folder_name}")
        print(f"Time for folder: {elapsed_time:.2f} seconds")

    # ---------------------------------------------------------------
    # Global timing
    # ---------------------------------------------------------------

    total_time = time.time() - start_time_global

    print("\nProcessing complete.")
    print(f"Total runtime: {total_time:.2f} seconds")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------

if __name__ == "__main__":
    main()