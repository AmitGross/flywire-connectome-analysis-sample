"""
FlyWire Connectome Analysis - Simple Tree Methods

This file contains a reduced subset of methods used to convert neuron
skeletons into simplified directed tree representations for downstream
analysis. The code supports compartment-aware preprocessing, pathway
compression, synapse aggregation on edges, and basic SI-related summaries.

This repository contains only a small sample of the full project.
"""

from math import log
from typing import List, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Basic SI utilities
# ---------------------------------------------------------------------

def calc_s(pre: int, post: int) -> float:
    """Binary entropy of pre/post counts."""
    total = pre + post
    if total <= 0:
        return 0.0

    p_pre = pre / total
    p_post = post / total

    entropy_pre = p_pre * log(p_pre, 2) if p_pre > 0 else 0
    entropy_post = p_post * log(p_post, 2) if p_post > 0 else 0
    return -(entropy_pre + entropy_post)


def SI_calc(neuron_id_data) -> Tuple[float, float]:
    """
    Compute Segregation Index (SI) and information gain (IG)
    from axon/dendrite pre-post synapse counts.

    Expected format:
    [nid, (axon_pre, axon_post), (dend_pre, dend_post)]
    """
    ax_pre = neuron_id_data[1][0]
    ax_post = neuron_id_data[1][1]
    dend_pre = neuron_id_data[2][0]
    dend_post = neuron_id_data[2][1]

    ent_ax = calc_s(ax_pre, ax_post)
    ent_dend = calc_s(dend_pre, dend_post)
    snorm = calc_s(ax_pre + dend_pre, ax_post + dend_post)

    total_syn = ax_pre + ax_post + dend_pre + dend_post
    if total_syn == 0 or snorm == 0:
        return 0.0, 0.0

    s_weighted = (
        ((ax_pre + ax_post) * ent_ax) +
        ((dend_pre + dend_post) * ent_dend)
    ) / total_syn

    si = 1 - (s_weighted / snorm)
    ig = snorm - s_weighted
    return si, ig


# ---------------------------------------------------------------------
# Split / compartment preprocessing
# ---------------------------------------------------------------------

def split_to_data(split):
    """
    Extract axon/dendrite/linker compartments and related metadata
    from a navis split object.
    """
    link = 0
    axon = 0
    dend = 0
    linker = 0
    lnodes = 0

    if len(split) < 2:
        raise ValueError("Split contains fewer than 2 compartments.")

    if len(split) == 2:
        axon = split[-1]
        dend = split[0]
    else:
        axon = split[-1]
        dend = split[0]
        link = split[1]
        linker = 1
        lnodes = link.nodes

    neuron_var = split[0].id
    axon_presynapses = len(axon.presynapses)
    axon_postsynapses = len(axon.postsynapses)
    dend_presynapses = len(dend.presynapses)
    dend_postsynapses = len(dend.postsynapses)

    anodes = axon.nodes
    dnodes = dend.nodes

    neuron_id = [
        neuron_var,
        [axon_presynapses, axon_postsynapses],
        [dend_presynapses, dend_postsynapses],
    ]

    root_comp = 0
    root_comp_l = 0
    root_comp_a = len(anodes[anodes["node_id"] == 1])
    root_comp_d = len(dnodes[dnodes["node_id"] == 1])

    if linker:
        root_comp_l = len(lnodes[lnodes["node_id"] == 1])

    total_roots = root_comp_l + root_comp_a + root_comp_d
    if total_roots > 1:
        print("Warning: more than one compartment contains node_id == 1")

    if root_comp_a:
        root_comp = "A"
    if root_comp_d:
        root_comp = "D"
    if root_comp_l:
        root_comp = "L"

    return axon, dend, link, neuron_id, anodes, dnodes, lnodes, linker, root_comp


def check_roots(anodes, dnodes, lnodes, root_comp, linker, link, nodes_df):
    """
    Determine effective axon/dendrite roots after split.
    """
    lroot = 0

    if root_comp == "D":
        aroot = anodes[anodes["parent_id"] == -1]["node_id"].values[0]
        if linker:
            lnodes = link.nodes
            lroot = lnodes[lnodes["parent_id"] == -1]["node_id"].values[0]
            droot = nodes_df[nodes_df["node"] == lroot]["parent"].values[0]
        else:
            droot = nodes_df[nodes_df["node"] == aroot]["parent"].values[0]

    elif root_comp == "A":
        droot = dnodes[dnodes["parent_id"] == -1]["node_id"].values[0]
        if linker:
            lroot = lnodes[lnodes["parent_id"] == -1]["node_id"].values[0]
            aroot = nodes_df[nodes_df["node"] == lroot]["parent"].values[0]
        else:
            aroot = nodes_df[nodes_df["node"] == droot]["parent"].values[0]

    else:
        droot = dnodes[dnodes["parent_id"] == -1]["node_id"].values[0]
        aroot = anodes[anodes["parent_id"] == -1]["node_id"].values[0]
        lroot = 1

    return aroot, droot, lroot


def tree_pre_process(nodes_df, synapses_df, allsynapses_, split):
    """
    Build full graph, separate axon/dendrite subgraphs, and extract roots.
    """
    g = create_neuron_tree_s_pass(nodes_df, synapses_df, allsynapses_)

    if g.has_edge(1, 1):
        g.remove_edge(1, 1)

    (
        axon,
        dend,
        link,
        neuron_id,
        anodes,
        dnodes,
        lnodes,
        linker,
        root_comp,
    ) = split_to_data(split)

    aroot, droot, lroot = check_roots(
        anodes, dnodes, lnodes, root_comp, linker, link, nodes_df
    )

    anodes_ids = list(anodes["node_id"])
    dnodes_ids = list(dnodes["node_id"])

    g2 = g.copy()
    gaxon = g2.subgraph(anodes_ids).copy()
    gdend = g2.subgraph(dnodes_ids).copy()

    return (
        gaxon,
        gdend,
        axon,
        dend,
        link,
        neuron_id,
        anodes,
        dnodes,
        lnodes,
        linker,
        root_comp,
        aroot,
        droot,
        lroot,
        anodes_ids,
        dnodes_ids,
    )


# ---------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------

def create_neuron_tree_s_pass(nodes_df, synapses_df, allsynapses_v):
    """
    Create a directed graph from node and synapse tables.
    """
    g = nx.DiGraph()

    for _, row in nodes_df.iterrows():
        g.add_node(
            row["node"],
            pos=(row["x"], row["y"], row["z"]),
            pre=0,
            post=0,
            synapses=[],
        )

    for _, row in nodes_df.iterrows():
        if row["parent"] != -1:
            g.add_edge(int(row["parent"]), int(row["node"]))

    for _, row in synapses_df.iterrows():
        if row["node"] in g.nodes:
            g.nodes[row["node"]]["pre"] += row["pre"]
            g.nodes[row["node"]]["post"] += row["post"]

            syn_v = allsynapses_v[
                allsynapses_v["node_id"] == row["node"]
            ][["node_id", "type", "x", "y", "z"]]

            g.nodes[row["node"]]["synapses"].append(syn_v)

    return g


def tree_rearrange_by_root(g, root):
    """
    Reorient a directed tree starting from a desired root.
    """
    new_tree = nx.DiGraph()
    undirected = g.to_undirected(as_view=True)

    for edge in nx.edge_bfs(undirected, source=root):
        if (edge[0], edge[1]) in g.edges:
            attrs = g.edges[edge[0], edge[1]]
        else:
            attrs = g.edges[edge[1], edge[0]]
        new_tree.add_edge(edge[0], edge[1], **attrs)

    for node in new_tree.nodes():
        new_tree.nodes[node].update(g.nodes[node])

    return new_tree


def remove_arbitrary_root(root_compv, axon_new_tree, dend_new_tree):
    """
    Remove the artificial root node from the compartment that contains it.
    """
    if root_compv == "D" and 1 in dend_new_tree:
        dend_new_tree.remove_node(1)
    elif root_compv == "A" and 1 in axon_new_tree:
        axon_new_tree.remove_node(1)

    return axon_new_tree, dend_new_tree


# ---------------------------------------------------------------------
# Tree simplification / compression
# ---------------------------------------------------------------------

def tree_leaf_pruning(g):
    """
    Remove leaf nodes with no pre/post synapses.
    """
    running = True
    while running:
        nodes_to_remove = []
        for node in list(g.nodes()):
            neighbors = list(g.neighbors(node))
            pre_var = g.nodes[node].get("pre", 0)
            post_var = g.nodes[node].get("post", 0)

            if pre_var == 0 and post_var == 0 and len(neighbors) == 0:
                nodes_to_remove.append(node)

        for node in nodes_to_remove:
            g.remove_node(node)

        if len(nodes_to_remove) == 0:
            running = False

    return g


def tree_emptylink_pruning(g):
    """
    Remove intermediate nodes that have no synapses and exactly one child.
    """
    nodes_to_remove = []

    for node in list(g.nodes()):
        neighbors = list(g.neighbors(node))
        pre_var = g.nodes[node].get("pre", 0)
        post_var = g.nodes[node].get("post", 0)

        if pre_var == 0 and post_var == 0 and len(neighbors) == 1:
            nodes_to_remove.append(node)

    for node in nodes_to_remove:
        preds = list(g.predecessors(node))
        if len(preds) == 0:
            continue

        parent = preds[0]
        child = list(g.neighbors(node))[0]
        g.remove_node(node)
        g.add_edge(parent, child)

    return g


def get_df_nodes_types(g):
    """
    Label each node as leaf / line / split.
    """
    data_list = []
    for node in g.nodes():
        nei_l = len(list(g.neighbors(node)))
        node_type = "leaf" if nei_l == 0 else "line" if nei_l == 1 else "split"
        data_list.append([node, node_type])

    return pd.DataFrame(data_list, columns=["nodeid", "type"])


def get_recursive_pathways(g, root_node=1):
    """
    Return pathways from each split/root to the next split/leaf.
    """
    df_nodes_var = get_df_nodes_types(g)
    outer_list = []

    def step2(current_node, parent, list_var):
        node_type = df_nodes_var[df_nodes_var["nodeid"] == current_node]["type"].values[0]

        if node_type == "leaf":
            if len(list_var) == 0:
                outer_list.append([parent, current_node])
            else:
                outer_list.append(list_var + [current_node])

        elif node_type == "split":
            if len(list_var) == 0:
                outer_list.append([parent, current_node])
            else:
                outer_list.append(list_var + [current_node])

            for child in list(g.neighbors(current_node)):
                if current_node != child:
                    step2(child, current_node, [])

        else:
            if len(list_var) == 0:
                to_add = [parent, current_node]
            else:
                to_add = list_var + [current_node]

            for child in list(g.neighbors(current_node)):
                if current_node != child:
                    step2(child, current_node, to_add)

    step2(root_node, root_node, [])
    return outer_list


def create_final_tree_structure(g, nodes_df, pathways, root=1):
    """
    Build compressed tree using only pathway endpoints.
    """
    gn = nx.DiGraph()

    for sublist in pathways:
        gn.add_edge(sublist[0], sublist[-1])

    for node in gn.nodes():
        node_data = nodes_df[nodes_df["node"] == node]
        if not node_data.empty:
            x = node_data.iloc[0]["x"]
            y = node_data.iloc[0]["y"]
            z = node_data.iloc[0]["z"]

            gn.nodes[node]["pos"] = (x, -y, z)
            gn.nodes[node]["synapses"] = g.nodes[node]["synapses"]

    return gn


def add_synapses_to_edges_pass_n(g, pathways, synapses_df, g_no_linker_nodes):
    """
    Aggregate synapses along each compressed path and assign them to edges.
    """
    for path in pathways:
        npath = path[1:]
        parent = path[0]
        node = npath[-1]

        pre_syn_var = 0
        post_syn_var = 0
        synapses_list = []

        for vnode in npath:
            pre_array = synapses_df[synapses_df["node"] == vnode]["pre"].values
            post_array = synapses_df[synapses_df["node"] == vnode]["post"].values

            synapses_list.extend(g_no_linker_nodes.nodes[vnode]["synapses"])

            if pre_array.size > 0:
                pre_syn_var += pre_array[0]
            if post_array.size > 0:
                post_syn_var += post_array[0]

        g.edges[parent, node]["pre"] = pre_syn_var
        g.edges[parent, node]["post"] = post_syn_var
        g.edges[parent, node]["synapses"] = synapses_list

    return g


def concat_synapses_in_graph(graph):
    """
    Concatenate edge synapse lists into DataFrames.
    """
    graph = graph.copy()

    for u, v, data in graph.edges(data=True):
        if "synapses" in data:
            synapses_list = data["synapses"]
            if (
                isinstance(synapses_list, list)
                and all(isinstance(df, pd.DataFrame) for df in synapses_list)
                and len(synapses_list) > 0
            ):
                graph[u][v]["synapses"] = pd.concat(synapses_list, ignore_index=True)

    return graph


def tree_process_synapses_on_edges(gtree, root, nodes_df, synapses_df):
    """
    Full simplification pipeline:
    prune -> compress -> aggregate synapses on compressed edges.
    """
    g_leafprun = tree_leaf_pruning(gtree)
    g_elinkprun = tree_emptylink_pruning(g_leafprun)
    pathways = get_recursive_pathways(g_elinkprun, root)
    gfinal_structure = create_final_tree_structure(g_elinkprun, nodes_df, pathways, root)
    gn_syn = add_synapses_to_edges_pass_n(gfinal_structure, pathways, synapses_df, g_elinkprun)
    gn_syn_f = concat_synapses_in_graph(gn_syn)

    return gn_syn_f, pathways


# ---------------------------------------------------------------------
# Root aggregation
# ---------------------------------------------------------------------

def assign_att_to_root(tree, root, childs):
    """
    Aggregate synapses and pre/post counts from child edges to a root edge.
    """
    combined_synapses = pd.DataFrame()
    total_pre = 0
    total_post = 0

    for connection in childs:
        edge_attributes = tree.get_edge_data(connection[0], connection[1])

        if edge_attributes:
            total_pre += edge_attributes.get("pre", 0)
            total_post += edge_attributes.get("post", 0)

            if "synapses" in edge_attributes:
                combined_synapses = pd.concat(
                    [combined_synapses, edge_attributes["synapses"]],
                    ignore_index=True,
                )

    return combined_synapses, total_pre, total_post


# ---------------------------------------------------------------------
# Main simplified tree conversion
# ---------------------------------------------------------------------

def swc_to_simple_tree(item, split, allsynapses, process_swc_pass_n, d=1750):
    """
    Convert a split neuron into a simplified combined axon-dendrite tree.

    Parameters
    ----------
    item : neuron object
        Neuron skeleton with synapses attached.
    split : split result
        Output of navis.split_axon_dendrite(...)
    allsynapses : pd.DataFrame
        Synapse table for the neuron / relevant subset.
    process_swc_pass_n : callable
        External helper from the full project that returns:
        nn, nodes_df, synapses_df, allsynapses_
    d : int
        Retained for API compatibility.

    Returns
    -------
    nx.DiGraph or Exception or str
    """
    nid = int(item.id)

    try:
        try:
            _, nodes_df, synapses_df, allsynapses_ = process_swc_pass_n(item, allsynapses)
        except Exception:
            return "no pre or post"

        (
            gaxon,
            gdend,
            axon,
            dend,
            link,
            neuron_id,
            anodes,
            dnodes,
            lnodes,
            linker,
            root_comp,
            aroot,
            droot,
            lroot,
            anodes_ids,
            dnodes_ids,
        ) = tree_pre_process(nodes_df, synapses_df, allsynapses_, split)

        dend_rearranged = tree_rearrange_by_root(gdend, droot)
        axon_rearranged = tree_rearrange_by_root(gaxon, aroot)
        axon_rearranged, dend_rearranged = remove_arbitrary_root(
            root_comp, axon_rearranged, dend_rearranged
        )

        if len(dend_rearranged.nodes) == 0 or len(axon_rearranged.nodes) == 0:
            return "no dend or axon"

        axon_final_tree, _ = tree_process_synapses_on_edges(
            axon_rearranged, aroot, nodes_df, synapses_df
        )
        dend_final_tree, _ = tree_process_synapses_on_edges(
            dend_rearranged, droot, nodes_df, synapses_df
        )

        si_ours, ig = SI_calc(neuron_id)

        combined_tree = nx.DiGraph()
        new_root = -1
        combined_tree.add_node(new_root)

        combined_tree = nx.compose(combined_tree, dend_final_tree)
        combined_tree = nx.compose(combined_tree, axon_final_tree)

        combined_tree.add_edge(new_root, droot)
        combined_tree.add_edge(new_root, aroot)

        droot_pos = dend_final_tree.nodes[droot].get("pos", (0, 0, 0))
        aroot_pos = axon_final_tree.nodes[aroot].get("pos", (0, 0, 0))
        centroid_pos = (
            (droot_pos[0] + aroot_pos[0]) / 2,
            (droot_pos[1] + aroot_pos[1]) / 2,
            (droot_pos[2] + aroot_pos[2]) / 2,
        )

        combined_tree.nodes[new_root]["pos"] = centroid_pos
        combined_tree.nodes[new_root]["rank"] = -1
        combined_tree.nodes[new_root]["aroot"] = aroot
        combined_tree.nodes[new_root]["droot"] = droot
        combined_tree.nodes[new_root]["id"] = nid
        combined_tree.nodes[new_root]["SI"] = si_ours

        axon_childs = list(combined_tree.out_edges(aroot))
        dend_childs = list(combined_tree.out_edges(droot))

        combined_synapses, total_pre, total_post = assign_att_to_root(
            combined_tree, aroot, axon_childs
        )
        combined_tree.add_edge(
            -1, aroot, pre=total_pre, post=total_post, synapses=combined_synapses
        )

        combined_synapses, total_pre, total_post = assign_att_to_root(
            combined_tree, droot, dend_childs
        )
        combined_tree.add_edge(
            -1, droot, pre=total_pre, post=total_post, synapses=combined_synapses
        )

        return combined_tree

    except Exception as e:
        return e
