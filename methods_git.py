import pandas as pd

from typing import Iterable
import sys
import networkx as nx
import subprocess
import uuid
edge_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
import matplotlib.pyplot as plt

from nglui.statebuilder import StateBuilder,ChainedStateBuilder
from caveclient import CAVEclient
from nglui.statebuilder.helpers import PointMapper ,from_client,AnnotationLayerConfig ,package_state,sort_dataframe_by_root_id
import os
import navis
import numpy as np
#%%ng core

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

def shorten_and_open_links(link):
    # Create a unique temporary HTML file
    temp_file = f"temp_link_{uuid.uuid4().hex}.html"
    with open(temp_file, "w") as file:
        file.write(f'<html><head><meta http-equiv="refresh" content="0; url={link}" /></head><body></body></html>')

    # Check if Chrome exists
    if not os.path.exists(chrome_path):
        print("Chrome executable not found. Please check the path.")
    else:
        try:
            subprocess.Popen([chrome_path, os.path.abspath(temp_file)])
            print(f"Opening link in Chrome via: {temp_file}")
        except Exception as e:
            print(f"Failed to open the link: {e}")


#%%

def make_pre_post_statebuilder(
    client: CAVEclient,
    show_inputs: bool = False,
    show_outputs: bool = False,
    contrast: list = None,
    view_kws: dict = None,
    point_column="ctr_pt_position",
    pre_pt_root_id_col="pre_pt_root_id",
    post_pt_root_id_col="post_pt_root_id",
    dataframe_resolution_pre=None,
    dataframe_resolution_post=None,
    input_layer_name="syns_in",
    output_layer_name="syns_out",
    input_layer_color='cyan',
    output_layer_color='red',
    split_positions=False,
):

    img_layer, seg_layer = from_client(client, contrast=contrast)
    seg_layer.add_selection_map(selected_ids_column="root_id")

    if view_kws is None:
        view_kws = {}
    sb1 = StateBuilder(layers=[img_layer, seg_layer], client=client, view_kws=view_kws)

    state_builders = [sb1]
    if show_inputs:
        # First state builder
        input_point_mapper = PointMapper(
            point_column=point_column,
            linked_segmentation_column=pre_pt_root_id_col,
            split_positions=split_positions,
        )
        inputs_lay = AnnotationLayerConfig(
            input_layer_name,
            mapping_rules=[input_point_mapper],
            linked_segmentation_layer=seg_layer.name,
            data_resolution=dataframe_resolution_post,
            color=input_layer_color,
        )
        sb_in = StateBuilder([inputs_lay], client=client)
        state_builders.append(sb_in)
    if show_outputs:
        output_point_mapper = PointMapper(
            point_column=point_column,
            linked_segmentation_column=post_pt_root_id_col,
            split_positions=split_positions,
        )
        outputs_lay = AnnotationLayerConfig(
            output_layer_name,
            mapping_rules=[output_point_mapper],
            linked_segmentation_layer=seg_layer.name,
            data_resolution=dataframe_resolution_pre,
            color=output_layer_color,
        )
        sb_out = StateBuilder([outputs_lay], client=client)
        state_builders.append(sb_out)
    return state_builders

def make_neuron_neuroglancer_link(
    client: CAVEclient,
    root_ids,
    return_as="html",
    shorten="always",
    show_inputs=False,
    show_outputs=False,
    sort_inputs=True,
    sort_outputs=True,
    sort_ascending=False,
    input_color='cyan',
    output_color='red',
    contrast=None,
    timestamp=None,
    view_kws=None,
    point_column="ctr_pt_position",
    pre_pt_root_id_col="pre_pt_root_id",
    post_pt_root_id_col="post_pt_root_id",
    input_layer_name="syns_in",
    output_layer_name="syns_out",
    ngl_url=None,
    link_text="Neuroglancer Link",
    target_site=None,
):

    if not isinstance(root_ids, Iterable):
        root_ids = [root_ids]
    df1 = pd.DataFrame({"root_id": root_ids})
    dataframes = [df1]
    data_resolution_pre = None
    data_resolution_post = None
    if show_inputs:
        syn_in_df = client.materialize.synapse_query(
            post_ids=root_ids,
            timestamp=timestamp,
            desired_resolution=client.info.viewer_resolution(),
            split_positions=True,
        )
        data_resolution_pre = syn_in_df.attrs["dataframe_resolution"]
        if sort_inputs:
            syn_in_df = sort_dataframe_by_root_id(
                syn_in_df, pre_pt_root_id_col, ascending=sort_ascending, drop=True
            )
        dataframes.append(syn_in_df)
    if show_outputs:
        syn_out_df = client.materialize.synapse_query(
            pre_ids=root_ids,
            timestamp=timestamp,
            desired_resolution=client.info.viewer_resolution(),
            split_positions=True,
        )
        data_resolution_post = syn_out_df.attrs["dataframe_resolution"]
        if sort_outputs:
            syn_out_df = sort_dataframe_by_root_id(
                syn_out_df, post_pt_root_id_col, ascending=sort_ascending, drop=True
            )
        dataframes.append(syn_out_df)
    sb = make_pre_post_statebuilder(
        client,
        show_inputs=show_inputs,
        show_outputs=show_outputs,
        contrast=contrast,
        point_column=point_column,
        view_kws=view_kws,
        pre_pt_root_id_col=pre_pt_root_id_col,
        post_pt_root_id_col=post_pt_root_id_col,
        input_layer_name=input_layer_name,
        output_layer_name=output_layer_name,
        input_layer_color=input_color,
        output_layer_color=output_color,
        dataframe_resolution_pre=data_resolution_pre,
        dataframe_resolution_post=data_resolution_post,
        split_positions=True,
    )
    return sb

def make_point_statebuilder(
    client: CAVEclient,
    point_column="pre_pt_position",
    linked_seg_column="pre_pt_root_id",
    data_resolution=None,
    group_column=None,
    tag_column=None,
    description_column=None,
    contrast=None,
    view_kws=None,
    point_layer_name="pts",
    color=None,
    split_positions=False,
):

    img_layer, seg_layer = from_client(client, contrast=contrast)
    point_mapper = PointMapper(
        point_column=point_column,
        linked_segmentation_column=linked_seg_column,
        tag_column=tag_column,
        description_column=description_column,
        group_column=group_column,
        split_positions=split_positions,
    )
    ann_layer = AnnotationLayerConfig(
        point_layer_name,
        mapping_rules=[point_mapper],
        linked_segmentation_layer=seg_layer.name,
        data_resolution=data_resolution,
        color=color,
    )
    if view_kws is None:
        view_kws = {}
    return StateBuilder(
        [img_layer, seg_layer, ann_layer],
        client=client,
        view_kws=view_kws,)


def make_synapse_neuroglancer_link(
    synapse_df,
    client: CAVEclient,
    return_as="html",
    shorten="always",
    contrast=None,
    point_column="ctr_pt_position",
    dataframe_resolution=None,
    group_connections=True,
    link_pre_and_post=True,
    ngl_url=None,
    view_kws=None,
    pre_post_columns=None,
    neuroglancer_link_text="Neuroglancer Link",
    color=None,
    split_positions=False,
    target_site=None,
):

    if point_column not in synapse_df.columns:
        raise ValueError(f"point_column={point_column} not in dataframe")
    if pre_post_columns is None:
        pre_post_columns = ["pre_pt_root_id", "post_pt_root_id"]
    if dataframe_resolution is None:
        dataframe_resolution = synapse_df.attrs.get("dataframe_resolution", None)

    if group_connections:
        group_column = pre_post_columns
    else:
        group_column = None
    if link_pre_and_post:
        linked_columns = pre_post_columns
    else:
        linked_columns = None
    c_v=color_synapses_inverse_pre_post(color)
    point_layer_name=c_v
    sb = make_point_statebuilder(
        client,
        point_column=point_column,
        linked_seg_column=linked_columns,
        group_column=group_column,
        data_resolution=dataframe_resolution,
        contrast=contrast,
        view_kws=view_kws,
        point_layer_name=point_layer_name,
        color=color,
        split_positions=split_positions,
    )
    return  sb

def make_synapse_neuroglancer_link_layer_name(
    synapse_df,
    client: CAVEclient,
    return_as="html",
    shorten="always",
    contrast=None,
    point_column="ctr_pt_position",
    dataframe_resolution=None,
    group_connections=True,
    link_pre_and_post=True,
    ngl_url=None,
    view_kws=None,
    pre_post_columns=None,
    neuroglancer_link_text="Neuroglancer Link",
    color=None,
    split_positions=False,
    target_site=None,point_layer_name=None
):

    if point_column not in synapse_df.columns:
        raise ValueError(f"point_column={point_column} not in dataframe")
    if pre_post_columns is None:
        pre_post_columns = ["pre_pt_root_id", "post_pt_root_id"]
    if dataframe_resolution is None:
        dataframe_resolution = synapse_df.attrs.get("dataframe_resolution", None)

    if group_connections:
        group_column = pre_post_columns
    else:
        group_column = None
    if link_pre_and_post:
        linked_columns = pre_post_columns
    else:
        linked_columns = None

    sb = make_point_statebuilder(
        client,
        point_column=point_column,
        linked_seg_column=linked_columns,
        group_column=group_column,
        data_resolution=dataframe_resolution,
        contrast=contrast,
        view_kws=view_kws,
        point_layer_name=point_layer_name,
        color=color,
        split_positions=split_positions,
    )
    return  sb
#%%neuroglancer base
def process_syndf_to_visual(synapse_table):
    synapse_table=synapse_table.copy()
    
        
    synapse_table.loc[:, 'pre_pt_position'] = synapse_table.apply(lambda row: np.array([row['pre_x']/4, row['pre_y']/4, row['pre_z']/40]), axis=1)
    synapse_table.loc[:, 'post_pt_position'] = synapse_table.apply(lambda row: np.array([row['post_x']/4, row['post_y']/4, row['post_z']/40]), axis=1)
    synapse_table.loc[:, 'pre_pt_root_id'] = synapse_table['pre']
    synapse_table.loc[:, 'post_pt_root_id'] = synapse_table['post']
    return synapse_table


def color_synapses_inverse_syn_type(color):
    if color == 'blue':
        return 'AA'
    elif color == 'yellow':
        return 'AD'
    elif color == 'green':
        return 'DD'
    elif color == 'red':
        return 'DA'
    elif color == 'black':
        return None  # Assuming 'black' does not correspond to a unique synapse type in the original function
    else:
        return None  # For colors that are not valid inputs

def color_synapses_inverse_pre_post(color):
    if color == 'blue':
        return 'post'
    elif color == 'yellow':
        return 'AA'
    elif color == 'green':
        return 'DD'
    elif color == 'red':
        return 'pre'
    elif color == 'black':
        return None  # Assuming 'black' does not correspond to a unique synapse type in the original function
    else:
        return None  # For colors that are not valid inputs


#%%ng practical


def shorthen_and_open_links(link):
    # Create a unique temporary HTML file for each link
    temp_file = f"temp_link_{uuid.uuid4().hex}.html"
    with open(temp_file, "w") as file:
        file.write(f'<html><head><meta http-equiv="refresh" content="0; url={link}" /></head><body></body></html>')
    
    # Check if Edge exists
    if not os.path.exists(edge_path):
        print("Edge executable not found. Please check the path.")
    else:
        # Open the temporary file in Edge
        try:
            subprocess.Popen([edge_path, os.path.abspath(temp_file)])
            print(f"Opening link via temporary HTML file: {temp_file}")
        except Exception as e:
            print(f"Failed to open the link: {e}")
            
#%%
def create_links_two_neurons_shared_synapses(pre,post,syn_df):
    client = CAVEclient('flywire_fafb_production')

    all_urls=[]


    # First, ensure dhi is a copy if you intend to modify it without affecting the original DataFrame
    presynapses = syn_df[(syn_df['pre'] == pre)&(syn_df['post'] == post)].copy()
    postsynapses = syn_df[(syn_df['pre'] == post)&(syn_df['post'] == pre)].copy()


    presynapses=process_syndf_to_visual(presynapses)
    postsynapses=process_syndf_to_visual(postsynapses)

    presynapses['color']='red'
    postsynapses['color']='cyan'
    
    
    rsyn=presynapses
    bsyn=postsynapses

    
    sbr=make_synapse_neuroglancer_link(
            rsyn, client,
            point_column='pre_pt_position',
            link_pre_and_post=True, return_as="url", color='red'
            );
    
    
    sbb=make_synapse_neuroglancer_link(
            bsyn, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url", color='cyan'
            );
    nlist=[pre,post]

    sbn=make_neuron_neuroglancer_link(root_ids=nlist,return_as="url",client=client)[0]
    
    root_ids=nlist
        
    df3 = pd.DataFrame({"root_id": nlist})

    ab=package_state(
        bsyn,
        sbb,
        client,   
        return_as= "url",
        shorten="always",
     
        link_text='nglink')
    
    ar=package_state(
        rsyn,
        sbr,
        client,   
        return_as= "url",
        shorten="always",
     
        link_text='nglink')
   
    # Assuming sbr and sbb are your state builders for red and blue synapses respectively
    state_builders = [sbr, sbb,sbn]
    # Create a ChainedStateBuilder with your state builders
    chained_builder = ChainedStateBuilder(state_builders)
    
    # Assuming rsyn and bsyn are your dataframes for red and blue synapses respectively
    data_list = [rsyn, bsyn,df3]
    
    # Render the combined state as a URL
    combined_url = chained_builder.render_state(
        data_list=data_list,
        return_as="url",
        target_site="seunglab"  # or whatever your target site is
    )
    
    #combined_url2='https://neuroglancer.neuvue.io'+combined_url[0][22:]
    all_urls.append(combined_url)
#all_urls_mod=['https://neuroglancer.neuvue.io'+i[0][22:] for i in all_urls]
    links_='https://neuroglancer.neuvue.io'+all_urls[0][22:]

    return all_urls
    
'''

gg=create_links_two_neurons_shared_synapses(720575940639628870,720575940623052981,allsynapses)
shorthen_and_open_links(gg[0])

'''

#%%
def create_links_many_neurons(nlist,syn_df):
    client = CAVEclient('flywire_fafb_production')

    all_urls=[]


    # First, ensure dhi is a copy if you intend to modify it without affecting the original DataFrame
    #presynapses = syn_df[(syn_df['pre'].isin(nlist))].copy().sample(1000)
    #postsynapses = syn_df[(syn_df['post'].isin(nlist))].copy().sample(1000)
    presynapses = syn_df[(syn_df['pre'].isin(nlist))].copy()
    postsynapses = syn_df[(syn_df['post'].isin(nlist))].copy()
    presynapses['color']='red'
    postsynapses['color']='cyan'
    presynapses=process_syndf_to_visual(presynapses)
    postsynapses=process_syndf_to_visual(postsynapses)

    
    rsyn=presynapses
    bsyn=postsynapses

    
    sbr=make_synapse_neuroglancer_link(
            rsyn, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url", color='red'
            );
    
    
    sbb=make_synapse_neuroglancer_link(
            bsyn, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url", color='cyan'
            );
    
    sbn=make_neuron_neuroglancer_link(root_ids=nlist,return_as="url",client=client)[0]
    


    

    root_ids=nlist
    
    
    df3 = pd.DataFrame({"root_id": nlist})
    '''
    
    ab=package_state(
        bsyn,
        sbb,
        client,   
        return_as= "url",
        shorten="always",
     
        link_text='nglink')
    
    ar=package_state(
        rsyn,
        sbr,
        client,   
        return_as= "url",
        shorten="always",
     
        link_text='nglink')
   
    
    
    '''
    
    # Assuming sbr and sbb are your state builders for red and blue synapses respectively
    state_builders = [sbr, sbb,sbn]
    # Create a ChainedStateBuilder with your state builders
    chained_builder = ChainedStateBuilder(state_builders)
    
    # Assuming rsyn and bsyn are your dataframes for red and blue synapses respectively
    data_list = [rsyn, bsyn,df3]
    
    # Render the combined state as a URL
    combined_url = chained_builder.render_state(
        data_list=data_list,
        return_as="url",
        target_site="seunglab"  # or whatever your target site is
    )
    
    #combined_url2='https://neuroglancer.neuvue.io'+combined_url[0][22:]
    all_urls.append(combined_url)
#all_urls_mod=['https://neuroglancer.neuvue.io'+i[0][22:] for i in all_urls]
    links_='https://neuroglancer.neuvue.io'+all_urls[0][22:]

    return all_urls
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\post_processing_data\article\synapses_783_article_princeton.ftr')
#%%
'''
gg=create_links_many_neurons([720575940625308162],allsynapses.sample(0))
shorthen_and_open_links(gg[0])
      ''' 
       #%%
def safe_sample(df, n=200):
    return df.sample(n=min(len(df), n), random_state=42)

def create_links_two_neurons_and_shared_synapses2(nlist,syn_df,sample=200):
    client = CAVEclient('flywire_fafb_production')
    pre=nlist[0]
    post=nlist[1]
    all_urls=[]

    
    # First, ensure dhi is a copy if you intend to modify it without affecting the original DataFrame
    #presynapses = syn_df[(syn_df['pre'].isin(nlist))].copy().sample(1000)
    #postsynapses = syn_df[(syn_df['post'].isin(nlist))].copy().sample(1000)
    presynapses_1 = safe_sample(syn_df[syn_df['pre'] == nlist[0]].copy(),n=sample)
    postsynapses_1 = safe_sample(syn_df[syn_df['post'] == nlist[0]].copy(),n=sample)
    presynapses_2 = safe_sample(syn_df[syn_df['pre'] == nlist[1]].copy(),n=sample)
    postsynapses_2 = safe_sample(syn_df[syn_df['post'] == nlist[1]].copy(),n=sample)

    presynapses_1['color']='red'
    postsynapses_1['color']='cyan'
    presynapses_1=process_syndf_to_visual(presynapses_1)
    postsynapses_1=process_syndf_to_visual(postsynapses_1)

    presynapses_2['color']='red'
    postsynapses_2['color']='cyan'
    presynapses_2=process_syndf_to_visual(presynapses_2)
    postsynapses_2=process_syndf_to_visual(postsynapses_2)
    
    rsyn_1=presynapses_1
    bsyn_1=postsynapses_1
    rsyn_2=presynapses_2
    bsyn_2=postsynapses_2
    presynapses_shared = syn_df[(syn_df['pre'] == pre)&(syn_df['post'] == post)].copy()
    postsynapses_shared = syn_df[(syn_df['pre'] == post)&(syn_df['post'] == pre)].copy()

    presynapses_shared=process_syndf_to_visual(presynapses_shared)
    postsynapses_shared=process_syndf_to_visual(postsynapses_shared)
    presynapses_shared['color']='red'
    postsynapses_shared['color']='cyan'
    
    rsyn_shared=presynapses_shared
    bsyn_shared=postsynapses_shared
    
    sbr_shared=make_synapse_neuroglancer_link_layer_name(
            rsyn_shared, client,
            point_column='pre_pt_position',
            link_pre_and_post=True, return_as="url", color='red',point_layer_name='shared_synapses_pre_synapse_neuron_A'
            );
    
    
    sbb_shared=make_synapse_neuroglancer_link_layer_name(
            bsyn_shared, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url", color='cyan',point_layer_name='shared_synapses_post_synapse_neuron_A'
            );
    
    
    sbr_1=make_synapse_neuroglancer_link_layer_name(
            rsyn_1, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url", color='red',point_layer_name='pre_synapses_n1'
            );
    
    
    sbb_1=make_synapse_neuroglancer_link_layer_name(
            bsyn_1, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url", color='cyan',point_layer_name='post_synapses_n1'
            );
    sbr_2=make_synapse_neuroglancer_link_layer_name(
            rsyn_2, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url", color='red',point_layer_name='pre_synapses_n2'
            );
    
    
    sbb_2=make_synapse_neuroglancer_link_layer_name(
            bsyn_2, client,
            point_column='post_pt_position',
            link_pre_and_post=True, return_as="url", color='cyan',point_layer_name='post_synapses_n2'
            );
    
    sbn=make_neuron_neuroglancer_link(root_ids=nlist,return_as="url",client=client)[0]
    


    

    root_ids=nlist
    
    
    df3 = pd.DataFrame({"root_id": nlist})
    
    
    
    # Assuming sbr and sbb are your state builders for red and blue synapses respectively
    state_builders = [sbr_1, sbb_1,sbr_shared,sbb_shared,sbr_2, sbb_2,sbn]
    # Create a ChainedStateBuilder with your state builders
    chained_builder = ChainedStateBuilder(state_builders)
    
    # Assuming rsyn and bsyn are your dataframes for red and blue synapses respectively
    data_list = [rsyn_1, bsyn_1,rsyn_shared,bsyn_shared,rsyn_2, bsyn_2,df3]
    
    # Render the combined state as a URL
    combined_url = chained_builder.render_state(
        data_list=data_list,
        return_as="url",
        target_site="seunglab"  # or whatever your target site is
    )
    
    #combined_url2='https://neuroglancer.neuvue.io'+combined_url[0][22:]
    all_urls.append(combined_url)
#all_urls_mod=['https://neuroglancer.neuvue.io'+i[0][22:] for i in all_urls]
    links_='https://neuroglancer.neuvue.io'+all_urls[0][22:]

    return all_urls


'''

gg=create_links_two_neurons_and_shared_synapses2([720575940608199748,720575940611514261],allsynapses,sample=200)
shorthen_and_open_links(gg[0])


'''
              
#%%

#%%neurons



def upload_swc(id_):
    
    input_base_dir = r"C:\Users\user\organised_work\data\783\recieved\swc\783"
    
    id_=str(id_)+'.swc'
    # Get the list of subdirectories in the input directory
    
    input_subfolders = {f.name for f in os.scandir(input_base_dir) if f.is_dir()}
    
    for folder_name in input_subfolders:
        input_folder = os.path.join(input_base_dir, folder_name)
        swc_path = os.path.join(input_folder, id_)
        
        try:
            swc = navis.read_swc(swc_path)
            #print(folder_name)
            #allsynapses2 = allsynapses[(allsynapses['pre'] == int(id_[:-4])) | (allsynapses['post'] == int(id_[:-4]))]

            return swc
        except:
            continue
'''
swc=upload_swc(720575940608404747)
'''


def attach_synapses(x,syn, min_score=30,neuropils=False,batch_size=100, progress=True):
   
   
    # Parse root IDs


        # Drop below threshold connections
        if min_score:
            syn = syn[syn.cleft_score >= min_score]


        if isinstance(x, navis.core.BaseNeuron):
            x = navis.NeuronList([x])

        if isinstance(x, navis.NeuronList):
            for n in x:
                presyn = postsyn = pd.DataFrame([])
                add_cols = ['neuropil'] if neuropils else []
                cols = ['pre_x', 'pre_y', 'pre_z',
                            'cleft_score', 'post'] + add_cols
                presyn = syn.loc[syn.pre == np.int64(n.id), cols
                                     ].rename({'pre_x': 'x',
                                               'pre_y': 'y',
                                               'pre_z': 'z',
                                               'post': 'partner_id'},
                                              axis=1)
                presyn['type'] = 'pre'
                cols = ['post_x', 'post_y', 'post_z',
                            'cleft_score', 'pre'] + add_cols
                postsyn = syn.loc[syn.post == np.int64(n.id), cols
                                      ].rename({'post_x': 'x',
                                                'post_y': 'y',
                                                'post_z': 'z',
                                                'pre': 'partner_id'},
                                               axis=1)
                postsyn['type'] = 'post'

                connectors = pd.concat((presyn, postsyn), axis=0, ignore_index=True)

                # Turn type column into categorical to save memory
                connectors['type'] = connectors['type'].astype('category')

                # If TreeNeuron, map each synapse to a node
                if isinstance(n, navis.TreeNeuron):
                    tree = navis.neuron2KDTree(n, data='nodes')
                    dist, ix = tree.query(connectors[['x', 'y', 'z']].values)

                    too_far = dist > 10_000
                    if any(too_far):
                        connectors = connectors[~too_far].copy()
                        ix = ix[~too_far]

                    connectors['node_id'] = n.nodes.node_id.values[ix]

                    # Add an ID column for navis' sake
                    connectors.insert(0, 'connector_id', np.arange(connectors.shape[0]))

                n.connectors = connectors


        return x



def attach_synapses_princeton(x,syn,neuropils=False,batch_size=100, progress=True):
   
   
        if isinstance(x, navis.core.BaseNeuron):
            x = navis.NeuronList([x])

        if isinstance(x, navis.NeuronList):
            for n in x:
                presyn = postsyn = pd.DataFrame([])
                add_cols = ['neuropil'] if neuropils else []
                cols = ['pre_x', 'pre_y', 'pre_z',
                            'post','synapse_id','size'] + add_cols
                presyn = syn.loc[syn.pre == np.int64(n.id), cols
                                     ].rename({'pre_x': 'x',
                                               'pre_y': 'y',
                                               'pre_z': 'z',
                                               'post': 'partner_id'},
                                              axis=1)
                presyn['type'] = 'pre'
                cols = ['post_x', 'post_y', 'post_z',
                            'pre','synapse_id','size'] + add_cols
                postsyn = syn.loc[syn.post == np.int64(n.id), cols
                                      ].rename({'post_x': 'x',
                                                'post_y': 'y',
                                                'post_z': 'z',
                                                'pre': 'partner_id'},
                                               axis=1)
                postsyn['type'] = 'post'

                connectors = pd.concat((presyn, postsyn), axis=0, ignore_index=True)

                # Turn type column into categorical to save memory
                connectors['type'] = connectors['type'].astype('category')

                # If TreeNeuron, map each synapse to a node
                if isinstance(n, navis.TreeNeuron):
                    tree = navis.neuron2KDTree(n, data='nodes')
                    dist, ix = tree.query(connectors[['x', 'y', 'z']].values)

                    too_far = dist > 10_000
                    if any(too_far):
                        connectors = connectors[~too_far].copy()
                        ix = ix[~too_far]

                    connectors['node_id'] = n.nodes.node_id.values[ix]

                    # Add an ID column for navis' sake
                    connectors.insert(0, 'connector_id', np.arange(connectors.shape[0]))

                n.connectors = connectors


        return x


def attach_synapses_princeton(x,syn,neuropils=False,batch_size=100, progress=True):
   
   
        if isinstance(x, navis.core.BaseNeuron):
            x = navis.NeuronList([x])

        if isinstance(x, navis.NeuronList):
            for n in x:
                presyn = postsyn = pd.DataFrame([])
                add_cols = ['neuropil'] if neuropils else []
                cols = ['pre_x', 'pre_y', 'pre_z',
                            'post','synapse_id','size'] + add_cols
                presyn = syn.loc[syn.pre == np.int64(n.id), cols
                                     ].rename({'pre_x': 'x',
                                               'pre_y': 'y',
                                               'pre_z': 'z',
                                               'post': 'partner_id'},
                                              axis=1)
                presyn['type'] = 'pre'
                cols = ['post_x', 'post_y', 'post_z',
                            'pre','synapse_id','size'] + add_cols
                postsyn = syn.loc[syn.post == np.int64(n.id), cols
                                      ].rename({'post_x': 'x',
                                                'post_y': 'y',
                                                'post_z': 'z',
                                                'pre': 'partner_id'},
                                               axis=1)
                postsyn['type'] = 'post'

                connectors = pd.concat((presyn, postsyn), axis=0, ignore_index=True)

                # Turn type column into categorical to save memory
                connectors['type'] = connectors['type'].astype('category')

                # If TreeNeuron, map each synapse to a node
                if isinstance(n, navis.TreeNeuron):
                    tree = navis.neuron2KDTree(n, data='nodes')
                    dist, ix = tree.query(connectors[['x', 'y', 'z']].values)

                    too_far = dist > 10_000
                    if any(too_far):
                        connectors = connectors[~too_far].copy()
                        ix = ix[~too_far]

                    connectors['node_id'] = n.nodes.node_id.values[ix]

                    # Add an ID column for navis' sake
                    connectors.insert(0, 'connector_id', np.arange(connectors.shape[0]))

                n.connectors = connectors


        return x
    

def attach_synapses_princeton_nonprocess(x,syn,neuropils=False,batch_size=100, progress=True):
   
   
        if isinstance(x, navis.core.BaseNeuron):
            x = navis.NeuronList([x])

        if isinstance(x, navis.NeuronList):
            for n in x:
                presyn = postsyn = pd.DataFrame([])
                add_cols = ['neuropil'] if neuropils else []
                cols = ['pre_x', 'pre_y', 'pre_z',
                            'post'] + add_cols
                presyn = syn.loc[syn.pre == np.int64(n.id), cols
                                     ].rename({'pre_x': 'x',
                                               'pre_y': 'y',
                                               'pre_z': 'z',
                                               'post': 'partner_id'},
                                              axis=1)
                presyn['type'] = 'pre'
                cols = ['post_x', 'post_y', 'post_z',
                            'pre'] + add_cols
                postsyn = syn.loc[syn.post == np.int64(n.id), cols
                                      ].rename({'post_x': 'x',
                                                'post_y': 'y',
                                                'post_z': 'z',
                                                'pre': 'partner_id'},
                                               axis=1)
                postsyn['type'] = 'post'

                connectors = pd.concat((presyn, postsyn), axis=0, ignore_index=True)

                # Turn type column into categorical to save memory
                connectors['type'] = connectors['type'].astype('category')

                # If TreeNeuron, map each synapse to a node
                if isinstance(n, navis.TreeNeuron):
                    tree = navis.neuron2KDTree(n, data='nodes')
                    dist, ix = tree.query(connectors[['x', 'y', 'z']].values)

                    too_far = dist > 10_000
                    if any(too_far):
                        connectors = connectors[~too_far].copy()
                        ix = ix[~too_far]

                    connectors['node_id'] = n.nodes.node_id.values[ix]

                    # Add an ID column for navis' sake
                    connectors.insert(0, 'connector_id', np.arange(connectors.shape[0]))

                n.connectors = connectors


        return x
def get_split(item,flow_thresh=1):
    try:    
        split=navis.split_axon_dendrite(item, metric='synapse_flow_centrality',reroot_soma=True,flow_thresh=flow_thresh)
        return split
    except:
        return 'split issue'
    
    
    
def heal_attach(item,allsynapses):
    try:
        healed_neurons=navis.heal_skeleton(item)
        healed_neurons_att_syn=attach_synapses(healed_neurons,allsynapses)
        return healed_neurons_att_syn
    except:
        return 'heal or attach issue'



'''
id_=720575940601603185
swc1=upload_swc(id_)
allsynapses2=allsynapses[(allsynapses['pre']==n1_id)|(allsynapses['post']==n1_id)]
swc1=heal_attach(swc1,allsynapses2)

'''
def load_swc_from_folder(directory):
    swc_list=[]
    i=0
    for filename in os.listdir(directory):

        if filename.endswith(".swc"):  # Check if the file is an SWC file
            full_path = os.path.join(directory, filename)
            swc = navis.read_swc(full_path)  # Load the SWC file
            swc_list.append(swc)
    return swc_list


'''
folder = r"C:\\Users\\user\\organised_work\\data\\783\\recieved\\swc\\783\\8"
swc_list = load_swc_from_folder(folder)
'''


def parents_check_on_comp(compartment):
        
    dend=compartment
    dn=dend.nodes
    parents=dend.nodes[dend.nodes['parent_id']==-1]
    if len(parents)==1:
        return True
    else:
        return False



#%%
def extract_subgraph(G, root_node):
    # Perform a breadth-first search (BFS) or depth-first search (DFS) from the root node
    nodes_in_subgraph = nx.descendants(G, root_node)
    nodes_in_subgraph.add(root_node)  # Include the root node itself
    return nodes_in_subgraph

def create_subgraphs_from_split(compartment):
    try:
        subgraphs=[]
    
        neuron_df=compartment.nodes
        # Get root nodes (where parent_id is -1)
        edges = neuron_df[neuron_df['parent_id'] != -1][['parent_id', 'node_id']].values
        G = nx.DiGraph()
        
        G.add_edges_from(edges)
        
        # Find all root nodes (where parent_id = -1)
        root_nodes = neuron_df[neuron_df['parent_id'] == -1]['node_id'].tolist()
        
        # List to store DataFrames for each subgraph
        subgraph_dfs = []
        
        # Function to extract a subgraph starting from a root node

        # Iterate over each root node and extract the corresponding subgraph
        for root in root_nodes:
            try:
                subgraph_nodes = extract_subgraph(G, root)
                subgraph_df = neuron_df[neuron_df['node_id'].isin(subgraph_nodes)].copy()
                subgraph_dfs.append(subgraph_df)
            except Exception as e:
                print ('a root was not found')
                subgraphs.append(pd.DataFrame(columns=['node_id',"parent_id"]))
        # Optional: Inspect the results
        for i, sub_df in enumerate(subgraph_dfs):
            subgraphs.append(sub_df)
        return subgraphs,root_nodes
    except Exception as e:
        print('error in creating subgraphs')

        return e
    
  
    

def get_synapses_from_subgraph(compartment):
    try:

        subgraph_count=[]
        extracted_subgraph_results=[]
        extracted_subgraphs_r=  create_subgraphs_from_split(compartment)
        extracted_subgraphs=extracted_subgraphs_r[0]
        extracted_subgraphs_roots=extracted_subgraphs_r[1]
        for subgraph in extracted_subgraphs:
            df_s1=subgraph
            conn=compartment.connectors
            synapses_value=conn.merge(df_s1,on='node_id',how='right').dropna()
            synapses_count=len(synapses_value)
            extracted_subgraph_results.append(synapses_count)
            
        return extracted_subgraph_results,extracted_subgraphs_roots
            
            
    except Exception as e:
        print('error in getting synapses')
        return e
    
    
#%%
def get_synapses(nid,syn_df,pos):
    return syn_df[syn_df[pos]==nid] 

def get_points(syn_df,pos):
    return syn_df[[pos,pos+'_x',pos+'_y',pos+'_z']]

def turn_to_tuple(points_df):
    points_df=points_df.copy()
    points_df.columns=['neuron','x','y','z']
    points_df=points_df.drop(columns=['neuron'])
    
    
    points_df['xyz'] = points_df.apply(lambda row: (int(row['x']/4), int(row['y']/4), int(row['z']/40)), axis=1)

    return points_df['xyz']

def get_shared_synapses_directed(pre_neuron,post_neuron,syn_df):
    shared_syn=syn_df[(syn_df['pre']==pre_neuron)&(syn_df['post']==post_neuron)]
    return shared_syn

def get_shared_synapses_undirected(pre_neuron,post_neuron,syn_df):
    shared_syn_dir1=syn_df[(syn_df['pre']==pre_neuron)&(syn_df['post']==post_neuron)]
    shared_syn_dir2=syn_df[(syn_df['post']==pre_neuron)&(syn_df['pre']==post_neuron)]
    shared_syn_com=pd.concat([shared_syn_dir1,shared_syn_dir2])
    
    return shared_syn_com
def divide_to_syn_type(syn_df,pos):
    AD=syn_df[syn_df['syn_type']=='AD']
    AD=AD[[pos,pos+'_x',pos+'_y',pos+'_z']]
    
    AA=syn_df[syn_df['syn_type']=='AA']
    AA=AA[[pos,pos+'_x',pos+'_y',pos+'_z']]

    DD=syn_df[syn_df['syn_type']=='DD']
    DD=DD[[pos,pos+'_x',pos+'_y',pos+'_z']]

    DA=syn_df[syn_df['syn_type']=='DA']
    DA=DA[[pos,pos+'_x',pos+'_y',pos+'_z']]


    return AD,AA,DD,DA


def divide_to_syn_type(syn_df,pos):
    AD=syn_df[syn_df['comp']=='AD']
    AD=AD[[pos,pos+'_x',pos+'_y',pos+'_z']]
    
    AA=syn_df[syn_df['comp']=='AA']
    AA=AA[[pos,pos+'_x',pos+'_y',pos+'_z']]

    DD=syn_df[syn_df['comp']=='DD']
    DD=DD[[pos,pos+'_x',pos+'_y',pos+'_z']]

    DA=syn_df[syn_df['comp']=='DA']
    DA=DA[[pos,pos+'_x',pos+'_y',pos+'_z']]


    return AD,AA,DD,DA
#%%
import pandas as pd
import copy

import pandas as pd
import copy
import os

def spheres(id_, syn_df):
    save_dir = r'C:\Users\user\organised_work\data\783\article\examples\spheres'
    
    # Make sure the save folder exists
    os.makedirs(save_dir, exist_ok=True)
    try:
        # Pre-synaptic
        a = get_synapses(id_, syn_df, 'pre')
        b = get_points(a, 'pre')
        cc = turn_to_tuple(b)
        cc = pd.DataFrame(cc)
    
        # Set column 'Coordinate 1'
        cc.columns = ['Coordinate 1']
    
        # Add the missing columns
        cc['Coordinate 2'] = ''
        cc['Ellipsoid Dimensions'] = '200 Ã— 200 Ã— 20'
        cc['Tags'] = ''
        cc['Description'] = ''
        cc['Segment IDs'] = ''
        cc['Parent ID'] = ''
        cc['Type'] = 'Ellipsoid'
        cc['ID'] = ''
    
        # Save pre
        pre_path = os.path.join(save_dir, f'{id_}_pre.csv')
        cc.to_csv(pre_path, index=False, encoding='cp1252')
    except:
        print('no pre synapses')
    # Post-synaptic
    a = get_synapses(id_, syn_df, 'post')
    b = get_points(a, 'post')
    cc = turn_to_tuple(b)
    cc = pd.DataFrame(cc)

    # Set column 'Coordinate 1'
    cc.columns = ['Coordinate 1']

    # Add the missing columns
    cc['Coordinate 2'] = ''
    cc['Ellipsoid Dimensions'] = '200 Ã— 200 Ã— 20'
    cc['Tags'] = ''
    cc['Description'] = ''
    cc['Segment IDs'] = ''
    cc['Parent ID'] = ''
    cc['Type'] = 'Ellipsoid'
    cc['ID'] = ''

    # Save post
    post_path = os.path.join(save_dir, f'{id_}_post.csv')
    cc.to_csv(post_path, index=False, encoding='cp1252')
    
#allsynapses=pd.read_feather(r'C:\Users\user\organised_work\data\783\generated\post_processing_data\article\synapses_783_article_princeton.ftr')
#%%
'''

gg=create_links_many_neurons([720575940642312136],allsynapses.sample(0))
shorthen_and_open_links(gg[0])
      '''
#%%
'''
# Construct the filename
spheres(720575940642312136,allsynapses)
'''
#%%

import pandas as pd
import os

def shared_spheres(id_1, id_2, syn_df):
    # Base sizes
    base_x = 160
    base_y = 160
    base_z = 16

    # Scaling factors and suffixes
    scales = {
        'half': 0.5,
        'x1': 1,
        'x2': 2,
        'x4': 4
    }

    # Define the save directory
    save_dir = rf'C:\Users\user\organised_work\data\783\article\examples\spheres\{id_1}_{id_2}_shared'
    os.makedirs(save_dir, exist_ok=True)

    # Filter synapses between id_1 and id_2
    syn_df = syn_df[
        ((syn_df['pre'] == id_1) & (syn_df['post'] == id_2)) |
        ((syn_df['pre'] == id_2) & (syn_df['post'] == id_1))
    ]

    # Pre-synaptic
    pre_df = syn_df[syn_df['pre'] == id_1]
    a = get_synapses(id_1, pre_df, 'pre')
    b = get_points(a, 'pre')
    cc_pre = turn_to_tuple(b)
    cc_pre = pd.DataFrame(cc_pre)
    cc_pre.columns = ['Coordinate 1']

    # Post-synaptic
    post_df = syn_df[syn_df['post'] == id_1]
    a = get_synapses(id_1, post_df, 'post')
    b = get_points(a, 'post')
    cc_post = turn_to_tuple(b)
    cc_post = pd.DataFrame(cc_post)
    cc_post.columns = ['Coordinate 1']

    # Save for each scaling
    for label, factor in scales.items():
        # Pre
        cc = cc_pre.copy()
        cc['Coordinate 2'] = ''
        cc['Ellipsoid Dimensions'] = f'{int(base_x * factor)} Ã— {int(base_y * factor)} Ã— {int(base_z * factor)}'
        cc['Tags'] = ''
        cc['Description'] = ''
        cc['Segment IDs'] = ''
        cc['Parent ID'] = ''
        cc['Type'] = 'Ellipsoid'
        cc['ID'] = ''

        pre_path = os.path.join(save_dir, f'{id_1}_pre_{label}.csv')
        cc.to_csv(pre_path, index=False, encoding='cp1252')

        # Post
        cc = cc_post.copy()
        cc['Coordinate 2'] = ''
        cc['Ellipsoid Dimensions'] = f'{int(base_x * factor)} Ã— {int(base_y * factor)} Ã— {int(base_z * factor)}'
        cc['Tags'] = ''
        cc['Description'] = ''
        cc['Segment IDs'] = ''
        cc['Parent ID'] = ''
        cc['Type'] = 'Ellipsoid'
        cc['ID'] = ''

        post_path = os.path.join(save_dir, f'{id_1}_post_{label}.csv')
        cc.to_csv(post_path, index=False, encoding='cp1252')
        
        #%%
def heal_attach_princeton(item,allsynapses):
    try:
        healed_neurons=navis.heal_skeleton(item)
        healed_neurons_att_syn=attach_synapses_princeton(healed_neurons,allsynapses)
        return healed_neurons_att_syn
    except:
        return 'heal or attach issue'
    
    
def heal_attach_princeton_non_process(item,allsynapses):
    try:
        healed_neurons=navis.heal_skeleton(item)
        healed_neurons_att_syn=attach_synapses_princeton_nonprocess(healed_neurons,allsynapses)
        return healed_neurons_att_syn
    except:
        return 'heal or attach issue'