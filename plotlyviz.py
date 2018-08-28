from __future__ import division
from kmapper.visuals import init_color_function,   _size_node, _format_projection_statistics, _format_cluster_statistics

import igraph as ig
import numpy as np
import plotly.graph_objs as go
from ast import literal_eval
import ipywidgets as ipw

pl_jet = [[0.0, 'rgb(0, 0, 127)'],#default coloscale
          [0.1, 'rgb(0, 0, 241)'],
          [0.2, 'rgb(0, 76, 255)'],
          [0.3, 'rgb(0, 176, 255)'],
          [0.4, 'rgb(41, 255, 205)'],
          [0.5, 'rgb(124, 255, 121)'],
          [0.6, 'rgb(205, 255, 41)'],
          [0.7, 'rgb(255, 196, 0)'],
          [0.8, 'rgb(255, 103, 0)'],
          [0.9, 'rgb(241, 7, 0)'],
          [1.0, 'rgb(127, 0, 0)']]



def pl_build_histogram(data, colorscale):
    # Build histogram of data based on values of color_function

    h_min, h_max = 0, 1
    hist, bin_edges = np.histogram(data, range=(h_min, h_max), bins=10)

    bin_mids = np.mean(np.array(list(zip(bin_edges, bin_edges[1:]))), axis=1)

    histogram = []
    max_bucket_value = max(hist)
    sum_bucket_value = sum(hist)
    for bar, mid in zip(hist, bin_mids):
        height = np.floor(((bar / max_bucket_value) * 100) + 0.5)
        perc = round((bar / sum_bucket_value) * 100., 1)
        color = _map_val2color(mid, 0., 1., colorscale)

        histogram.append({
            'height': height,
            'perc': perc,
            'color': color
        })

    return histogram



def pl_graph_data_distribution(graph, color_function, colorscale):

    node_averages = []
    for node_id, member_ids in graph["nodes"].items():
        member_colors = color_function[member_ids]
        node_averages.append(np.mean(member_colors))

    histogram = pl_build_histogram(node_averages, colorscale)

    return histogram


def mapper_graph(simplicial_complex, color_function, X, X_names,
                 lens, lens_names, custom_tooltips, colorscale):

    json_dict = {"nodes": [], "links": []}
    node_id_to_num = {}
    for i, (node_id, member_ids) in enumerate(simplicial_complex["nodes"].items()):
        node_id_to_num[node_id] = i
        projection_stats, cluster_stats, member_histogram = _pl_format_tooltip(member_ids,
                                                                               custom_tooltips,
                                                                               X,
                                                                               X_names,
                                                                               lens,
                                                                               lens_names,
                                                                               color_function,
                                                                               i,
                                                                               colorscale)#node_ID
        n = {"id": i,
             "name": node_id,
             "member_ids": member_ids,
             "color": _pl_color_function(member_ids, color_function),
             "size": _size_node(member_ids),
             "cluster": cluster_stats,
             "distribution": member_histogram,
             "projection": projection_stats,
             "custom_tooltips": custom_tooltips}

        json_dict["nodes"].append(n)
    for i, (node_id, linked_node_ids) in enumerate(simplicial_complex["links"].items()):
        for linked_node_id in linked_node_ids:
            l = {"source": node_id_to_num[node_id],
                 "target": node_id_to_num[linked_node_id]}
                 #"width": _size_link_width(graph, node_id, linked_node_id)}
            json_dict["links"].append(l)

    return json_dict


def pl_format_meta(graph, custom_meta=None):
    n = [l for l in graph["nodes"].values()]
    n_unique = len(set([i for s in n for i in s]))

    if custom_meta is None:
        custom_meta = graph['meta_data']
        clusterer=custom_meta['clusterer']
        custom_meta['clusterer']=clusterer.replace('\n', '<br>')
        if 'projection' in custom_meta.keys():
            projection=custom_meta['projection']
            custom_meta['projection']=projection.replace('\n', '<br>')

    mapper_summary = {
        "custom_meta": custom_meta,
        "n_nodes": len(graph["nodes"]),
        "n_edges": sum([len(l) for l in graph["links"].values()]),
        "n_total": sum([len(l) for l in graph["nodes"].values()]),
        "n_unique": n_unique
    }

    return mapper_summary


def mapper_graph_dict_and_info(simplicial_complex,
                               color_function=None,
                               colorscale=pl_jet,
                               custom_tooltips=None,
                               custom_meta=None,
                               X=None,
                               X_names=[],
                               lens=None,
                               lens_names=[]):
    """Generate data for mapper graph visualization and annotation. Returns
       the graph dictionary in  a json representation

    Parameters
    ----------
    simplicial_complex : dict
        Simplicial complex is the output from the `map` method.

    Example
    -------

    >>> mapper_graph_dict_and_info(simplicial_complex)

    """
    if not len(simplicial_complex['nodes']) > 0:
            raise Exception("Visualize requires a mapper with more than 0 nodes")

    color_function = init_color_function(simplicial_complex, color_function)

    json_graph = mapper_graph(simplicial_complex, color_function, X, X_names,
                              lens, lens_names, custom_tooltips, colorscale=colorscale)
    colorf_distribution = pl_graph_data_distribution(simplicial_complex, color_function, colorscale)
    mapper_summary = pl_format_meta(simplicial_complex, custom_meta)

    return json_graph,  mapper_summary, colorf_distribution



def plotly_graph(kmgraph, graph_layout='kk', colorscale=pl_jet,
                 showscale=True, factor_size=2,
                 edge_linecolor='rgb(200,200,200)', edge_linewidth=0.5,
                 node_linecolor='rgb(240,240,240)', node_linewidth=0):

    # kmgraph: a dict representing the mapper graph, returned by the function plotly_visualize
    # graph_layout: an igraph layout; recommended 'kk' (kamada-kawai) or 'fr' (fruchterman-reingold)
    # factor_size: a factor for the node size
    # return the plotly traces representing the graph edges and nodes

    # define an igraph.Graph instance of n_nodes
    n_nodes = len(kmgraph['nodes'])
    if n_nodes == 0:
        raise ValueError('Your graph has 0 nodes')
    G = ig.Graph(n=n_nodes)
    links = [(e['source'], e['target']) for e in kmgraph['links']]
    G.add_edges(links)
    layt = G.layout(graph_layout)#layt is a list of listsl an inner  list contains  node
                                 #coords

    hover_text = [node['name'] for node in kmgraph['nodes']]

    color_vals = [node['color'] for node in kmgraph['nodes']]
    node_size = np.array([factor_size * node['size'] for node in kmgraph['nodes']],
                    dtype=np.int)
    Xn, Yn, Xe, Ye = _get_plotly_data(links, layt)

    edge_trace = dict(type='scatter',
                      x=Xe,
                      y=Ye,
                      mode='lines',
                      line=dict(color=edge_linecolor,
                                width=edge_linewidth),
                      hoverinfo='none')

    node_trace = dict(type='scatter',
                      x=Xn,
                      y=Yn,
                      mode='markers',
                      marker=dict(size=node_size.tolist(),
                                  color=color_vals,
                                  colorscale=colorscale,
                                  showscale=showscale,
                                  line=dict(color=node_linecolor, width=node_linewidth),
                                  colorbar=dict(thickness=20,
                                                ticklen=4)),
                      text=hover_text,
                      hoverinfo='text')

    return [edge_trace, node_trace]


def get_kmgraph_meta(mapper_summary):
    #Extract info from mapper summary to be displayed under the graph plot
    d = mapper_summary['custom_meta']
    meta = "<b>N_cubes:</b> " + str(d['n_cubes']) +\
                " <b>Perc_overlap:</b> " + str(d['perc_overlap'])
    meta+= "<br><b>Nodes:</b> " + str(mapper_summary['n_nodes']) +\
                " <b>Edges:</b> " +str(mapper_summary['n_edges']) +\
                " <b>Total samples:</b> " + str(mapper_summary['n_total']) +\
                " <b>Unique_samples:</b> " +str(mapper_summary['n_unique'])

    return meta


def plot_layout(title='TDA KMapper', width=700, height=700,
                bgcolor='rgba(20,20,20, 0.8)', annotation_text=None,
                annotation_x=0, annotation_y=-0.01, top=100, left=60, right=60, bottom=60):
    # width, height: plot window width, height
    # bgcolor: plot background color; a rgb, rgba or hex color code
    # annotation_text: meta data to be displayed
    # annotation_x, annotation_y are the coordinates of the
    # point where we insert the annotation


    pl_layout = dict(title=title,
                     font=dict(size=12),
                     showlegend=False,
                     autosize=False,
                     width=width,
                     height=height,
                     xaxis=dict(visible=False),
                     yaxis=dict(visible=False),
                     hovermode='closest',
                     plot_bgcolor=bgcolor,
                     margin=dict(t=top, b=bottom, l=left, r=right)
                     )

    if annotation_text is None:
        return pl_layout
    else:
        annotations = [dict(showarrow=False,
                            text=annotation_text,
                            xref='paper',
                            yref='paper',
                            x=annotation_x,
                            y=annotation_y,
                            align='left',
                            xanchor='left',
                            yanchor='top',
                            font=dict(size=12))]
        pl_layout.update(annotations=annotations)
        return pl_layout


def get_node_hist(node_color_distribution, title='Graph Node Distribution',
                  width=600, height=350, top=60, left=60, bottom=60, right=60,
                  bgcolor='rgb(240,240,240)'):

    text = ["{perc}%".format(**locals()) for perc in [d['perc']
            for d in node_color_distribution]]

    pl_hist = go.Bar(y=[d['height'] for d in node_color_distribution],
                     marker=dict(color=[d['color'] for d in node_color_distribution]),
                     text=text,
                     hoverinfo='y+text')

    hist_layout = dict(title=title,
                       width=width, height=height,
                       font=dict(size=12),
                       xaxis=dict(showline=True, zeroline=False, showgrid=False, showticklabels=False),
                       yaxis=dict(showline=False, gridcolor='white'),
                       bargap=0.01,
                       margin=dict(l=left, r=right, b=bottom, t=top),
                       hovermode='x',
                       plot_bgcolor=bgcolor)

    return go.FigureWidget(data=[pl_hist], layout=hist_layout)


def get_summary_fig(mapper_summary, width=600, height=900, top=60,
                    left=60, bottom=60, right=60, bgcolor='rgb(240,240,240)'):
    #Define a d figure that displays the algorithms and sklearn called functions
    text = _text_mapper_summary(mapper_summary)

    data = [dict(type='scatter',
                 x=[0,   width],
                 y=[height,   0],#height
                 mode='text',
                 text=[text, ''],
                 textposition='bottom right',
                 hoverinfo='none')]

    layout = dict(title='Algorithms and Functions',
                  width=width, height=height,
                  font=dict(size=12),
                  xaxis=dict(visible=False),
                  yaxis=dict(visible=False, range=[0, height+5]),
                  margin=dict(t=top, b=bottom, l=left, r=right),
                  plot_bgcolor=bgcolor)

    return go.FigureWidget(data=data, layout=layout)


def hovering_widgets(kmgraph, graph_fw, ctooltips=False, width_figd=400,
                     height_figd=300, top_figd=100, textbox2_width=200):

    dnode = kmgraph['nodes'][0]
    fwc = get_node_hist(dnode['distribution'], title='Cluster Member Distribution',
                        width=width_figd, height=height_figd, top=top_figd)
    textbox1 = ipw.Text(value='{:d}'.format(dnode['cluster']['size']),
                        description='Cluster size:',
                        disabled=False,
                        continuous_update=True)

    textbox1.layout = dict(margin='10px 10px 10px 10px', width='200px')

    textbox2 = ipw.Textarea(value=str(dnode['member_ids']) if not ctooltips  else str(dnode['custom_tooltips']),
                            description='Member ids:',
                            disabled=False,
                            continuous_update=True)

    textbox2.layout = dict(margin='5px 5px 5px 10px', width= str(textbox2_width)+'px')

    def do_on_hover(trace, points, state):
        if not points.point_inds:
            return
        ind = points.point_inds[0]#get the index of the hovered node
        node = kmgraph['nodes'][ind]
        #on hover do:
        with fwc.batch_update():#update data in the cluster member hist
            fwc.data[0].text  =['{:.1f}%'.format(d['perc'])  for d in node['distribution']]
            fwc.data[0].y = [d['height'] for d in node['distribution']]
            fwc.data[0].marker.color = [d['color'] for d in node['distribution']]

        textbox1.value = '{:d}'.format(node['cluster']['size'])#get the cluster size
        textbox2.value = str(node['member_ids']) if not ctooltips  else str(node['custom_tooltips'])#get the cluster member ids
    trace = graph_fw.data[1]
    trace.on_hover(do_on_hover)
    return ipw.VBox([ipw.HBox([graph_fw, fwc]), textbox1, textbox2])


def _map_val2color(val, vmin, vmax, colorscale):
    # maps a value val in [vmin, vmax] to the corresponding color in the colorscale
    # returns the rgb color code of that color

    if vmin >= vmax:
        raise ValueError('vmin should be < vmax')

    plotly_scale, plotly_colors = list(map(float, np.array(colorscale)[:,0])), np.array(colorscale)[:,1]

    colors_01 = np.array(list(map(literal_eval,[color[3:]
                     for color in plotly_colors] )))/255.#color codes in [0,1]

    v= (val - vmin) / float((vmax - vmin)) #here val is mapped to v in[0,1]

    idx = 0
    #sequential search for the two   consecutive indices idx, idx+1 such that v belongs
    #to the interval  [plotly_scale[idx], plotly_scale[idx+1]

    while(v > plotly_scale[idx+1]):
        idx+= 1
    left_scale_val = plotly_scale[idx]
    right_scale_val = plotly_scale[idx+ 1]
    vv = (v - left_scale_val) / (right_scale_val - left_scale_val)

    #get the triplet of three values in [0,1] that represent the rgb color corresponding to val
    val_color01 = colors_01[idx] + vv * (colors_01[idx + 1] - colors_01[idx])
    val_color_0255 = list(map(np.uint8, 255*val_color01))

    return 'rgb'+str(tuple(val_color_0255))

def _get_plotly_data(E, coords):
    # E is the list of tuples representing the graph edges
    # coords is the list of node coordinates assigned by igraph.Layout
    N = len(coords)
    Xnodes = [coords[k][0] for k in range(N)]  # x-coordinates of nodes
    Ynodes = [coords[k][1] for k in range(N)]  # y-coordnates of nodes

    Xedges = []
    Yedges = []
    for e in E:
        Xedges.extend([coords[e[0]][0], coords[e[1]][0], None])
        Yedges.extend([coords[e[0]][1], coords[e[1]][1], None])

    return Xnodes, Ynodes, Xedges, Yedges


def _text_mapper_summary(mapper_summary):

    d = mapper_summary['custom_meta']

    if d['projection'] is not None:
        text = "<br><b>Projection: </b>" +  d['projection']
        text+= "<br><b>Clusterer: </b>" + d['clusterer'] +\
                "<br><b>Scaler: </b>" + d['scaler']

    return text

def _pl_format_tooltip(member_ids, custom_tooltips, X,
                    X_names, lens, lens_names, color_function, node_ID, colorscale):

    custom_tooltips = custom_tooltips[member_ids] if custom_tooltips is not None else member_ids

    custom_tooltips = list(custom_tooltips)

    projection_stats = _format_projection_statistics(member_ids, lens, lens_names)
    cluster_stats = _format_cluster_statistics(member_ids, X, X_names)
    member_histogram = pl_build_histogram(color_function[member_ids], colorscale)

    return projection_stats, cluster_stats, member_histogram




def _hover_format(member_ids, custom_tooltips, X,
                           X_names, lens, lens_names):
    #tooltip = _format_projection_statistics(member_ids, lens, lens_names)
    cluster_data = _format_cluster_statistics(member_ids, X, X_names)
    tooltip = ''
    custom_tooltips = custom_tooltips[member_ids] if custom_tooltips is not None else member_ids
    val_size=cluster_data['size']
    tooltip+="{val_size}".format(**locals())
    return tooltip


def _pl_color_function(member_ids, color_function):
    return np.mean(color_function[member_ids])
