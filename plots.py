import networkx as nx
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import random

def plot_graph(mtx: torch.Tensor, w=5, h=5, labels = False, nodecolor = '#17B6D1', layout='spring', scale_kamada_grid=2, k_spring=0.5, title="Graph Visualization"):
    adj_matrix = mtx.cpu().numpy()
    graph = nx.from_numpy_array(adj_matrix)

    if layout == 'spring':
        pos = nx.spring_layout(graph)
    elif layout == 'kamada':
        pos = nx.kamada_kawai_layout(graph)
    elif layout == 'kamada_grid':
        pos = kamada_kawai_grid_layout(graph, scale=2)

    plt.figure(figsize=(w, h))
    nx.draw(graph, pos, with_labels=labels, node_color=nodecolor, edge_color='gray', node_size=500, font_size=10)
    plt.title(title)
    plt.show()

def compare_graphs(mtx1: torch.Tensor, mtx2: torch.Tensor, 
                   w=10, h=5, labels = False, 
                   nodecolor1 = '#17B6D1', nodecolor2 = '#17B6D1', 
                   layout1='spring', layout2='spring', 
                   title1="Graph 1", title2="Graph 2"):
    adj_matrix1 = mtx1.cpu().numpy()
    adj_matrix2 = mtx2.cpu().numpy()
    
    graph1 = nx.from_numpy_array(adj_matrix1)
    graph2 = nx.from_numpy_array(adj_matrix2)
    
    if layout1 == 'spring':
        pos1 = nx.spring_layout(graph1)
    elif layout1 == 'kamada':
        pos1 = nx.kamada_kawai_layout(graph1)
    elif layout1 == 'kamada_grid':
        pos1 = kamada_kawai_grid_layout(graph1, scale=2)

    if layout2 == 'spring':
        pos2 = nx.spring_layout(graph2)
    elif layout2 == 'kamada':
        pos2 = nx.kamada_kawai_layout(graph2)
    elif layout2 == 'kamada_grid':
        pos2 = kamada_kawai_grid_layout(graph2, scale=2)

    plt.figure(figsize=(w, h))
    
    plt.subplot(1, 2, 1)
    nx.draw(graph1, pos1, with_labels=labels, node_color=nodecolor1, edge_color='gray', node_size=500, font_size=10)
    plt.title(title1)

    plt.subplot(1, 2, 2)
    nx.draw(graph2, pos2, with_labels=labels, node_color=nodecolor2, edge_color='gray', node_size=500, font_size=10)
    plt.title(title2)

    plt.show()
    
def hist_obs_samples(observables_samples: list, observable_data: torch.Tensor, w = 10, h = 8, scale = 0.6, color = "#11a39c", bins = 30, obs_labels = None):
    num_obs = observables_samples[0].shape[0]
    figsize = (num_obs * w * scale, h * scale)
    
    samples_np = torch.stack(observables_samples).cpu().numpy()
    data_np = observable_data.cpu().numpy()
    
    fig, axes = plt.subplots(1, num_obs, figsize=figsize, squeeze=False)
    
    for p in range(num_obs):
        ax = axes[0, p]

        obs_samples = samples_np[:, p]
        ax.hist(obs_samples, bins=bins, alpha=0.5, label='Sampled', color=color)

        if data_np.ndim == 1 or data_np.shape[0] < 2:
            ax.axvline(data_np[p], color='r', linestyle='--', label='Data')
        else:
            ax.hist(data_np[:, p], bins=bins, alpha=0.5, label='Data')
        if obs_labels is not None:
            ax.set_title(f'{obs_labels[p]}')
        else:
            ax.set_title(f'Observable {p}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def boxplot_obs_samples(observables_samples: list, observable_data: torch.Tensor, w = 10, h = 8, scale = 0.6, color = "#11a39c", obs_labels = None):
    num_obs = observables_samples[0].shape[0]
    figsize = (num_obs * w * scale, h * scale)
    

    samples_np = torch.stack(observables_samples).cpu().numpy()
    data_np = observable_data.cpu().numpy()
    
    fig, axes = plt.subplots(1, num_obs, figsize=figsize, squeeze=False)
    
    for p in range(num_obs):
        ax = axes[0, p]
        
        obs_samples = samples_np[:, p]
        ax.boxplot(obs_samples, vert=False, patch_artist=True, boxprops=dict(facecolor=color), medianprops=dict(color='red'))

        if data_np.ndim == 1 or data_np.shape[0] < 2:
            ax.axvline(data_np[p], color='r', linestyle='--', label='Data')
        else:
            ax.boxplot(data_np[:, p], vert=False, patch_artist=True, boxprops=dict(facecolor=color), medianprops=dict(color='red'))

        if obs_labels is not None:
            ax.set_title(f'{obs_labels[p]}')
        else:
            ax.set_title(f'Observable {p}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_params_iterations(params:list, w = 10, h = 8, scale = 0.6, color = "#17B6D1"):

    parlist_np = np.array([p.cpu().numpy() for p in params])
    w = int(w * scale)
    h = int(h * scale)

    plt.figure(figsize = (parlist_np.shape[1] * w, h))
    for p in range(parlist_np.shape[1]):
        plt.subplot(1,parlist_np.shape[1], p + 1)
        plt.plot(parlist_np[:,p], '.-', color = color)
        plt.title(f'Parameter {p}')

def plot_connected_components(list_of_graphs):
    n_connected = []
    for mtx in list_of_graphs:
        comps = connected_components(csr_matrix(mtx.numpy()))
        n_connected.append(comps[0])
    plt.figure()
    plt.plot(n_connected, color = '#11a39c', marker = 'o')
    plt.xlabel('Iteration')
    plt.ylabel('Number of connected components')
    plt.title('Connected components over iterations')
    plt.show()

def plot_stats_on_samples(true_vals, mean_samplers, std_samplers=None, observables_labels=None):
    n_stats = true_vals.shape[0]
    fig, axs = plt.subplots(1, n_stats, figsize=(5*n_stats, 5))

    for i in range(n_stats):
        if observables_labels is not None:
            axs[i].hist(mean_samplers[:, i].numpy(), bins=30, alpha=0.5, label=observables_labels[i], color="#11a39c")
            axs[i].set_title(f'Distribution of {observables_labels[i]}')
        else:
            axs[i].hist(mean_samplers[:, i].numpy(), bins=30, alpha=0.5, label=f"Observable {i}", color="#11a39c")
            axs[i].set_title(f'Distribution of Observable {i}')

        axs[i].axvline(true_vals[i].item(), color='red', linestyle='dashed', linewidth=1, label='True value')
        axs[i].axvline(mean_samplers[:, i].mean().item(), color='blue', linestyle='dashed', linewidth=1, label='Mean of samples')
        if std_samplers is not None:
            axs[i].axvline(mean_samplers[:, i].mean().item() + std_samplers[i].item(), color='blue', linestyle='dotted', linewidth=1, label = 'Standard Deviation')
            axs[i].axvline(mean_samplers[:, i].mean().item() - std_samplers[i].item(), color='blue', linestyle='dotted', linewidth=1)
            

        axs[i].set_xlabel('Value')
        axs[i].set_ylabel('Frequency')
        axs[i].legend()
        

    plt.tight_layout()
    plt.show()

def kamada_kawai_grid_layout(G, scale=3.0, seed=None):
    pos = {}
    components = list(nx.connected_components(G))
    rng = np.random.default_rng(seed)
    n = len(components)
    grid_size = int(np.ceil(np.sqrt(n)))
    for idx, component in enumerate(components):
        subgraph = G.subgraph(component)
        sub_pos = nx.kamada_kawai_layout(subgraph)
        row, col = divmod(idx, grid_size)
        offset = np.array([col * scale, -row * scale])
        for node, coords in sub_pos.items():
            pos[node] = coords + offset
    return pos

def plot_random_graphs(graph_list: list[torch.Tensor], n=4, w=5, h=5, 
                       labels=False, nodecolor='#17B6D1', layout="kamada_grid", scale_kamada_grid=2, k_spring=0.5):
    """
    Plotta n grafi scelti a caso da graph_list in una griglia 2x2.
    Parametri:
    - graph_list: lista di tensori torch rappresentanti matrici di adiacenza.
    - n: numero di grafi da plottare (default 4).
    - w, h: larghezza e altezza di ogni subplot (default 5).
    - labels: se True, mostra le etichette dei nodi (default False).
    - nodecolor: colore dei nodi (default '#17B6D1').
    - layout: tipo di layout per il posizionamento dei nodi ('spring', 'kamada', 'kamada_grid'; default 'kamada_grid').
    - scale_kamada_grid: scala per il layout kamada_grid (default 2).
    - k_spring: parametro k per il layout spring (default 0.5).

    """
    chosen_graphs = random.sample(graph_list, n)
    fig, axes = plt.subplots(2, 2, figsize=(2*w, 2*h))
    axes = axes.flatten()
    
    for i, mtx in enumerate(chosen_graphs):
        ax = axes[i]
        adj_matrix = mtx.cpu().numpy()
        graph = nx.from_numpy_array(adj_matrix)
        
        if layout == "spring":
            pos = nx.spring_layout(graph, seed=42, k=k_spring)
        elif layout == "kamada":
            pos = nx.kamada_kawai_layout(graph)
        elif layout == "kamada_grid":
            pos = kamada_kawai_grid_layout(graph, scale=scale_kamada_grid, seed=42)
        else:
            raise ValueError(f"Layout '{layout}' non riconosciuto.")
        

        nx.draw(
            graph, pos,
            with_labels=labels,
            node_color=nodecolor,
            edge_color='gray',
            node_size=300,
            font_size=10,
            ax=ax
        )
        ax.set_title(f"Graph {i+1}")

        ax.axis('on')

        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1.5)
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()