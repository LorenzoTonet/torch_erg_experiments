import networkx as nx
import torch
import matplotlib.pyplot as plt

def plot_graph(mtx: torch.Tensor, w=5, h=5):
    adj_matrix = mtx.cpu().numpy()
    graph = nx.from_numpy_array(adj_matrix)
    pos = nx.spring_layout(graph)
    plt.figure(figsize=(w, h))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Graph Visualization")
    plt.show()

def compare_graphs(mtx1: torch.Tensor, mtx2: torch.Tensor, w=10, h=5):
    adj_matrix1 = mtx1.cpu().numpy()
    adj_matrix2 = mtx2.cpu().numpy()
    
    graph1 = nx.from_numpy_array(adj_matrix1)
    graph2 = nx.from_numpy_array(adj_matrix2)
    
    pos1 = nx.spring_layout(graph1)
    pos2 = nx.spring_layout(graph2)
    
    plt.figure(figsize=(w, h))
    
    plt.subplot(1, 2, 1)
    nx.draw(graph1, pos1, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    plt.title("Graph 1")
    
    plt.subplot(1, 2, 2)
    nx.draw(graph2, pos2, with_labels=True, node_color='lightgreen', edge_color='gray', node_size=500, font_size=10)
    plt.title("Graph 2")
    
    plt.show()
    
def hist_obs_samples(observables_samples: list, observable_data: torch.Tensor, w = 10, h = 8, scale = 0.6):
    num_obs = observables_samples[0].shape[0]
    figsize = (num_obs * w * scale, h * scale)
    
    samples_np = torch.stack(observables_samples).numpy()
    data_np = observable_data.numpy()
    
    fig, axes = plt.subplots(1, num_obs, figsize=figsize, squeeze=False)
    
    for p in range(num_obs):
        ax = axes[0, p]

        obs_samples = samples_np[:, p]
        ax.hist(obs_samples, bins=30, alpha=0.5, label='Sampled')
        
        if data_np.ndim == 1 or data_np.shape[0] < 2:
            ax.axvline(data_np[p], color='r', linestyle='--', label='Data')
        else:
            ax.hist(data_np[:, p], bins=30, alpha=0.5, label='Data')
        
        ax.set_title(f'Observable {p}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()

def boxplot_obs_samples(observables_samples: list, observable_data: torch.Tensor, w = 10, h = 8, scale = 0.6):
    num_obs = observables_samples[0].shape[0]
    figsize = (num_obs * w * scale, h * scale)
    

    samples_np = torch.stack(observables_samples).numpy()
    data_np = observable_data.numpy()
    
    fig, axes = plt.subplots(1, num_obs, figsize=figsize, squeeze=False)
    
    for p in range(num_obs):
        ax = axes[0, p]
        
        obs_samples = samples_np[:, p]
        ax.boxplot(obs_samples, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue'), medianprops=dict(color='red'))
        
        if data_np.ndim == 1 or data_np.shape[0] < 2:
            ax.axvline(data_np[p], color='r', linestyle='--', label='Data')
        else:
            ax.boxplot(data_np[:, p], vert=False, patch_artist=True, boxprops=dict(facecolor='lightgreen'), medianprops=dict(color='red'))

        ax.set_title(f'Observable {p}')
        ax.legend()
    
    plt.tight_layout()
    plt.show()
