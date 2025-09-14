import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import networkx as nx

def avg_clustering_coeff(mtx):
    deg = torch.sum(mtx, dim=1)
    tri_diag = torch.diagonal(mtx @ mtx @ mtx) / 2

    valid = deg > 1
    local_clust = torch.zeros_like(deg)
    local_clust[valid] = tri_diag[valid] / (deg[valid] * (deg[valid] - 1))
    avg_clustering = torch.mean(local_clust[valid])
    #avg_clustering_coeff = nx.average_clustering(nx.from_numpy_array(mtx.numpy()))
    return avg_clustering

def connected_graphs(sampler, list_of_graphs: list[torch.Tensor], max_components: int = 1):
    if sampler is None:
        raise ValueError("Sampler must be provided to compute observables.")

    connected_graphs = [g for g in list_of_graphs if connected_components(csr_matrix(g.cpu().numpy()))[0] == max_components]
    connected_observables = [sampler.observables(g) for g in connected_graphs]
    return connected_graphs, connected_observables


def number_of_trees_over_major_components(sampler, list_of_graphs: list[torch.Tensor]):
    if sampler is None:
        raise ValueError("Sampler must be provided to compute observables.")

    trees_count = 0

    for g in list_of_graphs:
        # Assumendo g sia una matrice di adiacenza torch.Tensor
        nx_graph = nx.from_numpy_array(g.cpu().numpy())

        connected_components = nx.connected_components(nx_graph)
        maj_component = max(connected_components, key=len)  # componente più grande
        maj_graph = nx_graph.subgraph(maj_component).copy()

        trees_count += nx.is_tree(maj_graph)

    return trees_count