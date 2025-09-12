import torch


def avg_clustering_coeff(mtx):
    deg = torch.sum(mtx, dim=1)
    tri_diag = torch.diagonal(mtx @ mtx @ mtx) / 2

    valid = deg > 1
    local_clust = torch.zeros_like(deg)
    local_clust[valid] = tri_diag[valid] / (deg[valid] * (deg[valid] - 1))
    avg_clustering = torch.mean(local_clust[valid])
    #avg_clustering_coeff = nx.average_clustering(nx.from_numpy_array(mtx.numpy()))
    return avg_clustering