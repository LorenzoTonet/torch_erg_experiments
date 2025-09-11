from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.utils import laplacian_matrix
from src.torch_erg.samplers import GWGSampler, MHSampler
import torch
import numpy as np
import networkx as nx
import random as rnd


def mean_difference(sampler, real_graph, generated_samples:list[torch.Tensor]):
    real_obs = sampler.observables(real_graph)
    generated_obs = [sampler.observables(g) for g in generated_samples]

    mean_obs = torch.stack(generated_obs).mean(axis = 0)

    return abs(real_obs-mean_obs)


def number_of_isomorphic(real_graph, generated_samples:list[torch.Tensor]):
    ref = nx.from_numpy_array(real_graph.cpu().numpy())
    count = 0
    for g in generated_samples:
        if nx.is_isomorphic(nx.from_numpy_array(g.cpu().numpy()), ref):
            count += 1
    return count