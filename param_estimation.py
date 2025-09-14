from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.utils import laplacian_matrix
from src.torch_erg.samplers import BaseSampler, GWGSampler, MHSampler
import torch
import numpy as np
import networkx as nx
import random as rnd

import itertools

from eval_metrics import mean_difference

def get_stats_from_sampler(input_graph: torch.Tensor, 
                           sampler: BaseSampler, 
                           n_iter_training: int, 
                           n_iter_sampling: int, 
                           n_processes: int, 
                           init_betas: torch.Tensor, 
                           alpha=0.001, 
                           min_change=0.05, 
                           update_steps=3, 
                           equilibrium_steps=100, 
                           burn_in_fraction=0.01) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    
    """
    Executes multiple independent runs of parameter estimation and sampling using the provided sampler.
    Returns the mean and standard deviation of the sampled observables, as well as all individual means and parameter sets.
    Parameters:
        - input_graph: The input graph as a torch.Tensor.
        - sampler: An instance of BaseSampler (e.g., GWGSampler or MHSampler).
        - n_iter_training: Number of iterations for parameter estimation.
        - n_iter_sampling: Number of iterations for sampling.
        - n_processes: Number of independent runs to execute.
        - init_betas: Initial parameters for the sampler.
        - alpha: Learning rate for parameter updates.
        - min_change: Minimum change threshold for parameter updates.
        - update_steps: Frequency of parameter updates during training.
        - equilibrium_steps: Number of final steps to consider for parameter estimation.
        - burn_in_fraction: Fraction of initial samples to discard during sampling.
    Returns:
        - mean: Mean of the sampled observables across all runs.
        - std: Standard deviation of the sampled observables across all runs.
        - mean_values: List of mean observables from each run.
        - parameters_sets: List of parameter sets from each run.
    """

    input_obs = sampler.observables(input_graph)

    mean_values = []
    parameters_sets = []
    for i in range(n_processes):
        #train the model
        params, _ = sampler.param_run(graph=input_graph,
                            observables=input_obs,
                            params=init_betas,
                            niter=n_iter_training,
                            params_update_every=update_steps,
                            save_every=50,
                            save_params=True,
                            alpha=alpha,                      
                            min_change=min_change)

        params_for_estimates = torch.stack(params[-equilibrium_steps:]).mean(axis = 0)
        parameters_sets.append(params_for_estimates)
        #sample
        observables, _ = sampler.sample_run(graph=input_graph,
                            observables=input_obs,
                            params=params_for_estimates,
                            niter=n_iter_sampling,
                            save_every=50,
                            burn_in = burn_in_fraction)

        mean_values.append(torch.stack(observables).mean(axis=0))

    #return mean and variance of the means, all the means and all the parameters sets
    return torch.stack(mean_values).mean(axis=0), torch.stack(mean_values).std(axis=0), mean_values, parameters_sets

def grid_search_params(sampler, ordmat, betas,
                       obs_weight=None,
                       alphas=[0.001, 0.002, 0.005],
                       min_changes=[0.01, 0.1],
                       update_steps_list=[2, 4, 8],
                       betas_init_list=None,
                       niter=50000,
                       equilibrium_samples=100,
                       save_every=50):
    """
    Esegue una grid search per i parametri di param_run.
    Restituisce la combinazione con migliore mean_difference.
    """

    if obs_weight is not None and len(obs_weight) != len(betas):
        raise ValueError("Lunghezza di obs_weight deve essere uguale a quella di betas")
    if sum(obs_weight) != 1.0:
        raise ValueError("La somma di obs_weight deve essere 1.0")

    obs = sampler.observables(ordmat)
    
    if betas_init_list is None:
        betas_init_list = [betas]

    best_score = float("inf")
    best_result = None

    for alpha, min_change, update_steps, betas_init in itertools.product(
            alphas, min_changes, update_steps_list, betas_init_list):

        try:
            print(f"Eseguo con (alpha={alpha}, min_change={min_change}, update_steps={update_steps})")
            pars, _ = sampler.param_run(
                graph=ordmat,
                observables=obs,
                params=betas_init,
                niter=niter,
                params_update_every=update_steps,
                save_every=save_every,
                save_params=True,
                alpha=alpha,
                min_change=min_change
            )

            

            # stima parametri come media degli ultimi k
            params_for_estimates = torch.stack(pars[-equilibrium_samples:]).mean(axis=0)

            observables, graphs = sampler.sample_run(
                graph=ordmat,
                observables=obs,
                params=params_for_estimates,
                niter=niter,
                save_every=save_every
            )

            
            if obs_weight is not None:
                score = (obs_weight * mean_difference(sampler, ordmat, graphs)).mean()
            else:
                score = mean_difference(sampler, ordmat, graphs).mean()

            print(f"[alpha={alpha}, min_change={min_change}, update_steps={update_steps}] → score={score:.4f}")

            if score < best_score:
                best_score = score
                best_result = {
                    "alpha": alpha,
                    "min_change": min_change,
                    "update_steps": update_steps,
                    "betas_init": betas_init,
                    "score": score,
                    "params": params_for_estimates
                }

        except Exception as e:
            print(f"Errore con (alpha={alpha}, min_change={min_change}, update_steps={update_steps}): {e}")

    return best_result

def param_estimation_more_graphs(data:list[torch.Tensor], sampler, alpha, niter, min_change, update_steps, initial_params, change_input_every, equilibrium_steps):

    return_params = [initial_params]
    return_graphs = []

    number_of_swaps = niter//change_input_every

    for i in range(number_of_swaps):
        actual_data = rnd.choice(data)
        return_graphs.append(actual_data)
        actual_obs = sampler.observables(actual_data)

        params, graphs = sampler.param_run(graph=actual_data,
                      observables=actual_obs,
                      params=return_params[-1],
                      niter=change_input_every,
                      params_update_every=update_steps,
                      save_every=50,
                      save_params=True,
                      alpha=alpha,                      
                      min_change = min_change)
        
        for p in params: return_params.append(p)
        for g in graphs: return_graphs.append(g)
        

    return return_params, return_graphs

def perturb_params(params, magnitude):
    noise = torch.randn_like(params) * magnitude
    return params + noise

def param_estimation_with_perturbation(perturbation_at, perturbation_magnitude, data, sampler, alpha, niter, min_change, update_steps, initial_params):
    it_after_pert = niter-perturbation_at

    return_params = [initial_params]
    return_graphs = [data]

    obs = sampler.observables(data)

    params, graphs = sampler.param_run(graph=data,
                        observables=obs,
                        params=initial_params,
                        niter=perturbation_at,
                        params_update_every=update_steps,
                        save_every=50,
                        save_params=True,
                        alpha=alpha,
                        min_change=min_change)

    for p in params: return_params.append(p)
    for g in graphs: return_graphs.append(g)

    perturbed_params = perturb_params(return_params[-1], perturbation_magnitude)

    obs = sampler.observables(return_graphs[-1])
    params, graphs = sampler.param_run(graph=return_graphs[-1],
                        observables=obs,
                        params=perturbed_params,
                        niter=it_after_pert,
                        params_update_every=update_steps,
                        save_every=50,
                        save_params=True,
                        alpha=alpha,
                        min_change=min_change)
    
    for p in params: return_params.append(p)
    for g in graphs: return_graphs.append(g)

    return return_params, return_graphs