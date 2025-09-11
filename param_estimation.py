from src.torch_erg import load_pglib_opf as lp
from src.torch_erg.utils import laplacian_matrix
from src.torch_erg.samplers import GWGSampler, MHSampler
import torch
import numpy as np
import networkx as nx
import random as rnd

import itertools

from eval_metrics import mean_difference

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