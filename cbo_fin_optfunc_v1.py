import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import clone
from time import time
import pickle
import sys
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from cbo_fin_datagen_v1 import gen_cls_dataset, gen_p_ind_val, gen_checker_partitions,\
    gen_checker_par_unbalanced1, gen_data_func, latin_hypercube1, gen_training_data,\
    to_unit_cube1, from_unit_cube1, scaled_gen_ydataset_max, scaled_gen_ydataset_min
from cbo_fin_treelib_v1 import leaf_node_data
from cbo_fin_modeling_v1 import gpr_par, TreedGaussianProcess,compute_leaf_boundaries, train_clf_func
from cbo_fin_acqfunc_v1 import propose_location, propose_location_gp, propose_location_tgp,\
      propose_location_modular, ucb_sampling, adjust_leaf_bounds
from turbo_011525 import Turbo1, TurboM, evaluate_candidates
from cbo_fin_plot_v1 import plot_rewards_se


####################### Results logging functions ########################
def log_result(Y, label, t_start, t_end, y_opt, T, N, opt, verbose=True):
    y_opt_run = max(Y[N:]) if opt == 'max' else min(Y[N:])
    if verbose:
        print(f"{label} optimum {y_opt_run} and time taken = {round(t_end - t_start, 0)}s")
    return [min(abs(y_opt - Y[N:N + j + 1])) for j in range(T)]


def eval_append(X, Y, C, Pind, Pval, x_proposed, partitions, label_mat, partition_mat,
                func_gen_ydata, freq, const, alpha, std,):
    c_eval = gen_cls_dataset(x_proposed, partitions, label_mat)  # True cls eval of the new sample
    p_eval = gen_cls_dataset(x_proposed, partitions, partition_mat)  # True partition eval of the new sample
    y_eval = np.array(func_gen_ydata(x_proposed, c_eval, p_eval, partitions, partition_mat, freq, 
                                     const, alpha=alpha, std=std))  # func eval
    pind_eval, pval_eval = gen_p_ind_val(x_proposed, partitions) # Index and Threshold of the new sample
    X = np.append(X, x_proposed, axis=0)
    Y = np.append(Y, y_eval)
    C = np.append(C, c_eval)
    Pind = np.append(Pind, pind_eval)
    Pval = np.append(Pval, pval_eval)
    return X, Y, C, Pind, Pval


######################### Functions for each strategy ##################################
###### Partition BO where only the partitions are learned (using class info) but no hyperparameter learning ######
def PBO_MH(X, Y, C, Pind, Pval, kernel, obj_bounds, identifier, func_gen_ydata, d, T, partitions, label_mat,
            partition_mat, freq, const, max_depth=10, min_samp=1, restarts=1, normalize=True, tol=0.5,
            n_trials=10, emt_llh=-70, samp_fit=1, gp_alpha=1e-10, split_method='random', clf_method='dist',
            mh_sample=True, burn_in=100, ensemble=False, n_ensemble=1, n_restarts=1, delta=0.01, alpha=1,
            alpha_acq=1, std=0, b_tol=1e-5, acq=ucb_sampling, opt='max', unsampled_leaf_val=-10, verbose=True):
    beta_acq = 0
    for iteration in range(T):
        if (iteration%10==0) and verbose:
            print(iteration)
        clf, clf_leaf_nodes, _, _, _, _, gps =\
            gpr_par(X, Y, C, Pind, Pval, kernel, obj_bounds, identifier, max_depth=max_depth,
                    min_samp=min_samp, restarts=restarts, normalize=normalize, tol=tol, n_trials=n_trials,
                    emt_llh=emt_llh, samp_fit=samp_fit, gp_alpha=gp_alpha, split_method=split_method,
                    clf_method=clf_method, mh_sample=mh_sample, burn_in=burn_in, ensemble=ensemble,
                    n_ensemble=n_ensemble)
        u = X.shape[0]
        beta_acq = alpha_acq*(2*np.log((u**2)*2*(np.pi**2)/(3*delta))+2*d*np.log((u**2)*d*20*100*
                      np.sqrt(np.log(4*d/delta)))) # b =100, r = 20
        x_proposed, _, _ = propose_location_modular(acq, gps, clf_leaf_nodes, beta_acq, d, n_restarts,
                                                    obj_bounds, b_tol=b_tol, opt=opt,
                                                    unsampled_leaf_val=unsampled_leaf_val, call_for='cbo')
        X, Y, C, Pind, Pval = eval_append(X, Y, C, Pind, Pval, x_proposed, partitions, label_mat, partition_mat,
                                          func_gen_ydata, freq, const, alpha, std)
    return clf, gps, X, Y, C


def PtuRBO_MH(X, C, P, Pind, Pval, obj_bounds, identifier, func_gen_ydata, d, T, partitions, label_mat,
                partition_mat, freq, const, max_depth=10, min_samp=1, tol=0.5, n_trials=10,
                emt_llh=-70, samp_fit=1, split_method='random', clf_method='dist', mh_sample=True,
                burn_in=100, ensemble=False, n_ensemble=1, alpha=1, std=0, b_tol=1e-5,
                n_training_steps=100, unsampled_leaf_val=-30.0, verbose=True):
    fX = np.reshape(func_gen_ydata(X, C, P, partitions, partition_mat, freq, const, alpha=alpha,
                                   std=std), (-1,1))
    N = len(X)  # Initial number of samples
    for iteration in range(T):
        if verbose and (iteration%10==0):
            print(f"Iteration {iteration}")
            sys.stdout.flush()
        # Learning the partitions based on data 
        clf, clf_leaf_nodes, _, _, leaf_data_idx, clf_leaf_node_data, _ \
            = train_clf_func(X, C, Pind, Pval, obj_bounds, identifier=identifier, max_depth=max_depth,
                            min_samp=min_samp, tol=tol, n_trials=n_trials, emt_llh=emt_llh,
                            split_method=split_method, clf_method=clf_method, mh_sample=mh_sample,
                            burn_in=burn_in, ensemble=ensemble, n_ensemble=n_ensemble)
        turbos = {}
        X_next_leaf = np.empty((len(clf_leaf_nodes), d))
        Y_next_leaf = np.empty((len(clf_leaf_nodes), 1))
        # For each leaf node, we learn a surrogate model
        for idx, leaf_node in enumerate(clf_leaf_nodes):
            # Bounds for each leaf node
            leaf_bounds = leaf_node.data.bounds
            # Generate a candidate point for each leaf node
            if len(clf_leaf_node_data[idx]) >= samp_fit:
                X_tur = to_unit_cube1(clf_leaf_node_data[idx], leaf_bounds)
                fX_tur = fX[leaf_data_idx[idx]].ravel()
                leaf_lbs, leaf_ubs = adjust_leaf_bounds(leaf_bounds, obj_bounds, b_tol)
                turbos[leaf_node] = Turbo1(f=func_gen_ydata, lb=leaf_lbs, ub=leaf_ubs, n_init=N, max_evals=T+N,
                                            batch_size=1, verbose=False, use_ard=True, max_cholesky_size=2000,
                                            n_training_steps=n_training_steps, min_cuda=1024, device="cpu", 
                                            dtype="float64")
                # Initializing the length of the trust region such that entire leaf is covered by trust region 
                turbos[leaf_node].length = 2
                X_cand, y_cand, _ = turbos[leaf_node]._create_candidates(X_tur, fX_tur, length=2,
                                                                         n_training_steps=n_training_steps,
                                                                         hypers={})    
                indbest = np.argmin(y_cand)
                X_next = X_cand[indbest, :].reshape(1, d)
                X_next = from_unit_cube1(X_next, leaf_bounds)
                X_next_leaf[idx, :] = X_next[0,:].copy()
                Y_next_leaf[idx, :] = y_cand[indbest].copy()
            else:
                turbos[leaf_node] = None
                X_next = latin_hypercube1(1, d)
                X_next = from_unit_cube1(X_next, leaf_bounds)
                X_next_leaf[idx, :] = X_next[0,:].copy()
                Y_next_leaf[idx, :] = unsampled_leaf_val
        x_proposed = np.reshape(X_next_leaf[np.argmin(Y_next_leaf), :],(-1,d))
        X, fX, C, Pind, Pval = eval_append(X, fX, C, Pind, Pval, x_proposed, partitions, label_mat, partition_mat,
                                           func_gen_ydata, freq, const, alpha, std)
    return clf, turbos, X, fX, C


####### Partition BO with two stage approach - Learns partitions and then optimize #######
def PtuRBO_phased(X, C, P, Pind, Pval, obj_bounds, identifier, func_gen_ydata, d, T, partitions, label_mat,
                  partition_mat, freq, const, max_depth=10, min_samp=1, tol=0.5, n_trials=10,
                  emt_llh=-70, samp_fit=1, split_method='random', clf_method='dist', mh_sample=False,
                  burn_in=1, ensemble=False, n_ensemble=1, alpha=1, std=0, b_tol=1e-5, 
                  n_training_steps=100, unsampled_leaf_val=-30.0, verbose=True):
    if T%2 == 0:
        T_struc = T//2
        T_opt = T//2
    else:
        T_struc = (T//2) + 1
        T_opt = T//2
    # Generating data at random to learn structure - (T//2 points)
    X_rand = gen_training_data(d, obj_bounds, T_struc)
    C_rand = gen_cls_dataset(X_rand, partitions, label_mat)  # True cls eval of the random samples
    P_rand = gen_cls_dataset(X_rand, partitions, partition_mat)  # True partition eval of the random samples
    Pind_rand, Pval_rand = gen_p_ind_val(X_rand, partitions)  # Index and Threshold of the random samples
    
    # Append the structure data to the initial data
    X = np.append(X, X_rand, axis=0)
    C = np.append(C, C_rand)
    P = np.append(P, P_rand)
    Pind = np.append(Pind, Pind_rand)
    Pval = np.append(Pval, Pval_rand)
    # Evaluate the function at the initial + structure points with the given function
    fX = np.reshape(func_gen_ydata(X, C, P, partitions, partition_mat, freq, const, alpha=alpha,
                                   std=std), (-1, 1))
    N1 = len(X)  # Updated number of samples
    #### First phase: Learn the partitions
    clf, clf_leaf_nodes, _, _, leaf_data_idx, clf_leaf_node_data, _ \
    = train_clf_func(X, C, Pind, Pval, obj_bounds, identifier=identifier, max_depth=max_depth,
                            min_samp=min_samp, tol=tol, n_trials=n_trials, emt_llh=emt_llh,
                            split_method=split_method, clf_method=clf_method, mh_sample=mh_sample,
                            burn_in=burn_in, ensemble=ensemble, n_ensemble=n_ensemble)
    clf_leaf_node_fX = [fX[i] for i in leaf_data_idx]

    #### Second phase: Optimize with learnt partition structure
    # Initialize the surrogate models for each leaf node
    turbos = {}
    X_next_leaf = np.empty((len(clf_leaf_nodes), d))
    Y_next_leaf = np.empty((len(clf_leaf_nodes), 1))
    for idx, leaf_node in enumerate(clf_leaf_nodes):
        # Bounds for each leaf node
        leaf_bounds = leaf_node.data.bounds
        leaf_lbs, leaf_ubs = adjust_leaf_bounds(leaf_bounds, obj_bounds, b_tol)
        turbos[leaf_node] = Turbo1(f=func_gen_ydata, lb=leaf_lbs, ub=leaf_ubs, n_init=N1, max_evals=T_opt+N1,
                            batch_size=1, verbose=False, use_ard=True, max_cholesky_size=2000,
                            n_training_steps=n_training_steps, min_cuda=1024, device="cpu", dtype="float64")
        # Initializing the length of the trust region such that entire leaf is covered by trust region 
        turbos[leaf_node].length = 2
        # Initialize data to model and generate a candidate points for each leaf node
        if len(clf_leaf_node_data[idx]) >= samp_fit:
            # Warp the data to unit cube
            X_tur = to_unit_cube1(clf_leaf_node_data[idx], leaf_bounds)
            fX_tur = clf_leaf_node_fX[idx].ravel()  
            # Create candidates for the model
            X_cand, y_cand, _ = turbos[leaf_node]._create_candidates(X_tur, fX_tur, length=2,
                                                                    n_training_steps=n_training_steps,
                                                                    hypers={})
            # Select the best candidate
            indbest = np.argmin(y_cand)
            X_next = X_cand[indbest, :].reshape(1, d)
            # Undo the warping to get the next point in the leaf node space
            X_next = from_unit_cube1(X_next, leaf_bounds)
            X_next_leaf[idx, :] = X_next[0,:].copy()
            Y_next_leaf[idx, :] = y_cand[indbest].copy()
        else:
            X_next = gen_training_data(d, leaf_bounds, 1)
            X_next_leaf[idx, :] = X_next[0,:].copy()
            Y_next_leaf[idx, :] = unsampled_leaf_val
    
    # Select the candidates and perform optimization
    for iteration in range(T_opt):
        if verbose and (iteration%10==0):
            print(f"Iteration {iteration}")
            sys.stdout.flush()
        # select the leaf node with minimum value
        sel_leaf_idx = np.argmin(Y_next_leaf)
        sel_node = clf_leaf_nodes[sel_leaf_idx]
        x_proposed = np.reshape(X_next_leaf[sel_leaf_idx, :],(1,d))
        c_eval = gen_cls_dataset(x_proposed, partitions, label_mat)  # True cls eval of the new sample
        p_eval = gen_cls_dataset(x_proposed, partitions, partition_mat)  # True partition eval of the new sample
        y_eval = np.array(func_gen_ydata(x_proposed, c_eval, p_eval, partitions, partition_mat, freq, 
                                         const, alpha=alpha, std=std))  # func eval
        # Append to entire data set
        X = np.append(X, x_proposed, axis=0)
        C = np.append(C, c_eval)
        fX = np.append(fX, y_eval)
        # Append to only the leaf node data
        clf_leaf_node_data[sel_leaf_idx] = np.append(clf_leaf_node_data[sel_leaf_idx], x_proposed, axis=0)
        clf_leaf_node_fX[sel_leaf_idx] = np.append(clf_leaf_node_fX[sel_leaf_idx], y_eval)
        # Generate candidates for the model that was selected
        if iteration < T_opt - 1:
            leaf_bounds = sel_node.data.bounds
            X_tur = to_unit_cube1(clf_leaf_node_data[sel_leaf_idx], leaf_bounds)
            fX_tur = clf_leaf_node_fX[sel_leaf_idx].ravel()
            X_cand, y_cand, _ = turbos[sel_node]._create_candidates(X_tur, fX_tur,
                                                                    length=turbos[sel_node].length,
                                                                    n_training_steps=turbos[sel_node].n_training_steps,
                                                                    hypers={})
            indbest = np.argmin(y_cand)
            X_next = X_cand[indbest, :].reshape(1, d)
            X_next = from_unit_cube1(X_next, leaf_bounds)
            X_next_leaf[sel_leaf_idx, :] = X_next[0,:].copy()
            Y_next_leaf[sel_leaf_idx, :] = y_cand[indbest].copy()
    return clf, turbos, X, fX, C


# Function to run the MH ensemble version of PtuRBO with acq function is avg of ensemble
def PtuRBO_MH_ensemble(X, C, P, Pind, Pval, obj_bounds, identifier, func_gen_ydata, d, T, partitions, label_mat,
                        partition_mat, freq, const, max_depth=10, min_samp=1, tol=0.5, n_trials=10,
                        emt_llh=-70, samp_fit=1, split_method='random', clf_method='dist', mh_sample=True,
                        burn_in=100, ensemble=True, n_ensemble=5, alpha=1, std=0, b_tol=1e-5,
                        n_training_steps=100, unsampled_leaf_val=-30.0, verbose=True):
    # Evaluate the function at the initial points with the given function
    fX = np.reshape(func_gen_ydata(X, C, P, partitions, partition_mat, freq, const,
                                   alpha=alpha, std=std), (-1, 1))
    N = len(X)  # Initial number of samples
    for iteration in range(T):
        if verbose and (iteration%10==0):
            print(f"Iteration {iteration}")
            sys.stdout.flush()
        # Train the ensemble of Trees for the partitions
        _, clf_ensemble = train_clf_func(X, C, Pind, Pval, obj_bounds, identifier=identifier, max_depth=max_depth,
                            min_samp=min_samp, tol=tol, n_trials=n_trials, emt_llh=emt_llh,
                            split_method=split_method, clf_method=clf_method, mh_sample=mh_sample,
                            burn_in=burn_in, ensemble=ensemble, n_ensemble=n_ensemble)
        tree_data = {}
        # Prepare data for each tree in the ensemble
        for ens_idx in range(n_ensemble):
            clf_leaf_nodes, _, _, leaf_data_idx = leaf_node_data(X, clf_ensemble[ens_idx])
            tree_data[ens_idx] = {'leaf_nodes': clf_leaf_nodes,
                                  'leaf_data_idx': leaf_data_idx,
                                  'leaf_X': [X[idx] for idx in leaf_data_idx],
                                  'leaf_fX': [fX[idx].ravel() for idx in leaf_data_idx]}
        # Function to process each tree in the ensemble
        def process_tree(tree_idx):
            """ Process each tree in the ensemble to generate candidate points """
            if (iteration%10==0) and (tree_idx%10==0) and verbose:
                print(f"Processing tree {tree_idx}")
                sys.stdout.flush()
            turbos = {}
            gp_hypers = {}
            X_cand_list = []
            for idx, leaf_node in enumerate(tree_data[tree_idx]['leaf_nodes']):
                # Bounds for each leaf node
                leaf_bounds = leaf_node.data.bounds
                X_leaf = tree_data[tree_idx]['leaf_X'][idx]
                fX_leaf = tree_data[tree_idx]['leaf_fX'][idx]
                # Generate a candidate points for each leaf node
                if len(X_leaf) >= samp_fit:
                    X_tur = to_unit_cube1(X_leaf, leaf_bounds)
                    fX_tur = fX_leaf.ravel()
                    leaf_lbs, leaf_ubs = adjust_leaf_bounds(leaf_bounds, obj_bounds, b_tol)
                    turbos[leaf_node] = Turbo1(f=func_gen_ydata, lb=leaf_lbs, ub=leaf_ubs, n_init=N, max_evals=T+N,
                                                batch_size=1, verbose=False, use_ard=True, max_cholesky_size=2000,
                                                n_training_steps=n_training_steps, min_cuda=1024, device="cpu", 
                                                dtype="float64")
                    # Initializing the length of the trust region such that entire leaf is covered by trust region 
                    turbos[leaf_node].length = 2
                    X_cand, _, hyper_j = turbos[leaf_node]._create_candidates(X_tur, fX_tur, length=2,
                                                                              n_training_steps=n_training_steps,
                                                                              hypers={})
                    gp_hypers[leaf_node] = hyper_j
                    X_cand = from_unit_cube1(X_cand, leaf_bounds)
                    del X_tur, fX_tur  # Free memory
                else:
                    turbos[leaf_node] = None
                    gp_hypers[leaf_node] = {}
                    n_cand = min(100*d, 5000)
                    X_cand = latin_hypercube1(n_cand, d)
                    X_cand = from_unit_cube1(X_cand, leaf_bounds)
                    gp_hypers[leaf_node] = {}
                del X_leaf, fX_leaf  # Free memory
                X_cand_list.append(X_cand)
            return np.vstack(X_cand_list), turbos, gp_hypers
        all_results = Parallel(n_jobs=n_ensemble)(delayed(process_tree)(ens_idx) for ens_idx in range(n_ensemble))
        X_candidates_list, turbo_ensemble, hypers_ensemble = zip(*all_results)
        X_candidates = np.vstack(X_candidates_list)
        # Function to evaluate the candidate points
        def evaluate_tree(tree_idx):
            """ Evaluate the candidate points for each tree in the ensemble """
            if (iteration%10==0) and (tree_idx%10==0) and verbose:
                print(f"Evaluating tree {tree_idx}")
                sys.stdout.flush()
            # Dividing the candidate points based on the partitions of each tree
            _, _, _, cand_leaf_data_idx = leaf_node_data(X_candidates, clf_ensemble[tree_idx])     
            y_candidates_j = np.empty((len(X_candidates),))
            for idx, leaf_node in enumerate(tree_data[tree_idx]['leaf_nodes']):
                # Bounds for each leaf node
                leaf_bounds = leaf_node.data.bounds
                X_leaf = tree_data[tree_idx]['leaf_X'][idx]
                fX_leaf = tree_data[tree_idx]['leaf_fX'][idx]
                X_cand_leaf = X_candidates[cand_leaf_data_idx[idx]]
                if len(X_leaf) >= samp_fit:
                    X_tur = to_unit_cube1(X_leaf, leaf_bounds)
                    X_candidates_j = to_unit_cube1(X_cand_leaf, leaf_bounds)
                    fX_tur = fX_leaf.ravel()
                    y_candidates_j[cand_leaf_data_idx[idx]] = evaluate_candidates(
                                                    turbo_ensemble[tree_idx][leaf_node],
                                                    X_tur, fX_tur, X_candidates_j,
                                                    n_training_steps=n_training_steps,
                                                    hypers=hypers_ensemble[tree_idx][leaf_node]).reshape(-1,)
                    del X_tur, fX_tur, X_candidates_j  # Free memory
                else:
                    y_candidates_j[cand_leaf_data_idx[idx]] = unsampled_leaf_val
                del X_leaf, fX_leaf, X_cand_leaf # Free memory
            return y_candidates_j
        y_results = Parallel(n_jobs=n_ensemble)(delayed(evaluate_tree)(ens_idx) for ens_idx in range(n_ensemble))
        y_candidates = np.column_stack(y_results)
        x_proposed = np.reshape(X_candidates[np.argmin(y_candidates.mean(axis=1)), :],(-1,d))  # Select the best candidate point
        del X_candidates, y_candidates, tree_data # Free memory
        X, fX, C, Pind, Pval = eval_append(X, fX, C, Pind, Pval, x_proposed, partitions, label_mat, partition_mat,
                                           func_gen_ydata, freq, const, alpha, std)
    return clf_ensemble, turbo_ensemble, X, fX, C, hypers_ensemble


# Function to run the MH ensemble version of PtuRBO with acq function is min of ensemble
def PtuRBO_MH_ensemble_v2(X, C, P, Pind, Pval, obj_bounds, identifier, func_gen_ydata, d, T, partitions, label_mat,
                        partition_mat, freq, const, max_depth=10, min_samp=1, tol=0.5, n_trials=10,
                        emt_llh=-70, samp_fit=1, split_method='random', clf_method='dist', mh_sample=True,
                        burn_in=100, ensemble=True, n_ensemble=5, alpha=1, std=0, b_tol=1e-5,
                        n_training_steps=100, unsampled_leaf_val=-30.0, verbose=True):
    # Evaluate the function at the initial points with the given function
    fX = np.reshape(func_gen_ydata(X, C, P, partitions, partition_mat, freq, const,
                                    alpha=alpha, std=std), (-1, 1))
    N = len(X)  # Initial number of samples
    for iteration in range(T):
        if verbose and (iteration%10==0):
            print(f"Iteration {iteration}")
            sys.stdout.flush()
        # Train the ensemble of Trees for the partitions
        _, clf_ensemble = train_clf_func(X, C, Pind, Pval, obj_bounds, identifier=identifier, max_depth=max_depth,
                            min_samp=min_samp, tol=tol, n_trials=n_trials, emt_llh=emt_llh,
                            split_method=split_method, clf_method=clf_method, mh_sample=mh_sample,
                            burn_in=burn_in, ensemble=ensemble, n_ensemble=n_ensemble)
        # Store the classifier trained just on data
        tree_data = {}
        # Prepare data for each tree in the ensemble
        for ens_idx in range(n_ensemble):
            clf_leaf_nodes, _, _, leaf_data_idx = leaf_node_data(X, clf_ensemble[ens_idx])
            tree_data[ens_idx] = {'leaf_nodes': clf_leaf_nodes,
                                  'leaf_data_idx': leaf_data_idx,
                                  'leaf_X': [X[idx] for idx in leaf_data_idx],
                                  'leaf_fX': [fX[idx].ravel() for idx in leaf_data_idx]}
        # Function to generate and evaluate candidates for each tree in the ensemble
        def candidates_tree(tree_idx):
            turbos = {}
            gp_hypers = {}
            X_next_leaf_list = np.empty((len(tree_data[tree_idx]['leaf_nodes']), d))
            Y_next_leaf_list = np.empty((len(tree_data[tree_idx]['leaf_nodes']), 1))
            # For each leaf node, we learn a surrogate model
            for idx, leaf_node in enumerate(tree_data[tree_idx]['leaf_nodes']):
                # Bounds for each leaf node
                                # Bounds for each leaf node
                leaf_bounds = leaf_node.data.bounds
                X_leaf = tree_data[tree_idx]['leaf_X'][idx]
                fX_leaf = tree_data[tree_idx]['leaf_fX'][idx]
                # Generate a candidate points for each leaf node
                if len(X_leaf) >= samp_fit:
                    X_tur = to_unit_cube1(X_leaf, leaf_bounds)
                    fX_tur = fX_leaf.ravel()
                    leaf_lbs, leaf_ubs = adjust_leaf_bounds(leaf_bounds, obj_bounds, b_tol)
                    turbos[leaf_node] = Turbo1(f=func_gen_ydata, lb=leaf_lbs, ub=leaf_ubs, n_init=N, max_evals=T+N,
                                                batch_size=1, verbose=False, use_ard=True, max_cholesky_size=2000,
                                                n_training_steps=n_training_steps, min_cuda=1024, device="cpu", 
                                                dtype="float64")
                    # Initializing the length of the trust region such that entire leaf is covered by trust region 
                    turbos[leaf_node].length = 2
                    X_cand, y_cand, hyper_leaf = turbos[leaf_node]._create_candidates(X_tur, fX_tur, length=2,
                                                                              n_training_steps=n_training_steps,
                                                                              hypers={})
                    gp_hypers[leaf_node] = hyper_leaf
                    indbest = np.argmin(y_cand)
                    X_next = X_cand[indbest, :].reshape(1, d)
                    X_next = from_unit_cube1(X_next, leaf_bounds)
                    X_next_leaf_list[idx, :] = X_next[0,:].copy()
                    Y_next_leaf_list[idx, :] = y_cand[indbest].copy()
                else:
                    turbos[leaf_node] = None
                    gp_hypers[leaf_node] = {}
                    X_next = latin_hypercube1(1, d)
                    X_next = from_unit_cube1(X_next, leaf_bounds)
                    X_next_leaf_list[idx, :] = X_next[0,:].copy()
                    Y_next_leaf_list[idx, :] = unsampled_leaf_val
            # Select the best candidate point from the leaf nodes
            x_proposed_j = np.reshape(X_next_leaf_list[np.argmin(Y_next_leaf_list), :],(-1,d))
            y_proposed_j = np.min(Y_next_leaf_list)
            return x_proposed_j, y_proposed_j, turbos, gp_hypers

        all_results = Parallel(n_jobs=n_ensemble)(delayed(candidates_tree)(ens_idx) for ens_idx in range(n_ensemble))
        X_candidates_list, Y_candidates_list, turbo_ensemble, hypers_ensemble = zip(*all_results)
        X_candidates = np.vstack(X_candidates_list)
        Y_candidates = np.array(Y_candidates_list)
        x_proposed = np.reshape(X_candidates[np.argmin(Y_candidates), :],(-1,d)).copy()  # Select the best of the bestcandidate point
        del X_candidates, Y_candidates, tree_data # Free memory
        X, fX, C, Pind, Pval = eval_append(X, fX, C, Pind, Pval, x_proposed, partitions, label_mat, partition_mat,
                                           func_gen_ydata, freq, const, alpha, std)
    return clf_ensemble, turbo_ensemble, X, fX, C, hypers_ensemble


###### Treed GP based BO where partitions are learned without class info and BO in each sub-region independently ######
def tgp_bo(X, Y, C, func_gen_ydata, kernel, bnds, d, T, partitions, label_mat, partition_mat,
           freq, beta, const, initial_pt=None, n_restarts=1, delta=0.01, alpha=1, alpha_acq=1,
           max_depth=5, restarts=1, normalize=True, std=0, gp_alpha=1e-10, b_tol=1e-5, samp_fit=1,
           acq=ucb_sampling, opt='max'):
    C = []
    for _ in range(T):
        tgp = TreedGaussianProcess(max_depth=max_depth, kernel=kernel, normalize=normalize,
                                   restarts=restarts, gp_alpha=gp_alpha)
        tgp.fit(X, Y, samp_fit=samp_fit)
        gps = tgp.models
        tgp_bnds = compute_leaf_boundaries(tgp.tree, bnds, feature_names=None)
        clf_leaf_nodes = list(gps.keys())
        u = X.shape[0]
        beta_acq = alpha_acq*(2*np.log((u**2)*2*(np.pi**2)/(3*delta))+2*d*np.log((u**2)*d*20*100*
                    np.sqrt(np.log(4*d/delta)))) # b =100, r = 20
        x_proposed, _, _ = propose_location_tgp(acq, gps, tgp_bnds, clf_leaf_nodes, beta_acq,
                                                d, n_restarts, initial_pt, bnds, b_tol=b_tol,
                                                opt=opt)
        c_eval = gen_cls_dataset(x_proposed, partitions, label_mat)  # True cls eval of the new sample
        p_eval = gen_cls_dataset(x_proposed, partitions, partition_mat)  # True partition eval of the new sample
        y_eval = np.array(func_gen_ydata(x_proposed, c_eval, p_eval, partitions, partition_mat, freq, 
                                         const, alpha=alpha, std=std))  # func eval
        X = np.append(X, x_proposed, axis=0)
        C += [c_eval]
        Y = np.append(Y, y_eval)
    return tgp, X, Y, C


###### Standard BO ######
def gp_bo(X, Y, C, func_gen_ydata, kernel, bnds, d, T, partitions, label_mat, partition_mat,
          freq, beta, const, initial_pt=None, n_restarts=1, delta=0.01, alpha=1, alpha_acq=1,
          restarts=1, normalize=True, std=0, gp_alpha=1e-10, acq=ucb_sampling, opt='max'):
    bo_gp = GaussianProcessRegressor(clone(kernel), normalize_y=normalize,
                                                      n_restarts_optimizer=restarts, alpha=gp_alpha)
    C = []
    beta_acq = 0
    for j in range(T):
        bo_gp.fit(X, Y)
        u = X.shape[0]
        beta_acq = alpha_acq*(2*np.log((u**2)*2*(np.pi**2)/(3*delta))+2*d*np.log((u**2)*d*20*100*
                      np.sqrt(np.log(4*d/delta)))) # b =100, r = 20
        x_proposed, _ = propose_location_gp(acq, bo_gp, beta_acq, d, n_restarts, 
                                                initial_pt, bnds, opt=opt)
        c_eval = gen_cls_dataset(x_proposed, partitions, label_mat)
        p_eval = gen_cls_dataset(x_proposed, partitions, partition_mat)
        y_eval = np.array(func_gen_ydata(x_proposed, c_eval, p_eval, partitions, partition_mat, freq,
                                         const, alpha=alpha, std=std))
        X = np.append(X, x_proposed, axis=0)
        C += [c_eval]
        Y = np.append(Y, y_eval)
    return bo_gp, X, Y, C

#############################################################################################
######### Parallel implementation of macro runs for Ensemble based PTuRBO and other baselines
def compare_TuRBO(N, k, p, tup, obj_bounds, identifier, func_gen_ydata, d, T, freq, const, runs, y_opt, kernel,
                  file_name, max_depth=10, min_samp=1, tol=0.5, n_trials=10, emt_llh=-70, samp_fit=1, 
                  split_method='random', clf_method='dist',  burn_in=100, n_ensemble=5, alpha=1, std=0, b_tol=1e-5,
                  n_training_steps=100, unsampled_leaf_val=-30.0, n_jobs=10, verbose=True, balanced=True, opt='min',
                  initial_pt=None, n_restarts=1, delta=0.01, alpha_acq=1, restarts=1, normalize=True, gp_alpha=1e-10,
                  acq=ucb_sampling):

    epturbo_CART_mh_rewards = []
    epturbo_dist_mh_rewards = []
    epturbo_dist_CART_mh_rewards = []
    pturbo_dist_mh_rewards = []
    pturbo_dist_CART_mh_rewards = []
    phased_pturbo_dist_rewards = []
    phased_pturbo_dist_CART_rewards = []
    tgpbo_rewards = []
    turbo1_rewards = []
    turbo5_rewards = []
    gpbo_rewards = []
    data = {}

    obj_lb = np.array([bound[0] for bound in obj_bounds])
    obj_ub = np.array([bound[1] for bound in obj_bounds])
    beta = 0

    if opt=='max':
        def func_gen_ydata_neg(X, C, P, partitions, partition_mat, freq, beta, const, alpha=0.1, std=0):
            """Output with different function in each class"""
            return -1*func_gen_ydata(X, C, P, partitions, partition_mat, freq, const, alpha=alpha, std=std)
    else:
        def func_gen_ydata_neg(X, C, P, partitions, partition_mat, freq, beta, const, alpha=0.1, std=0):
            """Output with different function in each class"""
            return func_gen_ydata(X, C, P, partitions, partition_mat, freq, const, alpha=alpha, std=std)
    
    """ Partitions and Label Mat for ground truth function """
    if balanced:
        partitions, label_mat = gen_checker_partitions(d, k, tup, obj_bounds)
    else:
        partitions, label_mat = gen_checker_par_unbalanced1(d, k, tup, obj_bounds)
    partition_mat = np.arange(p).reshape(tup)
    if verbose:
        print(partitions, '\n', label_mat, '\n', partition_mat)

    def run_epturbo(X, C, P, Pind, Pval, clf_method):
        return PtuRBO_MH_ensemble_v2(
            X, C, P, Pind, Pval, obj_bounds, identifier, func_gen_ydata, d, T, partitions, label_mat,
            partition_mat, freq, const, max_depth=max_depth, min_samp=min_samp, tol=tol, n_trials=n_trials,
            emt_llh=emt_llh, samp_fit=samp_fit, split_method=split_method, clf_method=clf_method, mh_sample=True,
            burn_in=burn_in, ensemble= True, n_ensemble=n_ensemble, alpha=alpha, std=std, b_tol=b_tol,
            n_training_steps=n_training_steps, unsampled_leaf_val=unsampled_leaf_val, verbose=verbose)
    
    def run_pturbo(X, C, P, Pind, Pval, clf_method):
        return PtuRBO_MH(
            X, C, P, Pind, Pval, obj_bounds, identifier, func_gen_ydata, d, T, partitions, label_mat,
            partition_mat, freq, const, max_depth=max_depth, min_samp=min_samp, tol=tol, n_trials=n_trials,
            emt_llh=emt_llh, samp_fit=samp_fit, split_method=split_method, clf_method=clf_method, mh_sample=True,
            burn_in=burn_in, ensemble= False, n_ensemble=n_ensemble, alpha=alpha, std=std, b_tol=b_tol,
            n_training_steps=n_training_steps, unsampled_leaf_val=unsampled_leaf_val, verbose=verbose)
    
    def run_ppturbo(X, C, P, Pind, Pval, clf_method):
        return PtuRBO_phased(
            X, C, P, Pind, Pval, obj_bounds, identifier, func_gen_ydata, d, T, partitions, label_mat,
            partition_mat, freq, const, max_depth=max_depth, min_samp=min_samp, tol=tol, n_trials=n_trials,
            emt_llh=emt_llh, samp_fit=samp_fit, split_method=split_method, clf_method=clf_method, mh_sample=False,
            burn_in=burn_in, ensemble= False, n_ensemble=n_ensemble, alpha=alpha, std=std, b_tol=b_tol,
            n_training_steps=n_training_steps, unsampled_leaf_val=unsampled_leaf_val, verbose=verbose)

    def single_run(run_id):
        warnings.filterwarnings('ignore')
        print(f"\nStarting Run {run_id + 1}/{runs}")
        sys.stdout.flush()
        run_rewards = {}
        ## Generate data
        X, Y, C, P, Pind, Pval = gen_data_func(N, func_gen_ydata, obj_bounds, d, freq,
                                               const, partitions, partition_mat, label_mat,
                                               alpha=alpha, std=std)
        if verbose:
            print("=============== Running EPturBO_CART_MH ==================")
        t1 = time()
        clf1, gps1, X1, Y1, _, _ = run_epturbo(X, C, P, Pind, Pval, 'CART')
        t2 = time()
        run_rewards['epturbo_cart'] = log_result(Y1, "EPturBO CART MH Ensemble", t1, t2, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running EPturBO_dist_MH ==================")
        t3 = time()
        clf2, gps2, X2, Y2, _, _ = run_epturbo(X, C, P, Pind, Pval, 'dist')
        t4 = time()
        run_rewards['epturbo_dist'] = log_result(Y2, "EPturBO Dist MH Ensemble", t3, t4, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running EPturBO_dist_CART_MH ==================")
        t5 = time()
        clf3, gps3, X3, Y3, _, _ = run_epturbo(X, C, P, Pind, Pval, 'dist_CART')
        t6 = time()
        run_rewards['epturbo_dist_cart'] = log_result(Y3, "EPturBO Dist CART MH Ensemble", t5, t6, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running PturBO_dist_MH ==================")
        t7 = time()
        clf4, gps4, X4, Y4, _ = run_pturbo(X, C, P, Pind, Pval, 'dist')
        t8 = time()
        run_rewards['pturbo_dist'] = log_result(Y4, "PturBO Dist", t7, t8, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running PturBO_Dist_CART ==================")
        t9 = time()
        clf5, gps5, X5, Y5, _ = run_pturbo(X, C, P, Pind, Pval, 'dist_cart')
        t10 = time()
        run_rewards['pturbo_dist_cart'] = log_result(Y5, "PturBO Dist CART", t9, t10, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running Phased Pturbo Dist ==================")
        t11 = time()
        clf6, gps6, X6, Y6, _ = run_ppturbo(X, C, P, Pind, Pval, 'dist')
        t12 = time()
        run_rewards['phased_pturbo_dist'] = log_result(Y6, "Phased PturBO Dist", t11, t12, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running Phased Pturbo Dist CART ==================")
        t13 = time()
        clf7, gps7, X7, Y7, _ = run_ppturbo(X, C, P, Pind, Pval, 'dist_cart')
        t14 = time()
        run_rewards['phased_pturbo_dist_cart'] = log_result(Y7, "Phased PturBO Dist CART", t13, t14, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running TGP BO ==================")
        t15 = time()
        bo_tgp, X8, Y8, _ \
        = tgp_bo(X, Y, C, func_gen_ydata, kernel, obj_bounds, d, T, partitions, label_mat, partition_mat,
                freq, beta, const, initial_pt=initial_pt, n_restarts=n_restarts, delta=delta, 
                alpha=alpha, alpha_acq=alpha_acq, max_depth=min(10, max_depth), restarts=restarts,
                normalize=normalize, std=std, gp_alpha=gp_alpha, b_tol=b_tol, samp_fit=samp_fit,
                acq=acq, opt=opt)
        t16 = time()
        run_rewards['tgp_bo'] = log_result(Y8, "TGP BO", t15, t16, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running TuRBO ==================")
        # TuRBO - 1
        t17 = time()
        turbo1 = Turbo1(f=func_gen_ydata_neg, lb=obj_lb, ub=obj_ub, n_init=N, max_evals=T+N, batch_size=5,
                        verbose=True, use_ard=True, max_cholesky_size=2000, n_training_steps=n_training_steps,
                        min_cuda=1024, device="cpu", dtype="float64")
        turbo1.optimize(X, C, P, partitions, label_mat, partition_mat, freq, 0, const, alpha=alpha, std=std)
        t18 = time()
        X9 = turbo1.X
        if opt == 'max':
            Y9 = -1*turbo1.fX.ravel()
        else:
            Y9 = turbo1.fX.ravel()
        run_rewards['turbo1'] = log_result(Y9, "TuRBO 1", t17, t18, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        
        # TuRBO - 5
        t19 = time()
        turbo_m = TurboM(f=func_gen_ydata_neg, lb=obj_lb, ub=obj_ub, n_init=N, max_evals=T+N, n_trust_regions=5,
                         batch_size=5, verbose=True, use_ard=True, max_cholesky_size=2000, n_training_steps=n_training_steps,
                         min_cuda=1024, device="cpu", dtype="float64")
        turbo_m.optimize(X, C, P, partitions, label_mat, partition_mat, freq, 0, const, alpha=alpha, std=std)
        t20 = time()
        X10 = turbo_m.X
        if opt == 'max':
            Y10 = -1*turbo_m.fX.ravel()
        else:
            Y10 = turbo_m.fX.ravel()
        run_rewards['turbo5'] = log_result(Y10, "TuRBO 5", t19, t20, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        if verbose:
            print("=============== Running Standard BO ==================")
        t21 = time()
        bo_gp, X11, Y11, _ \
        = gp_bo(X, Y, C, func_gen_ydata, kernel, obj_bounds, d, T, partitions, label_mat, partition_mat,
                freq, beta, const, initial_pt=initial_pt, n_restarts=n_restarts, delta=delta, 
                alpha=alpha, alpha_acq=alpha_acq, restarts=restarts, normalize=normalize, std=std,
                gp_alpha=gp_alpha, acq=acq, opt=opt)
        t22 = time()
        run_rewards['gp_bo'] = log_result(Y11, "GP_BO", t21, t22, y_opt, T, N, opt, verbose)
        sys.stdout.flush()
        round_data = {"X":X, "Y":Y, "clf_epturbo_cart":clf1, "gps_epturbo_cart":gps1, "clf_epturbo_dist":clf2, "gps_epturbo_dist":gps2,
                      "clf_epturbo_dist_cart":clf3, "gps_epturbo_dist_cart":gps3, "clf_pturbo_dist":clf4, "gps_pturbo_dist":gps4,
                      "clf_pturbo_dist_cart":clf5, "gps_pturbo_dist_cart":gps5, "clf_phased_pturbo_dist":clf6, "gps_phased_pturbo_dist":gps6,
                      "clf_phased_pturbo_dist_cart":clf7, "gps_phased_pturbo_dist_cart":gps7, "tgp_bo":bo_tgp, "bo_gp":bo_gp, 
                      "X_epturbo_cart":X1, "Y_epturbo_cart":Y1, "X_epturbo_dist":X2, "Y_epturbo_dist":Y2, "X_epturbo_dist_cart":X3, 
                      "Y_epturbo_dist_cart":Y3, "X_pturbo_dist":X4, "Y_pturbo_dist":Y4, "X_pturbo_dist_cart":X5, "Y_pturbo_dist_cart":Y5,
                      "X_phased_pturbo_dist":X6, "Y_phased_pturbo_dist":Y6, "X_phased_pturbo_dist_cart":X7, "Y_phased_pturbo_dist_cart":Y7,
                      "X_tgp":X8, "Y_tgp":Y8, "X_t1":X9, "Y_t1":Y9, "X_t5":X10, "Y_t5":Y10, "X_bo":X11, "Y_bo":Y11}

        run_rewards['round_data'] = round_data
        return run_rewards
    
    total_time_start = time()
    print(f"Starting parallel runs for {runs} iterations...")
    # Run the function in parallel
    results_list = Parallel(n_jobs=n_jobs)(delayed(single_run)(i) for i in range(runs))
    
    total_time_end = time()
    print(f"Total time taken for all runs: {round(total_time_end - total_time_start, 0)} seconds")

    # Collect results
    for i, result in enumerate(results_list):
        epturbo_CART_mh_rewards.append(result['epturbo_cart'])
        epturbo_dist_mh_rewards.append(result['epturbo_dist'])
        epturbo_dist_CART_mh_rewards.append(result['epturbo_dist_cart'])
        pturbo_dist_mh_rewards.append(result['pturbo_dist'])
        pturbo_dist_CART_mh_rewards.append(result['pturbo_dist_cart'])
        phased_pturbo_dist_rewards.append(result['phased_pturbo_dist'])
        phased_pturbo_dist_CART_rewards.append(result['phased_pturbo_dist_cart'])
        tgpbo_rewards.append(result['tgp_bo'])
        turbo1_rewards.append(result['turbo1'])
        turbo5_rewards.append(result['turbo5'])
        gpbo_rewards.append(result['gp_bo'])
        data[i] = result['round_data']

    data['params'] = {"alpha_acq":alpha_acq, "freq":freq, "alpha_fun":alpha, "beta_fun":beta, 
                      "const_fun":const, "tol":tol, "n_trials":n_trials, "emt_llh":emt_llh,
                      "samp_fit":samp_fit, "obj_bnds":obj_bounds, "obj_func":func_gen_ydata,
                      "kernel":kernel, "tup": tup, "unsampled_leaf_val": unsampled_leaf_val,
                      "n_ensemble": n_ensemble, "N": N, "k": k, "p": p, "d": d, "T": T,
                      "burn_in": burn_in, "initial_pt": initial_pt,
                      "opt_restarts": n_restarts, "delta": delta, "gp_restarts": restarts,
                      "normalize": normalize, "std": std, "balanced": balanced,
                      "split_method": split_method, "acq": acq, "opt": opt, "y_opt": y_opt, "runs": runs,
                      "n_training_steps": n_training_steps}
    
    df_epturbo_cart = pd.DataFrame(epturbo_CART_mh_rewards)
    df_epturbo_dist = pd.DataFrame(epturbo_dist_mh_rewards)
    df_epturbo_dist_cart = pd.DataFrame(epturbo_dist_CART_mh_rewards)
    df_pturbo_dist = pd.DataFrame(pturbo_dist_mh_rewards)
    df_pturbo_dist_cart = pd.DataFrame(pturbo_dist_CART_mh_rewards)
    df_ppturbo_dist = pd.DataFrame(phased_pturbo_dist_rewards)
    df_ppturbo_dist_cart = pd.DataFrame(phased_pturbo_dist_CART_rewards)
    df_tgpbo = pd.DataFrame(tgpbo_rewards)
    df_tur1 = pd.DataFrame(turbo1_rewards)
    df_tur5 = pd.DataFrame(turbo5_rewards)
    df_bo = pd.DataFrame(gpbo_rewards)

    epturbo_cart_desc = df_epturbo_cart.describe()
    epturbo_dist_desc = df_epturbo_dist.describe()
    epturbo_dist_cart_desc = df_epturbo_dist_cart.describe()
    pturbo_dist_desc = df_pturbo_dist.describe()
    pturbo_dist_cart_desc = df_pturbo_dist_cart.describe()
    ppturbo_dist_desc = df_ppturbo_dist.describe()
    ppturbo_dist_cart_desc = df_ppturbo_dist_cart.describe()
    tgpbo_desc = df_tgpbo.describe()
    tur1_desc = df_tur1.describe()
    tur5_desc = df_tur5.describe()
    gpbo_desc = df_bo.describe()

    rewards_desc_list = [epturbo_cart_desc, epturbo_dist_desc, epturbo_dist_cart_desc, pturbo_dist_desc, pturbo_dist_cart_desc,
                          ppturbo_dist_desc, ppturbo_dist_cart_desc, tgpbo_desc, tur1_desc, tur5_desc, gpbo_desc]
    color_list = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'cyan', 'magenta', 'lime']
    # color_list = [ 'tab:green', 'tab:orange', 'tab:purple', 'tab:cyan', 'tab:olive', 'tab:gray', 'tab:pink']
    label_list = ['EPtuRBO CART','EPtuRBO Dist', 'EPtuRBO Dist CART', 'PTuRBO Dist', 'PTuRBO Dist CART',
                  'Phased PTuRBO Dist', 'Phased PTuRBO Dist CART', 'TGP BO','TuRBO-1','TuRBO-5','BO']
    title = "Regret vs Iterations"
    y_label = "Simple Regret"
    plot_rewards_se(rewards_desc_list, T, y_opt, runs, color_list, label_list,
                     title, y_label, d, p, k, N, file_name, ylim=(-0.01,0.99))
    # plot_rewards_se(rewards_desc_list, T, y_opt, runs, color_list, label_list, file_name)
    data_file_name = file_name + ".pkl"
    pickle_out = open(data_file_name, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    return df_epturbo_cart, df_epturbo_dist, df_epturbo_dist_cart, df_pturbo_dist, df_pturbo_dist_cart, df_ppturbo_dist, df_ppturbo_dist_cart, df_tgpbo, df_tur1, df_tur5, df_bo, data

