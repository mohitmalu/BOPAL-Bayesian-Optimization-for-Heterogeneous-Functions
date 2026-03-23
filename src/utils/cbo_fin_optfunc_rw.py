import numpy as np
import pandas as pd
from sklearn.base import clone
from time import time
import pickle
import sys
from joblib import Parallel, delayed
import warnings

from cbo_fin_datagen_v1 import rw_data, gen_rw_data
from cbo_fin_acqfunc_v1 import ucb_sampling
from turbo_011525 import Turbo1, TurboM
from cbo_fin_optfunc_v1 import gp_bo, tgp_bo, PtuRBO_MH, PtuRBO_MH_ensemble_v2, PtuRBO_phased
from cbo_fin_plot_v1 import plot_rewards_se


####################### Results logging functions ########################
def log_result(Y, label, t_start, t_end, y_opt, T, N, opt, verbose=True):
    y_opt_run = max(Y[N:]) if opt == 'max' else min(Y[N:])
    if verbose:
        print(f"{label} optimum {y_opt_run} and time taken = {round(t_end - t_start, 0)}s")
    return [min(Y[N:N + j + 1]-y_opt) for j in range(T)]


#############################################################################################
######### Parallel implementation of macro runs for Ensemble based PTuRBO and other baselines
def compare_TuRBO_rw(N, k, p, tup, obj_bounds, identifier, func_gen_ydata, d, T, freq, const, runs, y_opt, kernel,
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
    partitions = [[-1, 0.85, 0.95, 1], [-1, 0.85, 0.95, 1]]
    partition_mat = np.arange(3*3).reshape(3,3)
    label_mat= np.array([[1,1,1],
                        [1,0,1],
                        [1,1,1]])

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
        X, Y, C, P, Pind, Pval = gen_rw_data(N, func_gen_ydata, obj_bounds, d, freq,
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
                     title, y_label, d, p, k, N, file_name, ylim=(-0.2,0.8))
    # plot_rewards_se(rewards_desc_list, T, y_opt, runs, color_list, label_list, file_name)
    data_file_name = file_name + ".pkl"
    pickle_out = open(data_file_name, "wb")
    pickle.dump(data, pickle_out)
    pickle_out.close()
    return df_epturbo_cart, df_epturbo_dist, df_epturbo_dist_cart, df_pturbo_dist, df_pturbo_dist_cart, df_ppturbo_dist, df_ppturbo_dist_cart, df_tgpbo, df_tur1, df_tur5, df_bo, data

