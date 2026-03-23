import numpy as np
import math
from numpy.random import RandomState
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from datetime import datetime
from cbo_fin_datagen_v1 import scaled_gen_ydataset_max, scaled_gen_ydataset_min
from cbo_fin_optfunc_v1 import compare_TuRBO
from cbo_fin_acqfunc_v1 import ucb_sampling, lcb_sampling
import sys

import warnings
warnings.filterwarnings('ignore')

if __name__=='__main__':
   # Global Initialization
    rstate = 123  # Random State 123,130,175,
    prng = RandomState(rstate)
    verbose = True  # Verbose flag for printing
    dt = datetime.now().strftime("%m%d%y%H%M%S")
    N = 5  # number of data points exp - 1000, 2000, 5000
    d = 15  # dimension -  exp - 2, 4, 8
    d_par = 3  # No of dimensions with partitions
    obj_bounds = [(-10, 10)] * d  # bounds on the input data - same as space_bounds
    T = 300  # No of iterations of cbo and bo
    runs = 10  # Number of Macro reps
    balanced = False  # Balanced or unbalanced partitions

    # Tree Related Initialization
    max_depth = 20  # Max depth of the tree
    min_samp = 1  # minimum number of leaf samples    
    identifier = 'pbo_mh_tree'  # Tree identifier
    burn_in = 100  # MH burnin samples
    tol = 0.5  # Tolerance for proposal distribution
    n_trials = 10  # Number of tries to find a new tree candidate with split
    emt_llh = -70  # LLH for empty leaf node
    split_method = 'random'  # Method to select distance information 'best'/ 'random'
    clf_method = 'dist_CART'  # Method to generate tree - 'dist', 'dist_CART', 'CART'
    n_ensemble = 5  # Number of trees in the ensemble
    n_jobs = runs  # Number of jobs for parallelization

    # Gaussian Process Initialization
    # kernel_def = ConstantKernel(1)*RBF(length_scale=np.ones((d,)), length_scale_bounds=(1e-5,1e5))+WhiteKernel()  # kernel for learning
    kernel_def = ConstantKernel(1,  constant_value_bounds=(1e-2,20))*\
                                Matern(length_scale=np.ones((d,)), length_scale_bounds=(1e-3,10), nu=2.5)+\
                                WhiteKernel(noise_level_bounds=(1e-7,1))  # kernel for learning
    restarts = 3  # No of restarts of gp - theta initial is uniform random - For hyperparameters
    theta_initial = np.zeros_like(kernel_def.theta)  # theta initial
    normalize = True  # Normalization of y while training GP's
    samp_fit = 1  # Number of samples in subregion to fit GP
    gp_alpha = 1e-10  # Noise nugget in GPR (sklearn)
    n_training_steps = 100  # Number of training steps for GPR (Gpytorch)
    unsampled_leaf_val = -10  # Value of unsampled leaf node

    # UCB initialization
    n_restarts = 3  # No of restarts of minimize optimizer - For UCB
    initial_pt = None  # Initial point for minimize optimizer - For UCB
    delta = 0.01  # Probability of error
    alpha_acq = 1  # Multiple for beta in UCB (exploration)
    uct_acq_const=5  # Exploration factor for UCT
    b_tol = 1e-3  # Bound tolerance for acquisition minimization i.e., x in [lbs+btol, ubs-btol]

    # Objective Function Initialization
    std = np.sqrt(0.0001)  # np.sqrt(0.001)  # Noisey evaluations
    het_ratio = 1  # factor Gap between 2 set of frequencies
    alpha = 10  # multiplicative factor 10 - 2d, 20 - 3d, 30 - 4d  
    opt = 'min'

    # For Standard test functions since the max value it takes is 0
    if opt=='max':
        acq = ucb_sampling
        func_gen_ydata = scaled_gen_ydataset_max
    else:
        acq = lcb_sampling
        func_gen_ydata = scaled_gen_ydataset_min


    # Variable Initializations (Abalation Study - [k, tup, idx1, idx2, alpha])
    ###################################################################################
    # each subregion index - (p-1,p-1...(d bits))_p and sub-region class = sum of 
    # sub-region index % k - Example - (k, p, d) = (3, 4, 3) - 64 sub-regions indexed
    # from (0 to 63) - region 6 index - (0 1 2)_4 = 0 * 4**2 + 1 * 4**1 + 2 * 4**0
    # region 6 class - 0+1+2 = 3%k = 3%3 = 0
    ###################################################################################
    # exp_init = [(2,2,1,0,10), (2,3,7,2,10), (2,4,14,5,10), (2,5,23,6,10)] # Partition Test 2D
    # exp_init = [(3,5,23,6,10), (4,5,21,6,10), (5,5,22,6,10), (6,5,23,6,10)] # Class Test 2D
    
    # exp_init = [(2,2,7,0,10), (2,3,25,4,10), (2,4,63,5,20), (2,5,123,6,20)] # Partition dim = 3
    # exp_init = [(3,5,122,6,20), (4,5,121,6,20), (5,5,123,6,20), (6,5,115,6,20)] # Class dim = 3
    
    # exp_init = [(2,2,14,0,10), (2,3,79,4,20), (2,4,254,5,30)] # Partition dim = 4
    # exp_init = [(3,4,253,5,30), (4,4,252,5,30), (5,4,254,5,30), (6,4,241,5,30)] # Class dim = 4
    
    # exp_init = [(2,2,31,0,10), (2,3,241,4,30)] # Partition dim = 5
    # exp_init = [(3,3,242,4,30), (4,3,241,4,30), (5,3,234,4,30), (6,3,237,4,30)] # Class dim = 5
    
    # exp_init = [(2,2,62,0,20), (3,2,60,0,20), (4,2,62,0,20), (5,2,63,0,20), (6,2,32,0,20)] # comparison with base line dim = 6
    
    exp_init = [(2,3,25,4,10)] # Single test case - par
    # i - class, j - tup, ii - large sub-region, jj - small sub-region, a1 - alpha 
    for i,j,ii,jj,a1 in exp_init:
        np.random.seed(rstate)
        k = i  # Number of classes - exp - 2, 3, 4, 5
        tup1 = [j,]*d_par
        tup2 = [1,]*(d-d_par)
        tup1.extend(tup2)
        tup = tuple(tup1)
        del tup1, tup2
        p = math.prod(tup)  # Number of partitions - exp - 4, 9, 16, 25
        freq = np.ones((k,d)) # Freq set to ones (No effect)
        alpha = a1
        const = (-1/alpha) * prng.permutation(p).reshape(-1, 1)  # Fixed Offset/Intercept random
        # For easy setup the larger sub-region gets the optimum value
        const[ii] = -(p+(2*alpha))/alpha  # Sub optimal value
        const[jj] = -(p+(5*alpha))/alpha  # Optimal value
        if opt=='max':
            y_opt = 1  # max(const*alpha).item()  # Maximum function evaluation
        else:
            y_opt = 0  # min(const*alpha).item()  # Minimum function evaluation

        str1 = "exp_baselines/all_parallel_"
        str2 = "_normalized_alpha_vs_nosupp_base_scaled_"
        str3 = "bal_"+str(balanced)+"_"+str(d)+"d_"+str(d_par)+"dpar_"+str(N)+"N_"+str(k)+"c_"+str(p)+"p_matern_"+str(T)+"_iterations"+str(runs)+"runs_"+str(n_ensemble)+"ensembles_"\
               +str(unsampled_leaf_val)+"leaf_val_"+str(het_ratio)+"het_"+str(alpha_acq)+"alp-acq_"+str(std)+"std_"+dt

        """ File name and Main Function """
        if balanced:   
            if std !=0:
                file_name = str1+"bal"+str2+"noisy_"+str3
            else:
                file_name = str1+"bal"+str2+str3
        else:       
            if const[ii] < const[jj]:
                if std !=0:
                    file_name = str1+"easy"+str2+"noisy_"+str3
                else:
                    file_name = str1+"easy"+str2+str3
            else:   
                if std !=0:
                    file_name = str1+"hard"+str2+"noisy_"+str3
                else:
                    file_name = str1+"hard"+str2+str3
        print(file_name)
        sys.stdout.flush()

        df_epturbo_cart, df_epturbo_dist, df_epturbo_dist_cart, df_pturbo_dist, df_pturbo_dist_cart,\
        df_ppturbo_dist, df_ppturbo_dist_cart, df_tgpbo, df_tur1, df_tur5, df_bo, data = \
        compare_TuRBO(N, k, p, tup, obj_bounds, identifier, func_gen_ydata, d, T, freq, const, runs, y_opt, kernel_def,
                  file_name, max_depth=max_depth, min_samp=min_samp, tol=tol, n_trials=n_trials, emt_llh=emt_llh, samp_fit=samp_fit, 
                  split_method=split_method, clf_method=clf_method,  burn_in=burn_in, n_ensemble=n_ensemble, alpha=alpha, std=std,
                  b_tol=b_tol, n_training_steps=n_training_steps, unsampled_leaf_val=unsampled_leaf_val, n_jobs=n_jobs, verbose=verbose,
                  balanced=balanced, opt=opt, initial_pt=initial_pt, n_restarts=n_restarts, delta=delta, alpha_acq=alpha_acq,
                  restarts=restarts, normalize=normalize, gp_alpha=gp_alpha, acq=acq)

        