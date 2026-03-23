import numpy as np
from scipy.optimize import minimize, Bounds


def adjust_leaf_bounds(leaf_bounds, objective_bounds, b_tol):
    """ Adjusting the bounds for leaf node to be within the objective function bounds """ 
    leaf_lbs = np.array([leaf_bounds[dim][0] for dim in range(len(leaf_bounds))]) + b_tol
    leaf_ubs = np.array([leaf_bounds[dim][1] for dim in range(len(leaf_bounds))]) - b_tol
    if not(np.all(leaf_lbs<=leaf_ubs)):
        leaf_lbs = np.maximum(np.array([objective_bounds[dim][0] for dim in range(len(leaf_bounds))])+b_tol,
                            np.array([leaf_bounds[dim][0] for dim in range(len(leaf_bounds))]))
        leaf_ubs = np.minimum(np.array([objective_bounds[dim][1] for dim in range(len(leaf_bounds))])-b_tol,
                            np.array([leaf_bounds[dim][1] for dim in range(len(leaf_bounds))]))
    return leaf_lbs, leaf_ubs


def propose_location_leaf_init(leaf_lbs, leaf_ubs, d):
    """ Given the leaf node, the gps, initialization point and leaf bounds initialize the variables
    for finding the next sampling point for each leaf node """
    initial_pt = (leaf_ubs - leaf_lbs) * np.random.random_sample(size=(d, )) + leaf_lbs
    x0 = initial_pt.copy().reshape(d, )
    x_min_leaf = initial_pt.copy().reshape(d, )
    acq_min_leaf = np.inf
    return  x0, x_min_leaf, acq_min_leaf


def propose_location_opt_func(n_restarts, obj_fun, x0, leaf_ubs, leaf_lbs, d):
    ### Optimization process
    min_bounds = Bounds(lb=leaf_lbs, ub=leaf_ubs)
    if n_restarts == 0:
        result = minimize(obj_fun, x0=x0, bounds=min_bounds, method='L-BFGS-B')
        acq_min_leaf = result.fun
        x_min_leaf = result.x
    else:
        for _ in range(n_restarts + 1):
            x0 = (leaf_ubs - leaf_lbs) * np.random.random_sample(size=(d, )) + leaf_lbs
            result = minimize(obj_fun, x0=x0, bounds=min_bounds, method='L-BFGS-B')
            if result.fun < acq_min_leaf:
                acq_min_leaf = result.fun
                x_min_leaf = result.x
    return acq_min_leaf, x_min_leaf


# This is modularized version of propose_location function
def propose_location_modular(acquisition, gps, clf_leaf_nodes, beta_acq, d, n_restarts,
                             objective_bounds, b_tol=1e-5, opt='max', unsampled_leaf_val=-10,
                             call_for = 'cbo', bnds=None):
    """ Next sampling point based on the maximizer of the acquisition function
    acquisition - Sampling strategy
    gps - Gaussian Processes
    clf_leaf_nodes - leaf nodes for the learnt classification tree
    beta_acq - constant to balance between exploration and exploitation
    d - dimension
    n_restarts - no: of restarts for the optimizer
    initial_pt - starting point of optimizer if n_restarts = 0"""
    x_min_par = list([])
    acq_min_par = list([])
    if call_for == 'cbo':
        leaf_ids = [leaf.identifier for leaf in clf_leaf_nodes]
    for leaf_node in clf_leaf_nodes:
        ### Initialize variables for each leaf
        if call_for == 'cbo':
            leaf_bounds = leaf_node.data.bounds
        elif call_for == 'tgp':
            leaf_bounds = bnds[leaf_node]
        # Adjust bounds for leaf node
        leaf_lbs, leaf_ubs = adjust_leaf_bounds(leaf_bounds, objective_bounds, b_tol)
        # Initializations for each leaf node
        x0, x_min_leaf, acq_min_leaf = propose_location_leaf_init(leaf_lbs, leaf_ubs, unsampled_leaf_val, d)
        ### Check if leaf node is sampled and fitted with GP
        if not hasattr(gps[leaf_node],'kernel_'):
            # acq_min_par = [-np.inf]  # This ensures that the unsampled leaf is selected 
            acq_min_par += [unsampled_leaf_val]
            x_min_par += [x_min_leaf]
            continue
        ### Define the objective function based on optimization type
        def obj_fun(x, opt=opt):
            # Objective function to be minimized (+LCB for min and -UCB for max)
            multiplier = 1
            if opt=='max':
                multiplier = -1
            return multiplier*acquisition(x.reshape(1, d), gps[leaf_node], beta_acq)
        ### Optimization process
        acq_min_leaf, x_min_leaf = propose_location_opt_func(n_restarts, obj_fun, x0, leaf_ubs, leaf_lbs, d)
        acq_min_par += [acq_min_leaf]
        x_min_par += [x_min_leaf]
    ### Find the minimum UCB value and corresponding leaf
    acq_min = min(acq_min_par)
    arg_acq_min = acq_min_par.index(acq_min)
    if call_for == 'cbo':
        leaf_index = leaf_ids[arg_acq_min]
    elif call_for == 'tgp':
        leaf_index = clf_leaf_nodes[arg_acq_min]
    x_min = x_min_par[arg_acq_min]
    return x_min.reshape(1, d), acq_min, leaf_index


# This can be parallelized using joblib
def propose_location(acquisition, gps, clf_leaf_nodes, beta_acq, d, n_restarts, initial_pt,
                     objective_bounds, b_tol=1e-5, opt='max', unsampled_leaf_val=-10):
    """ Next sampling point based on the maximizer of the acquisition function
    acquisition - Sampling strategy
    gps - Gaussian Processes
    clf_leaf_nodes - leaf nodes for the learnt classification tree
    beta_acq - constant to balance between exploration and exploitation
    d - dimension
    n_restarts - no: of restarts for the optimizer
    initial_pt - starting point of optimizer if n_restarts = 0"""
    x_min_par = list([])
    ucb_min_par = list([])
    leaf_ids = [leaf.identifier for leaf in clf_leaf_nodes]
    for leaf_node in clf_leaf_nodes:
        ### Initialize variables for each leaf
        # Adjust bounds for leaf node
        leaf_bounds = leaf_node.data.bounds
        leaf_lbs = np.array([leaf_bounds[dim][0] for dim in range(len(leaf_bounds))]) + b_tol
        leaf_ubs = np.array([leaf_bounds[dim][1] for dim in range(len(leaf_bounds))]) - b_tol
        if not(np.all(leaf_lbs<=leaf_ubs)):
            leaf_lbs = np.maximum(np.array([objective_bounds[dim][0] for dim in range(len(leaf_bounds))])+b_tol,
                                np.array([leaf_bounds[dim][0] for dim in range(len(leaf_bounds))]))  # max(lb+tol, leaf_lb)
            leaf_ubs = np.minimum(np.array([objective_bounds[dim][1] for dim in range(len(leaf_bounds))])-b_tol,
                                np.array([leaf_bounds[dim][1] for dim in range(len(leaf_bounds))]))  # min(ub-tol, leaf_ub)
        min_bounds = Bounds(lb=leaf_lbs, ub=leaf_ubs)
        # If initial point is not provided for optimization then assign one randomly with in the leaf bounds
        if initial_pt == None:
            initial_pt = (leaf_ubs - leaf_lbs) * np.random.random_sample(size=(d, )) + leaf_lbs
        x0 = initial_pt.copy().reshape(d, )
        # Initialize the x_min and ucb_min for each leaf
        x_min_leaf = initial_pt.copy().reshape(d, )
        ucb_min_leaf = np.inf
        if not hasattr(gps[leaf_node],'kernel_'):
            # ucb_min_par += [-np.inf]  # This ensures that the unsampled leaf is selected 
            ucb_min_par += [unsampled_leaf_val]
            x_min_par += [x_min_leaf]
            continue

        ### Define the objective function based on optimization type
        def obj_fun(x, opt=opt):
            # Objective function to be minimized (+LCB for min and -UCB for max)
            multiplier = 1
            if opt=='max':
                multiplier = -1
            return multiplier*acquisition(x.reshape(1, d), gps[leaf_node], beta_acq)
        
        ### Optimization process
        if n_restarts == 0:
            result = minimize(obj_fun, x0=x0, bounds=min_bounds, method='L-BFGS-B')
            ucb_min_leaf = result.fun
            x_min_leaf = result.x
        else:
            for _ in range(n_restarts + 1):
                x0 = (leaf_ubs - leaf_lbs) * np.random.random_sample(size=(d, )) + leaf_lbs
                result = minimize(obj_fun, x0=x0, bounds=min_bounds, method='L-BFGS-B')
                if result.fun < ucb_min_leaf:
                    ucb_min_leaf = result.fun
                    x_min_leaf = result.x
        ucb_min_par += [ucb_min_leaf]
        x_min_par += [x_min_leaf]
        initial_pt = None # Because the bounds for each region is different
    
    ### Find the minimum UCB value and corresponding leaf
    ucb_min = min(ucb_min_par)
    arg_ucb_min = ucb_min_par.index(ucb_min) 
    leaf_index = leaf_ids[arg_ucb_min]
    x_min = x_min_par[arg_ucb_min]
    return x_min.reshape(1, d), ucb_min, leaf_index


# This can be parallelized using joblib
def propose_location_tgp(acquisition, gps, bnds, clf_leaf_nodes, beta_acq, d, n_restarts, initial_pt,
                         objective_bounds, b_tol=1e-5, opt='max', unsampled_leaf_val=-10):
    """ Next sampling point based on the maximizer of the acquisition function
    acquisition - Sampling strategy
    gps - Gaussian Processes
    clf_leaf_nodes - leaf nodes for the learnt classification tree
    beta_acq - constant to balance between exploration and exploitation
    d - dimension
    n_restarts - no: of restarts for the optimizer
    initial_pt - starting point of optimizer if n_restarts = 0"""
    x_min_par = list([])
    ucb_min_par = list([])
    for leaf_node in clf_leaf_nodes:
        ### Initialize variables for each leaf
        leaf_bounds = bnds[leaf_node]  ### This is the change for TGP
        leaf_lbs = np.array([leaf_bounds[dim][0] for dim in range(len(leaf_bounds))]) + b_tol
        leaf_ubs = np.array([leaf_bounds[dim][1] for dim in range(len(leaf_bounds))]) - b_tol
        if not(np.all(leaf_lbs<=leaf_ubs)):
            leaf_lbs = np.maximum(np.array([objective_bounds[dim][0] for dim in range(len(leaf_bounds))])+b_tol,
                                  np.array([leaf_bounds[dim][0] for dim in range(len(leaf_bounds))]))
            leaf_ubs = np.minimum(np.array([objective_bounds[dim][1] for dim in range(len(leaf_bounds))])-b_tol,
                                  np.array([leaf_bounds[dim][1] for dim in range(len(leaf_bounds))]))
        if initial_pt == None:
            initial_pt = (leaf_ubs - leaf_lbs) * np.random.random_sample(size=(d, )) + leaf_lbs
        min_bounds = Bounds(lb=leaf_lbs, ub=leaf_ubs)
        x0 = initial_pt.copy().reshape(d, )
        x_min_leaf = initial_pt.copy().reshape(d, )
        ucb_min_leaf = np.inf
        if not hasattr(gps[leaf_node],'kernel_'):
            # ucb_min_par += [-np.inf]  # This ensures that the unsampled leaf is selected 
            ucb_min_par += [unsampled_leaf_val]
            x_min_par += [x_min_leaf]
            continue
        
        ### Define objective function
        def obj_fun(x, opt=opt):
            # Objective function to be minimized (+LCB for min and -UCB for max)
            multiplier = 1
            if opt=='max':
                multiplier = -1
            return multiplier*acquisition(x.reshape(1, d), gps[leaf_node], beta_acq)
            
        ### Optimization process
        if n_restarts == 0:
            result = minimize(obj_fun, x0=x0, bounds=min_bounds, method='L-BFGS-B')
            ucb_min_leaf = result.fun
            x_min_leaf = result.x
        else:
            for _ in range(n_restarts + 1):
                x0 = (leaf_ubs - leaf_lbs) * np.random.random_sample(size=(d, )) + leaf_lbs
                result = minimize(obj_fun, x0=x0, bounds=min_bounds, method='L-BFGS-B')
                if result.fun < ucb_min_leaf:
                    ucb_min_leaf = result.fun
                    x_min_leaf = result.x
        ucb_min_par += [ucb_min_leaf]
        x_min_par += [x_min_leaf]
        initial_pt = None # Because the bounds for each region is different
    ucb_min = min(ucb_min_par)
    arg_ucb_min = ucb_min_par.index(ucb_min)
    leaf_id = clf_leaf_nodes[arg_ucb_min]
    x_min = x_min_par[arg_ucb_min]
    return x_min.reshape(1, d), ucb_min, leaf_id


def propose_location_gp(acquisition, gp, beta_acq, d, n_restarts, initial_pt, obj_bounds, opt='max'):
    """ Next sampling point based on the maximizer of the acquisition function
    acquisition - Sampling strategy
    gp - Gaussian Processes
    beta - constant to balance between exploration and exploitation
    bounds - [(lb,ub)]*d
    d - dimension
    n_restarts - no: of restarts for the optimizer
    initial_pt - starting point of optimizer if n_restarts = 0"""
    acq_min = np.inf
    lbs = np.array([obj_bounds[dim][0] for dim in range(len(obj_bounds))])
    ubs = np.array([obj_bounds[dim][1] for dim in range(len(obj_bounds))])
    if initial_pt ==  None:
        initial_pt = (ubs - lbs) * np.random.random_sample(size=(d, )) + lbs
    x0 = initial_pt.copy().reshape(d, )
    def obj_fun(x, opt=opt):
    # Objective function to be minimized (+LCB for min and -UCB for max)
        multiplier = 1
        if opt=='max':
            multiplier = -1
        return multiplier*acquisition(x.reshape(1, d), gp, beta_acq)
    if n_restarts == 0:
        result = minimize(obj_fun, x0=x0, bounds=obj_bounds, method='L-BFGS-B')
        x_min = result.x
    else:
        for _ in range(n_restarts + 1):
            x0 = (ubs - lbs) * np.random.random_sample(size=(d, )) + lbs
            # print(x0)
            result = minimize(obj_fun, x0=x0, bounds=obj_bounds, method='L-BFGS-B')
            if result.fun < acq_min:
                x_min = result.x
    return x_min.reshape(1, d), acq_min


def ucb_sampling(x, gp, beta_acq):
    """Computing upper confidence bound at a given point."""
    mu, sig = gp.predict(x, return_std=True)
    ucb_x = mu + (np.sqrt(beta_acq) * sig)
    return ucb_x


def lcb_sampling(x, gp, beta_acq):
    """Computing lower confidence bound at a given point."""
    mu, sig = gp.predict(x, return_std=True)
    lcb_x = mu - (np.sqrt(beta_acq) * sig)
    return lcb_x

