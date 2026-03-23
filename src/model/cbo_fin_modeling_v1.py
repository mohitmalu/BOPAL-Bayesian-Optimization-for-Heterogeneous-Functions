import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.utils.optimize import _check_optimize_result
from sklearn.base import clone
from scipy.optimize import minimize
from scipy.linalg import cholesky, cho_solve
from operator import itemgetter
from copy import deepcopy

### For Base Line ###
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor

from cbo_fin_treelib_v1 import TreeClassifier, leaf_node_data
from cbo_fin_treesampling_v1 import metropolis_hastings


def remove_boundary_paridx(Pind, Pval, bounds):
    '''Remove all dimension index and thresholds that are boundaries'''
    mask = (Pind == 0)&(Pval == bounds[0][0])
    for dim in range(len(bounds)):
        mask1 = (Pind == dim)
        for bound in bounds[dim]:
            mask2 = (Pval == bound)
            mask = mask|(mask1 & mask2)
    Pind_new = Pind[~mask]
    Pval_new = Pval[~mask]
    return Pind_new, Pval_new


def train_clf_func(X, C, Pind, Pval, obj_bounds, identifier, max_depth=10,
                   min_samp=1, tol=0.5, n_trials=10, emt_llh=-70, split_method='random',
                   clf_method = 'dist', mh_sample=True, burn_in=100, ensemble=True, n_ensemble=1):
    '''Learn the tree based on the data
       (Classification tree + distance info + MH sampling)'''
    Pind_new, Pval_new = remove_boundary_paridx(Pind, Pval, obj_bounds)
    # Initialize the tree classifier
    clf = TreeClassifier(identifier=identifier, min_samples_leaf=min_samp, max_depth=max_depth)
    # Fit the classifer to the data
    clf.new_fit(X, C, Pind_new, Pval_new, obj_bounds, split_method=split_method, clf_method=clf_method)
    # MH sampling algo to sample new trees
    if mh_sample:
        if ensemble:
            clf_ensemble = {}
            for ens_idx in range(n_ensemble):
                # MH sampling algo to sample new trees
                clf_sample = metropolis_hastings(burn_in, clf, X, C, min_samp=min_samp,
                                                 tol=tol, n_trials=n_trials, emt_llh=emt_llh)
                clf_ensemble[ens_idx] = deepcopy(clf_sample)
            return clf, clf_ensemble
        else:
            clf_sample = metropolis_hastings(burn_in, clf, X, C, min_samp=min_samp, tol=tol,
                                         n_trials=n_trials, emt_llh=emt_llh)
    else:
        if ensemble:
            clf_ensemble = {}
            for ens_idx in range(n_ensemble):
                clf_ensemble[ens_idx] = deepcopy(clf)
            return clf, clf_ensemble
        else:
            clf_sample = clf
    clf_leaf_nodes, clf_leaf_cls, clf_leaf_id, leaf_data_idx = leaf_node_data(X, clf_sample)
    clf_leaf_node_data = [X[indices] for indices in leaf_data_idx]  # Data points in each leaf node
    clf_leaf_data_cls = [C[indices] for indices in leaf_data_idx]  # Class of each datapoint in each leaf node
    return clf_sample, clf_leaf_nodes, clf_leaf_cls, clf_leaf_id, leaf_data_idx, clf_leaf_node_data, clf_leaf_data_cls


def clf_data_func(clf):
    ''' Output leafs, leaf classes and leaf ids '''
    clf_leaf_nodes = np.array(clf.leaves())  # Leaf Nodes of the tree
    clf_leaf_cls = np.array([leaf_node.data.predicted_class for leaf_node in clf_leaf_nodes])  # Class of the leaf node
    clf_leaf_id = np.array([leaf_node.identifier for leaf_node in clf_leaf_nodes])
    return clf_leaf_nodes, clf_leaf_cls, clf_leaf_id


def gpr_par(X, Y, C, Pind, Pval, kernel, obj_bounds, identifier, max_depth=10, min_samp=1,
            restarts=1, normalize=True, tol=0.5, n_trials=10, emt_llh=-70, samp_fit=1,
            gp_alpha=1e-10, split_method='random', clf_method='dist', mh_sample=True, burn_in=100,
            ensemble=False, n_ensemble=1):
    """bo_par is a function to train a gps in each partition individually/independently"""
    # Learn the tree based on the data (Classification tree + distance info + MH sampling)
    clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_id, leaf_data_idx, clf_leaf_node_data, _ \
        = train_clf_func(X, C, Pind, Pval, obj_bounds, identifier=identifier, max_depth=max_depth,
                        min_samp=min_samp, tol=tol, n_trials=n_trials, emt_llh=emt_llh,
                        split_method=split_method, clf_method=clf_method, mh_sample=mh_sample,
                        burn_in=burn_in, ensemble=ensemble, n_ensemble=n_ensemble)
    # Initialize GP's for each leaf node
    gps = {leaf_node:GaussianProcessRegressor(clone(kernel),
                                              normalize_y=normalize,
                                              n_restarts_optimizer=restarts, 
                                              alpha=gp_alpha) for leaf_node in clf_leaf_nodes}
    # Train GP's with corresponding partition data
    for idx, leaf_node in enumerate(clf_leaf_nodes):
        if len(clf_leaf_node_data[idx]) >= samp_fit:
            gps[leaf_node].fit(clf_leaf_node_data[idx], Y[leaf_data_idx[idx]])
        else:
            # print(len(clf_leaf_node_data[idx]))
            continue
    return clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_id, leaf_data_idx, clf_leaf_node_data, gps


def objective_func(theta, gps_cls_fitted, eval_gradient=True):
    """Objective function - log marginal likelihood (lml) for each gp"""
    lml_par = []
    grad_par = []
    if eval_gradient:
        for gpr_i in gps_cls_fitted:
            lml_i, grad_i = gpr_i.log_marginal_likelihood(theta, eval_gradient=True, clone_kernel=False)
            lml_par += [lml_i]
            grad_par += [grad_i]
        lml = np.sum(np.array(lml_par))
        grad = np.sum(np.array(grad_par), axis=0)
        return -lml, -grad
    else:
        for gpr_i in gps_cls_fitted:
            lml_i = gpr_i.log_marginal_likelihood(theta, clone_kernel=False)
            lml_par += [lml_i]
        lml = np.sum(np.array(lml_par))
        return -lml


def constrained_opt_func(obj_func, initial_theta, obj_bounds=[], args=(), opt_method="L-BFGS-B"):
    opt_res = minimize(obj_func, initial_theta, args=args, method=opt_method, jac=True, bounds=obj_bounds)
    _check_optimize_result("lbfgs", opt_res)
    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min


def gpr_hetero_ll(clf_leaf_nodes, clf_leaf_cls, gps, theta_initial, restarts, gradient_eval=True,
                  GPR_CHOLESKY_LOWER=True):
    optima = []
    for label in sorted(set(clf_leaf_cls)):
        optimum = []
        cls_leaf_nodes = clf_leaf_nodes[clf_leaf_cls == label]
        gps_cls = [gps[leaf_node] for leaf_node in cls_leaf_nodes]
        gps_cls_fitted = [gps[leaf_node] for leaf_node in cls_leaf_nodes if hasattr(gps[leaf_node], "kernel_")]
        gps_ker_bounds = gps_cls[0].kernel.bounds
        if len(gps_cls_fitted) > 1:
            if restarts == 0:
                optimum += [(constrained_opt_func(objective_func, theta_initial, obj_bounds=gps_ker_bounds, 
                                                  args=(gps_cls_fitted, gradient_eval), opt_method="L-BFGS-B"))]
            else:
                if not np.isfinite(gps_cls_fitted[0].kernel.bounds).all():
                    raise ValueError("Multiple optimizer restarts (n_restarts_optimizer>0) "
                                     "requires that all bounds are finite.")
                for _ in range(restarts):
                    theta_initial1 = gps_cls_fitted[0]._rng.uniform(gps_ker_bounds[:, 0], gps_ker_bounds[:, 1])
                    optimum.append(constrained_opt_func(objective_func, theta_initial1, obj_bounds=gps_ker_bounds, 
                                                  args=(gps_cls_fitted, gradient_eval), opt_method="L-BFGS-B"))
            lml_values = list(map(itemgetter(1), optimum))
            # Looping through GP's to update newly learnt kernel hyperparameters
            for gpr_i in gps_cls_fitted:
                gpr_i.kernel_.theta = optimum[np.argmin(lml_values)][0]
                gpr_i.log_marginal_likelihood_value_ = gpr_i.log_marginal_likelihood(gpr_i.kernel_.theta,
                                                                                     clone_kernel=False)
                K = gpr_i.kernel_(gpr_i.X_train_)
                K[np.diag_indices_from(K)] += gpr_i.alpha
                try:
                    gpr_i.L_ = cholesky(K, lower=GPR_CHOLESKY_LOWER, check_finite=False)
                except np.linalg.LinAlgError as exc:
                    exc.args = (
                        (
                            f"The kernel, {gpr_i.kernel_}, is not returning a positive "
                            "definite matrix. Try gradually increasing the 'alpha' "
                            "parameter of your GaussianProcessRegressor estimator."
                        ),
                    ) + exc.args
                    raise
                gpr_i.alpha_ = cho_solve((gpr_i.L_, GPR_CHOLESKY_LOWER), gpr_i.y_train_, check_finite=False)
            optima += [optimum[np.argmin(lml_values)]]
    return gps, optima


def gpr_hetero(X, Y, C, Pind, Pval, kernel, obj_bounds, identifier, theta_initial, max_depth=10, min_samp=1,
               restarts=1, normalize=True, tol=0.5, n_trials=10, emt_llh=-70, samp_fit=1, gp_alpha=1e-10,
               split_method='random', clf_method='dist', mh_sample=True, burn_in=100, ensemble=False, n_ensemble=1,
               gradient_eval=True, GPR_CHOLESKY_LOWER=True):
    """bo_hetero is a function for training gps in each partition with new ll objective
    that depends on the data of all the partitions that belong to the same class"""
    # bo_par function to generate partitions and learn gps in individual patitions
    # Learn the tree based on the data (Classification tree + distance info + MH sampling)
    clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_id, leaf_data_idx, clf_leaf_node_data, gps = \
        gpr_par(X, Y, C, Pind, Pval, kernel, obj_bounds, identifier, max_depth=max_depth,
                min_samp=min_samp, restarts=restarts, normalize=normalize, tol=tol, n_trials=n_trials,
                emt_llh=emt_llh, samp_fit=samp_fit, gp_alpha=gp_alpha, split_method=split_method,
                clf_method=clf_method, mh_sample=mh_sample, burn_in=burn_in, ensemble=ensemble,
                n_ensemble=n_ensemble)
    # bo_hetero_ll to update gps with new kernel hyperparameters
    gps, optima = gpr_hetero_ll(clf_leaf_nodes, clf_leaf_cls, gps, theta_initial, restarts,
                                gradient_eval=gradient_eval, GPR_CHOLESKY_LOWER=GPR_CHOLESKY_LOWER)
    
    return clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_id, leaf_data_idx, clf_leaf_node_data, gps


def gpr_cls(X, Y, C, Pind, Pval, kernel, obj_bounds, identifier, max_depth=10, min_samp=1,
            restarts=1, normalize=True, tol=0.5, n_trials=10, emt_llh=-70, samp_fit=1, gp_alpha=1e-10,
            split_method='random', clf_method='dist', mh_sample=True, burn_in=100, ensemble=False, n_ensemble=1):
    """gpr_cls is a function to train a gps in each class independently"""
    # Learn the tree based on the data (Classification tree + distance info + MH sampling)
    clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_id, leaf_data_idx, clf_leaf_node_data, _ \
        = train_clf_func(X, C, Pind, Pval, obj_bounds, identifier=identifier, max_depth=max_depth,
                        min_samp=min_samp, tol=tol, n_trials=n_trials, emt_llh=emt_llh,
                        split_method=split_method, clf_method=clf_method, mh_sample=mh_sample,
                        burn_in=burn_in, ensemble=ensemble, n_ensemble=n_ensemble)
    # Unique leaf classes of the current tree
    clf_classes = sorted(set(clf_leaf_cls))
    # Initialize GP's for each Class
    gps = {label:GaussianProcessRegressor(clone(kernel), normalize_y=normalize,
                            n_restarts_optimizer=restarts, alpha=gp_alpha) for label in clf_classes}
    # Train GP's with corresponding Class data
    for idx, label in enumerate(clf_classes):
        # Select all the leafs for which the class is j
        data_idx_leaf = leaf_data_idx[clf_leaf_cls==label]
        # The index array of size input to select all the points in each class
        indices = np.any(data_idx_leaf, axis=0)
        # Fitting a GP for each class
        if len(X[indices]) >= samp_fit:
            gps[label].fit(X[indices], Y[indices])
        else:
            continue
    return clf, clf_leaf_nodes, clf_leaf_cls, clf_leaf_id, leaf_data_idx, clf_leaf_node_data, gps


def gpr_test(X, Y, clf, gps):
    """gpr_test is the function to compute the mse in each partition"""
    clf_leaf_nodes, _, _, leaf_data_idx = leaf_node_data(X, clf)
    mse_par = []
    y_pred = []
    n_par = []
    for idx, leaf_node in enumerate(clf_leaf_nodes):   # For every leaf node compute mse
        if len(X[leaf_data_idx[idx]]) != 0:  # Check if there are any test data points in given node i
            n_par += [len(X[leaf_data_idx[idx]])]  # number of test samples in each partition
            if hasattr(gps[leaf_node], "kernel_"): 
                ypred = gps[leaf_node].predict(X[leaf_data_idx[idx]])
                err = mean_squared_error(ypred, Y[leaf_data_idx[idx]])
            else:
                ypred = np.zeros_like(Y[leaf_data_idx[idx]])
                err = mean_squared_error(ypred, Y[leaf_data_idx[idx]])
            y_pred += [ypred]
            mse_par += [err]
        else:
            n_par += [0]
            mse_par += [0]
    return mse_par, n_par


class TreedGaussianProcess:
    def __init__(self, max_depth=5, kernel=None, normalize=True, restarts=1, gp_alpha=1e-10):
        self.max_depth = max_depth
        self.tree = DecisionTreeRegressor(max_depth=max_depth)
        self.models = {}
        self.kernel = kernel
        self.normalize = normalize
        self.n_restarts = restarts
        self.gp_alpha = gp_alpha
        
    def fit(self, X, y, samp_fit=1):
        # Step 1: Fit decision tree to partition the space
        self.tree.fit(X, y)
        
        # Step 2: Fit separate GP model for each leaf node in the tree
        leaf_indices = self.tree.apply(X)
        unique_leaves = np.unique(leaf_indices)
        
        for leaf in unique_leaves:
            # Get the data points in this leaf
            leaf_mask = (leaf_indices == leaf)
            X_leaf, y_leaf = X[leaf_mask], y[leaf_mask]
            
            # Fit a Gaussian Process model to this subset of data
            gp = GaussianProcessRegressor(kernel=self.kernel, normalize_y=self.normalize,
                                          n_restarts_optimizer=self.n_restarts, alpha=self.gp_alpha)
            if len(X_leaf)>=samp_fit:
                gp.fit(X_leaf, y_leaf)
            self.models[leaf] = gp
    
    def predict(self, X, return_std=True):
        predictions = np.zeros(X.shape[0])
        std_devs = np.zeros(X.shape[0])
        leaf_indices = self.tree.apply(X)
        
        for idx, leaf in enumerate(leaf_indices):
            gp = self.models[leaf]
            y_pred, sigma = gp.predict(X[idx].reshape(1, -1), return_std=return_std)
            predictions[idx] = y_pred
            std_devs[idx] = sigma
        
        return predictions, std_devs
    
def compute_leaf_boundaries(tree, obj_bounds, feature_names=None):
    # Initialize a list to store boundaries for each leaf
    leaf_boundaries = []

    # Recursive helper function to traverse the tree and calculate boundaries
    def traverse(node_id, feature_ranges):
        # Check if this node is a leaf node
        if tree.tree_.children_left[node_id] == -1 and tree.tree_.children_right[node_id] == -1:
            # Add the final ranges to leaf_boundaries
            leaf_boundaries.append((node_id, feature_ranges.copy()))
            return
        # Get the feature and threshold for the current split
        feature = tree.tree_.feature[node_id]
        threshold = tree.tree_.threshold[node_id]
        # Traverse left (feature <= threshold)
        if tree.tree_.children_left[node_id] != -1:
            left_ranges = feature_ranges.copy()
            left_ranges[feature] = (left_ranges[feature][0], min(left_ranges[feature][1], threshold))
            traverse(tree.tree_.children_left[node_id], left_ranges)
        # Traverse right (feature > threshold)
        if tree.tree_.children_right[node_id] != -1:
            right_ranges = feature_ranges.copy()
            right_ranges[feature] = (max(right_ranges[feature][0], threshold), right_ranges[feature][1])
            traverse(tree.tree_.children_right[node_id], right_ranges)
    
    traverse(0, obj_bounds)
    result = {}
    for node_id, feature_ranges in leaf_boundaries:
        if feature_names:
            result[node_id] = {feature_names[dim]: feature_ranges[dim] for dim in range(len(feature_ranges))}
        else:
            result[node_id] = [feature_ranges[dim] for dim in range(len(feature_ranges))]
    return result

