import numpy as np
from cbo_fin_treelib_v1 import leaf_node_data, Node_data
from copy import deepcopy

#================= Node Functions ===================#
def mnodes_func(clf, change_identifier=True):
    ''' Binary split tree - Merge Nodes - Nodes before leaves '''
    mnodes = []
    for node in clf.all_nodes():
        children = clf.children(node.identifier)
        if len(children) != 0:
            if change_identifier:   # If change identifier is True, then consider merge nodes based on CART
                if (children[0].is_leaf() and children[1].is_leaf()) and node.data.change:
                    mnodes += [node]
            else:   # If change identifier is False, then consider all merge nodes
                if (children[0].is_leaf() and children[1].is_leaf()):
                    mnodes+=[node]
    return mnodes


def inodes_func(clf, change_identifier=True):
    ''' Binary split tree - Internal Nodes - Nodes that are not leaves and merge nodes'''
    inodes = []
    for node in clf.all_nodes():
        children = clf.children(node.identifier)
        if len(children) !=0:
            if change_identifier:  # If change identifier is True, then consider internal nodes based on CART
                if ((not children[0].is_leaf()) and (not children[1].is_leaf())) and node.data.change:
                    inodes += [node]
            else:  # If change identifier is False, then consider all internal nodes
                if ((not children[0].is_leaf()) and (not children[1].is_leaf())):
                    inodes += [node]
    return inodes


def snodes_func(clf, min_samp=1, change_identifier=True):
    ''' Binary split tree - split nodes - Nodes that are leaves'''
    lnodes = clf.leaves()
    if change_identifier: # If change identifier is True, then consider split nodes based on CART
        snodes = [leaf for leaf in lnodes if (leaf.data.sample_size>=min_samp and leaf.data.change)]
    else:  # If change identifier is False, then consider all split nodes
        snodes = [leaf for leaf in lnodes if leaf.data.sample_size>=min_samp]
    return snodes

#================ Probability of split ===================#
def p_split(depth, alpha = 0.95, beta = 0.5):
    ''' Probability of splitting a leaf node - 
        depends on depth, control alpha and beta 
        to make the tree shallow or deep'''
    psplit = alpha*(1+depth)**(-beta)
    return psplit

#================ Proposal Functions ======================#
# ========================= Continuous Split and Change MCMC ====================== #

def ct_split(clf, X, C, min_samp=1, tol=0.5, n_trials=10, change_identifier=True):
    '''Split a uniformly sampled leaf nodes (Discrete)
       Split feature selected uniformly (Discrete)
       Split threshold selected unifromly between the node bounds (Continuous)'''
    idx, thr = None, None
    psplit_node = None  # Probability of split for the node
    psplit_child = None # Probability of split for the children
    splitting = True
    p_tran_f = 0
    n_idx = 0  # index for randomized split nodes array
    f_idx = 0  # index for randomized features array
    t_idx = 0  # index for thresholds counts
    snodes = snodes_func(clf, min_samp=min_samp, change_identifier=change_identifier)
    if len(snodes)!=0:
        random_node_idx_array = np.random.choice(len(snodes), (len(snodes),), False)
        while splitting:
            if f_idx==0:  # Check if feature index is reset (select new node)
                p_tran_f = 1/(len(snodes)-n_idx)
                node = snodes[random_node_idx_array[n_idx]]
                node_bounds = node.data.bounds
                node_depth = clf.depth(node)
                psplit_node = p_split(node_depth)
                psplit_child = p_split(node_depth+1)
                node_data_idx = np.full(X[:,0].shape, True)
                for dim_idx in range(len(node_bounds)):
                    node_data_idx &= (X[:,dim_idx]>node_bounds[dim_idx][0])
                    node_data_idx &= (X[:,dim_idx]<=node_bounds[dim_idx][1])
                xl = X[node_data_idx]
                cl = C[node_data_idx]
                random_feature_idx_array = np.random.choice(len(node_bounds), (len(node_bounds),), False) # Randomly select the feature index 
            if t_idx==0:  # Check if threshold index is reset (select new feature)
                idx = random_feature_idx_array[f_idx]
                thr_min = node_bounds[idx][0]
                thr_max = node_bounds[idx][1]
            if (thr_min+tol) < (thr_max-tol):  # Check if the threshold is valid
                thr = np.random.uniform(thr_min+tol, thr_max-tol)
                indices_left = xl[:, idx] <= thr
                cl_left = cl[indices_left]
                cl_right = cl[~indices_left]
                samples_per_class_left = [np.sum(cl_left == label) for label in range(clf.n_classes_)]
                samples_per_class_right = [np.sum(cl_right == label) for label in range(clf.n_classes_)]
                if cl_left.size == 0: # Left if unsampled
                    predicted_class_left = -1
                    gini_left = 0
                else: # Left if sampled
                    predicted_class_left = np.argmax(samples_per_class_left)
                    gini_left = clf._gini(cl_left)
                if cl_right.size == 0: # Right if unsampled
                    predicted_class_right = -1
                    gini_right = 0
                else:  # Right if sampled
                    predicted_class_right = np.argmax(samples_per_class_right)
                    gini_right = clf._gini(cl_right)
                if (predicted_class_left != predicted_class_right):  # Accept the split if the classes are different
#                     |((predicted_class_left==-1) & (predicted_class_right==-1))):
                    splitting = False   
            else:  # Select new feature if the threshold lim is not valid
                f_idx+=1
                t_idx=0
            t_idx+=1
            if t_idx>=n_trials: # Try atleast n_trials values for each node and each variable
                f_idx+=1
                t_idx=0
            if f_idx>=len(node_bounds):  # Select new node if looped all features
                n_idx+=1
                f_idx=0
                t_idx=0
            if n_idx >= len(snodes):  # If looped through all the nodes, then stop splitting
                splitting = False
                p_tran_f = 0
                psplit_node = 0
                psplit_child = 0
                idx, thr = None, None
        if p_tran_f>0:  # If a valid split is found
            node.data.feature_index = idx
            node.data.threshold = thr
            ##### Left Node #####
            bounds_left = node_bounds.copy()
            bd_l = list(bounds_left[idx])  # Tuple (immutable objects) so change to list (mutable object)
            bd_l[1] = thr
            bounds_left[idx] = tuple(bd_l)
            data_left = Node_data(gini_left, cl_left.size, samples_per_class_left,
                                  predicted_class_left, bounds_left, None, None)
            clf.create_node(identifier = clf.node_num, data = data_left, parent = node.identifier)
            clf.node_num += 1
            ##### Right Node #####
            bounds_right = node_bounds.copy()
            bd_r = list(bounds_right[idx])  # Tuple (immutable objects) so change to list (mutable object)
            bd_r[0] = thr
            bounds_right[idx] = tuple(bd_r)
            data_right = Node_data(gini_right, cl_right.size, samples_per_class_right,
                                   predicted_class_right, bounds_right, None, None)
            clf.create_node(identifier = clf.node_num, data = data_right, parent = node.identifier)
            clf.node_num += 1
    return idx, thr, p_tran_f, psplit_node, psplit_child

def ct_merge(clf, X, C, change_identifier=True):
    """Merge a uniformly selected node whose childern are leaf nodes"""
    idx, thr = None, None
    mnodes = mnodes_func(clf, change_identifier=change_identifier)
    p_tran_f = 0
    if len(mnodes)!=0:
        p_tran_f = 1/len(mnodes)
        node = np.random.choice(mnodes) # Randomly select a merge node
        node_depth = clf.depth(node)
        node_bounds = node.data.bounds
        psplit_node = p_split(node_depth)
        psplit_child = p_split(node_depth+1)
        children = clf.children(node.identifier)
        node_data_idx = np.full(X[:,0].shape, True)
        for dim in range(len(node_bounds)):
            node_data_idx &= (X[:,dim]>node_bounds[dim][0])
            node_data_idx &= (X[:,dim]<=node_bounds[dim][1])
        xl = X[node_data_idx]
        cl = C[node_data_idx]
        samples_per_class = [np.sum(cl == label) for label in range(clf.n_classes_)]
        node.data.gini = clf._gini(cl)
        node.data.sample_size = cl.size
        node.data.samples_per_class = samples_per_class
        node.data.predicted_class = np.argmax(samples_per_class)
        node.data.feature_index = idx
        node.data.threshold = thr
        clf.remove_node(children[0].identifier)
        clf.remove_node(children[1].identifier)
    return idx, thr, p_tran_f, psplit_node, psplit_child

def ct_change(clf, X, C, tol=0.5, n_trials=10, change_identifier=True):
    """Change a uniformly selected node whose childern are leaf nodes"""
    idx, thr = None, None
    changing = True
    n_idx = 0  # index for randomized node array 
    f_idx = 0  # index for randomized feature array
    t_idx = 0  # index for threshold counts
    p_tran_f = 0
    mnodes = mnodes_func(clf, change_identifier=change_identifier)
    if len(mnodes)!=0:
        random_node_idx_array = np.random.choice(len(mnodes), (len(mnodes), ), False)
        while changing:
            if f_idx==0:  # Select new node if looped through all features
                p_tran_f = 1/(len(mnodes)-n_idx)
                node = mnodes[random_node_idx_array[n_idx]]
                children = clf.children(node.identifier)
                node_bounds = node.data.bounds
                node_data_idx = np.full(X[:, 0].shape, True)
                for dim in range(len(node_bounds)):
                    node_data_idx &= (X[:, dim] > node_bounds[dim][0])
                    node_data_idx &= (X[:, dim] <= node_bounds[dim][1])
                xl = X[node_data_idx]
                cl = C[node_data_idx]
                random_feature_idx_array = np.random.choice(len(node_bounds), (len(node_bounds),), False)
            if t_idx==0:  # Select new variable only if values exhausted
                idx = random_feature_idx_array[f_idx]
                thr_min = node_bounds[idx][0]
                thr_max = node_bounds[idx][1] 
            if (thr_min+tol)<(thr_max-tol): # Check if the threshold lim's are valid
                thr = np.random.uniform(thr_min+tol, thr_max-tol)
                indices_left = xl[:,idx]<=thr
                cl_left = cl[indices_left]
                cl_right = cl[~indices_left]
                samples_per_class_left = [np.sum(cl_left == label) for label in range(clf.n_classes_)]
                samples_per_class_right = [np.sum(cl_right == label) for label in range(clf.n_classes_)]
                if cl_left.size == 0:  # Left if unsampled
                    predicted_class_left = -1
                    gini_left = 0
                else:  # Left if sampled
                    predicted_class_left = np.argmax(samples_per_class_left)
                    gini_left = clf._gini(cl_left)
                if cl_right.size == 0:  # Right if unsampled
                    predicted_class_right = -1
                    gini_right = 0
                else:  # Right if sampled
                    predicted_class_right = np.argmax(samples_per_class_right)
                    gini_right = clf._gini(cl_right)
                if predicted_class_left != predicted_class_right:  # Accept the change if the classes are different
                    changing = False
            else:  # Select new feature if the threshold lim is not valid
                f_idx+=1
                t_idx=0
            t_idx+=1
            if t_idx >= n_trials: # Try atleast n_trial values for each node and each variable
                f_idx += 1
                t_idx = 0
            if f_idx >= len(node_bounds): # Select new node if looped all features
                n_idx += 1
                f_idx = 0
                t_idx = 0
            if n_idx >= len(mnodes): # If looped through all the nodes, then stop changing
                changing = False
                p_tran_f = 0
                idx, thr = None, None
        if p_tran_f>0:
            node.data.feature_index = idx
            node.data.threshold = thr
            ##### Left Node #####
            bounds_left = node_bounds.copy()
            bd_l = list(bounds_left[idx])  # Tuple (immutable objects) so change to list (mutable object)
            bd_l[1] = thr
            bounds_left[idx] = tuple(bd_l)
            children[0].data.gini = gini_left
            children[0].data.sample_size = cl_left.size
            children[0].data.samples_per_class = samples_per_class_left
            children[0].data.predicted_class = predicted_class_left
            children[0].data.bounds = bounds_left
            ##### Right Node ####
            bounds_right = node_bounds.copy()
            bd_r = list(bounds_right[idx])  # Tuple (immutable objects) so change to list (mutable object)
            bd_r[0] = thr
            bounds_right[idx] = tuple(bd_r)
            children[1].data.gini = gini_right
            children[1].data.sample_size = cl_right.size
            children[1].data.samples_per_class = samples_per_class_right
            children[1].data.predicted_class = predicted_class_right
            children[1].data.bounds = bounds_right
    return idx, thr, p_tran_f


def prior(clf):
    # Computing the prior on the tree i.e., log(prob(split for all nodes)) - (For now ignore the Prob(idx) and Prob(thr))
    node_array = np.array(clf.all_nodes())
    leaf_array = np.array(clf.leaves())
    psplit_node_array = [p_split(clf.depth(node)) for node in node_array if node not in leaf_array]
    psplit_leaf_array = [(1-p_split(clf.depth(leaf))) for leaf in leaf_array]
    return np.log(np.prod(np.array(psplit_node_array))*np.prod(np.array(psplit_leaf_array)))


def target_distribution(clf, X, C, emt_llh = -70):
    # computing likelihood of a tree based on the (categorical distribution) 
    # mle estimate of probabilities of each class in a given partition
    clf_leaf_nodes, _, _, leaf_data_idx = leaf_node_data(X, clf)
    llh_leaf = []
    for idx in range(len(clf_leaf_nodes)):
        cl = C[leaf_data_idx[idx]]
        num_samp = cl.size
        if num_samp!=0:
            samp_classes = np.array([np.sum(cl==label) for label in range(clf.n_classes_)]) # Samples per label
            prob_classes = (samp_classes/num_samp)  # Probability of each label
            # Adding exp(emt_llh) very small value to ensure the log(0) does not occur when some classes are not present 
            llh_leaf += [np.sum(samp_classes*np.log(prob_classes + np.exp(emt_llh)))] # Log likelihood of the leaf
        else:
            llh_leaf += [emt_llh]
    llh = np.sum(np.array(llh_leaf))
    return llh, llh_leaf


def proposal_distribution(clf, X, C, min_samp=1, tol=0.5, n_trials=10, p_tol = 1e-70, change_identifier=True):
    # Define the proposal distribution to find a Tree
    clf_prop = deepcopy(clf)
    set_k = np.random.choice(3,(3,),False)  # Randomized operation choice array
    choice_idx = 0
    p_transition_f = 1.0
    p_transition_b = 1.0
    sample = True
    while sample:
        if set_k[choice_idx]==0:
            # Split operation set to 0
            splitnodes = snodes_func(clf_prop, min_samp=min_samp, change_identifier=change_identifier)
            if len(splitnodes)!=0:
                sample = False
                idx, thr, p_tran_f, psplit_node, psplit_child = ct_split(clf_prop, X, C, min_samp=min_samp,
                                                                         tol=tol, n_trials=n_trials)
                # Check the number of merge nodes 
                mnodes = mnodes_func(clf_prop, change_identifier=change_identifier)
                # Compute the forward and backward transition probabilities
                p_transition_f = (p_tran_f*(1-psplit_node)*len(mnodes))
                p_transition_b = (psplit_node*((1-psplit_child)**2))
                # If any of the probabilities is zero then do not accept the split
                if (p_transition_f == 0) or (p_transition_b == 0):
                    clf_prop = deepcopy(clf)
                    sample = True
                    choice_idx+=1
            else:
                choice_idx+=1
        elif set_k[choice_idx]==1:
            # Merge operation set to 1
            mergenodes = mnodes_func(clf_prop)
            if len(mergenodes)!=0:
                sample = False
                idx, thr, p_tran_f, psplit_node, psplit_child = ct_merge(clf_prop, X, C)
                splitnodes = snodes_func(clf_prop, min_samp=min_samp)
                p_transition_f = (p_tran_f*psplit_node*((1-psplit_child)**2)*len(splitnodes))
                p_transition_b = (1-psplit_node)
                if (p_transition_f == 0) or (p_transition_b == 0):
                    clf_prop = deepcopy(clf)
                    sample = True
                    choice_idx+=1
            else:
                choice_idx+=1
        elif set_k[choice_idx]==2:
            # Change operation set to 2
            mergenodes = mnodes_func(clf_prop)
            if len(mergenodes)!=0:
                sample = False
                idx, thr, p_transition_f = ct_change(clf_prop, X, C, tol=tol, n_trials=n_trials)
                p_transition_b = p_transition_f
                if (p_transition_f == 0):
                    clf_prop = deepcopy(clf)
                    sample = True
                    choice_idx+=1
            else:
                choice_idx+=1
        if choice_idx==3:
            sample = False
            p_transition_f = 0.0
            p_transition_b = 0.0
    return clf_prop, set_k, choice_idx, np.log(p_transition_f+p_tol), np.log(p_transition_b+p_tol)


def metropolis_hastings(iterations, clf, X, C, min_samp=1, tol=0.5, n_trials=10, emt_llh = -70, p_tol=1e-70):
    # Metropolis-Hastings MCMC algorithm
    clf_init = deepcopy(clf)
    for _ in range(iterations):
        clf_prop, _, _, lp_transition_f, lp_transition_b = proposal_distribution(clf_init, X, C, min_samp=min_samp,
                                                                                     tol=tol, n_trials=n_trials,
                                                                                     p_tol=p_tol)
        llh_prop, _ = target_distribution(clf_prop, X, C, emt_llh=emt_llh)
        llh_curr, _ = target_distribution(clf, X, C, emt_llh=emt_llh)
        log_acceptance_ratio = min(0, ((llh_prop + lp_transition_b) - (llh_curr + lp_transition_f)))
        if np.log(np.random.uniform()) < log_acceptance_ratio:
            clf_init = deepcopy(clf_prop)
    return clf_init

