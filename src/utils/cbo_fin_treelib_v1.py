import numpy as np
import treelib


class Node_data():
    def __init__(self, gini, sample_size, samples_per_class, predicted_class, bounds, feature_index, threshold,
                 change=True):
        self.gini = gini
        self.sample_size = sample_size
        self.samples_per_class = samples_per_class
        self.predicted_class = predicted_class
        self.bounds = bounds
        self.feature_index = feature_index
        self.threshold = threshold
        self.change = change


def leaf_node_data(X, clf):
    """Partitioning the input data at leaf node"""
    node_rules = []
    node_rule = []
    leafnodes = []
    leafclass = []
    leaf_id = []
    indices_list = []
    root_node = clf.get_node(0)
    
    def recurse(X, node, node_rule, node_rules, lnode_id, leafnode, leafcls):
        # Function to recurse through the nodes to collect the indices that satisfy the conditions of the node
        # rule is where the list of indicies for each node is collected until leaf node is reached i.e., [[Nx1],[Nx1],...,[Nx1]]
        # rules is final list of all the indices at each node till leaf node is reached [[list of indicies for a leaf node],...]
        if not node.is_leaf():
            children = clf.children(node.identifier)
            left_indices, right_indices = list(node_rule), list(node_rule)
            # Compare the selected column values against the threshold for both left and right
            left_indices += [X[:, node.data.feature_index] <= node.data.threshold]  # True for left node indices
            recurse(X, children[0], left_indices, node_rules, lnode_id, leafnode, leafcls)
            right_indices += [X[:, node.data.feature_index] > node.data.threshold]  # True for right node indices
            recurse(X, children[1], right_indices, node_rules, lnode_id, leafnode, leafcls)
        else:
            leafnode += [node]
            leafcls += [node.data.predicted_class]
            lnode_id += [node.identifier]
            node_rules += [node_rule]

    recurse(X, root_node, node_rule, node_rules, leaf_id, leafnodes, leafclass)
    for idx in range(len(node_rules)):
        indices = np.full(X[:, 0].shape, True)
        for indices_idx in range(len(node_rules[idx])):
            # All the rules for each node is logically anded to give the final set of points in each leaf node
            indices = np.logical_and(indices,node_rules[idx][indices_idx])
        indices_list += [indices] # P is list with the len of number of leaf nodes
    return np.array(leafnodes), np.array(leafclass), np.array(leaf_id), np.array(indices_list)


# ======================== Decision Tree Classifier ======================= #
class TreeClassifier(treelib.Tree):
    def __init__(self, identifier, max_depth=0, min_samples_leaf=1):
        self.max_depth = max_depth   # Maximum depth of the tree
        self.max_depth_ = max_depth  # Maximum depth of the tree
        self.min_samples_leaf = min_samples_leaf  # Minimum number of samples required to be at a leaf node
        self.n_classes_ = 0  # Number of classes in the data
        self.n_features_ = 0  # Number of features (dim) in the data
        self.node_num = 0  # Node number to be assigned to the next node
        self.initialized = False  # Flag to check if the tree is initialized using the data
        treelib.Tree.__init__(self,identifier=identifier)
        
    def _gini(self, C):
        """Compute Gini impurity of a non-empty node"""
        n_samples = C.size
        return 1.0 - sum((np.sum(C == label) / n_samples) ** 2 for label in range(self.n_classes_))
    
    def _predict(self, input, predict_class=True):
        """Predict class / leaf_node_id for a single sample."""
        node = self.get_node(0)  # Start at the root node (get_node -  Treelib function)
        while not node.is_leaf():
            if input[node.data.feature_index] <= node.data.threshold:
                node = self.children(node.identifier)[0]
            else:
                node = self.children(node.identifier)[1]
        if predict_class:
            return node.data.predicted_class
        else:
            return node.identifier
    
    def initialize(self, X, C, Pind, Pval, bounds, split_method='random', clf_method='CART'):
        """To initialize the root_node"""
        self.n_classes_ = np.max(C)+1  #len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_features_ = X.shape[1]  # Number of features (dim) in the data
        self.max_depth = 0  # Maximum depth of the tree
        if self.initialized:  # If tree is already initialized, re-initialize it i.e., reset the count and identifier
            treelib.Tree.__init__(self, identifier=self.identifier)
            self.node_num = 0
        self._tree(X, C, Pind, Pval, bounds, select_list=None, parent=None, split_method=split_method, clf_method=clf_method)
        self.initialized = True
    
    def new_fit(self, X, C, Pind, Pval, bounds, split_method='random', clf_method='CART'):
        """Fit the data with the distance information"""
        self.n_classes_ = np.max(C)+1 #len(set(y))  # classes are assumed to go from 0 to n-1
        self.n_features_ = X.shape[1]
        self.max_depth = self.max_depth_
        if self.initialized:
            treelib.Tree.__init__(self, identifier=self.identifier)
            self.node_num = 0
        self._tree(X, C, Pind, Pval, bounds, select_list=None, parent=None, split_method=split_method, clf_method=clf_method)
        self.initialized = True

    def predict(self, X, predict_class=True):
        """Predict class / leaf_node_id for X."""
        return [self._predict(inputs, predict_class=predict_class) for inputs in X]

    def _best_split(self, X, C):
        """Find the best split for a node.
        "Best" means that the average impurity of the two children, weighted by their
        population, is the smallest possible. Additionally it must be less than the
        impurity of the current node.
        To find the best split, we loop through all the features, and consider all the
        midpoints between adjacent training samples as possible thresholds. We compute
        the Gini impurity of the split generated by that particular feature/threshold
        pair, and return the pair with smallest impurity.
        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        n_samples = C.size
        if n_samples <= 1:
            return None, None
        # Count of each class in the current node.
        samples_per_class = [np.sum(C == label) for label in range(self.n_classes_)]
        # Gini of current node.
        best_gini = 1.0 - sum((l_count / n_samples) ** 2 for l_count in samples_per_class)
        best_idx, best_thr = None, None
        # Loop through all features.
        for dim_idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, dim_idx], C)))
            num_left = [0] * self.n_classes_
            num_right = samples_per_class.copy()
            for samp_id in range(1, n_samples):  # possible split positions
                samp_label = classes[samp_id - 1]
                num_left[samp_label] += 1
                num_right[samp_label] -= 1
                gini_left = 1.0 - sum((num_left[x]/samp_id)**2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x]/(n_samples-samp_id))**2 for x in range(self.n_classes_))
                # The Gini impurity of a split is the weighted average of the Gini impurity of the children.
                gini = (samp_id*gini_left + (n_samples-samp_id)*gini_right)/n_samples
                # The following condition is to make sure we don't try to split two
                # points with identical values for that feature, as it is impossible
                # (both have to end up on the same side of a split).
                if thresholds[samp_id] == thresholds[samp_id - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = dim_idx
                    best_thr = (thresholds[samp_id - 1] + thresholds[samp_id - 1]) / 2  # midpoint
        return best_idx, best_thr

    def _best_split_new(self, X, C, list_pind_pval):
        """Find the best split for a node from the list of indexes and thresholds.
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        n_samples = C.size
        if n_samples <= 1:
            return None, None
        # Count of each class in the current node.
        samples_per_class = [np.sum(C == label) for label in range(self.n_classes_)]
        # Gini of current node.
        best_gini = 1.0 - sum((l_count/n_samples) ** 2 for l_count in samples_per_class)
        best_idx, best_thr = None, None
        # Loop through all the indexes and thresholds
        for dim_idx, threshold in list_pind_pval:
            # print(idx.astype(int), threshold)
            # Split the node according to each feature/threshold pair
            # and count the resulting population for each class in the children
            indices_left = X[:, dim_idx.astype(int)] <= threshold
            C_left = C[indices_left]
            C_right = C[~indices_left]
            if len(C_left) == 0 or len(C_right) == 0:
                continue
            num_left = [np.sum(C_left == label) for label in range(self.n_classes_)]
            num_right = [np.sum(C_right == label) for label in range(self.n_classes_)]
            n_samples_left = sum(num_left)
            gini_left = 1.0 - sum((num_left[x]/n_samples_left)**2 for x in range(self.n_classes_))
            gini_right = 1.0 - sum((num_right[x]/(n_samples-n_samples_left))**2 for x in range(self.n_classes_))
            # The Gini impurity of a split is the weighted average of the Gini
            # impurity of the children.
            gini = (n_samples_left*gini_left+(n_samples-n_samples_left)*gini_right)/n_samples
            if gini < best_gini:
                best_gini = gini
                best_idx = dim_idx.astype(int)
                best_thr = threshold  # given pval (threshold)

        return best_idx, best_thr
    
    def _split_new(self, list_pind_pval):
        """Find the random split for a node from the list of indexes and thresholds.
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        i = np.random.randint(len(list_pind_pval))
        best_idx, best_thr = list_pind_pval[i]
        return best_idx.astype(int), best_thr

    def _tree(self, X, C, Pind, Pval, bounds, select_list=None, parent=None, split_method='random', clf_method='CART'):
        """ Build a initial decision tree from the given closest distance to boundary of data points
        X - Sampled data points
        C - labels of the datapoints
        Pind - Dim along which the given point has the closest partition boundary
        Pval - Closest threshold along the given dim for each data point
        bounds - Boundaries of input space - [(lb, ub)]*d
        select_list - list of ind,thr selected from the given data set for the split
        parent - Parent node (None if root node)
        split_method - best, random
        clf_method - Method to generate tree - 'dist', 'dist_CART', 'CART'
        """
        if select_list is None:
            select_list_ = []
        else:
            select_list_ = select_list.copy()  # To ensure every node has it's own select list
        list_pind_pval = sorted(set(zip(Pind, Pval))) # Unique sorted list of closest distance
        
        # Removing the feature indexes and thresholds that were previously selected for split
        # list_pind_pval = [i for i in list_pind_pval_ if i not in select_list_]
        for idx in range(len(select_list_)):
            try:
                list_pind_pval.remove(select_list_[idx])
            except ValueError:
                continue
        if C.size !=0:  # Node Data values based on data points
            samples_per_class = [np.sum(C == label) for label in range(self.n_classes_)]
            predicted_class = np.argmax(samples_per_class)
            data = Node_data(gini = self._gini(C),
                             sample_size = C.size, 
                             samples_per_class = samples_per_class,
                             predicted_class = predicted_class, 
                             bounds = bounds,
                             feature_index = None,
                             threshold = None,
                             change = True)
        else:
            samples_per_class = [np.sum(C == label) for label in range(self.n_classes_)]
            predicted_class = -1
            data = Node_data(gini = 0,
                             sample_size = C.size, 
                             samples_per_class = samples_per_class,
                             predicted_class = predicted_class, 
                             bounds = bounds,
                             feature_index = None,
                             threshold = None,
                             change = True)
        # Create a new node with the data
        self.create_node(identifier=self.node_num, data=data, parent=parent)
        node = self.get_node(self.node_num)
        self.node_num += 1
        
        # Split recursively until max depth is reached reducing gini as much as possible
        if (self.level(node.identifier) < self.max_depth):
            dim_idx, par_thr = None, None
            if clf_method == 'CART':
                dim_idx, par_thr = self._best_split(X, C)
            elif clf_method == 'dist':
                if len(list_pind_pval)!=0:
                    if split_method=='random':
                        dim_idx, par_thr = self._split_new(list_pind_pval)
                    else:
                        dim_idx, par_thr = self._best_split_new(X, C, list_pind_pval)
                if dim_idx is not None:  # If split is possible based on distance information
                    select_list_.append((dim_idx, par_thr))  # Append the selected distance info to the list
                    node.data.change = False  # Node split based on distance cannot be changed
            elif clf_method == 'dist_CART':
                if len(list_pind_pval) != 0:  # First split based on distance information
                    if split_method=='random':
                        dim_idx, par_thr = self._split_new(list_pind_pval)
                    else:
                        dim_idx, par_thr = self._best_split_new(X, C, list_pind_pval)
                if dim_idx is not None:  # If split is possible based on distance information
                    select_list_.append((dim_idx, par_thr))  # Append the selected distance info to the list 
                    node.data.change = False  # Node split based on distance cannot be changed
                else:  # Split based on data and gini index (CART)
                    dim_idx, par_thr = self._best_split(X, C)
            # If split is possible then assign the node - idx, thr and call _tree for each node     
            if dim_idx is not None:
                node.data.feature_index = dim_idx
                node.data.threshold = par_thr
                indices_left = X[:, dim_idx] <= par_thr
                indices_left_thr = (((Pval <= par_thr) & (Pind == dim_idx)) | (Pind != dim_idx))
                indices_right_thr = (((Pval > par_thr) & (Pind == dim_idx)) | (Pind != dim_idx))
                
                ##### Left Node ###### Bound[idx]->[lb,ub] divided into left[idx]->[lb, thr) and right[idx]->[thr,ub]
                bounds_left = bounds.copy()
                bd_l = list(bounds_left[dim_idx])  # Tuple (immutable objects) so change to list (mutable object)
                bd_l[1] = par_thr
                bounds_left[dim_idx] = tuple(bd_l)
                X_left, C_left, Pind_left, Pval_left = \
                    X[indices_left], C[indices_left], Pind[indices_left_thr],Pval[indices_left_thr]
                self._tree(X_left, C_left, Pind_left, Pval_left, bounds_left,
                           select_list_, parent=node.identifier, split_method=split_method,
                           clf_method=clf_method)
                
                ##### Right Node ##### Bound[idx]->[lb,ub] divided into left[idx]->[lb, thr) and right[idx]->[thr,ub]
                bounds_right = bounds.copy()
                bd_r = list(bounds_right[dim_idx])  # Tuple (immutable objects) so change to list (mutable object)
                bd_r[0] = par_thr
                bounds_right[dim_idx] = tuple(bd_r)
                X_right, C_right, Pind_right, Pval_right = \
                    X[~indices_left], C[~indices_left], Pind[indices_right_thr], Pval[indices_right_thr]
                self._tree(X_right, C_right, Pind_right, Pval_right, bounds_right,
                           select_list_, parent=node.identifier, split_method=split_method,
                           clf_method=clf_method)

                
