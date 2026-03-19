import numpy as np
from cbo_func_HA import HA

# Input Sampling
def gen_training_data(d, bnds, N):
    """ Uniformly sampled input data
    bnds - bounds - bounds on space; data - [(lb,ub)]*dimension; datatype - list[tup(float, float)]
    N - training_budget - Number of data points - datatype - int """
    lbs = np.array([bnds[i][0] for i in range(len(bnds))])
    ubs = np.array([bnds[i][1] for i in range(len(bnds))])
    X = (ubs - lbs) * np.random.random_sample(size=(N, d)) + lbs  # uniformly sampled x_data
    return X

def to_unit_cube1(point, bnds):
    lb = np.array([bnds[i][0] for i in range(len(bnds))])
    ub = np.array([bnds[i][1] for i in range(len(bnds))])
    assert np.all(lb < ub)
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    assert (np.all(point >= lb) and np.all(point <= ub))
    new_point = (point-lb)/(ub-lb)
    return new_point

def from_unit_cube1(point, bnds):
    lb = np.array([bnds[i][0] for i in range(len(bnds))])
    ub = np.array([bnds[i][1] for i in range(len(bnds))])
    assert np.all(lb < ub) 
    assert lb.ndim == 1 
    assert ub.ndim == 1 
    assert point.ndim  == 2
    assert (np.all(point >= 0) and np.all(point <= 1))
    new_point = point * (ub - lb) + lb
    return new_point

def latin_hypercube1(n, dims):
    points = np.zeros((n, dims))
    centers = (1.0 + 2.0 * np.arange(0.0, n)) 
    centers = centers / float(2 * n)
    for i in range(0, dims):
        points[:, i] = centers[np.random.permutation(n)]

    perturbation = np.random.uniform(-1.0, 1.0, (n, dims)) 
    perturbation = perturbation / float(2 * n)
    points += perturbation
    return points

# Partitions
def gen_checker_partitions(d, k, tup, bnds):
    """ Generating equisize partitions for space
     tup - tuple of partitions along each dimension
     d - dimension
     k - number of classes
     bounds - boundaries of input space
     returns partitions - list of lists and label matrix - array"""
    # Evenly sampling each axis for partitions
    # 1st cord - vertical down, rows, 2nd cord - horizontal right, columns
    # lowest point top left corner
    partitions = list([])
    for i in range(d):
        x1 = np.linspace(bnds[i][0], bnds[i][1], tup[i] + 1)
        # returns array with [lb, par, ub] - size (tup[i]+1) x 1
        partitions += [x1]
    label_mat = np.indices(tup).sum(axis=0) % k  # generating class label for each partition
    return partitions, label_mat 


def gen_checker_par_unbalanced1(d, k, tup, bnds):
    """ Generating non uniform partitions for space
     tup - tuple of partitions along each dimension
     d - dimension
     k - number of classes
     bounds - boundaries of input space
     returns partitions - list of lists and label matrix - array"""
    # Unevenly sampling each axis for point of partition
    partitions = list([])
    for i in range(d):
        width_x = (bnds[i][1] - bnds[i][0])*2/(tup[i]*(tup[i]+1))
        partition_i = np.array([(bnds[i][0] + (j*(j+1)*width_x)/2) for j in range(tup[i])])
        partition_i = np.append(partition_i, [bnds[i][1]]) 
        # returns array with [lb, par, ub] - size (tup[i]+1) x 1
        partitions += [partition_i] # non uniform sized partitions
    label_mat = np.indices(tup).sum(axis=0) % k # generating class label for each partition
    return partitions, label_mat


# Class Label
def label_fun(x, partitions, label_mat):
    """Label generation function for general partitions"""
    # x is the d dimensional input point - input shape (-1,)
    # par is the list of partition arrays along each dimension, len(par) == d, len(par[i]) == tup[i]+1
    # label_mat is the label matrix
    d = len(partitions)
    ind = ()
    for dim in range(d):
        dim_ind = 1 * (partitions[dim] >= x[dim])
        ind_i = np.where(dim_ind == 1)[0][0]-1
        if ind_i == -1:
            ind_i = 0
        ind += (ind_i,)
    c = label_mat[ind]
    return c


def gen_cls_dataset(X, partitions, label_mat):
    """Class dateset generation function for general partitions"""
    C = np.empty((len(X, )))
    for idx in range(len(X)):
        C[idx] = label_fun(X[idx], partitions, label_mat)
    return C.astype(int)


# Objective Functions
def modified_levy(x, freq):
    '''Modified Levy Function - min at (0,...,0), value - 0'''
    d = len(x)
    a = (1+(((x*freq)-1)/4)+(1/4))  # adding 1/4 to shift the minimum to origin
    a1 = np.sin(np.pi*a[0])**2
    a2 = sum([((a[j]-1)**2)*(1+10*(np.sin(np.pi*a[j])**2)) for j in np.arange(1,d-1)])
    a3 = ((a[d-1]-1)**2)*(1+np.sin(2*np.pi*a[d-1]))
    return a1+a2+a3


def modified_sphere(x, freq):
    '''Modified Sphere Function - min at +/-(sqrt(x1),...,sqrt(xd)), value - 0'''
    d = len(x)
    a = x*freq
    a1 = (10**(-0.5*d))*(sum([a[j]**2 for j in range(d)]))
    return a1


def modified_rosenbrock(x, freq):
    '''Modified Rosenbrock Function - min at (0,...,0), value - 0'''
    d = len(x)
    a = x*freq+1  # adding 1 to shift the minimum to origin 
    a1 = (10**(-2*d))*(sum([100*((a[j+1] - a[j]**2)**2) + (a[j]-1)**2 for j in range(d-1)]))
    return a1


def modified_rastrigin(x, freq):
    '''Modified Rastrigin Function - min at (0,...,0), value - 0'''
    d = len(x)
    a = x*freq
    a1 = (1e-2)*((100*d)+sum([(a[j]**2)-(100*np.cos(2*np.pi*a[j]/3)) for j in range(d)]))
    return a1


def modified_griewank(x, freq):
    '''Modified Griewank Function - min at (0,...,0), value - 0'''
    d = len(x)
    a = x*freq
    a1 = sum([(a[j]**2)/4000 for j in range(d)])
    a2 = np.prod([np.cos(a[j]/np.sqrt(j+1)) for j in range(d)])
    return 3+3*(a1-a2)


def modified_ackley(x, freq):
    '''Modified Ackley Function - min at (0,...,0), value - 0'''
    d = len(x)
    a = x*freq
    a1 = 10-(10*np.exp(-0.1*np.sqrt(sum([(a[j]**2)/d for j in range(d)]))))
    a2 = np.exp(1)-np.exp(sum([np.cos(2*np.pi*a[j]/3)/d for j in range(d)]))
    return a1+a2


def gen_ydataset(X, C, P, partitions, partition_mat, freq, const, alpha=0.1, std=0):
    """Output with different function in each class"""
    Y = np.empty((len(X, )))
    cen = [np.convolve(partitions[dim], np.ones(2), 'valid') / 2 for dim in range(len(partitions))]
    for samp_idx in range(len(X)):
        par_ind = np.where(partition_mat == P[samp_idx])
        par_cen = np.array([cen[dim][par_ind[dim][0]] for dim in range(len(cen))])
        if C[samp_idx] == 0:  # Modified Ackley Function 
            Y[samp_idx] = modified_ackley(X[samp_idx]-par_cen, freq[C[samp_idx]]) + const[P[samp_idx]] + std*(np.random.randn())
        elif C[samp_idx] == 1:  # Modified Rastrigin
            Y[samp_idx] = modified_rastrigin(X[samp_idx]-par_cen, freq[C[samp_idx]]) + const[P[samp_idx]] + std*(np.random.randn())
        elif C[samp_idx] == 2:  # Modified Griewank Function
            Y[samp_idx] = modified_griewank(X[samp_idx]-par_cen, freq[C[samp_idx]]) + const[P[samp_idx]] + std*(np.random.randn())
        elif C[samp_idx] == 3:  # Modified Sphere
            Y[samp_idx] = modified_sphere(X[samp_idx]-par_cen, freq[C[samp_idx]]) + const[P[samp_idx]] + std*(np.random.randn())
        elif C[samp_idx] == 4:  # Modified Levy
            Y[samp_idx] = modified_levy(X[samp_idx]-par_cen, freq[C[samp_idx]]) + const[P[samp_idx]] + std*(np.random.randn())
        elif C[samp_idx] == 5:  # Modified Rosenbrock
            Y[samp_idx] = modified_rosenbrock(X[samp_idx]-par_cen, freq[C[samp_idx]]) + const[P[samp_idx]] + std*(np.random.randn())
        else:  # Harmonic function
            Y[samp_idx] = np.cos(np.sum((X[samp_idx]-par_cen)*freq[C[samp_idx]])) + const[P[samp_idx]] + std*(np.random.randn())
    Y = alpha*Y.reshape(-1, )
    return Y


def p_ind_val(x, partitions):
    """ Function to compute the closest partition index and value for each datapoint
        x - single data point of dimension d
        partition - list of lists(per dimension partitions) of length 'd' """
    c_par = []
    dist = []
    for d_idx in range(len(x)):
        d1 = abs(partitions[d_idx] - x[d_idx])  # len(d1) - length of par - distance from each partition along dimension d_idx
        d2 = np.array(d1)
        c_par_idx = np.argmin(d2)  # index of partition with minimum distance along dimension d_idx
        dist_idx = np.min(d2)  # minimum of the distance to partitions along dimension d_idx
        c_par += [c_par_idx]  # list of indices of closest partitions along all dimensions
        dist += [dist_idx]  # list of minimum distances of closest partition along all dimensions
    min_dist = min(dist)  # Minimum of distance from the closest partitions along all dimensions
    p_ind = dist.index(min_dist)  # Dimension along which the closest partition lies
    p_val = partitions[p_ind][c_par[p_ind]]  # Closest Partition 
    return p_ind, p_val


def gen_p_ind_val(X, partitions):
    """ Function to compute closest partition index and partition value for the dataset"""
    data_Pind = np.empty((len(X),))
    data_Pval = np.empty((len(X),))
    for data_idx in range(len(X)):
        p_ind, p_val = p_ind_val(X[data_idx], partitions)
        data_Pind[data_idx] = int(p_ind)
        data_Pval[data_idx] = p_val
    return data_Pind, data_Pval


def gen_data_func(N, func_gen_ydata, bnds, d, freq, const, partitions, partition_mat,
                  label_mat, alpha=1, std=0):
    """ Function to generate data
    Inputs: 
    dimension - d
    # classes - k
    # partitions - p
    partition tuple - tup
    bounds - bnds
    initial budget - N
    frequency vector - freq
    exp factor - beta
    partition intercept vec - const
    multiplicative const - alpha
    noise standard deviation - std
    balanced partitions - balanced (True / False)
    
    Outputs - X, Y, C, P, pind_dataset, pval_dataset, partitions, label_mat, partition_mat
    """
    # X = gen_training_data(d, bnds, N)  # Generating Uniformly Random input data (X)
    x = latin_hypercube1(N, d)   # Generating data based on latin_hypercube in unit square
    X = from_unit_cube1(x, bnds)   # Rescaling the data from unit_square to corresponding bounded space
    C = gen_cls_dataset(X, partitions, label_mat)  # cls labels for the generated data
    P = gen_cls_dataset(X, partitions, partition_mat)  # partition labels for the generated data
    Y = func_gen_ydata(X, C, P, partitions, partition_mat, freq, const, alpha=alpha, std=std)  # Output y_dataset
    data_Pind, data_Pval = gen_p_ind_val(X, partitions)
    return X, Y, C, P, data_Pind, data_Pval


# Additional copy for the pickle files
def scaled_gen_ydataset(X, C, P, partitions, partition_mat, freq, const, alpha=10, std=0):
    '''Function set to minimization problem and rescaled to have the min as 0'''
    return 1 + (-1/min(const*alpha).item())*gen_ydataset(X, C, P, partitions, partition_mat, freq, 
                                                         const, alpha=alpha, std=std)

def scaled_gen_ydataset_min(X, C, P, partitions, partition_mat, freq, const, alpha=10, std=0):
    '''Function set to minimization problem and rescaled to have the min as 0'''
    return 1 + (-1/min(const*alpha).item())*gen_ydataset(X, C, P, partitions, partition_mat, freq, 
                                                         const, alpha=alpha, std=std)

def scaled_gen_ydataset_max(X, C, P, partitions, partition_mat, freq, const, alpha=10, std=0):
    '''Function set to maximization problem and rescaled to have the max as 1'''
    return (1/min(const*alpha).item())*gen_ydataset(X, C, P, partitions, partition_mat, freq, 
                                                    const, alpha=alpha, std=std)



def rw_data(X, C, P, partitions=None, partition_mat=None, freq=None, const=None, alpha=10, std=0):
    '''Function set to minimization problem and rescaled to have the min as 0'''
    func_rw = HA()
    return np.array([func_rw.get_cost(xi) for xi in X])


def gen_rw_data(N, rw_data, bnds, dim, freq, const, partitions, partition_mat,
                  label_mat, alpha=1, std=0):
    partitions = [[-1, 0.85, 0.95, 1], [-1, 0.85, 0.95, 1]]
    partition_mat = np.arange(3*3).reshape(3,3)
    label_mat= np.array([[1,1,1],
                        [1,0,1],
                        [1,1,1]])
    x = latin_hypercube1(N, dim)   # Generating data based on latin_hypercube in unit square
    X = from_unit_cube1(x, bnds)   # Rescaling the data from unit_square to corresponding bounded space
    C = gen_cls_dataset(X, partitions, label_mat)  # cls labels for the generated data
    P = gen_cls_dataset(X, partitions, partition_mat)  # partition labels for the generated data
    Y = rw_data(X, C, P, partitions, partition_mat, freq, const, alpha=alpha, std=std)
    data_Pind, data_Pval = gen_p_ind_val(X, partitions)
    return X, Y, C, P, data_Pind, data_Pval


