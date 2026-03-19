import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('FocalBO-main'), '..')))
from cbo_func_HA import HA


class Rastrigin:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Rastrigin'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        f = 10*self.dim + np.sum(x**2-np.cos(2*np.pi*x))
        if not self.minimize:
            return -f
        else:
            return f


class Ackley:
    def __init__(self, dim=10, minimize=True):
        self.dim = dim
        self.lb = -5 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.name = 'Ackley'
        self.minimize = minimize

    def __call__(self, x):
        assert len(x) == self.dim
        assert x.ndim == 1
        # print(x)
        assert np.all(x <= self.ub) and np.all(x >= self.lb), x
        a, b, c = 20.0, 0.2, 2*np.pi
        f = -a*np.exp(-b*np.sqrt(np.mean(x**2)))
        f -= np.exp(np.mean(np.cos(c*x)))
        f += a + np.exp(1)
        if not self.minimize:
            return -f
        else:
            return f


class Levy:
    def __init__(self, dim=1, minimize=True):
        assert dim > 0
        self.dim = dim
        self.minimize = minimize
        self.lb = -5 * np.ones(dim)
        self.ub = 5 * np.ones(dim)
        self.name='Levy'

    def __call__(self, x):
        x = np.array(x)
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)

        w = []
        for idx in range(0, len(x)):
            w.append(1 + (x[idx] - 1) / 4)
        w = np.array(w)

        term1 = (np.sin(np.pi*w[0]))**2
        term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
        term2 = 0
        for idx in range(1, len(w)):
            wi = w[idx]
            new = (wi-1)**2 * (1 + 10 * (np.sin(np.pi * wi + 1))**2)
            term2 = term2 + new

        result = term1 + term2 + term3

        if not self.minimize:
            return -result
        else:
            return result
        
class Synthetic:
    def __init__(self, dim=2, minimize=True, balanced=True, k=3, tup=(3,3), partition=9, std=0.1, partition_mat=None, label_mat=None, partitions=None, const=None, alpha=1):
        self.name = 'Synthetic'
        self.dim = dim
        self.lb = -10 * np.ones(dim)
        self.ub = 10 * np.ones(dim)
        self.minimize = minimize
        self.balanced = balanced
        self.k = k  # number of classes
        self.tup = tup  # tuple of partitions along each dimension
        self.partition = partition  # total number of partitions
        self.partitions = partitions # list of partition arrays along each dimension
        self.partition_mat = partition_mat # partition matrix with partition index for each partition
        self.label_mat = label_mat # label matrix with class label for each partition
        self.obj_bnds = [(self.lb[i], self.ub[i]) for i in range(self.dim)]
        self.freq = np.ones((k, dim))
        self.const = const
        self.std = std  # noise std dev
        self.alpha = alpha
        self.scale = -1/min(self.const*self.alpha)
        # self.optimal_value = 0  # known optimal value

    def __call__(self, x):
        x = np.array(x.cpu())
        assert len(x) == self.dim
        assert x.ndim == 1
        assert np.all(x <= self.ub) and np.all(x >= self.lb)
        p = label_fun(x, self.partitions, self.partition_mat)
        c = label_fun(x, self.partitions, self.label_mat)
        f = self.scale*gen_synthetic(x, p, c, self.partitions, self.partition_mat, self.freq, self.const, self.std, alpha=self.alpha)
        if not self.minimize:
            return -f - 1 # maximum value of 0
        else:
            return 1 + f  # minimum value of 0


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


def gen_synthetic(x, p, c, partitions, partition_mat, freq, const, std, alpha):
    """Output with different function in each class"""
    cen = [np.convolve(partitions[dim], np.ones(2), 'valid') / 2 for dim in range(len(partitions))]
    par_ind = np.where(partition_mat == p)
    par_cen = np.array([cen[dim][par_ind[dim][0]] for dim in range(len(cen))])
    if c == 0:  # Modified Ackley Function 
        y = modified_ackley(x-par_cen, freq[c]) + const[p] + std*(np.random.randn())
    elif c == 1:  # Modified Rastrigin
        y = modified_rastrigin(x-par_cen, freq[c]) + const[p] + std*(np.random.randn())
    elif c == 2:  # Modified Griewank Function
        y = modified_griewank(x-par_cen, freq[c]) + const[p] + std*(np.random.randn())
    elif c == 3:  # Modified Sphere
        y = modified_sphere(x-par_cen, freq[c]) + const[p] + std*(np.random.randn())
    elif c == 4:  # Modified Levy
        y = modified_levy(x-par_cen, freq[c]) + const[p] + std*(np.random.randn())
    elif c == 5:  # Modified Rosenbrock
        y = modified_rosenbrock(x-par_cen, freq[c]) + const[p] + std*(np.random.randn())
    else:  # Harmonic function
        y = np.cos(np.sum((x-par_cen)*freq[c])) + const[p] + std*(np.random.randn())
    y = alpha*y
    return y


def gen_rw(x, p, c, partitions, partition_mat, freq, const, std, alpha):
    """Output with different function in each class"""
    y = rw_data(x, freq)
    return y


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


def rw_data(x, freq=None):
    '''Function set to minimization problem and rescaled to have the min as 0'''
    func_rw = HA()
    return np.array(func_rw.get_cost(x))