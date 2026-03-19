import os
import time
import argparse
from copy import deepcopy
# import multiprocessing as mp
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
# from functions.design_bench_fun import DesignBenchFunction # recommend to use another python env
import gpytorch
from botorch.test_functions import SyntheticTestFunction
from botorch.utils.transforms import unnormalize, normalize
from botorch import test_functions
from numpy.random import RandomState
# from botorch.models import ApproximateGPyTorchModel

from gp_model import train_gp, train_fvgp

from turbo_botorch import TurboState
from acqf import focal_acqf_opt_sample
from functions.synthetic_fun import Synthetic, gen_synthetic, label_fun, gen_checker_par_unbalanced1, gen_checker_partitions

from os import listdir
from os.path import isfile, join
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('FocalBO-main'), '..')))
from cbo_fin_datagen_v1 import gen_checker_partitions, gen_checker_par_unbalanced1, gen_cls_dataset, scaled_gen_ydataset_min, scaled_gen_ydataset_max
from cbo_fin_treelib_v1 import leaf_node_data
from joblib import Parallel, delayed

mypath = '/home/mmalu/cbo_code_011525/exp_baselines/'
f_idx = 17
strings = ['all_parallel_hard_normalized_alpha_vs_nosupp_base_scaled_noisy_bal_False', 'all_parallel_easy_normalized_alpha_vs_nosupp_base_scaled_noisy_bal_False',
           'all_parallel_bal_normalized_alpha_vs_nosupp_base_scaled_noisy_bal_True']
f = []


onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for i in onlyfiles:
    for j in strings:
        if (i.startswith(j)) and (i.endswith('.pkl')) and ('matern' in i) and (('dpar' in i) or ('15d' in i)) and ('300_iterations' in i):
            f += [i]
f.sort()
print(f_idx, f[f_idx])
sys.stdout.flush()
data = pkl.load(open(join(mypath,f[f_idx]), 'rb'))

# Initiailization for test function
dim = data['params']['d']
bnds = data['params']['obj_bnds']
minimize = False
balanced = data['params']['balanced']
k = data['params']['k']
tup = data['params']['tup']
partition = data['params']['p']
std = data['params']['std']
alpha = data['params']['alpha_fun']
if balanced:
    partitions, label_mat = gen_checker_partitions(dim, k, tup, bnds)
else:
    partitions, label_mat = gen_checker_par_unbalanced1(dim, k, tup, bnds)
partition_mat = np.arange(partition).reshape(tup)
const = data['params']['const_fun']

# Initialization for focal bo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float
torch.manual_seed(0)
gp_model = 'fvgp'
max_loop_num = 1
init = False
algo = 'bo'
acqf = 'ts'
n_initial_samples = 5 # number of initial random samples
n_evals = data['params']['T']  + n_initial_samples # Total number of evals
n_repeat = 1
n_samples_per_step = 1  # number of new samples per step
induce_size = 5
f_objective = 'func'
use_depth = 1
auto = 1

# Define the objective function
if f_objective == 'func':
    bounds = torch.cat((-10*torch.ones(1, dim), 10*torch.ones(1, dim))).to(dtype=dtype, device=device) # bounds of the problem
    objective = Synthetic(dim=dim, minimize=minimize, balanced=balanced,
                           k=k, tup=tup, partition=partition, std=std, 
                           partition_mat=partition_mat, label_mat=label_mat, 
                           partitions=partitions, const=const, alpha=alpha)

    
def eval_objective(x):
    unnorm_x = unnormalize(x, bounds)
    y = torch.zeros(x.shape[0], 1).to(dtype=dtype, device=device)
    for i in range(x.shape[0]):
        y_int = objective(unnorm_x[i])
        y[i] = y_int[0]
    return y

cur_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
obj_name = objective.name
save_path = f'results/hard_hd_{cur_time}_{objective.name}_{algo}_{acqf}_d_{dim}_p_{partition}_{n_evals}_{n_initial_samples}_{n_samples_per_step}_{induce_size}_{gp_model}_{use_depth}'
save_path += f'_{max_loop_num}_{init}'
auto_suffix = '_auto' if auto == 1 else ''
save_path += auto_suffix
if not os.path.exists(save_path):
    os.mkdir(save_path)


def single_run(run_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float
    gp_model = 'fvgp'
    max_loop_num = 1
    init = False
    algo = 'bo'
    acqf = 'ts'
    n_initial_samples = 5 # number of initial random samples
    n_evals = data['params']['T'] + n_initial_samples  # number of optimization steps
    n_samples_per_step = 1  # number of new samples per step
    induce_size = 5
    use_depth = 1
    auto = 1
    rep = run_id
    print(f"\n+++++++++++++++++ Starting Run {run_id + 1}/{runs} ++++++++++++++++++++++++")
    sys.stdout.flush()
    unnorm_initial_X = torch.from_numpy(data[run_id]['X']).to(dtype=dtype, device=device)
    initial_X = normalize(unnorm_initial_X, bounds)
    initial_Y = -1*torch.from_numpy(data[run_id]['Y'].reshape(-1, 1)).to(dtype=dtype, device=device)
    with gpytorch.settings.max_cholesky_size(float('inf')):
        start_time = time.time()
        model = likelihood = None
        max_loop_num = max_loop_num
        
        if not isinstance(objective, SyntheticTestFunction):
            train_X = initial_X[:n_initial_samples].clone()
            train_Y = initial_Y[:n_initial_samples].clone()
        best_observed_value_index = train_Y.argmax()

        state = TurboState(dim, batch_size=n_samples_per_step, length=1)
        try:
            print(f"{obj_name} {rep} Init {len(train_X)} Best regret: {objective.optimal_value - train_Y[best_observed_value_index]}")
            sys.stdout.flush()
        except:
            print(f"{obj_name} {rep} Init {len(train_X)} Best reward: {train_Y[best_observed_value_index]}")
            sys.stdout.flush()
            
        all_loop_nums = []
        all_max_depth = []
        all_depth_record = torch.zeros(0, 1).to(dtype=dtype, device=device)
        while len(train_X) < n_evals:
            if len(train_X) % 5 == 0:
                print(f'=====Start optimization round with {len(train_X)} samples for run {run_id + 1}=====')
                sys.stdout.flush()
            norm_Y = (train_Y-train_Y.mean())/train_Y.std()
            norm_Y = norm_Y.ravel()
            train_dataset = TensorDataset(train_X, norm_Y)
            # train_dataset = TensorDataset(train_X, train_Y)
            train_loader = DataLoader(train_dataset, batch_size=30000, shuffle=True)
            x_center = 0.5 * torch.ones(1, dim).to(dtype=dtype, device=device) # center of the input region
                
            if gp_model == 'gp' or len(train_X) < induce_size:
                model, likelihood = train_gp(train_X, norm_Y, num_epochs=num_epochs)
                tr = cand_tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)
                # model = train_gp(train_X, train_Y, num_epochs=num_epochs)
            else:
                if gp_model == 'fvgp':
                    try:
                        base_length = state.length
                        in_center_num = 0
                        increase_step = 0.01
                        tr =  torch.cat((torch.zeros(dim).reshape(1, -1), torch.ones(dim).reshape(1, -1))).to(dtype=dtype, device=device)
                        cand_tr = tr
                        # print(tr)
                    except Exception as e:
                        print('error', e)
                        # tr = torch.cat((torch.zeros(1, dim), torch.ones(1, dim))).to(dtype=dtype, device=device)
                        raise NotImplementedError
                    
                    model, likelihood = train_fvgp(train_loader, tr, num_epochs=num_epochs, induce_size=induce_size, 
                                                    init=init)
            # Define the acquisition function
            if gp_model == 'fvgp' or use_depth == 1: # sparse to dense optimization for fvgp and vanilla BO
                train_new_gp = True if gp_model == 'fvgp' else False
                new_X, new_depth = focal_acqf_opt_sample(model, likelihood, deepcopy(state), x_center, acqf, 
                                        train_loader, num_epochs=num_epochs, 
                                        max_loop_num=max_loop_num, batch_size=n_samples_per_step, init=init, tr=tr, cand_tr=cand_tr, train_new_gp=train_new_gp)
                # print('x next', new_X)
                depth_num = []
                for l_idx in range(max_loop_num):
                    depth_num.append((new_depth == l_idx).sum().item())
                print(f'Batch number of each depth {depth_num}')
                all_depth_record = torch.cat((all_depth_record, new_depth), 0)
            
            # Update the training data, make sure to convert back to cuda bc of vecchia
            # Evaluate the objective function at the new samples
            device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            train_X, train_Y, norm_Y = train_X.to(device=device), train_Y.to(device=device), norm_Y.to(device=device), 
            
            new_Y = eval_objective(new_X.to(device=device)).reshape(-1, 1)
            
            train_X = torch.cat([train_X, new_X.to(device=device)])
            train_Y = torch.cat([train_Y, new_Y.to(device=device)])
            
            # Print the location of the best observed value
            best_observed_value_index = train_Y.argmax()
            try:
                print(f'{obj_name} {gp_model} {algo} {rep} Eval {len(train_X)}'\
                    f'Best this round: {objective.optimal_value - new_Y.max()} '\
                    f'Best regret: {objective.optimal_value - train_Y[best_observed_value_index]}')\
                
            except:
                print(f'{obj_name} {gp_model} {algo} {rep} Eval {len(train_X)}'\
                    f'Best this round: {new_Y.max()} '\
                    f'Best reward: {train_Y[best_observed_value_index]}')
                
            if gp_model == 'fvgp' or use_depth ==1 and len(train_X) >= induce_size:
                budget_per_loop = []
                total_budget = n_samples_per_step
                n_per_loop = int(np.ceil(n_samples_per_step/max_loop_num))
                val_per_depth = []
                
                for i in range(max_loop_num):
                    b = min(n_per_loop, total_budget)
                    budget_per_loop.append(b)
                    total_budget -= b
                
                budget_per_loop.reverse()
                start = 0
                for b in budget_per_loop:
                    val_per_depth.append(new_Y[start:start+b].mean())
                    start += b
                
                # check the best depth
                max_idx = new_Y.argmax()
                
                depth = new_depth[max_idx].item()  
                if auto == 1:
                    if depth + 1 < max_loop_num:
                        max_loop_num -= 1
                    else:
                        # max_loop_num = min(dim, max_loop_num+1)
                        max_loop_num += 1
                    # max_loop_num = depth + 1
                print(f'Max point find in depth {depth+1}, set max loop num to {max_loop_num}')
                    
                all_loop_nums.append(max_loop_num)
                all_max_depth.append(depth+1)

            # Save results
            best_x = train_X[best_observed_value_index].detach().cpu().numpy()
            np.savez(f'{save_path}/{run_id}.npz', x=train_X.detach().cpu().numpy(), y=train_Y.detach().cpu().numpy(), 
                        best_x=best_x, loop_num=np.array(all_loop_nums), 
                        max_depth=np.array(all_max_depth), depth_record=all_depth_record.detach().cpu().numpy())
        print("--- %s seconds ---" % (time.time() - start_time))


num_epochs = 200
n_jobs = 10
runs = 10
Parallel(n_jobs=n_jobs)(delayed(single_run)(i) for i in range(runs))





