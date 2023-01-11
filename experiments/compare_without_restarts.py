"""
Compare the performance of stacked NESTA with and without restarts, under a 
fixed iteration budget.

The comparison is made for solving a Fourier imaging problem via TV 
minimization.
"""
import math
import nestanet.operators as n_op
import nestanet.sampling as n_sp
import nestanet.nn as n_nn
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# create results directory if not already present
demos_path = Path(__file__).parent
results_dir = demos_path / 'results'
results_dir.mkdir(exist_ok=True)

# use GPU if available
device_g = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### load image

with Image.open(demos_path / "images/GLPU_phantom_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# fixed parameters
eta = 1e-7          # noise level
sample_rate = 0.15  # sample rate
outer_iters = 200   # num of restarts + 1
r = math.exp(-1)    # decay factor 
zeta = 1e-9         # CS error parameter
delta = 2e-4        # rNSP parameter

# smoothing parameter for NESTA (without restarts)
nesta_mu = 10**np.arange(-3,-8,-1, dtype=float) 

# inferred parameters (mu and inner_iters are defined later)
eps0 = np.linalg.norm(X,'fro')

N, _ = X.shape          # image size (assumed to be N by N)
m = sample_rate*N*N     # expected number of measurements


### generate sampling mask

uni_hist = n_sp.uniform_hist_2d(N)
uni_probs = n_sp.bernoulli_sampling_probs_2d(uni_hist,N,m/2)
uni_mask = n_sp.generate_sampling_mask_from_probs(uni_probs)

var_hist = n_sp.optimal_hist_2d(N)
var_probs = n_sp.bernoulli_sampling_probs_2d(var_hist,N,m/2)
var_mask = n_sp.generate_sampling_mask_from_probs(var_probs)

mask, _, _, _, _ = n_sp.stacked_scheme_2d(uni_mask, var_mask, N, N)

m_exact = np.sum(mask)
m_uni_exact = np.sum(uni_mask)
m_var_exact = np.sum(var_mask)

mask_t = (torch.from_numpy(mask)).bool().to(device=device_g)

print('Image size (number of pixels):', (N,N))
print('Number of uniform measurements:', m_uni_exact)
print('Number of variable measurements:', m_var_exact)
print('Number of uniquely sampled frequencies:', m_exact)
print('Target sample rate:', sample_rate)
print('Effective sample rate:', m_exact/(N*N))


### generate functions for measurement and weight operators

B = lambda x, mode: n_op.fourier_2d(x,mode,N,mask_t,device_g)*(N/math.sqrt(m))
W = lambda x, mode: n_op.discrete_gradient_2d(x,mode,N,N)
L_W = 2*math.sqrt(2)
c_B = N*N/m


### define the inverse problem

X_vec_t = torch.from_numpy(np.reshape(X,N*N))
X_vec_t = X_vec_t.to(device_g)

noise1 = torch.randn(m_exact) + 1j*torch.rand(m_exact)
noise2 = torch.randn(m_exact) + 1j*torch.rand(m_exact)

e1 = eta * noise1 / (math.sqrt(2) * torch.linalg.norm(noise1,2))
e2 = eta * noise2 / (math.sqrt(2) * torch.linalg.norm(noise2,2))

e1 = e1.to(device_g)
e2 = e2.to(device_g)

y1 = B(X_vec_t,1) + e1
y2 = B(X_vec_t,1) + e2


### compute restarted NESTA solution

norm_fro_X = np.linalg.norm(X,'fro')
print('Frobenius norm of X:', norm_fro_X)

inner_iters = math.ceil(math.sqrt(2)/(r*N*delta))
total_iters = outer_iters*inner_iters
print('Inner iterations:', inner_iters)
print('Total iterations:', total_iters)

z0 = torch.zeros(N*N,dtype=X_vec_t.dtype)
z0 = z0.to(device_g)
    
mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

eval_fns = {
    'l2_err' : lambda x : torch.linalg.norm(X_vec_t - x,2),
}

l2_errs_dict = {}

# compute restarted NESTA solution and compute relative error of iterates
_, re_ev_values = n_nn.restarted_nesta_stacked(
    y1, y2, z0, B, W, c_B, L_W, 
    inner_iters, outer_iters, eta, mu, eval_fns)

r_errs = [val for rvals in re_ev_values['l2_err'] for val in rvals]
l2_errs_dict['restarts'] = torch.as_tensor(r_errs).numpy()

# compute NESTA solutions and relative error of iterates
for n_mu in nesta_mu:
    _, n_ev_values = n_nn.nesta_stacked(
        y1, y2, z0, B, W, c_B, L_W,
        total_iters, eta, n_mu, eval_fns)

    n_errs = [val for val in n_ev_values['l2_err']]
    n_mu_str = "{:.0e}".format(n_mu)
    l2_errs_dict[n_mu_str] = torch.as_tensor(n_errs).numpy()

### save results

np.savez(results_dir / 'compare_without_restarts-results.npz', **l2_errs_dict)
