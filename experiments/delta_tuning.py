"""
Linear decay in reconstruction error when using restarted stacked NESTA.

Here NESTA is solving a Fourier imaging problem via TV minimization.
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

with Image.open(demos_path / "images/GPLU_phantom_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# fixed parameters
eta = 1e-5          # noise level
sample_rate = 0.125 # sample rate
outer_iters = 50    # num of restarts + 1
r = math.exp(-1)    # decay factor
zeta = 1e-9         # CS error parameter
delta = np.logspace(-3.76,-3.4,num=10)  # rNSP parameter

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

mask, u1_mask, u2_mask, perm1, perm2 = n_sp.stacked_scheme_2d(uni_mask, var_mask, N, N)

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

B = lambda x, mode: n_op.fourier_2d(x,mode,N,mask_t,use_gpu=True)*(N/math.sqrt(m))
W = lambda x, mode: n_op.discrete_gradient_2d(x,mode,N,N)
L_W = 2*math.sqrt(2)
c_B = N*N/m


### reconstruct image using restarted NESTA for each eta value

# create variables that are only need to be created once
X_vec_t = torch.from_numpy(np.reshape(X,N*N))
X_vec_t = X_vec_t.to(device_g)

norm_fro_X = np.linalg.norm(X,'fro')
print('Frobenius norm of X:', norm_fro_X)

### define the inverse problem

noise1 = torch.randn(m_exact) + 1j*torch.rand(m_exact)
noise2 = torch.randn(m_exact) + 1j*torch.rand(m_exact)

e1 = eta * noise1 / (math.sqrt(2) * torch.linalg.norm(noise1,2))
e2 = eta * noise2 / (math.sqrt(2) * torch.linalg.norm(noise2,2))

e1 = e1.to(device_g)
e2 = e2.to(device_g)

y1 = B(X_vec_t,1) + e1
y2 = B(X_vec_t,1) + e2


eval_fns = {
    'l2_err' : lambda x : torch.linalg.norm(X_vec_t - x,2),
}

l2_errs_dict = {}

for delta_val in delta:

    inner_iters = math.ceil(math.sqrt(2)/(r*N*delta_val))-1
    print('Inner iterations:', inner_iters+1, 'for delta =', delta_val)

    mu = []
    eps = eps0
    for k in range(outer_iters):
        mu.append(r*delta_val*eps)
        eps = r*eps + zeta

    ### compute restarted NESTA solution
    
    z0 = torch.zeros(N*N,dtype=X_vec_t.dtype)
    z0 = z0.to(device_g)

    _, re_ev_values = n_nn.restarted_nesta_stacked(
        y1, y2, z0, B, W, c_B, L_W, 
        inner_iters, outer_iters, eta, mu, eval_fns)


    ### extract restart values
    
    errs = [res[-1] for res in re_ev_values['l2_err']]
    
    delta_val_str = "{:.2f}".format(math.log10(delta_val))
    print('delta_val_str:', delta_val_str)
    l2_errs_dict[delta_val_str] = torch.as_tensor(errs).numpy()


### save results

np.savez(results_dir / 'delta_tuning-results.npz', **l2_errs_dict)
