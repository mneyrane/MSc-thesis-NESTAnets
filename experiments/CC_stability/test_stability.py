"""
Test script for parameter tuning the stability experiments, used as a 
template for writing a computer cluster version.

Here NESTA is solving a Fourier imaging problem via TV minimization.
"""
import math
import nestanet.operators as n_op
import nestanet.sampling as n_sp
import nestanet.nn as n_nn
import nestanet.stability as n_st
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# create results directory if not already present
demos_path = Path(__file__).parent.parent
results_dir = demos_path / 'results'
results_dir.mkdir(exist_ok=True)

# use GPU if available
device_g = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### load image

with Image.open(demos_path / "images/brain_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# fixed parameters
eta = 1e-2         # noise level
sample_rate = 0.25 # sample rate
outer_iters = 8    # num of restarts + 1
r = math.exp(-1)   # decay factor
zeta = 1e-9        # CS error parameter
delta = 5e-4       # rNSP parameter

pga_num_iters = 25
pga_lr = 1.0
stab_eta = 1000* (eta / math.sqrt(2))
# needs to be scaled by math.sqrt(2) since the perturbation noise level 
# refers to the noise level of e1, the first block of the noise vector e 
# (we assume e1 == e2)


# inferred parameters 
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


### setup worst-case perturbation computation

X_vec_t = torch.from_numpy(np.reshape(X,N*N))
X_vec_t = X_vec_t.to(device_g)

z0 = torch.zeros(N*N,dtype=X_vec_t.dtype)
z0 = z0.to(device_g)


### define reconstruction map

mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

inner_iters = math.ceil(math.sqrt(2)/(r*N*delta))
print('Inner iterations:', inner_iters)

def R(y):
    xout, _ = n_nn.restarted_nesta_stacked(
        y, y, z0, B, W, c_B, L_W,
        inner_iters, outer_iters, eta, mu)

    return xout

# compute measurements and image reconstruction
y = B(X_vec_t,1)
X_rec_t = R(y)

X_rec_t = X_rec_t.cpu()
X_rec = np.reshape(X_rec_t.numpy(),(N,N))
print("l2 error:", np.linalg.norm(X-X_rec,'fro'))

### compute worst-case perturbation

e_pert_t = n_st.adv_perturbation(
    X_vec_t, B, R, c_B=c_B, eta=stab_eta, 
    lr=pga_lr, num_iters=pga_num_iters, device=device_g)

p_pert_t = B(e_pert_t,0)/c_B
X_pert_rec_t = R(y+e_pert_t)

p_pert_t = p_pert_t.cpu()
X_pert_rec_t = X_pert_rec_t.cpu()

p_pert = np.reshape(p_pert_t.numpy(),(N,N))
X_pert_rec = np.reshape(X_pert_rec_t.numpy(),(N,N))

print("Pert size:", np.linalg.norm(p_pert,'fro'))
print("Pert l2 reconstruction error:", np.linalg.norm(X_rec-X_pert_rec,'fro'))

np.savez(results_dir / 'test_stability-results.npz', recovery=X_rec, adv_recovery=X_pert_rec, adv_noise=p_pert)
