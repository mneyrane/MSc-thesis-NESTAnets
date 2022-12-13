# -*- coding: utf-8 -*-
"""
Generate plots of exponential decay in reconstruction error when using
restarted NESTA to solve TV minimization with Fourier measurements.
"""
import math
import nestanet.operators as n_op
import nestanet.sampling as n_sp
import nestanet.nn as n_nn
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image

device_t = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### load image

with Image.open("images/GPLU_phantom_512.png") as im:
    X = np.asarray(im).astype(float) / 255


### parameters

# fixed parameters
eta = [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5]  # noise level
sample_rate = 0.15  # sample rate
outer_iters = 25    # num of restarts + 1
r = math.exp(-1)    # decay factor
zeta = 1e-9         # CS error parameter
delta = 20*1e-5     # rNSP parameter

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

mask_t = (torch.from_numpy(mask)).bool().to(device=device_t)

print('Image size (number of pixels):', (N,N))
print('Number of uniform measurements:', m_uni_exact)
print('Number of variable measurements:', m_var_exact)
print('Number of uniquely sampled frequencies:', m_exact)
print('Target sample rate:', sample_rate)
print('Effective sample rate:', 2*m_exact/(N*N))


### generate functions for measurement and weight operators

B = lambda x, mode: n_op.fourier_2d(x,mode,N,mask_t,use_gpu=False)*(N/math.sqrt(m))
W = lambda x, mode: n_op.discrete_gradient_2d(x,mode,N,N)
L_W = 2*math.sqrt(2)
c_B = N*N/m


### reconstruct image using restarted NESTA for each eta value

# create variables that are only need to be created once
X_vec_t = torch.from_numpy(np.reshape(X,N*N))

norm_fro_X = np.linalg.norm(X,'fro')
print('Frobenius norm of X:', norm_fro_X)

inner_iters = math.ceil(math.sqrt(2)/(r*N*delta))-1
print('Inner iterations:', inner_iters+1)

mu = []
eps = eps0
for k in range(outer_iters):
    mu.append(r*delta*eps)
    eps = r*eps + zeta

rel_errs_dict = dict()

for noise_level in eta:
    
    ### define the inverse problem

    noise1 = torch.randn(m_exact) + 1j*torch.rand(m_exact)
    noise2 = torch.randn(m_exact) + 1j*torch.rand(m_exact)
    
    e1 = noise_level * noise1 / (math.sqrt(2) * torch.linalg.norm(noise1,2))
    e2 = noise_level * noise2 / (math.sqrt(2) * torch.linalg.norm(noise2,2))

    y1 = B(X_vec_t,1) + e1
    y2 = B(X_vec_t,1) + e2
    
    
    ### compute restarted NESTA solution
    
    z0 = torch.zeros(N*N,dtype=X_vec_t.dtype)
        
    y1 = y1.to(device_t)
    y2 = y2.to(device_t)
    z0 = z0.to(device_t)

    _, iterates = n_nn.restarted_nesta_bernoulli_wqcbp(
        y1, y2, z0, B, W, c_B, L_W, 
        inner_iters, outer_iters, noise_level, mu, True)


    ### extract restart values

    final_its = [torch.reshape(its[-1],(N,N)) for its in iterates]

    rel_errs = list()

    for X_final in final_its:
        X_final = X_final.cpu().numpy()
        rel_errs.append(np.linalg.norm(X-X_final,'fro')/norm_fro_X)

    rel_errs_dict[noise_level] = rel_errs


### plots

sns.set(context='paper', style='whitegrid', font='sans', font_scale=1.4)#, rc={'text.usetex' : True})

for noise_level in eta:
    end_idx = len(rel_errs_dict[noise_level])+1
    plt.semilogy(
        range(1,end_idx), 
        rel_errs_dict[noise_level], 
        label='$\\eta = 10^{%d}$' % math.log10(noise_level),
        marker='o',
        markersize=4,
        linewidth=2)

plt.xlabel('Restart')
plt.ylabel('Relative error')
plt.legend(loc='lower left')
plt.savefig('restarts-plot.pdf', bbox_inches='tight', dpi=300)
