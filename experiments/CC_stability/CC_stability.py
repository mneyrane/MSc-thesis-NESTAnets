"""
Modification of stability experiments (test_stability.py) to run on
Alliance clusters.
"""
import argparse
import os
import re
import math
import nestanet.operators as n_op
import nestanet.sampling as n_sp
import nestanet.nn as n_nn
import nestanet.stability as n_st
import torch
import numpy as np
from pathlib import Path
from PIL import Image

# use GPU if available
device_g = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


### parse command-line args

parser = argparse.ArgumentParser(description="Restarted NESTA worst-case perturbation")
parser.add_argument('--save-dir', type=str, action='store', help='directory to save results')
parser.add_argument('--image-path', type=str, action='store', help='image path')
parser.add_argument('--eta', type=float, action='store', help='solver noise level')
parser.add_argument('--eta-pert', type=float, action='store', help='perturbation noise level')

args = parser.parse_args()


### load data

with Image.open(args.image_path) as im:
    X = np.asarray(im).astype(float) / 255

script_dir = Path(__file__).parent

mask = np.load(script_dir / 'CC_stability-mask.npy')

# obtain sample rate from mask details
with open(script_dir / 'CC_stability-mask_details.txt', 'r') as f:
    mask_details = f.read()
    match = re.search(r'^Target sample rate: (.+)$', mask_details, flags=re.MULTILINE)
    sample_rate = float(match.group(1))


### change to a new directory to save results in

os.makedirs(args.save_dir)
os.chdir(args.save_dir)


### parameters
eta = args.eta           # noise level
#sample_rate =           # sample rate -- read from file above
outer_iters = 8          # num of restarts + 1
r = math.exp(-1)         # decay factor
zeta = 1e-9              # CS error parameter
delta = 5e-4             # rNSP parameter

pga_num_iters = 500      # gradient ascent iterations
pga_lr = 1.0             # gradient ascent step size
eta_pert = args.eta_pert / math.sqrt(2) # perturbation noise level 
# needs to be scaled by math.sqrt(2) since the perturbation noise level 
# 'stab_eta' refers to the noise level of e1, the first block of the
# noise vector e (we assume e1 == e2)

# inferred parameters
# (some of these are defined early since they we will define the
#  reconstruction map via an anonymous function)
eps0 = np.linalg.norm(X,'fro')

N, _ = X.shape          # image size (assumed to be N by N)
m = sample_rate*N*N     # expected number of measurements

m_exact = np.sum(mask)
mask_t = (torch.from_numpy(mask)).bool().to(device=device_g)


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

np.savez('results.npz', recovery=X_rec, adv_recovery=X_pert_rec, adv_noise=p_pert)

with open('out_values.txt','w') as ofd:
    print("rec error:", np.linalg.norm(X_rec-X,'fro'), file=ofd)
    print("eta:", args.eta, file=ofd)
    print("eta pert:", args.eta_pert, file=ofd)
    print("pert size:", np.linalg.norm(p_pert,'fro'), file=ofd)
    print("pert rec error:", np.linalg.norm(X_rec-X_pert_rec,'fro'), file=ofd)
