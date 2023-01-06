import nestanet.sampling as n_sp
import numpy as np
import argparse
from pathlib import Path
from PIL import Image

parser = argparse.ArgumentParser(description="Worst-case perturbation - generate mask")

# side length of N by N image
parser.add_argument('-N', type=int, action='store', help='side length of NxN image')

# target sampling rate
parser.add_argument('-r', type=float, action='store', help='target sampling rate')

args = parser.parse_args()

N = args.N
m = N*N * args.r

# save to the directory where this script resides
script_dir = Path(__file__).parent

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

with open(script_dir / 'CC_stability-mask_details.txt', 'w') as f:
    print('Image size (number of pixels):', (N,N), file=f)
    print('Number of uniform measurements:', m_uni_exact, file=f)
    print('Number of variable measurements:', m_var_exact, file=f)
    print('Number of uniquely sampled frequencies:', m_exact, file=f)
    print('Target sample rate:', args.r, file=f)
    print('Effective sample rate:', m_exact/(N*N), file=f)

np.save(script_dir / 'CC_stability-mask.npy', mask)