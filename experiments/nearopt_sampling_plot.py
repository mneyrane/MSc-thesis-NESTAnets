import nestanet.sampling as n_sp
import numpy as np
from pathlib import Path
from PIL import Image

N = 512
sample_rate = 0.10

m = sample_rate * N*N

demos_path = Path(__file__).parent
plots_dir = demos_path / 'plots'
plots_dir.mkdir(exist_ok=True)

var_hist = n_sp.optimal_hist_2d(N)
var_probs = n_sp.bernoulli_sampling_probs_2d(var_hist,N,m)
var_mask = n_sp.generate_sampling_mask_from_probs(var_probs)

var_probs_im = (255*var_probs).astype(np.uint8)

# save Bernoulli probabilities
Image.fromarray(var_probs_im).save(plots_dir/'nearopt_sampling-probs.png')

# save example mask
Image.fromarray(var_mask).save(plots_dir/'nearopt_sampling-mask.png')
