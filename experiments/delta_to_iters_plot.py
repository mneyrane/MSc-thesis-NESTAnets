"""
Compares the effect of the delta parameter when reconstructing an image 
using restarted NESTA.

Here NESTA is solving a Fourier imaging problem via TV minimization.
"""
import math
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from PIL import Image


demos_path = Path(__file__).parent
results_dir = demos_path / 'results'
plots_dir = demos_path / 'plots'
plots_dir.mkdir(exist_ok=True)

### load image

with Image.open(demos_path / "images/GLPU_phantom_512.png") as im:
    X = np.asarray(im).astype(float) / 255

### functions

N, _ = X.shape # image size (assumed to be N by N)
r = math.exp(-1)
delta = np.logspace(-4,-2,num=200)

f_to_inner_iters = lambda delta : math.ceil(math.sqrt(2)/(r*N*delta))
f_to_rNSP_ratio = lambda delta : 2*delta*(N**2)

to_inner_iters = np.vectorize(f_to_inner_iters)
to_rNSP_ratio = np.vectorize(f_to_rNSP_ratio)

### generate plots

sns.set(context='paper', style='ticks', font='Arimo', font_scale=1.5)

# to inner iterations figure
plt.figure()

plt.semilogx(
    delta, 
    to_inner_iters(delta),
    linewidth=2.5)

#plt.xlabel('$\\delta$')
#plt.ylabel('Inner iterations')
plt.savefig(plots_dir / 'delta_to_inner_iters.pdf', dpi=300)

# to rNSP ratio figure
plt.figure()

plt.semilogx(
    delta, 
    to_rNSP_ratio(delta),
    linewidth=2.5)

#plt.xlabel('$\\delta$')
#plt.ylabel('$\\sqrt{s}/C$')
plt.savefig(plots_dir / 'delta_to_rNSP_ratio.pdf', dpi=300)
