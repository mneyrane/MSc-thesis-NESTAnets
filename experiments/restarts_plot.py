"""
Generate plots from restarts.py results.
"""
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

script_path = Path(__file__)
results_dir = script_path.parent / 'results'
plots_dir = script_path.parent / 'plots'
plots_dir.mkdir(exist_ok=True)

### load results

with np.load(results_dir / 'restarts-results.npz') as data:
    results = dict(data)

sns.set(context='paper', style='whitegrid', font='sans', font_scale=1.5, rc={'text.usetex' : True})

for noise_level in results:
    end_idx = len(results[noise_level])+1
    plt.semilogy(
        range(1,end_idx), 
        results[noise_level], 
        label='$\\eta = 10^{%d}$' % math.log10(float(noise_level)),
        marker='o',
        markersize=5,
        linewidth=2.5)

plt.xticks([0,4,8,12,16,20])
#plt.xlabel('Restart')
#plt.ylabel('$\\lVert x_k^\\star - x \\rVert$')
plt.legend(loc='lower left')
plt.savefig(plots_dir / 'restarts-plot.pdf', bbox_inches='tight', dpi=300)
