"""
Generate plots from compare_without_restarts.py results.
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

with np.load(results_dir / 'compare_without_restarts-results.npz') as data:
    results = dict(data)

sns.set(context='paper', style='whitegrid', font='sans', font_scale=1.5, rc={'text.usetex' : True})

for method in results:
    if method == "restarts":
        label = "Restarted NESTA"
    else: # smoothing parameter for no-restart NESTA
        label = "No re., $\\mu = 10^{%d}$" % math.log10(float(method))

    end_idx = len(results[method])+1
    plt.semilogy(
        range(1,end_idx), 
        results[method], 
        label=label,
        linewidth=2.5)

plt.xlim(left=0, right=5000)
plt.legend(loc='lower right')
plt.savefig(
    plots_dir / 'compare_without_restarts-plot.pdf', 
    bbox_inches='tight',
    dpi=300)
