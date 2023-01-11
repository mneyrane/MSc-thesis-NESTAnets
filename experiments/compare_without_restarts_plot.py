import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

demos_path = Path(__file__).parent
results_dir = demos_path / 'results'
plots_dir = demos_path / 'plots'
plots_dir.mkdir(exist_ok=True)

### load results

with np.load(results_dir / 'compare_without_restarts-results.npz') as data:
    results = dict(data)

sns.set(context='paper', style='ticks', font='Arimo', font_scale=1.5)

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
    dpi=300)
