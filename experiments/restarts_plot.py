import math
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

demos_path = Path(__file__).parent
results_dir = demos_path / 'results'
plots_dir = demos_path / 'plots'
plots_dir.mkdir(exist_ok=True)

### load results

with np.load(results_dir / 'restarts-results.npz') as data:
    results = dict(data)

sns.set(context='paper', style='ticks', font='Arimo', font_scale=1.5)

cmap = mpl.colormaps['rainbow']
colors = cmap(np.linspace(1,0,num=len(results)))

for i, noise_level in enumerate(results):
    end_idx = len(results[noise_level])+1
    plt.semilogy(
        range(1,end_idx), 
        results[noise_level], 
        label='$\\eta = 10^{%d}$' % math.log10(float(noise_level)),
        color=colors[i],
        marker='o',
        markersize=4.5,
        linewidth=2.5)

plt.xticks([0,4,8,12,16,20])
#plt.xlabel('Restart')
#plt.ylabel('$\\lVert x_k^\\star - x \\rVert$')
plt.legend(loc='lower left')
plt.savefig(plots_dir / 'restarts-plot.pdf', dpi=300)
