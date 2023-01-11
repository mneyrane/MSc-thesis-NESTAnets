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

with np.load(results_dir / 'delta_tuning-results.npz') as data:
    results = dict(data)

sns.set(context='paper', style='ticks', font='Arimo', font_scale=1.5)

cmap = mpl.colormaps['rainbow']
colors = cmap(np.linspace(0,1,num=len(results)))

for i, delta_val in enumerate(results):
    end_idx = len(results[delta_val])+1
    plt.semilogy(
        range(1,end_idx), 
        results[delta_val], 
        label='$\\log_{10}(\\delta) = %.2f$' % (float(delta_val)),
        color=colors[i],
#        marker='o',
#        markersize=5,
        linewidth=2.5)

plt.xlim(0,50)
plt.xticks([0,10,20,30,40,50])
#plt.xlabel('Restart')
#plt.ylabel('$\\lVert x_k^\\star - x \\rVert$')
plt.legend(loc='center right')
plt.savefig(plots_dir / 'delta_tuning-plot.pdf', dpi=300)
