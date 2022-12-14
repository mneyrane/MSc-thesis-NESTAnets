"""
Generate plots from compare_without_restarts.py results.
"""
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path

script_path = Path(__file__)
results_dir = script_path.parent / 'results'
plots_dir = script_path.parent / 'plots'
plots_dir.mkdir(exist_ok=True)

### load results

with np.load(results_dir / 'eta_zeta_tuning-results.npz') as data:
    results = dict(data)

sns.set(context='paper', style='whitegrid', font='sans', font_scale=1.5, rc={'text.usetex' : True})

xticklabels = np.log10(results['eta']).astype(int)
yticklabels = np.log10(results['zeta']).astype(int)
sns.heatmap(
    results['errs'], 
    xticklabels=xticklabels, yticklabels=yticklabels, 
    norm=LogNorm(), cmap='viridis')
plt.yticks(rotation=0)
#plt.xlabel('$\\log_{10}(\\zeta)$')
#plt.ylabel('$\\log_{10}(\\eta)$')
plt.savefig(plots_dir / 'eta_zeta_tuning-plot.pdf', bbox_inches='tight', dpi=300)
