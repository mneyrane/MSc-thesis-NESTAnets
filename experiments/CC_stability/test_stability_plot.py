import math
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from pathlib import Path
from PIL import Image

demos_path = Path(__file__).parent.parent
results_dir = demos_path / 'results'
plots_dir = demos_path / 'plots'
plots_dir.mkdir(exist_ok=True)

### load results

with np.load(results_dir / 'test_stability-results.npz') as data:
    results = dict(data)

# recovery from true measurements
true_rec = np.clip(np.abs(results['recovery'])*255,0,255).astype('uint8')
Image.fromarray(true_rec).save(plots_dir / 'test_stability-true_rec.png')

# recovery from adversarial measurements
adv_rec = np.clip(np.abs(results['adv_recovery'])*255,0,255).astype('uint8')
Image.fromarray(adv_rec).save(plots_dir / 'test_stability-adv_rec.png')


### plots

sns.set(context='paper', style='ticks', font='Arimo', font_scale=1.5)

cmap = mpl.colormaps['plasma']

# adversarial noise
plt.figure()
sns.heatmap(
    np.abs(results['adv_noise']), 
    xticklabels=[], yticklabels=[], 
    cmap=cmap, norm=clr.PowerNorm(4/5)
)
plt.savefig(plots_dir / 'test_stability-adv_noise.png', dpi=300)

# elementwise reconstruction error
plt.figure()
sns.heatmap(
    np.abs(results['recovery'] - results['adv_recovery']), 
    xticklabels=[], yticklabels=[], 
    cmap=cmap, norm=clr.PowerNorm(2/5)
)
plt.savefig(plots_dir / 'test_stability-error.png', dpi=300)
