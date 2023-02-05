import sys
import re
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr
from pathlib import Path
from PIL import Image

# expecting input path to data directory

try:
    data_dir = Path(sys.argv[1])
    assert data_dir.is_dir()
except (IndexError, AssertionError):
    print("Need argument to be a path to results directory.", file=sys.stderr)
    sys.exit(1)
except:
    print("Failed to open results directory path.", file=sys.stderr)
    sys.exit(1)

demos_path = Path(__file__).parent.parent
results_dir = demos_path / 'results'
plots_dir = demos_path / 'plots'
plots_dir.mkdir(exist_ok=True)


regex = re.compile(r'RESULTS-.+-(\d)')


sns.set(context='paper', style='ticks', font='Arimo', font_scale=1.5)
cmap = 'plasma'
fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(9.5,16))

with open(results_dir/'CC_stability_best_results.txt') as fd:
    for line in fd:
        exp_str = line.strip('\n')
        exp_path = Path(exp_str)

        match = regex.match(exp_str)
        label = int(match.group(1))

        with np.load(data_dir/exp_path/'results.npz') as data:
            results = dict(data)
        
        #''' 
        pos0 = axs[label][0].imshow(
            np.abs(results['adv_noise']),
            #xticklabels=[], yticklabels=[], 
            cmap=cmap, norm=clr.PowerNorm(4/5))

        cb0 = fig.colorbar(pos0, ax=axs[label][0])

        axs[label][0].axis('off')
        cb0.outline.set_visible(False)

        pos1 = axs[label][1].imshow(
            np.abs(results['recovery'] - results['adv_recovery']),
            #xticklabels=[], yticklabels=[], 
            cmap=cmap, norm=clr.PowerNorm(2/5))

        cb1 = fig.colorbar(pos1, ax=axs[label][1])

        axs[label][1].axis('off')
        cb1.outline.set_visible(False)

        #'''
        
        ''' old heatmap generation
        # show adversarial perturbation rescaled
        sns.heatmap(
            np.abs(results['adv_noise']), 
            xticklabels=[], yticklabels=[], 
            cmap=cmap, norm=clr.PowerNorm(4/5), ax=axs[label][0])

        # show absolute difference of truth and perturbed reconstruction 
        sns.heatmap(
            np.abs(results['recovery'] - results['adv_recovery']),
            xticklabels=[], yticklabels=[],
            cmap=cmap, norm=clr.PowerNorm(2/5), ax=axs[label][1])
        #'''

        # save perturbed and reconstruction of perturbed image
        
        im_rec = np.clip(np.abs( results['recovery'] )*255,0,255).astype('uint8')
        Image.fromarray(im_rec).save(plots_dir / ('CC_stability_'+exp_str+'-im_rec.png'))
        
        adv_rec = np.clip(np.abs( results['adv_recovery'] )*255,0,255).astype('uint8')
        Image.fromarray(adv_rec).save(plots_dir / ('CC_stability_'+exp_str+'-adv_rec.png'))

fig.savefig(plots_dir / 'CC_stability-subplots.pdf', bbox_inches='tight', dpi=300)
