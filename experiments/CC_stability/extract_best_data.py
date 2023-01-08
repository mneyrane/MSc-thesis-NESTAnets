import sys
import re
from pathlib import Path

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

# create results directory if not already present

demos_path = Path(__file__).parent.parent
results_dir = demos_path / 'results'
results_dir.mkdir(exist_ok=True)

# get top results from each noise level index

TOP_X_RESULTS = 1
NOISE_LEVEL_LABELS = (0,1,2,3)
rec_err_regex = re.compile(r'pert rec error: (.+)')

best_results = []

for i in NOISE_LEVEL_LABELS:
    folders = list(data_dir.glob('RESULTS-*-%s' % i))
    assert len(folders) > 0 # DEBUG

    results_err_pairs = []

    for path in folders:
        try:
            with open(path/'out_values.txt') as fd:
                text = fd.read()
                match_re = rec_err_regex.search(text)
                value_re = float(match_re.group(1))
                
                # select best based on reconstruction error            
                results_err_pairs.append((str(path.name),value_re))
        except:
            print('WARNING! Cannot read out_values.txt in %s, ignoring...' % path.name, file=sys.stderr)

    sorted_pairs = sorted(results_err_pairs, key=lambda x : x[1], reverse=True)
    assert sorted_pairs[0][1] >= sorted_pairs[-1][1] # DEBUG
    best_results.extend([x[0] for x in sorted_pairs[:TOP_X_RESULTS]])

assert len(best_results) == len(NOISE_LEVEL_LABELS) * TOP_X_RESULTS # DEBUG

with open(results_dir/'CC_stability_best_results.txt','w') as fd:
    for e in best_results:
        fd.write(str(e) + '\n')
