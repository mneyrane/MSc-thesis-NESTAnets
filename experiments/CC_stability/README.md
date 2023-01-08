# Stability experiment notes

The code here pertains to the worst-case perturbation experiment, and is mostly structured to run several distributed instances on computing clusters. 
`CC_stability_batch.sh` is a job script for [Slurm](https://slurm.schedmd.com/) that runs several instances of the experiment with different parameters.
The actual experiment is written in `CC_stability.py`, which is based off the desktop-runnable test version `test_stability.py`.

For the cluster experiments, we have

- `CC_stability_batch.sh` - the main script to run several experiment instances
- `CC_stability.py`- the main script uses this with specific parameters as command-line arguments
- `generate_mask.py` - generate a mask (to be committed to the repository) for the main script
- `extract_best_data.py` - obtain folder names of results that produced the worst-case perturbation for each respective noise level
- `plot_best_results.py` - plots the data from those experiments found by `extract_best_data.py`

We ran the experiments on Alliance Canada clusters. 
For more general information, please see the [Digital Research Alliance of Canada webpage](https://alliancecan.ca/en/services/advanced-research-computing).
