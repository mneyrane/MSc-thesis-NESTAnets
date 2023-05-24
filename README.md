# MSc-thesis-NESTAnets

Numerical experiments for my [master's thesis](https://summit.sfu.ca/item/36038), showcasing unrolled NESTA (NESTAnets) to recover images from Fourier measurements via TV minimization.

## Requirements

The experiments are written in [Python](https://www.python.org/downloads/) and can be run on any Linux distribution, provided the right Python version is packaged.

To run the experiments without issues, these were run with Python 3.10 and using

| Package | Version |
| ------- | ------- |
| `matplotlib` | 3.6.2 |
| `numpy` | 1.24.1 |
| `Pillow` | 9.4.0 |
| `scipy` | 1.10.0 |
| `seaborn` | 0.12.2 |
| `torch` | 1.13.1 |

We recommend using these versions or later versions. For convenience, a `requirements.txt` is provided in the repository for ease of installation via `pip`.

## Running the experiments

To run any of the experiments, we recommend using a Python [virtual environment](https://docs.python.org/3.10/library/venv.html) to set things up.

Below we assume the Bash shell is used. Proceeding, first create the virtual environment and source it:

```shell
$ mkdir env
$ python3 -m venv env
$ source env/bin/activate
```

Afterwards, clone the repository and then install the `nestanet` package defined in `setup.py`. This will install the requirements above as dependencies.

```shell
(env) $ git clone https://github.com/mneyrane/AS-NESTA-net.git
(env) $ cd AS-NESTA-net
(env) $ pip install -e .
```

Alternatively, if in the future some incompatible changes are made to the required packages, modify the final `pip` command above to

```shell
(env) $ pip install -r requirements.txt
```

All the experiments can be run on a desktop computer except the cluster version of the stability experiment (`CC_stability_batch.sh` and `CC_stability.py`). For further details, see `experiments/CC_stability/README.md`.

## Issues

You can post questions, requests, and bugs in [Issues](https://github.com/mneyrane/MSc-thesis-NESTAnets/issues).

## Acknowledgements

The unrolled NESTA implementation and experiments are directly adapted and extended from the NESTANet[^1] paper (by myself and Ben Adcock), which itself is adapted from the unrolled primal-dual iteration [FIRENETs](https://github.com/Comp-Foundations-and-Barriers-of-AI/firenet).

###### Footnotes

[^1]: You may instead be looking for the experiments of the related paper [NESTANets: stable, accurate and efficient neural networks for analysis-sparse inverse problems](https://doi.org/10.1007/s43670-022-00043-5), by myself and Ben Adcock. They are [here](https://github.com/mneyrane/AS-NESTA-net).
