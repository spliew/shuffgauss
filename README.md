# Overview
This package implements numerical methods for calculating the differential privacy of variants of Gaussian mechanism in the shuffle model using Rényi differential privacy (RDP).
The following implementations are available:
- `SubGaussRDPtoDP`: Subsampled Gaussian mechanism
- `ShuffGaussRDPtoDP`: Shuffle Gaussian mechanism
- `SubShuffGaussRDPtoDP`: Subsampled Shuffle Gaussian mechanism
- `ApproxSCIGaussRDPtoDP`: [Shuffled Check-in](http://arxiv.org/abs/2206.03151) Gaussian mechanism

Our implementation particularly allows fast comparisons of (epsilon, delta) at different numbers of composition. See [Usage](#usage).

# Installation

To install,
```bash
git clone https://github.com/spliew/shuffgauss
cd shuffgauss
pip install -e .
```

This code supports Python 3.8 and newer. See [pyproject.toml](./pyproject.toml) for other requirements.

# Usage

```python
import shuffgauss as sg

# setting up parameters
sigma = 1 # gauss std deviation
n = 6e5 # total number of users
m = 6e4 # expected no. of subsampled users. we assume sampling_rate = m/n
delta = 1/n  # differential privacy parameter
mxlmbda = 20 # maximum rdp moment to calculate

# shuffle gaussian mechanism
sf = sg.ShuffGaussRDPtoDP(sigma, n, mxlmbda)
sf.get_shuff() # prepare the calculation
print(sf.get_eps(delta, 1)) # calculate epsilon when no of composition is 1, return a tuple of (epsilon, lmbda_min)
print(sf.get_eps(delta, 10)) # calculate epsilon when no of composition is 10

# shuffled checkin gaussian mechanism
sci = sg.ApproxSCIGaussRDPtoDP(sigma, n, m, mxlmbda)
sci.get_subshuff() # prepare the calculation
print(sci.get_eps(delta, 1)) # calculate epsilon when no of composition is 1
print(sci.get_eps(delta, 10)) # calculate epsilon when no of composition is 10
```

# Citation
If you use this code in your work, please cite our paper:

```
@article{liew2022shuffle,
  title={Shuffle Gaussian Mechanism for Differential Privacy},
  author={Liew, Seng Pei and Takahashi, Tsubasa},
  journal={arXiv preprint arXiv:2206.09569},
  year={2022}
}
```


This code is heavily influenced by [autodp](https://github.com/yuxiangw/autodp).
