This repository is a class project for Berkeley [Stat 238](https://stat238.berkeley.edu/spring-2025/) - Bayesian Statistics, taught by Prof. Alexander Strang, in which we implement a version of [Gaussian Process Factor Analysis](https://users.ece.cmu.edu/~byronyu/software.shtml). We made use of some utility function from [Elephant](https://elephant.readthedocs.io/en/latest/tutorials/gpfa.html).

### Structure of the repo
The repo can be walked through in 4 notebooks:

0. readme. It details the derivations and explains numerical implementation variants.
1. makeData. It generates simulated data for testing.
2. vanillaGPFA. It runs our implementation.
3. appendix. It compares various implementations of the matrix inversion of the form used in this package. In a setting where the time series is long (per trial), the use of numpy broadcasting in our implementation of the inversion is faster than the Elephant implementation. 

Shijie Gu & Clay Smyth, April 3 - 8, 2025

