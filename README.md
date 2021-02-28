# CMU-02750-HW1
Spring 2021 Automation of Scientific Research course project - Study of Query Selection Methods in Active Learning

### Summary
The first part of this project explores heuristic query selection methods with pool-based sampling (Uncertainty 
Sampling, Density-based Sampling, Query-by-Committee) using the `modAL` library. The second part of this project
implements the Importance Weighted Active Learning (IWAL) algorithm from Beygelzimer et al. (2009) with bootstrap 
rejection threshold and hinge loss. IWAL is developed as a Python package with unit tests (pytest) and documentation.


### Project Structure
The IWAL algorithm is implemented as Python package `iwal` and is found under /packages.