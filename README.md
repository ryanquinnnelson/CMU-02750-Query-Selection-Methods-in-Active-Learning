# CMU-02750-HW1
Spring 2021 Automation of Scientific Research course project - Study of Query Selection Methods in Active Learning (HW1)

### Summary
There are two parts to this project:
- The first part of this project explores heuristic query selection methods with pool-based sampling (Uncertainty Sampling, Density-based Sampling, Query-by-Committee) using the `modAL` library. 
- The second part of this project implements the Importance Weighted Active Learning (IWAL) algorithm from Beygelzimer et al. (2009) with bootstrap rejection threshold and hinge loss. IWAL is developed as a Python package with unit tests (pytest) and documentation.

Analysis was performed using Jupyter Notebook and Python.


### Project Structure
The IWAL algorithm is implemented as Python package `iwal` and is found under /packages.


### Explanation of IWAL
#### Formal Version
IWAL is a Type I (hypothesis elimination) active learning algorithm used for binary and multiclass classification on any data access model. The algorithm labels instances in the disagreement region. To correct for sampling bias, IWAL uses an importance weighting strategy carefully chosen to control variance. Called "loss-weighting", this strategy defines the importance weight for a labeled instance to be inversely proportional to the range of predictions made for that instance over a bounded hypothesis space (i.e. close to optimal). 

The reason IWAL is consistent (i.e. converges to the optimal model) is that the rejection threshold it uses to decide whether or not to label an instance is bounded away from zero. With every instance having a chance of being selected for labeling, IWAL will eventually uncover all regions of disagreement.

#### Informal Version
*In layman's terms, the more disagreement about the predicted label of a given instance, the more likely the algorithm is to select that instance for labeling. This introduces bias: the instance is more likely to show up in the training set than the test set. Bias is corrected for by reducing the influence this labeled instance has within the training set by the same amount. IWAL is defined so every instance it considers has a chance of being selected for labeling (even if it is very small). This ensures that the algorithm will consistently find the best model over the long-run, regardless of the data it sees along the way.*


