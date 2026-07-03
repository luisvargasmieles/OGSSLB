
# OGSSLB

This repository contains the R package OGSSLB, which implements the
method Outcome-Guided Spike-and-Slab Lasso Biclustering.

## Installation

You can install the development version of OGSSLB like so:

``` r
library(devtools)
install_github("luisvargasmieles/OGSSLB")
```


Numerical experiments testing this package can be found at [OGSSLB-examples](https://github.com/luisvargasmieles/OGSSLB-examples).

## Update (July 2026)

To significantly accelerate execution on high-dimensional datasets, this version introduces **Hyperparameter Thinning** for the internal SOUL algorithm step. 

Instead of running the computationally heavy ULA (Unadjusted Langevin Algorithm) loops to estimate the L2 regularization hyperparameter of the disease-bicluster classification model at every single EM iteration, thinning allows you to freeze and reuse the hyperparameter, running the SOUL optimization only every m-th iteration. Meanwhile, the underlying outcome guidance regression weights continue to optimize at every iteration, ensuring your clinical outcomes stay perfectly synchronized with the evolving biclusters without sacrificing algorithm stability.

### Example

``` r
library(OGSSLB)

# Run OGSSLB with hyperparameter thinning enabled
results <- OGSSLB(
  X = X_matrix, 
  Y = Y_matrix, 
  # ... other model parameters ...
  use_thinning_SOUL = TRUE,       # Toggle hyperparameter thinning scheduling
  thinning_factor_SOUL = 5        # Run the SOUL algorithm every 5 EM iterations
)
```

## References

- Luis A. Vargas-Mieles, Paul D. W. Kirk, Chris Wallace "Outcome-guided spike-and-slab Lasso Biclustering: A Novel Approach for Enhancing Biclustering Techniques for Gene Expression Analysis," arXiv preprint arXiv:2412.08416.
- Gemma E. Moran, Veronika Ročková, Edward I. George "Spike-and-slab Lasso biclustering," The Annals of Applied Statistics, Ann. Appl. Stat. 15(1), 148-173, (March 2021)