
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

Instead of running the ULA (Unadjusted Langevin Algorithm) loops to estimate the L2 regularization hyperparameter of the disease-bicluster classification model at every single EM iteration, thinning allows you to freeze and reuse the hyperparameter, running the SOUL optimization only every m-th iteration. Meanwhile, the outcome guidance regression weights continue to optimize at every iteration, speeding up the bicluster estimation process.

### Example

``` r
library(OGSSLB)

# Run OGSSLB with hyperparameter thinning enabled
results <- OGSSLB(
  Y = disease_per_sample_binary_indicator_matrix,
  X = gene_expression_matrix,
  # ... other model parameters ...
  use_thinning_SOUL = TRUE,       # Toggle hyperparameter thinning scheduling
  thinning_factor_SOUL = 5        # Run the SOUL algorithm every 5 EM iterations
)
```

## References

- Luis A. Vargas-Mieles, Paul D. W. Kirk, Chris Wallace, "Outcome-guided spike-and-slab Lasso Biclustering: A Novel Approach for Enhancing Biclustering Techniques for Gene Expression Analysis." Stat Comput 35, 179 (2025). [doi.org/10.1007/s11222-025-10709-4](https://doi.org/10.1007/s11222-025-10709-4)
- Gemma E. Moran, Veronika Ročková, Edward I. George, "Spike-and-slab Lasso biclustering." Ann. Appl. Stat. 15(1), 148-173, (March 2021). [doi.org/10.1214/20-AOAS1385](https://doi.org/10.1214/20-AOAS1385)