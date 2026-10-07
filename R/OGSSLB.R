# This is a modification from the SSLB method from Gemma Moran's published R package
# available at https://github.com/gemoran/SSLB, to adapt our outcome-guided proposed method
# Reference: Moran, G. E., Rockova, V., & George, E. I. (2021). Spike-and-slab lasso biclustering.
# The Annals of Applied Statistics, 15(1), 148-173. DOI: 10.1214/20-AOAS1385
# Extended for OG-SSLB: Vargas-Mieles, L. A., Kirk, P. D. W., & Wallace, C. (2025). 
# Outcome-guided spike-and-slab Lasso Biclustering. Statistics and Computing, 35(179).

#' Outcome-Guided Spike-and-Slab Lasso Biclustering (OG-SSLB)
#'
#' @param Y Matrix of outcome variables Y \\in \\{0, 1\\}^{N \\times C}, representing disease presence/absence.
#' @param X Matrix of gene expression data X \\in R^{N \\times G}.
#' @param K_init Initial overestimate of the number of biclusters (K^*).
#' @param lambda1 Slab parameter (\\lambda_1) for the gene loading matrix \\Lambda.
#' @param lambda0s Sequence of spike parameters (\\lambda_0) for the gene loading matrix \\Lambda.
#' @param lambda1_tilde Slab parameter (\\tilde{\\lambda}_1) for the sample loading matrix Z.
#' @param lambda0_tildes Sequence of spike parameters (\\tilde{\\lambda}_0) for the sample loading matrix Z.
#' @param weights Initial matrix of weights W \\in R^{(K+1) \\times C} for the multinomial logistic regression.
#' @param IBP Indicator variable (1 or 0) for using the Indian Buffet Process prior.
#' @param a Beta prior hyperparameter for the gene indicators \\Gamma.
#' @param b Beta prior hyperparameter for the gene indicators \\Gamma.
#' @param a_tilde Beta prior hyperparameter for the finite approximation of sample indicators \\tilde{\\Gamma}.
#' @param b_tilde Beta prior hyperparameter for the finite approximation of sample indicators \\tilde{\\Gamma}.
#' @param alpha IBP parameter (\\tilde{\\alpha}) for the sample indicators.
#' @param d Pitman-Yor extension parameter (d \\in [0, 1)) for the sample indicators.
#' @param zeta_w \\ell_2 regularization hyperparameter (\\zeta_w) for the multinomial logistic regression weights.
#' @param EPSILON Convergence tolerance.
#' @param MAX_ITER Maximum number of EM iterations.
#' @param plot_conv Logical; whether to plot convergence progress.
#' @param iter_em_to_plot Iteration interval for plotting convergence.
#' @param dir_save_weight_grad Directory to save weights and gradients data if plot_conv = TRUE.
#' @param manual_set_stepsize_hyperparam_logreg Logical; manually set step size for hyperparameter estimation.
#' @param perc_max_stepsize_grad_desc Percentage scaling for maximum gradient descent step size.
#' @param l2_reg_log_reg \\ell_2 regularization hyperparameter for the multinomial logistic regression weights.
#' @param stepsize_graddesc_logreg Step size (\\delta_{AGD}) for Accelerated Gradient Descent.
#' @param use_thinning_SOUL Logical; enable thinning for SOUL algorithm MCMC samples.
#' @param thinning_factor_SOUL Thinning factor for SOUL MCMC sampling.
#' @param n_iter_burnIn_ULA_SOUL Number of burn-in iterations (N_0) for Unadjusted Langevin Algorithm in SOUL.
#' @param n_iter_ULA_SOUL Number of tracking iterations (n) for Unadjusted Langevin Algorithm in SOUL.
#' @param niter_graddesc_logreg Number of AGD iterations for W maximization.
#' @param niter_expgrad_graddesc_logreg Number of gradient evaluations per AGD iteration.
#' @param niter_exp_y Number of Monte Carlo samples (M) to approximate expected indicator variables.
#' 
#' @return A list containing the processed gene expression matrix (\code{X}), the estimated gene loading matrix (\code{B}), 
#' the estimated sample bicluster membership matrix (\code{Gamma_tilde}), the estimated covariance matrix (\code{ML}), 
#' the final number of biclusters (\code{K}), the initial gene loading matrix (\code{init_B}), 
#' the estimated multinomial logistic regression weights (\code{W}), the optimisation step size, 
#' the final \\ell_2 regularisation parameter, and convergence information.
OGSSLB <- function(Y,
                   X,
                   K_init,
                   lambda1 = 1,
                   lambda0s = c(1, 5, 10, 50, 100, 500, 1000, 10000,
                                100000, 1000000, 10000000),
                   lambda1_tilde = 1,
                   lambda0_tildes = c(1, rep(5, length(lambda0s) - 1)),
                   weights = matrix(
                    0.01 * rnorm((K_init + 1) * ncol(Y)),
                    nrow = K_init + 1,
                    ncol = ncol(Y)),
                   IBP = 1,
                   a = 1 / K_init,
                   b = 1,
                   a_tilde = 1 / K_init,
                   b_tilde = 1,
                   alpha = 1 / N,
                   d = 0,
                   EPSILON = 0.01,
                   MAX_ITER = 500,
                   plot_conv = FALSE,
                   iter_em_to_plot = 3,
                   dir_save_weight_grad = "conv_data",
                   manual_set_stepsize_hyperparam_logreg = FALSE,
                   perc_max_stepsize_grad_desc = 0.475,
                   l2_reg_log_reg = 0.5,
                   stepsize_graddesc_logreg = 0.1,
                   use_thinning_SOUL = FALSE,
                   thinning_factor_SOUL = 5,
                   n_iter_burnIn_ULA_SOUL = 500,
                   n_iter_ULA_SOUL = 100,
                   niter_graddesc_logreg = 200,
                   niter_expgrad_graddesc_logreg = 30,
                   niter_exp_y = 50) {

  N <- nrow(X)
  G <- ncol(X)

  # check if there's only one disease (no HC) in the Y variable
  if (is.vector(Y)) Y <- matrix(Y, ncol = 1)

  if (missing(K_init)) {
    stop("Must provide initial value of K (K_init)")
  }

  sigs <- apply(X, 2, sd)

  sigquant <- 0.5
  sigdf <- 3

  sigest <- quantile(sigs, 0.05)
  qchi <- qchisq(1 - sigquant, sigdf)
  xi <- sigest^2 * qchi / sigdf
  eta <- sigdf
  sigmas_median <- sigest^2
  sigmas_init <- rep(sigmas_median, G)
  sigma_min <- sigest^2 / G


  B_init <- matrix(rexp(G * K_init, rate = 1), nrow = G, ncol = K_init)
  Tau_init <- matrix(100, nrow = N, ncol = K_init)
  thetas_init <- rep(0.5, K_init)
  nus_init <- sort(rbeta(K_init, 1, 1), decreasing = T)
  theta_tildes_init <- rep(0.5, K_init)

  nlambda <- length(lambda0s)

  # if plot_conv = T; a folder with name given in variable "dir_save_weight_grad"
  # will be created in current directory, if it hasn't been created yet.

  if (plot_conv) {
    # Check if the folder exists
    if (!dir.exists(dir_save_weight_grad)) {
      # Create the folder if it doesn't exist
      dir.create(dir_save_weight_grad)
    }
  }

  res <- .cOGSSLB(Y, X, B_init, sigmas_init, Tau_init, thetas_init, theta_tildes_init, 
      nus_init, lambda1, lambda0s, lambda1_tilde, lambda0_tildes, weights, a, b, 
      a_tilde, b_tilde, alpha, d, eta, xi, sigma_min, IBP, EPSILON, MAX_ITER,
      plot_conv, iter_em_to_plot, dir_save_weight_grad, l2_reg_log_reg,
      stepsize_graddesc_logreg, n_iter_burnIn_ULA_SOUL, n_iter_ULA_SOUL,
      niter_graddesc_logreg, niter_expgrad_graddesc_logreg, niter_exp_y,
      manual_set_stepsize_hyperparam_logreg, perc_max_stepsize_grad_desc,
      use_thinning_SOUL, thinning_factor_SOUL)

  X <- res$X
  Tau <- res$Tau
  Gamma_tilde <- res$Gamma_tilde
  B <- res$B
  ML <- res$ML
  W <- res$W

  thetas <- lapply(res$thetas, as.vector)
  thetas <- thetas[!sapply(thetas, is.null)]

  theta_tildes <- lapply(res$theta_tildes, as.vector)
  theta_tildes <- theta_tildes[!sapply(theta_tildes, is.null)]

  nus <- lapply(res$nus, as.vector)
  nus <- nus[!sapply(nus, is.null)]

  sigmas <- res$sigmas

  iter <- as.vector(res$iter)

  keep_path <- which(iter > 0)


  iter <- iter[keep_path]

  K <- 0

  if (lambda1 != lambda0s[nlambda]) {
    X[Gamma_tilde < 0.5] <- 0
    Tau[Gamma_tilde < 0.5] <- 0
    keep <- which(apply(X, 2, function(x) sum(x != 0) > 1))
    X <- as.matrix(X[, keep])
    B <- as.matrix(B[, keep])
    ML <- as.matrix(ML[keep, keep])
    Tau <- as.matrix(Tau[, keep])
    Gamma_tilde <- as.matrix(Gamma_tilde[, keep])
    W <- as.matrix(W[c(1, keep + 1), ])
    thetas <- thetas[keep]
    theta_tildes <- theta_tildes[keep]
    nus <- nus[keep]

    K <- length(keep)
  }

  names_out <- c("X", "B", "Gamma_tilde",
                 "ML", "K", "init_B", "W",
                 "stepsize", "l2_reg_param",
                 "weights_progress", "samples_progress",
                 "l2_hyp_progress")

  out <- vector("list", length(names_out))
  names(out) <- names_out

  out$X <- X
  out$Gamma_tilde <- Gamma_tilde
  out$B <- B
  out$K <- K
  out$ML <- ML
  out$init_B <- B_init
  out$W <- W
  out$stepsize <- res$stepsize_logreg
  out$l2_reg_param <- res$lambda_l2_reg

  return(out)
}