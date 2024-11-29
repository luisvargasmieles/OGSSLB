
OGSSLB <- function(Dis,
                   Y,
                   K_init,
                   lambda1 = 1,
                   lambda0s = c(1, 5, 10, 50, 100, 500, 1000, 10000,
                                100000, 1000000, 10000000),
                   lambda1_tilde = 1,
                   lambda0_tildes = c(1, rep(5, length(lambda0s) - 1)),
                   weights = matrix(
                    0.01 * rnorm((K_init + 1) * ncol(Dis)),
                    nrow = K_init + 1,
                    ncol = ncol(Dis)),
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
                   n_iter_burnIn_ULA_SOUL = 500,
                   n_iter_ULA_SOUL = 100,
                   niter_graddesc_logreg = 200,
                   niter_expgrad_graddesc_logreg = 30,
                   niter_exp_y = 50) {

  N <- nrow(Y)
  G <- ncol(Y)

  # check if there's only one disease (no HC) in the Dis variable
  if (is.vector(Dis)) Dis <- matrix(Dis, ncol = 1)

  if (missing(K_init)) {
    stop("Must provide initial value of K (K_init)")
  }

  sigs <- apply(Y, 2, sd)

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

  sourceCpp("functions/cOGSSLB.cpp")

  res <- cOGSSLB(Dis, Y, B_init, sigmas_init, Tau_init, thetas_init, theta_tildes_init, 
      nus_init, lambda1, lambda0s, lambda1_tilde, lambda0_tildes, weights, a, b, 
      a_tilde, b_tilde, alpha, d, eta, xi, sigma_min, IBP, EPSILON, MAX_ITER,
      plot_conv, iter_em_to_plot, dir_save_weight_grad, l2_reg_log_reg,
      stepsize_graddesc_logreg, n_iter_burnIn_ULA_SOUL, n_iter_ULA_SOUL,
      niter_graddesc_logreg, niter_expgrad_graddesc_logreg, niter_exp_y,
      manual_set_stepsize_hyperparam_logreg, perc_max_stepsize_grad_desc)

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
  # out$weights_progress <- res$weight_progress
  # out$samples_progress <- res$samples_weight_progress
  # out$l2_hyp_progress <- res$list_lambda_l2_progress

  return(out)
}