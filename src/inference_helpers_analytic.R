library(cmdstanr)
library(posterior)
library(future.apply)
library(here)

get_point_estimates_gp <- function(fit) {
  draws_df <- posterior::as_draws_df(fit$draws())
  
  point_est <- list(
    rho_eta = apply(
      as.matrix(draws_df[, grep("^rho_eta\\[", names(draws_df)), drop = FALSE]),
      2, median
    ),
    lambda_eta = median(draws_df$lambda_eta),
    
    # IMPORTANT: if you estimated simulator noise in step 1
    sigma_sim = if ("sigma" %in% names(draws_df)) {
      median(draws_df$sigma)
    } else {
      NULL
    }
  )
  
  # optional init for tf
  if (length(grep("^tf\\[", names(draws_df))) > 0) {
    point_est$tf_init <- apply(
      as.matrix(draws_df[, grep("^tf\\[", names(draws_df)), drop = FALSE]),
      2, median
    )
  }
  
  point_est
}

make_gp_point_data <- function(stan_data, point_est) {
  list(
    n = stan_data$n,
    m = stan_data$m,
    n_pred = stan_data$n_pred,
    p = stan_data$p,
    q = stan_data$q,
    
    y = stan_data$y,
    eta = stan_data$eta,
    
    xf = stan_data$xf,
    xc = stan_data$xc,
    tc = stan_data$tc,
    x_pred = stan_data$x_pred,
    
    rho_eta = as.vector(point_est$rho_eta),
    lambda_eta = point_est$lambda_eta,
    
    # optional
    sigma_sim = point_est$sigma_sim
  )
}