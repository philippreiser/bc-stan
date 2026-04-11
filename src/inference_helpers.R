library(cmdstanr)
library(posterior)
library(future.apply)
library(here)

get_point_estimates_gp <- function(fit) {
  draws_df <- as_draws_df(fit$draws())
  
  f_sim_cols <- grep("^f_sim\\[", names(draws_df))
  if (length(f_sim_cols) == 0)
    stop("f_sim not found in draws — ensure f_sim is a transformed parameter in Step 1")
  
  list(
    rho   = apply(as.matrix(draws_df[, grep("^rho\\[", names(draws_df)), drop = FALSE]), 2, median),
    alpha = median(draws_df$alpha),
    f_sim = apply(as.matrix(draws_df[, f_sim_cols, drop = FALSE]), 2, median)
  )
}

make_gp_point_data <- function(stan_data, point_est) {
  list(
    N_sim  = stan_data$N_sim,
    N_real = stan_data$N_real,
    N_pred = stan_data$N_pred,
    p      = stan_data$p,
    q      = stan_data$q,
    y_real = stan_data$y_real,
    x_real = stan_data$x_real,
    x_sim  = stan_data$x_sim,
    w_sim  = stan_data$w_sim,
    x_pred = stan_data$x_pred,
    rho    = as.vector(point_est$rho),
    alpha  = point_est$alpha,
    f_sim  = as.vector(point_est$f_sim)
  )
}

# --- helper: extract sampled draws from first-step GP fit ---
get_gp_draws_for_epost <- function(fit, number_samples_1 = 100) {
  draws_df <- posterior::as_draws_df(fit$draws())
  n_total <- nrow(draws_df)
  
  if (number_samples_1 > n_total) {
    stop("number_samples_1 is larger than the available number of posterior draws.")
  }
  
  draw_ids <- seq_len(number_samples_1)
  
  rho_eta_mat <- as.matrix(draws_df[, grep("^rho_eta\\[", names(draws_df)), drop = FALSE])
  eta_std_mat <- as.matrix(draws_df[, grep("^eta_std\\[", names(draws_df)), drop = FALSE])
  
  tf_mat <- NULL
  if (length(grep("^tf\\[", names(draws_df))) > 0) {
    tf_mat <- as.matrix(draws_df[, grep("^tf\\[", names(draws_df)), drop = FALSE])
  }
  
  list(
    draw_ids = draw_ids,
    rho_eta_mat = rho_eta_mat,
    lambda_eta = draws_df$lambda_eta,
    eta_std_mat = eta_std_mat,
    tf_mat = tf_mat
  )
}

# --- helper: build data for one second-step GP fit ---
make_gp_epost_data <- function(stan_data, gp_draws, i) {
  draw_id <- gp_draws$draw_ids[i]
  
  list(
    n = stan_data$n,
    m = stan_data$m,
    n_pred = stan_data$n_pred,
    p = stan_data$p,
    q = stan_data$q,
    y = stan_data$y,
    xf = stan_data$xf,
    xc = stan_data$xc,
    tc = stan_data$tc,
    x_pred = stan_data$x_pred,
    rho_eta = as.vector(gp_draws$rho_eta_mat[draw_id, ]),
    lambda_eta = gp_draws$lambda_eta[draw_id],
    eta_std = as.vector(gp_draws$eta_std_mat[draw_id, ])
  )
}

# --- E-Post second step for GP ---
inference_e_post_gp <- function(
    surrogate_model_fit,
    inference_model,
    stan_data,
    number_samples_1 = 100,
    number_samples_2 = 250,
    number_chains_2 = 1,
    iter_warmup_2 = 500,
    adapt_delta_2 = 0.8,
    seed = 123,
    workers = 4
) {
  gp_draws <- get_gp_draws_for_epost(
    fit = surrogate_model_fit,
    number_samples_1 = number_samples_1
  )
  
  fits_csv_files <- c()
  csv_dir <- file.path(
    here(),
    "fitted_models/_imodel_fits_cmdstan",
    basename(tempdir())
  )
  dir.create(csv_dir, showWarnings = FALSE, recursive = TRUE)
  
  fit_one_draw <- function(i) {
    message("Fitting GP E-Post model: draw ", i, " / ", number_samples_1)
    
    data_i <- make_gp_epost_data(stan_data, gp_draws, i)
    
    init_value <- 0
    if (!is.null(gp_draws$tf_mat)) {
      draw_id <- gp_draws$draw_ids[i]
      tf_init <- as.vector(gp_draws$tf_mat[draw_id, ])
      init_value <- function() {
        list(
          tf = tf_init,
          sigma = 0.1
        )
      }
    }
    
    fit_i <- inference_model$sample(
      data = data_i,
      seed = seed + i,
      chains = number_chains_2,
      parallel_chains = 1,
      iter_warmup = iter_warmup_2,
      iter_sampling = number_samples_2,
      adapt_delta = adapt_delta_2,
      output_dir = csv_dir,
      init = init_value,
      refresh = 0
    )
    
    fit_i$output_files()
  }
  
  future::plan(future::multisession, workers = workers)
  fits_epost <- future.apply::future_lapply(
    seq_len(number_samples_1),
    fit_one_draw,
    future.seed = TRUE
  )
  
  fits_csv_files <- do.call(c, fits_epost)
  inference_fit <- cmdstanr::as_cmdstan_fit(fits_csv_files)
  
  unlink(csv_dir, recursive = TRUE)
  inference_fit
}

inference_step_gp <- function(
    surrogate_model_fit,
    inference_model,
    stan_data,
    type = c("point", "e_post"),
    number_samples_1 = 100,
    number_samples_2 = 250,
    number_chains_2 = 1,
    iter_warmup_2 = 500,
    adapt_delta_2 = 0.8,
    seed = 123,
    workers = 4
) {
  type <- match.arg(type)
  
  if (type == "point") {
    gp_point <- get_point_estimates_gp(surrogate_model_fit)
    stan_data_inf <- make_gp_point_data(stan_data, gp_point)
    
    # init_value <- function() {
    #   list(
    #     tf = gp_point$tf_init,
    #     sigma = 0.1
    #   )
    # }
    
    inference_model$sample(
      data = stan_data_inf,
      seed = seed,
      chains = number_chains_2,
      parallel_chains = number_chains_2,
      iter_warmup = iter_warmup_2,
      iter_sampling = number_samples_2,
      adapt_delta = adapt_delta_2,
      # init = init_value,
      refresh = 0
    )
    
  } else {
    inference_e_post_gp(
      surrogate_model_fit = surrogate_model_fit,
      inference_model = inference_model,
      stan_data = stan_data,
      number_samples_1 = number_samples_1,
      number_samples_2 = number_samples_2,
      number_chains_2 = number_chains_2,
      iter_warmup_2 = iter_warmup_2,
      adapt_delta_2 = adapt_delta_2,
      seed = seed,
      workers = workers
    )
  }
}