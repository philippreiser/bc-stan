library(cmdstanr) # library(rstan)
library(bayesplot)
library(posterior)
library(ggplot2)
source("util.R")

# read in field and computer simulation data
df_sim <- readRDS(file="log_trend_simulation.Rda")
df_real <- readRDS(file="log_sin_real_x1_slice.Rda")
df_real_oos_ood <- readRDS(file="log_sin_real_oos_ood_x1_uniform.Rda")

# subsample data for debugging
set.seed(123)
n_sim_small  <- 30
n_real_small <- 10
n_real_oos_ood_small <- 30

df_sim  <- df_sim[sample(nrow(df_sim),  n_sim_small), ]
df_real <- df_real[sample(nrow(df_real), n_real_small), ]
df_real_oos_ood <- df_real_oos_ood[sample(nrow(df_real_oos_ood), n_real_oos_ood_small), ]

ggplot() +
  geom_point(data = df_sim, aes(x = x1, y = y_noisy), color = "black") +
  geom_point(data = df_real, aes(x = x1, y = y_noisy), color = "blue") +
  geom_point(data = df_real_oos_ood, aes(x = x1, y = y_noisy), color = "lightblue") +
  theme_minimal() +
  labs(x = "x", y = "y")

# get dimensions of dataset
p <- 1 # number of input factors
q <- 1 # number of calibration parameters
n <- nrow(df_real) # sample size of observed field data
m <- nrow(df_sim) # sample size of computer simulation data


# extract data from DATAFIELD (Table 3) and DATACOMP (Table 4) 
y <- df_real$y_noisy # observed output
xf <- matrix(df_real$w1, ncol=1) # observed input
eta <- df_sim$y_noisy # simulation output
xc <- matrix(df_sim$w1, ncol=1) # simulation input
tc <- matrix(df_sim$w2, ncol=1) # calibration parameters

x_pred <- matrix(df_real_oos_ood$w1, ncol=1) # design points for predictions
n_pred <- nrow(x_pred) # number of predictions

# standardization of output y and eta
eta_mu <- mean(eta, na.rm = TRUE) # mean value
eta_sd <- sd(eta, na.rm = TRUE) # standard deviation
y <- (y - eta_mu) / eta_sd
eta <- (eta - eta_mu) / eta_sd

# Put design points xf and xc on [0,1]
x <- rbind(as.matrix(xf), as.matrix(xc))
for (i in (1:ncol(x))){
  x_min <- min(x[,i], na.rm = TRUE)
  x_max <- max(x[,i], na.rm = TRUE)
  xf[,i] <- (xf[,i] - x_min) / (x_max - x_min)
  xc[,i] <- (xc[,i] - x_min) / (x_max - x_min)
  x_pred[,i] <- (x_pred[,i] - x_min) / (x_max - x_min)
}

# Put calibration parameters t on domain [0,1]
for (j in (1:ncol(tc))){
  tc_min <- min(tc[,j], na.rm = TRUE)
  tc_max <- max(tc[,j], na.rm = TRUE)
  tc[,j] <- (tc[,j] - tc_min) / (tc_max - tc_min)
}

# create data as list for input to Stan
stan_data <- list(n=n, m=m, n_pred=n_pred, p=p, y=y, q=q, eta=eta, 
                  xf=as.matrix(xf), xc=as.matrix(xc), 
                  x_pred=as.matrix(x_pred), tc=as.matrix(tc))

# set stan to execute multiple Markov chains in parallel
# rstan_options(auto_write = TRUE)
# options(mc.cores = parallel::detectCores())

# run model in cmdstanr
mod <- cmdstan_model("bcWithPredLatentDiscr.stan")
# mod <- cmdstan_model("bcWithPred.stan")
fit <- mod$sample(
  data = stan_data,
  iter_warmup = 1000,
  iter_sampling = 1000, 
  chains = 4,
  parallel_chains = 4
)

# run model in stan
# To run without predictive inferenec: 
# comment lines 60-63 and 85
# uncomment lines 65-68 and 86
# fit <- stan(file = "bcWithPred.stan", 
#             data = stan_data, 
#             iter = 500, 
#             chains = 3)

#fit <- stan(file = "bcWithoutPred.stan",
#            data = stan_data,
#            iter = 500,
#            chains = 3)

# plot traceplots, excluding warm-up
# stan_trace(fit, pars = c("tf", "beta_eta", "beta_delta", 
#                          "lambda_eta", "lambda_delta", "lambda_e"))

# extract draws
draws <- fit$draws()

# traceplots
mcmc_trace(
  as_draws_array(draws),
  regex_pars = c("tf", "beta_eta", "beta_delta",
                 "lambda_eta", "lambda_delta", "lambda_e",
                 "eta_std", "delta_std", "sigma_sim")
)
fit$summary()

# summarize results
# print(fit, pars = c("tf", "beta_eta", "beta_delta", 
#                     "lambda_eta", "lambda_delta", "lambda_e"))

fit$summary(
  variables = c("tf", "beta_eta", "beta_delta",
                "lambda_eta", "lambda_delta", "lambda_e",
                "eta_std", "delta_std")
)

# posterior probability distribution of tf
# stan_hist(fit, pars = c("tf"))

# scale calibration parameters back to original scale
draws_array <- as_draws_array(draws)
tf_draws_array <- subset_draws(draws_array, variable="tf")
tf_draws <- as_draws_matrix(tf_draws_array)
tf_scaled <- as.vector(tf_draws)
tf_original <- tf_scaled * (tc_max - tc_min) + tc_min
tf_original_draws <- as_draws_array(array(tf_original,
                                          dim = c(length(tf_original), 1, 1),
                                          dimnames = list(NULL, NULL, "tf_original")))

mcmc_hist(tf_original_draws, regex_pars="tf")

mcmc_hist(
  as_draws_array(draws),
  regex_pars = "tf"
)

# extract predictions, excluding warm-up and 
# samples <- rstan::extract(fit)

samples <- as_draws_df(fit$draws())
y_pred_array <- fit$draws("y_pred")
y_pred_mat <- as_draws_matrix(y_pred_array)

# remove chain columns if present
y_pred <- y_pred_mat[, grepl("^y_pred", colnames(y_pred_mat))]

# rescale
y_pred <- y_pred * eta_sd + eta_mu

# get predictive inference y_pred and convert back to original scale
# y_pred <- samples$y_pred * eta_sd + eta_mu 
#y_pred <- y.pred(x_pred, samples, stan_data) * eta_sd + eta_mu 

n_samples <- nrow(y_pred)

# for loop to visualize predictions at different input x
for (i in (1:p)) {
  field_data <- data.frame(yf=df_real$y_noisy, 
                           xf=signif(df_real$w1, 3))
  # pred_data <- matrix(data=t(y_pred), 
  #                     nrow=length(y_pred), ncol = 1)
  pred_i <- y_pred[, i]
  plot_data <- data.frame(apply(field_data, 2, rep, 
                                n_samples), pred = as.vector(pred_i))
  # save plot as png file
  png(paste("plot", i, ".png", sep = ""))
  plt <- ggplot(data = plot_data, aes(y=pred, x=xf, group=xf)) +
    geom_boxplot(outlier.size=0.2) +
    geom_point(data = field_data, aes(x=xf, y=yf), 
               color="#D55E00", size=0.8) 
  print(plt)
  dev.off()
}


# plot spagehtti plot predictions
library(posterior)
library(tidyverse)

# --- Extract predictions ---
y_pred_array <- fit$draws("y_pred")
y_pred_mat   <- as_draws_matrix(y_pred_array)

# keep only prediction columns
y_pred <- y_pred_mat[, grepl("^y_pred", colnames(y_pred_mat))]

# --- Reverse standardization ---
y_pred <- y_pred * eta_sd + eta_mu

# --- Field data ---
xf <- as.numeric(signif(df_real_oos_ood$w1, 3))
field_data <- data.frame(
  xf = xf,
  yf = df_real_oos_ood$y_noisy
)

# --- Convert to long format for spaghetti ---
n_draws  <- nrow(y_pred)
n_points <- ncol(y_pred)

plot_df <- as.data.frame(y_pred)
colnames(plot_df) <- paste0("x", 1:n_points)
plot_df$draw <- 1:n_draws

plot_df_long <- plot_df %>%
  pivot_longer(-draw, names_to = "x_index", values_to = "y") %>%
  mutate(
    x_index = as.numeric(gsub("x", "", x_index)),
    x = xf[x_index]
  )

# --- Posterior predictive spaghetti plot ---
ggplot() +
  geom_line(
    data = plot_df_long,
    aes(x = x, y = y, group = draw),
    alpha = 0.05,
    color = "steelblue"
  ) +
  geom_point(
    data = field_data,
    aes(x = xf, y = yf),
    color = "#D55E00",
    size = 2
  ) +
  theme_minimal() +
  labs(
    title = "KOH Posterior Predictive",
    x = "x",
    y = "y"
  )


# plot spagehtti mean predictions
library(posterior)
library(tidyverse)

# --- Extract predictions ---
y_pred_array <- fit$draws("mu_pred")
y_pred_mat   <- as_draws_matrix(y_pred_array)

# keep only prediction columns
y_pred <- y_pred_mat[, grepl("^mu_pred", colnames(y_pred_mat))]

# --- Reverse standardization ---
y_pred <- y_pred * eta_sd + eta_mu

# --- Field data ---
xf <- as.numeric(signif(df_real_oos_ood$w1, 3))
field_data <- data.frame(
  xf = xf,
  yf = df_real_oos_ood$y_noisy
)

# --- Convert to long format for spaghetti ---
n_draws  <- nrow(y_pred)
n_points <- ncol(y_pred)

plot_df <- as.data.frame(y_pred)
colnames(plot_df) <- paste0("x", 1:n_points)
plot_df$draw <- 1:n_draws

plot_df_long <- plot_df %>%
  pivot_longer(-draw, names_to = "x_index", values_to = "y") %>%
  mutate(
    x_index = as.numeric(gsub("x", "", x_index)),
    x = xf[x_index]
  )

# --- Posterior predictive spaghetti plot ---
ggplot() +
  geom_line(
    data = plot_df_long,
    aes(x = x, y = y, group = draw),
    alpha = 0.05,
    color = "steelblue"
  ) +
  geom_point(
    data = field_data,
    aes(x = xf, y = yf),
    color = "#D55E00",
    size = 2
  ) +
  theme_minimal() +
  labs(
    title = "KOH Posterior Predictive",
    x = "x",
    y = "mu_pred"
  )


library(cmdstanr)
library(posterior)
library(tidyverse)
library(viridis)
library(latex2exp)

plot_pp_koh <- function(fit,
                        x_pred,
                        df_real,
                        pred_var = c("y_pred", "mu_pred"),
                        x_var_real = "w1",
                        y_var_real = "y_noisy",
                        tf_var = "tf[1]",
                        eta_mu = 0,
                        eta_sd = 1,
                        alpha = 0.1,
                        tf_lims = NULL,
                        tf_breaks = waiver(),
                        ylims = NULL,
                        x_lab = TeX("$x$"),
                        y_lab = "y",
                        title = NULL) {
  
  pred_var <- match.arg(pred_var)
  
  # posterior draws
  pred_mat <- fit$draws(pred_var) |>
    posterior::as_draws_matrix()
  pred_mat <- pred_mat[, grepl(paste0("^", pred_var, "\\["), colnames(pred_mat)), drop = FALSE]
  
  tf_draws <- fit$draws(tf_var) |>
    posterior::as_draws_matrix()
  tf_draws <- as.numeric(tf_draws[, 1])
  
  # reverse standardization
  pred_mat <- pred_mat * eta_sd + eta_mu
  
  x_vals <- as.numeric(x_pred)
  
  # long format
  plot_df <- as.data.frame(pred_mat)
  colnames(plot_df) <- paste0("x", seq_len(ncol(plot_df)))
  plot_df$draw <- seq_len(nrow(plot_df))
  plot_df$tf_draw <- tf_draws
  
  plot_df_long <- plot_df %>%
    tidyr::pivot_longer(
      cols = starts_with("x"),
      names_to = "x_index",
      values_to = pred_var
    ) %>%
    dplyr::mutate(
      x_index = as.numeric(gsub("x", "", x_index)),
      x_plot = x_vals[x_index]
    )
  
  # observed data
  field_data <- df_real %>%
    dplyr::transmute(
      x_plot = .data[[x_var_real]],
      y_obs = .data[[y_var_real]]
    )
  
  p <- ggplot() +
    geom_line(
      data = plot_df_long,
      aes(
        x = x_plot,
        y = .data[[pred_var]],
        color = tf_draw,
        group = draw
      ),
      alpha = alpha
    ) +
    scale_color_viridis_c(
      TeX("$t_f$"),
      limits = tf_lims,
      option = "F",
      breaks = tf_breaks
    ) +
    geom_point(
      data = field_data,
      aes(x = x_plot, y = y_obs),
      color = "#D55E00",
      size = 1
    ) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    labs(
      title = title,
      x = x_lab,
      y = y_lab
    )
  
  if (!is.null(ylims)) {
    p <- p + ylim(ylims)
  }
  
  p
}
  
p_y <- plot_pp_koh(
  fit = fit,
  x_pred = df_real_oos_ood$w1,
  df_real = df_real_oos_ood,
  pred_var = "y_pred",
  tf_var = "tf[1]",
  eta_mu = eta_mu,
  eta_sd = eta_sd,
  alpha = 0.1,
  ylims = c(-0.5, 9),
  title = "KOH Posterior Predictive"
)
p_y


# calc ELPD
get_lpd_matrix_gp <- function(mu_pred_mat, sigma_draws, y_test) {
  S <- nrow(mu_pred_mat)
  N <- ncol(mu_pred_mat)
  lpd_mat <- matrix(NA_real_, nrow = S, ncol = N)
  
  for (s in 1:S) {
    lpd_mat[s, ] <- dnorm(y_test, mean = mu_pred_mat[s, ], sd = sigma_draws[s], log = TRUE)
  }
  
  lpd_mat
}

calc_elpd_from_lpd_matrix <- function(lpd_mat) {
  pointwise_lpd <- apply(lpd_mat, 2, function(x) {
    matrixStats::logSumExp(x) - log(length(x))
  })
  sum(pointwise_lpd)
}

sigma_draws <- sqrt(1 / posterior::as_draws_df(fit$draws("lambda_e"))$lambda_e)

mu_pred_mat <- fit$draws("mu_pred") |>
  posterior::as_draws_matrix()

mu_pred_mat <- mu_pred_mat[, grepl("^mu_pred\\[", colnames(mu_pred_mat)), drop = FALSE]

# backtransform if needed
mu_pred_mat <- mu_pred_mat * eta_sd + eta_mu
sigma_draws <- sigma_draws * eta_sd

lpd_mat <- get_lpd_matrix_gp(mu_pred_mat, sigma_draws, y_test = df_real_oos_ood$y_noisy)
elpd <- calc_elpd_from_lpd_matrix(lpd_mat)
elpd
elpd_mean <- elpd / ncol(lpd_mat)
