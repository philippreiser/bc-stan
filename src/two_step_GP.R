library(cmdstanr)
library(bayesplot)
library(posterior)
library(ggplot2)
library(tidyverse)
library(viridis)
library(latex2exp)

source("util.R")
source("inference_helpers.R")

# ── Data ─────────────────────────────────────────────────────────────────────

df_sim        <- readRDS("log_trend_simulation.Rda")
df_real       <- readRDS("log_sin_real_x1_slice.Rda")
df_real_oos   <- readRDS("log_sin_real_oos_ood_x1_uniform.Rda")

# subsample for debugging
set.seed(123)
df_sim      <- df_sim[sample(nrow(df_sim),  30), ]
df_real     <- df_real[sample(nrow(df_real), 10), ]
df_real_oos <- df_real_oos[sample(nrow(df_real_oos), 30), ]

ggplot() +
  geom_point(data = df_sim,      aes(x = x1, y = y_noisy), color = "black") +
  geom_point(data = df_real,     aes(x = x1, y = y_noisy), color = "blue") +
  geom_point(data = df_real_oos, aes(x = x1, y = y_noisy), color = "lightblue") +
  theme_minimal() +
  labs(x = "x", y = "y")

# ── Dimensions ───────────────────────────────────────────────────────────────

p      <- 1
q      <- 1
N_sim  <- nrow(df_sim)
N_real <- nrow(df_real)
N_pred <- nrow(df_real_oos)

# ── Raw data extraction ───────────────────────────────────────────────────────

y_sim  <- df_sim$y_noisy
x_sim  <- matrix(df_sim$w1,  ncol = 1)
w_sim  <- matrix(df_sim$w2,  ncol = 1)
y_real <- df_real$y_noisy
x_real <- matrix(df_real$w1, ncol = 1)
x_pred <- matrix(df_real_oos$w1, ncol = 1)

# ── Standardize outputs ───────────────────────────────────────────────────────

y_sim_mu <- mean(y_sim, na.rm = TRUE)
y_sim_sd <- sd(y_sim,   na.rm = TRUE)
y_sim  <- (y_sim  - y_sim_mu) / y_sim_sd
y_real <- (y_real - y_sim_mu) / y_sim_sd

# ── Scale inputs to [0, 1] ────────────────────────────────────────────────────

x_all <- rbind(x_real, x_sim)
for (i in seq_len(p)) {
  x_min <- min(x_all[, i], na.rm = TRUE)
  x_max <- max(x_all[, i], na.rm = TRUE)
  x_real[, i] <- (x_real[, i] - x_min) / (x_max - x_min)
  x_sim[, i]  <- (x_sim[, i]  - x_min) / (x_max - x_min)
  x_pred[, i] <- (x_pred[, i] - x_min) / (x_max - x_min)
}

# ── Scale calibration parameters to [0, 1] ────────────────────────────────────

w_sim_min <- numeric(q)
w_sim_max <- numeric(q)
for (j in seq_len(q)) {
  w_sim_min[j] <- min(w_sim[, j], na.rm = TRUE)
  w_sim_max[j] <- max(w_sim[, j], na.rm = TRUE)
  w_sim[, j] <- (w_sim[, j] - w_sim_min[j]) / (w_sim_max[j] - w_sim_min[j])
}

# ── Stan data ─────────────────────────────────────────────────────────────────

stan_data <- list(
  N_sim  = N_sim,
  N_real = N_real,
  N_pred = N_pred,
  p      = p,
  q      = q,
  y_sim  = y_sim,
  x_sim  = x_sim,
  w_sim  = w_sim,
  y_real = y_real,
  x_real = x_real,
  x_pred = x_pred
)

# ── Step 1: Fit GP surrogate on simulator data ────────────────────────────────

mod_step1 <- cmdstan_model("GP_step_1.stan")
fit_step1 <- mod_step1$sample(
  data            = stan_data,
  iter_warmup     = 1000,
  iter_sampling   = 1000,
  chains          = 4,
  parallel_chains = 4,
  seed            = 101
)

mcmc_trace(
  as_draws_array(fit_step1$draws()),
  regex_pars = c("rho", "alpha", "sigma")
)
fit_step1$summary()

# ── Step 2: Calibration via analytic GP conditional ──────────────────────────

gp_point       <- get_point_estimates_gp(fit_step1)
stan_data_inf  <- make_gp_point_data(stan_data, gp_point)

mod_step2 <- cmdstan_model("GP_step_2.stan")
fit_step2 <- mod_step2$sample(
  data            = stan_data_inf,
  seed            = 123,
  chains          = 4,
  parallel_chains = 4,
  iter_warmup     = 1000,
  iter_sampling   = 1000
)

mcmc_trace(
  as_draws_array(fit_step2$draws()),
  regex_pars = c("w_real", "sigma")
)
fit_step2$summary()

# ── Posterior predictive plot ─────────────────────────────────────────────────

plot_pp <- function(fit,
                    x_pred,
                    df_real,
                    pred_var  = c("y_pred", "mu_pred"),
                    x_var     = "w1",
                    y_var     = "y_noisy",
                    w_var     = "w_real[1]",
                    y_sim_mu  = 0,
                    y_sim_sd  = 1,
                    alpha     = 0.1,
                    w_lims    = NULL,
                    w_breaks  = waiver(),
                    ylims     = NULL,
                    x_lab     = TeX("$x$"),
                    y_lab     = "y",
                    title     = NULL) {
  
  pred_var <- match.arg(pred_var)
  
  pred_mat <- fit$draws(pred_var) |>
    as_draws_matrix()
  pred_mat <- pred_mat[, grepl(paste0("^", pred_var, "\\["), colnames(pred_mat)), drop = FALSE]
  pred_mat <- pred_mat * y_sim_sd + y_sim_mu
  
  w_draws <- fit$draws(w_var) |>
    as_draws_matrix() |>
    as.numeric()
  
  x_vals <- as.numeric(x_pred)
  
  plot_df <- as.data.frame(pred_mat)
  colnames(plot_df) <- paste0("x", seq_len(ncol(plot_df)))
  plot_df$draw    <- seq_len(nrow(plot_df))
  plot_df$w_draw  <- w_draws
  
  plot_df_long <- plot_df |>
    pivot_longer(starts_with("x"), names_to = "x_index", values_to = pred_var) |>
    mutate(
      x_index = as.numeric(gsub("x", "", x_index)),
      x_plot  = x_vals[x_index]
    )
  
  field_data <- df_real |>
    transmute(x_plot = .data[[x_var]], y_obs = .data[[y_var]])
  
  p <- ggplot() +
    geom_line(
      data = plot_df_long,
      aes(x = x_plot, y = .data[[pred_var]], color = w_draw, group = draw),
      alpha = alpha
    ) +
    scale_color_viridis_c(TeX("$w$"), limits = w_lims, option = "F", breaks = w_breaks) +
    geom_point(data = field_data, aes(x = x_plot, y = y_obs), color = "lightblue", size = 1) +
    theme_minimal() +
    theme(legend.position = "bottom") +
    labs(title = title, x = x_lab, y = y_lab)
  
  if (!is.null(ylims)) p <- p + ylim(ylims)
  p
}

plot_pp(
  fit      = fit_step2,
  x_pred   = df_real_oos$w1,
  df_real  = df_real_oos,
  pred_var = "y_pred",
  w_var    = "w_real",
  y_sim_mu = y_sim_mu,
  y_sim_sd = y_sim_sd,
  alpha    = 0.1,
  ylims    = c(-0.5, 12),
  title    = "GP Step 2 Posterior Predictive"
)

# ── ELPD ──────────────────────────────────────────────────────────────────────

get_lpd_matrix <- function(mu_pred_mat, sigma_draws, y_test) {
  matrix(
    mapply(function(s) dnorm(y_test, mean = mu_pred_mat[s, ], sd = sigma_draws[s], log = TRUE),
           seq_len(nrow(mu_pred_mat))),
    nrow = nrow(mu_pred_mat),
    byrow = TRUE
  )
}

calc_elpd <- function(lpd_mat) {
  pointwise <- apply(lpd_mat, 2, function(x) matrixStats::logSumExp(x) - log(length(x)))
  list(elpd = sum(pointwise), elpd_mean = mean(pointwise))
}

sigma_draws <- as_draws_df(fit_step2$draws("sigma"))$sigma

mu_pred_mat <- fit_step2$draws("f_pred") |>
  as_draws_matrix()
mu_pred_mat <- mu_pred_mat[, grepl("^f_pred\\[", colnames(mu_pred_mat)), drop = FALSE]
mu_pred_mat <- mu_pred_mat * y_sim_sd + y_sim_mu

y_test <- df_real_oos$y_noisy * y_sim_sd + y_sim_mu

lpd_mat <- get_lpd_matrix(mu_pred_mat, sigma_draws * y_sim_sd, y_test)
elpd    <- calc_elpd(lpd_mat)
cat("ELPD:     ", elpd$elpd, "\n")
cat("ELPD/obs: ", elpd$elpd_mean, "\n")
