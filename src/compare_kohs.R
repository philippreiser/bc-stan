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
n_sim_small  <- 20
n_real_small <- 8
n_real_oos_ood_small <- 10

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
y_pred <- matrix(df_real_oos_ood$y_noisy, ncol=1)
n_pred <- nrow(x_pred) # number of predictions

# standardization of output y and eta
eta_mu <- mean(eta, na.rm = TRUE) # mean value
eta_sd <- sd(eta, na.rm = TRUE) # standard deviation
y <- (y - eta_mu) / eta_sd
y_pred <- (y_pred - eta_mu) / eta_sd
eta <- (eta - eta_mu) / eta_sd

# Put design points xf and xc on [0,1]
x <- rbind(as.matrix(xf), as.matrix(xc))
for (i in (1:ncol(x))){
  x_min <- min(x[,i], na.rm = TRUE)
  x_max <- max(x[,i], na.rm = TRUE)
  xf[,i] <- (xf[,i] - x_min) / (x_max - x_min)
  xc[,i] <- (xc[,i] - x_min) / (x_max - x_min)
  x_pred[,i] <- (x_pred[,i] - x_min) / (x_max - x_min)
  x_pred[, i] <- seq(0, 1, length = n_real_oos_ood_small)
}

# Put calibration parameters t on domain [0,1]
for (j in (1:ncol(tc))){
  tc_min <- min(tc[,j], na.rm = TRUE)
  tc_max <- max(tc[,j], na.rm = TRUE)
  tc[,j] <- (tc[,j] - tc_min) / (tc_max - tc_min)
}

# create data as list for input to Stan
bc_data <- list(n=n, m=m, n_pred=n_pred, p=p, y=y, q=q, eta=eta, 
                  xf=as.matrix(xf), xc=as.matrix(xc), 
                  x_pred=as.matrix(x_pred), tc=as.matrix(tc))

bc_to_koh_data <- function(stan_data_bc,
                           w_prior_mean = 0.5,
                           w_prior_sigma = 0.2) {
  x_sim  <- lapply(seq_len(stan_data_bc$m), function(i) {
    as.vector(stan_data_bc$xc[i, ])
  })
  
  w_sim <- lapply(seq_len(stan_data_bc$m), function(i) {
    as.vector(stan_data_bc$tc[i, ])
  })
  
  x_real <- lapply(seq_len(stan_data_bc$n), function(i) {
    as.vector(stan_data_bc$xf[i, ])
  })
  
  x_pred <- lapply(seq_len(stan_data_bc$n_pred), function(i) {
    as.vector(stan_data_bc$x_pred[i, ])
  })
  
  list(
    N_sim = stan_data_bc$m,
    N_real = stan_data_bc$n,
    N_pred = stan_data_bc$n_pred,
    p = stan_data_bc$p,
    q = stan_data_bc$q,
    x_sim = x_sim,
    w_sim = w_sim,
    y_sim = as.vector(stan_data_bc$eta),
    x_real = x_real,
    y_real = as.vector(stan_data_bc$y),
    x_pred = x_pred,
    w_prior_mean = w_prior_mean,
    w_prior_sigma = w_prior_sigma
  )
}

koh_data <- bc_to_koh_data(stan_data)

mod_bc <- cmdstan_model("bcWithPred.stan")
fit_bc <- mod_bc$sample(
  data = bc_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

mod_koh <- cmdstan_model("GP_KOH.stan")
fit_koh <- mod_koh$sample(
  data = koh_data,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  iter_warmup = 1000,
  iter_sampling = 1000
)

#-----------------------------
# 5. Extract posterior predictions
#-----------------------------

y_pred_bc <- fit_bc$draws("y_pred")
y_pred_koh <- fit_koh$draws("f_delta_pred")

# Convert to matrix
y_pred_bc <- as_draws_matrix(y_pred_bc)
y_pred_koh <- as_draws_matrix(y_pred_koh)

#-----------------------------
# 6. Compare qualitatively
#-----------------------------

# Compute summaries
summary_df <- data.frame(
  x_test = x_pred,
  mean_bc = colMeans(y_pred_bc),
  mean_koh = colMeans(y_pred_koh),
  sd_bc = apply(y_pred_bc, 2, sd),
  sd_koh = apply(y_pred_koh, 2, sd),
  y_test = y_pred
  
)

real_df = data.frame(
  x_real = xf,
  y_real = y
)

# Plot comparison
ggplot(summary_df) +
  geom_line(aes(x = x_test, y = mean_bc), color = "red", linewidth = 1) +
  geom_line(aes(x = x_test, y = mean_koh), color = "blue", linewidth = 1) +
  geom_ribbon(aes(x = x_test, ymin = mean_bc - 2*sd_bc, ymax = mean_bc + 2*sd_bc),
              fill = "red", alpha = 0.1) +
  geom_ribbon(aes(x = x_test, ymin = mean_koh - 2*sd_koh, ymax = mean_koh + 2*sd_koh),
              fill = "blue", alpha = 0.1) +
  # geom_point(aes(x = x_test, y = y_test), color = "black") +
  # geom_point(aes(x=x_real, y=y_real), color="grey", data=real_df)+
  labs(title = "Qualitative comparison of Stan models",
       y = "Prediction", x = "x") +
  theme_minimal()

