functions {
  vector gp_conditional_rng(
    array[] vector xw_pred,   // (N_pred) array of vectors (p+q)
    array[] vector xw_sim_real,    // (N_total) array of vectors (p+q)
    vector f,             // (N_total,) latent GP values
    real alpha,               // GP marginal std
    array[] real rho,         // (p+q) ARD length scales
    real jitter
  ) {
    int N_total = size(xw_sim_real);
    int N_pred = size(xw_pred);

    // --- Build K(xw_sim_real, xw_sim_real) — noiseless, latent covariance ---
    matrix[N_total, N_total] K = gp_exp_quad_cov(xw_sim_real, alpha, rho);
    K += diag_matrix(rep_vector(jitter, N_total));
    matrix[N_total, N_total] L = cholesky_decompose(K);

    // --- Build K(xw_sim_real, xw_pred) — cross covariance ---
    matrix[N_total, N_pred] K_cross = gp_exp_quad_cov(xw_sim_real, xw_pred, alpha, rho);

    // --- Build K(xw_pred, xw_pred) ---
    matrix[N_pred, N_pred] K_pred = gp_exp_quad_cov(xw_pred, alpha, rho);
    K_pred += diag_matrix(rep_vector(jitter, N_pred));

    // --- Analytic conditional ---
    // mu_pred  = K_cross' * K^{-1} * f
    // cov_pred = K_pred - K_cross' * K^{-1} * K_cross
    vector[N_total] a = mdivide_right_tri_low(
      mdivide_left_tri_low(L, f)', L)';
    vector[N_pred] mu_pred = K_cross' * a;

    matrix[N_total, N_pred] V = mdivide_left_tri_low(L, K_cross);
    matrix[N_pred, N_pred] cov_pred = K_pred - V' * V;

    return multi_normal_rng(mu_pred, cov_pred);
  }
}
data {
  int<lower=1> N_sim;              // number of simulator data
  int<lower=1> p;                  // dimension of x
  int<lower=1> q;                  // dimension of calibration parameter w
  vector[N_sim] y_sim;             // simulator outputs
  array[N_sim] vector [p] x_sim;          // simulator x inputs
  array[N_sim] vector [q] w_sim;          // simulator w inputs
  int<lower=1> N_real;                  // number of field observations
  array[N_real] vector[p] x_real;       // field x inputs (known)
  vector[N_real] y_real;                // field observations
  real<lower=0> alpha_sim;              // likelihood weight for sim data
  real<lower=0> alpha_real;            // likelihood weight for real data
  int<lower=1> N_pred;              // number of prediction points
  array[N_pred] vector[p] x_pred;   // prediction x inputs
  vector[q] w_prior_mean;           // prior mean for w_real
  vector<lower=0>[q] w_prior_sigma; // prior sd for w_real
}
transformed data {
  real jitter = 1e-8;
  int N_total = N_sim + N_real;
}
parameters {
  array[p+q] real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=1e-6> sigma;
  vector[N_total] eta_std;               // latent GP in non-centered form
  vector<lower=0, upper=1>[q] w_real;
}
transformed parameters {
  vector[N_total] f;
  array[N_total] vector[p+q] xw_sim_real;
  {
    matrix[N_total, N_total] K;
    xw_sim_real[1:N_sim, 1:p] = x_sim;
    xw_sim_real[1:N_sim, (p+1):(p+q)] = w_sim;
    xw_sim_real[(N_sim+1):N_total, 1:p] = x_real;
    xw_sim_real[(N_sim+1):N_total, (p+1):(p+q)] = rep_array(w_real, N_real);
    K = gp_exp_quad_cov(xw_sim_real, alpha, rho);
    K += diag_matrix(rep_vector(jitter, N_total));
    f = cholesky_decompose(K) * eta_std;
  }
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ normal(0, 0.5);
  w_real ~ normal(w_prior_mean, w_prior_sigma);
  eta_std ~ std_normal();
  
  target += alpha_sim * normal_lpdf(y_sim | f[1:N_sim], sigma);
  target += alpha_real * normal_lpdf(y_real | f[(N_sim + 1):N_total], sigma);
}
generated quantities {
  array[N_pred] vector[p+q] xw_pred;
  for (j in 1:N_pred) {
    xw_pred[j, 1:p]         = x_pred[j];
    xw_pred[j, (p+1):(p+q)] = w_real;
  }
  vector[N_pred] f_pred = gp_conditional_rng(
    xw_pred, xw_sim_real, f, alpha, rho, jitter
  );

  vector[N_pred] y_pred;
  for (j in 1:N_pred)
    y_pred[j] = normal_rng(f_pred[j], sigma);
}

