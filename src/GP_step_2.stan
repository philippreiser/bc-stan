functions {

  vector gp_conditional_rng(
    array[] vector xw_pred,   // (N_pred) array of vectors (p+q)
    array[] vector xw_sim,    // (N_sim) array of vectors (p+q)
    vector f_sim,             // (N_sim,) fixed latent GP values from Step 1
    real alpha,               // GP marginal std
    array[] real rho,         // (p+q) ARD length scales
    real jitter
  ) {
    int N_sim = size(xw_sim);
    int N_pred = size(xw_pred);

    // --- Build K(xw_sim, xw_sim) — noiseless, latent covariance ---
    matrix[N_sim, N_sim] K = gp_exp_quad_cov(xw_sim, alpha, rho);
    K += diag_matrix(rep_vector(jitter, N_sim));
    matrix[N_sim, N_sim] L = cholesky_decompose(K);

    // --- Build K(xw_sim, xw_pred) — cross covariance ---
    matrix[N_sim, N_pred] K_cross = gp_exp_quad_cov(xw_sim, xw_pred, alpha, rho);

    // --- Build K(xw_pred, xw_pred) ---
    matrix[N_pred, N_pred] K_pred = gp_exp_quad_cov(xw_pred, alpha, rho);
    K_pred += diag_matrix(rep_vector(jitter, N_pred));

    // --- Analytic conditional ---
    // mu_pred  = K_cross' * K^{-1} * f_sim
    // cov_pred = K_pred - K_cross' * K^{-1} * K_cross
    vector[N_sim] a = mdivide_right_tri_low(
      mdivide_left_tri_low(L, f_sim)', L)';
    vector[N_pred] mu_pred = K_cross' * a;

    matrix[N_sim, N_pred] V = mdivide_left_tri_low(L, K_cross);
    matrix[N_pred, N_pred] cov_pred = K_pred - V' * V;

    return multi_normal_rng(mu_pred, cov_pred);
  }
}

data {
  int<lower=1> N_real;             // number of field observations
  int<lower=1> N_sim;              // number of simulator points
  int<lower=1> N_pred;             // number of prediction points
  int<lower=1> p;                  // dimension of x
  int<lower=1> q;                  // dimension of w
  vector[N_real] y_real;           // field observations
  matrix[N_real, p] x_real;        // field x inputs
  matrix[N_sim, p] x_sim;          // simulator x inputs
  matrix[N_sim, q] w_sim;          // simulator w inputs (known)
  matrix[N_pred, p] x_pred;        // prediction x inputs
  // Fixed from Step 1
  array[p+q] real<lower=0> rho;
  real<lower=0> alpha;
  vector[N_sim] f_sim;
}

transformed data {
  real jitter = 1e-8;
  int n_star = N_real + N_pred;

  // Simulator joint input matrix — fixed, built once here
  array[N_sim] vector[p+q] xw_sim;
  for (i in 1:N_sim) {
    xw_sim[i, 1:p] = x_sim[i]';
    xw_sim[i, (p+1):(p+q)] = w_sim[i]';
  }
}

parameters {
  real<lower=0, upper=1> w_real;  // scalar calibration parameter
  real<lower=0> sigma;            // field observation noise
}

transformed parameters {
  array[N_real] vector[p+q] xw_real;
  for (i in 1:N_real) {
    xw_real[i, 1:p] = x_real[i]';
    xw_real[i, (p+1):(p+q)] = rep_vector(w_real, q);
  }

  array[N_pred] vector[p+q] xw_pred;
  for (i in 1:N_pred) {
    xw_pred[i, 1:p] = x_pred[i]';
    xw_pred[i, (p+1):(p+q)] = rep_vector(w_real, q);
  }

  // GP conditional mean at real points — promoted so generated quantities can access it
  vector[N_real] mu_real;
  {
    matrix[N_sim, N_sim] K = gp_exp_quad_cov(xw_sim, alpha, rho);
    K += diag_matrix(rep_vector(jitter, N_sim));
    matrix[N_sim, N_sim] L = cholesky_decompose(K);
    vector[N_sim] a = mdivide_right_tri_low(
      mdivide_left_tri_low(L, f_sim)', L)';
    matrix[N_sim, N_real] K_cross = gp_exp_quad_cov(xw_sim, xw_real, alpha, rho);
    mu_real = K_cross' * a;
  }
}

model {
  sigma ~ normal(0, 0.5);
  target += normal_lpdf(y_real | mu_real, sigma);
}

generated quantities {
  vector[N_pred] f_pred = gp_conditional_rng(
    xw_pred, xw_sim, f_sim, alpha, rho, jitter
  );

  vector[N_pred] y_pred;
  for (j in 1:N_pred)
    y_pred[j] = normal_rng(f_pred[j], sigma);
}
