functions {

  // Analytic GP conditional for latent f at field/prediction points
  // given fixed simulator latent values f_sim and GP hyperparameters.
  // Uses the NOISELESS simulator covariance (no sigma on diagonal),
  // since f_sim are latent values, not noisy observations.
  vector gp_conditional_rng(
    matrix x_star,          // (n_star, p+q) query points (field + pred)
    matrix x_sim,           // (m, p+q) simulator inputs
    vector f_sim,           // (m,) fixed latent GP values from Step 1
    row_vector beta_eta,    // (p+q) ARD length scale params
    real lambda_eta,        // GP variance = 1/lambda_eta
    real jitter
  ) {
    int m = rows(x_sim);
    int n_star = rows(x_star);

    // --- Build K(x_sim, x_sim) — noiseless, latent covariance ---
    matrix[m, m] K_sim;
    K_sim = diag_matrix(rep_vector(1.0 / lambda_eta + jitter, m));
    for (i in 1:(m - 1)) {
      for (j in (i + 1):m) {
        row_vector[cols(x_sim)] d = x_sim[i] - x_sim[j];
        real v = exp(-dot_product(beta_eta .* d, d)) / lambda_eta;
        K_sim[i, j] = v;
        K_sim[j, i] = v;
      }
    }
    matrix[m, m] L_sim = cholesky_decompose(K_sim);

    // --- Build K(x_sim, x_star) — cross covariance ---
    matrix[m, n_star] K_cross;
    for (i in 1:m) {
      for (j in 1:n_star) {
        row_vector[cols(x_sim)] d = x_sim[i] - x_star[j];
        K_cross[i, j] = exp(-dot_product(beta_eta .* d, d)) / lambda_eta;
      }
    }

    // --- Build K(x_star, x_star) ---
    matrix[n_star, n_star] K_star;
    K_star = diag_matrix(rep_vector(1.0 / lambda_eta + jitter, n_star));
    for (i in 1:(n_star - 1)) {
      for (j in (i + 1):n_star) {
        row_vector[cols(x_star)] d = x_star[i] - x_star[j];
        real v = exp(-dot_product(beta_eta .* d, d)) / lambda_eta;
        K_star[i, j] = v;
        K_star[j, i] = v;
      }
    }

    // --- Analytic conditional ---
    // mu*  = K_cross' * K_sim^{-1} * f_sim
    // cov* = K_star  - K_cross'  * K_sim^{-1} * K_cross
    vector[m] alpha_vec = mdivide_right_tri_low(
      mdivide_left_tri_low(L_sim, f_sim)', L_sim)';
    vector[n_star] mu_star = K_cross' * alpha_vec;

    matrix[m, n_star] V = mdivide_left_tri_low(L_sim, K_cross);
    matrix[n_star, n_star] cov_star = K_star - V' * V;

    return multi_normal_rng(mu_star, cov_star);
  }
}

data {
  int<lower=1> n;              // number of field observations
  int<lower=1> m;               // number of simulator points
  int<lower=1> n_pred;          // number of prediction points
  int<lower=1> p;               // dimension of x
  int<lower=1> q;               // dimension of t

  vector[n] y;                  // field observations
  matrix[n, p] xf;              // field x inputs
  matrix[m, p] xc;              // simulator x inputs
  matrix[m, q] tc;              // simulator t inputs (known)
  matrix[n_pred, p] x_pred;     // prediction x inputs

  // Fixed from Step 1
  row_vector<lower=1e-6, upper=1>[p+q] rho_eta;
  real<lower=1e-6> lambda_eta;
  vector[m] f_sim;
}

transformed data {
  real jitter = 1e-8;
  row_vector[p+q] beta_eta = -4.0 * log(rho_eta);
  int n_star = n + n_pred;

  // Concatenate field + prediction query points (x part only — t part depends on tf)
  // Full [n + n_pred, p+q] matrix built in transformed parameters once tf is known
}

parameters {
  row_vector<lower=1e-6, upper=1>[q] tf;   // calibration parameters
  real<lower=1e-6> sigma;                  // field observation noise
}

transformed parameters {
  // Build full query matrix: field points with inferred tf, pred points with inferred tf
  matrix[n_star, p+q] x_star;

  x_star[1:n, 1:p] = xf;
  x_star[1:n, (p+1):(p+q)] = rep_matrix(tf, n);
  x_star[(n+1):n_star, 1:p] = x_pred;
  x_star[(n+1):n_star, (p+1):(p+q)] = rep_matrix(tf, n_pred);

  // Simulator input matrix
  matrix[m, p+q] x_sim;
  x_sim[1:m, 1:p] = xc;
  x_sim[1:m, (p+1):(p+q)] = tc;
}

model {
  // Priors
  // tf ~ normal(0, 1);
  sigma ~ normal(0, 0.5);

  // Analytic GP conditional mean at field points given f_sim
  // mu_field = K(xf_aug, x_sim) * K(x_sim, x_sim)^{-1} * f_sim
  // We compute this inline (cheaper than full RNG in model block)
  matrix[m, m] K_sim;
  K_sim = diag_matrix(rep_vector(1.0 / lambda_eta + jitter, m));
  for (i in 1:(m - 1)) {
    for (j in (i + 1):m) {
      row_vector[p+q] d = x_sim[i] - x_sim[j];
      real v = exp(-dot_product(beta_eta .* d, d)) / lambda_eta;
      K_sim[i, j] = v;
      K_sim[j, i] = v;
    }
  }
  matrix[m, m] L_sim = cholesky_decompose(K_sim);
  vector[m] alpha_vec = mdivide_right_tri_low(
                          mdivide_left_tri_low(L_sim, f_sim)',
                          L_sim)';

  // Cross covariance: field points only (for likelihood)
  matrix[m, n] K_cross_field;
  for (i in 1:m) {
    for (j in 1:n) {
      row_vector[p+q] d = x_sim[i] - x_star[j];
      K_cross_field[i, j] = exp(-dot_product(beta_eta .* d, d)) / lambda_eta;
    }
  }
  vector[n] mu_field = K_cross_field' * alpha_vec;

  // Field likelihood conditioned on GP conditional mean
  target += normal_lpdf(y | mu_field, sigma);
}

generated quantities {
  // Draw full predictive sample at field + prediction points
  vector[n + n_pred] f_star = gp_conditional_rng(
    x_star, x_sim, f_sim, beta_eta, lambda_eta, jitter
  );

  vector[n_pred] mu_pred = f_star[(n+1):(n+n_pred)];
  vector[n_pred] y_pred;
  for (j in 1:n_pred)
    y_pred[j] = normal_rng(mu_pred[j], sigma);

  vector[n] log_lik;
  for (i in 1:n)
    log_lik[i] = normal_lpdf(y[i] | f_star[i], sigma);
}
