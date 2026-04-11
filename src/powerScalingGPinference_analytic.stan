data {
  int<lower=1> n;                  // number of field data
  int<lower=1> m;                  // number of simulator data
  int<lower=1> n_pred;             // number of prediction points
  int<lower=1> p;                  // dimension of x
  int<lower=1> q;                  // dimension of calibration parameter t

  vector[n] y;                     // field observations
  vector[m] eta;                   // simulator outputs

  matrix[n, p] xf;                 // field x inputs
  matrix[m, p] xc;                 // simulator x inputs
  matrix[m, q] tc;                 // simulator t inputs
  matrix[n_pred, p] x_pred;        // prediction x inputs

  row_vector<lower=1e-6, upper=1>[p+q] rho_eta;
  real<lower=1e-6> lambda_eta;
}

transformed data {
  real jitter = 1e-8;
}

parameters {
  row_vector<lower=1e-6, upper=1>[q] tf;
  real<lower=1e-6> sigma;
}

transformed parameters {
  row_vector[p+q] beta_eta;
  vector[n] mu_field;
  matrix[n, n] Sigma_field;

  beta_eta = -4.0 * log(rho_eta);

  {
    // Inputs
    matrix[m, p+q] u_sim;
    matrix[n, p+q] u_field;

    // Kernel blocks
    matrix[m, m] K_cc;
    matrix[n, m] K_fc;
    matrix[n, n] K_ff;
    matrix[m, m] L_cc;
    matrix[m, n] L_div;

    // Build simulator inputs
    u_sim[:, 1:p] = xc;
    u_sim[:, (p+1):(p+q)] = tc;

    // Build field inputs using unknown tf
    u_field[:, 1:p] = xf;
    u_field[:, (p+1):(p+q)] = rep_matrix(tf, n);

    // --- K_cc ---
    K_cc = diag_matrix(rep_vector(1 / lambda_eta + square(sigma), m));
    for (i in 1:(m - 1)) {
      for (j in (i + 1):m) {
        row_vector[p+q] d = u_sim[i] - u_sim[j];
        real sq = dot_product(beta_eta .* d, d);
        real v = exp(-sq) / lambda_eta;
        K_cc[i, j] = v;
        K_cc[j, i] = v;
      }
    }
    K_cc += diag_matrix(rep_vector(jitter, m));

    // --- K_fc ---
    for (i in 1:n) {
      for (j in 1:m) {
        row_vector[p+q] d = u_field[i] - u_sim[j];
        real sq = dot_product(beta_eta .* d, d);
        K_fc[i, j] = exp(-sq) / lambda_eta;
      }
    }

    // --- K_ff ---
    K_ff = diag_matrix(rep_vector(1 / lambda_eta, n));
    for (i in 1:(n - 1)) {
      for (j in (i + 1):n) {
        row_vector[p+q] d = u_field[i] - u_field[j];
        real sq = dot_product(beta_eta .* d, d);
        real v = exp(-sq) / lambda_eta;
        K_ff[i, j] = v;
        K_ff[j, i] = v;
      }
    }
    K_ff += diag_matrix(rep_vector(jitter, n));

    // Conditional GP formulas
    L_cc = cholesky_decompose(K_cc);
    L_div = mdivide_left_tri_low(L_cc, K_fc');

    mu_field = L_div' * mdivide_left_tri_low(L_cc, eta);
    Sigma_field = K_ff - L_div' * L_div;
    Sigma_field = 0.5 * (Sigma_field + Sigma_field'); // symmetrize
  }
}

model {
  sigma ~ normal(0, 0.5);

  // y | eta, tf, sigma
  y ~ multi_normal_cholesky(
    mu_field,
    cholesky_decompose(Sigma_field + diag_matrix(rep_vector(square(sigma) + 1e-8, n)))
  );
}

generated quantities {
  vector[n_pred] mu_pred;
  vector[n_pred] y_pred;
  vector[n] log_lik_real;

  {
    // Inputs
    matrix[m, p+q] u_sim;
    matrix[n_pred, p+q] u_pred;

    // Kernel blocks
    matrix[m, m] K_cc;
    matrix[n_pred, m] K_pc;
    matrix[n_pred, n_pred] K_pp;
    matrix[m, m] L_cc;
    matrix[m, n_pred] L_div;
    matrix[n_pred, n_pred] Sigma_pred;

    // Build simulator inputs
    u_sim[:, 1:p] = xc;
    u_sim[:, (p+1):(p+q)] = tc;

    // Build prediction inputs using current tf draw
    u_pred[:, 1:p] = x_pred;
    u_pred[:, (p+1):(p+q)] = rep_matrix(tf, n_pred);

    // --- K_cc ---
    K_cc = diag_matrix(rep_vector(1 / lambda_eta + square(sigma), m));
    for (i in 1:(m - 1)) {
      for (j in (i + 1):m) {
        row_vector[p+q] d = u_sim[i] - u_sim[j];
        real sq = dot_product(beta_eta .* d, d);
        real v = exp(-sq) / lambda_eta;
        K_cc[i, j] = v;
        K_cc[j, i] = v;
      }
    }
    K_cc += diag_matrix(rep_vector(jitter, m));

    // --- K_pc ---
    for (i in 1:n_pred) {
      for (j in 1:m) {
        row_vector[p+q] d = u_pred[i] - u_sim[j];
        real sq = dot_product(beta_eta .* d, d);
        K_pc[i, j] = exp(-sq) / lambda_eta;
      }
    }

    // --- K_pp ---
    K_pp = diag_matrix(rep_vector(1 / lambda_eta, n_pred));
    for (i in 1:(n_pred - 1)) {
      for (j in (i + 1):n_pred) {
        row_vector[p+q] d = u_pred[i] - u_pred[j];
        real sq = dot_product(beta_eta .* d, d);
        real v = exp(-sq) / lambda_eta;
        K_pp[i, j] = v;
        K_pp[j, i] = v;
      }
    }
    K_pp += diag_matrix(rep_vector(jitter, n_pred));

    // Conditional prediction
    L_cc = cholesky_decompose(K_cc);
    L_div = mdivide_left_tri_low(L_cc, K_pc');

    mu_pred = L_div' * mdivide_left_tri_low(L_cc, eta);
    Sigma_pred = K_pp - L_div' * L_div;
    Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred');

    // posterior predictive observations
    {
      vector[n_pred] f_pred = multi_normal_rng(mu_pred, Sigma_pred);
      for (j in 1:n_pred) {
        y_pred[j] = normal_rng(f_pred[j], sigma);
      }
    }
  }

  for (i in 1:n) {
    log_lik_real[i] = normal_lpdf(y[i] | mu_field[i], sqrt(Sigma_field[i, i] + square(sigma)));
  }
}
