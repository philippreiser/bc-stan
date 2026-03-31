data {
  int<lower=1> n;
  int<lower=1> m;
  int<lower=1> n_pred;
  int<lower=1> p;
  int<lower=1> q;

  vector[n] y;
  vector[m] eta;

  matrix[n, p] xf;
  matrix[m, p] xc; 
  matrix[m, q] tc; 

  vector[n_pred] x_pred;
}

parameters {
  real<lower=1e-6> lambda_eta;
  real<lower=1e-6> lambda_delta;
  real<lower=1e-6> lambda_e;

  real<lower=0> rho_eta;
  real<lower=0> rho_delta;
}

transformed parameters {

  matrix[m, m] K_eta;
  matrix[n, n] K_delta;
  matrix[m+n, m+n] Sigma;
  matrix[m+n, m+n] L_Sigma;

  // --- Build K_eta ---
  for (i in 1:m) {
    for (j in i:m) {
      real sqdist = square(xc[i] - xc[j]);
      real val = exp(-0.5 * sqdist / square(rho_eta)) / lambda_eta;
      K_eta[i,j] = val;
      K_eta[j,i] = val;
    }
    K_eta[i,i] += 1e-6;
  }

  // --- Build K_delta ---
  for (i in 1:n) {
    for (j in i:n) {
      real sqdist = square(xf[i] - xf[j]);
      real val = exp(-0.5 * sqdist / square(rho_delta)) / lambda_delta;
      K_delta[i,j] = val;
      K_delta[j,i] = val;
    }
    K_delta[i,i] += 1e-6;
  }

  // --- Block covariance ---
  Sigma = rep_matrix(0, m+n, m+n);

  Sigma[1:m, 1:m] = K_eta;
  Sigma[(m+1):(m+n), (m+1):(m+n)] = K_delta;

  // observation noise only for y
  for (i in 1:n)
    Sigma[m+i, m+i] += lambda_e;

  L_Sigma = cholesky_decompose(Sigma);
}

model {

  // Priors
  lambda_eta ~ lognormal(0, 0.5);
  lambda_delta ~ lognormal(0, 0.5);
  lambda_e ~ lognormal(0, 0.5);

  rho_eta ~ normal(0, 1);
  rho_delta ~ normal(0, 1);

  // Likelihood
  vector[m+n] z;
  z[1:m] = eta;
  z[(m+1):(m+n)] = y;

  z ~ multi_normal_cholesky(rep_vector(0, m+n), L_Sigma);
}

generated quantities {

  vector[n_pred] y_pred;

  matrix[n_pred, m+n] K_star;
  matrix[n_pred, n_pred] K_starstar;

  vector[m+n] z;
  vector[m+n] alpha;
  vector[n_pred] mu_pred;
  matrix[n_pred, n_pred] cov_pred;
  matrix[n_pred, n_pred] L_pred;

  // reconstruct z
  z[1:m] = eta;
  z[(m+1):(m+n)] = y;

  // --- Cross covariance ---
  for (i in 1:n_pred) {
    for (j in 1:m) {
      real sqdist = square(x_pred[i] - xc[j]);
      K_star[i,j] =
        exp(-0.5 * sqdist / square(rho_eta)) / lambda_eta;
    }
    for (j in 1:n) {
      real sqdist = square(x_pred[i] - xf[j]);
      K_star[i,m+j] =
        exp(-0.5 * sqdist / square(rho_delta)) / lambda_delta;
    }
  }

  // --- Predictive covariance ---
  for (i in 1:n_pred) {
    for (j in i:n_pred) {
      real sqdist = square(x_pred[i] - x_pred[j]);
      real val =
        exp(-0.5 * sqdist / square(rho_delta)) / lambda_delta;
      K_starstar[i,j] = val;
      K_starstar[j,i] = val;
    }
    K_starstar[i,i] += 1e-6;
  }

  // Solve Sigma^{-1} z via Cholesky
  alpha = mdivide_left_tri_low(L_Sigma, z);
  alpha = mdivide_right_tri_low(alpha', L_Sigma)';

  mu_pred = K_star * alpha;

  // conditional covariance
  matrix[m+n, n_pred] v =
    mdivide_left_tri_low(L_Sigma, K_star');

  cov_pred =
    K_starstar - v' * v;

  L_pred =
    cholesky_decompose(
      cov_pred + diag_matrix(rep_vector(1e-8, n_pred))
    );

  y_pred =
    multi_normal_cholesky_rng(mu_pred, L_pred);
}
