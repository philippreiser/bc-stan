data {
  int<lower=1> n;                  // number of field data
  int<lower=1> m;                  // number of simulator data
  int<lower=1> n_pred;             // number of prediction points
  int<lower=1> p;                  // dimension of x
  int<lower=1> q;                  // dimension of calibration parameter t

  vector[n] y;                     // field observations

  matrix[n, p] xf;                 // field x inputs
  matrix[m, p] xc;                 // simulator x inputs
  matrix[m, q] tc;                 // simulator t inputs
  matrix[n_pred, p] x_pred;        // prediction x inputs

  row_vector<lower=1e-6, upper=1>[p+q] rho_eta;
  real<lower=1e-6> lambda_eta;
  vector[n + m + n_pred] eta_std;
}

transformed data {
  int<lower=1> N = n + m + n_pred;
  real jitter = 1e-8;
}

parameters {
  row_vector<lower=1e-6, upper=1>[q] tf;
  real<lower=1e-6> sigma;
}

transformed parameters {
  row_vector[p+q] beta_eta;
  vector[N] f_eta;

  beta_eta = -4.0 * log(rho_eta);

  {
    matrix[N, p+q] xt_eta;
    matrix[N, N] K_eta;

    // field inputs use unknown tf
    xt_eta[1:n, 1:p] = xf;
    xt_eta[1:n, (p+1):(p+q)] = rep_matrix(tf, n);

    // simulator inputs use known tc
    xt_eta[(n+1):(n+m), 1:p] = xc;
    xt_eta[(n+1):(n+m), (p+1):(p+q)] = tc;

    // prediction inputs use same inferred tf
    xt_eta[(n+m+1):(n+m+n_pred), 1:p] = x_pred;
    xt_eta[(n+m+1):(n+m+n_pred), (p+1):(p+q)] = rep_matrix(tf, n_pred);

    K_eta = diag_matrix(rep_vector(1 / lambda_eta, N));

    for (i in 1:(N - 1)) {
      for (j in (i + 1):N) {
        row_vector[p+q] d = xt_eta[i] - xt_eta[j];
        real sq = dot_product(beta_eta .* d, d);
        real v = exp(-sq) / lambda_eta;
        K_eta[i, j] = v;
        K_eta[j, i] = v;
      }
    }

    K_eta += diag_matrix(rep_vector(jitter, N));

    f_eta = cholesky_decompose(K_eta) * eta_std;
  }
}

model {
  sigma ~ normal(0, 0.5);

  // only real-data likelihood in step 2
  target += normal_lpdf(y | f_eta[1:n], sigma);
}

generated quantities {
  vector[n_pred] mu_pred;
  vector[n_pred] y_pred;
  vector[n] log_lik_real;

  for (j in 1:n_pred) {
    mu_pred[j] = f_eta[n + m + j];
    y_pred[j] = normal_rng(mu_pred[j], sigma);
  }

  for (i in 1:n) {
    log_lik_real[i] = normal_lpdf(y[i] | f_eta[i], sigma);
  }
}
