data {
  int<lower=1> m;                  // number of simulator data
  int<lower=1> p;                  // dimension of x
  int<lower=1> q;                  // dimension of calibration parameter t

  vector[m] eta;                   // simulator outputs

  matrix[m, p] xc;                 // simulator x inputs
  matrix[m, q] tc;                 // simulator t inputs
}

parameters {
  row_vector<lower=1e-6, upper=1>[p+q] rho_eta;
  real<lower=1e-6> lambda_eta;     // GP precision -> var = 1/lambda_eta
  real<lower=1e-6> sigma;          // observation noise / nugget
}

transformed parameters {
  row_vector[p+q] beta_eta;
  beta_eta = -4.0 * log(rho_eta);
}

model {
  matrix[m, p+q] xt_eta;
  matrix[m, m] K_eta;

  // build simulator inputs u_c = (x_c, t_c)
  xt_eta[1:m, 1:p] = xc;
  xt_eta[1:m, (p+1):(p+q)] = tc;

  // covariance matrix
  K_eta = diag_matrix(rep_vector(1 / lambda_eta + square(sigma), m));

  for (i in 1:(m - 1)) {
    for (j in (i + 1):m) {
      row_vector[p+q] d = xt_eta[i] - xt_eta[j];
      real sq = dot_product(beta_eta .* d, d);
      real v = exp(-sq) / lambda_eta;
      K_eta[i, j] = v;
      K_eta[j, i] = v;
    }
  }

  // priors
  rho_eta ~ beta(1.0, 0.3);
  lambda_eta ~ gamma(10, 10);
  sigma ~ normal(0, 0.5);

  // marginal GP likelihood
  eta ~ multi_normal_cholesky(
    rep_vector(0, m),
    cholesky_decompose(K_eta + diag_matrix(rep_vector(1e-8, m)))
  );
}