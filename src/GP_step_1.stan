data {
  int<lower=1> m;                  // number of simulator data
  int<lower=1> n_pred;             // number of prediction points
  int<lower=1> p;                  // dimension of x
  int<lower=1> q;                  // dimension of calibration parameter t

  vector[m] eta;                   // simulator outputs

  matrix[m, p] xc;                 // simulator x inputs
  matrix[m, q] tc;                 // simulator t inputs

  // vector[q] tf_prior_mean;
  // vector<lower=0>[q] tf_prior_sigma;
}

transformed data {
  int<lower=1> N = m;
  real jitter = 1e-8;
}

parameters {
  row_vector<lower=1e-6, upper=1>[p+q] rho_eta;

  real<lower=1e-6> lambda_eta;     // GP precision -> var = 1/lambda_eta
  real<lower=1e-6> sigma;          // shared noise scale

  vector[N] eta_std;               // latent GP in non-centered form
}

transformed parameters {
  row_vector[p+q] beta_eta;
  vector[N] f_eta;

  beta_eta = -4.0 * log(rho_eta);

  {
    matrix[N, p+q] xt_eta;
    matrix[N, N] K_eta;

    // simulator inputs use known tc
    xt_eta[1:m, 1:p] = xc;
    xt_eta[1:m, (p+1):(p+q)] = tc;

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
  // priors
  // tf ~ normal(tf_prior_mean', tf_prior_sigma');
  rho_eta ~ beta(1.0, 0.3);
  lambda_eta ~ gamma(10, 10);
  sigma ~ normal(0, 0.5);
  eta_std ~ std_normal();

  // simulation likelihood
  target += normal_lpdf(eta | f_eta[1:m], sigma);
}
