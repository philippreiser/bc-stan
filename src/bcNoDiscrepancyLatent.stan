data {
  int<lower=1> n;              // field data
  int<lower=1> m;              // simulator data
  int<lower=1> p;              // number of observable inputs x
  int<lower=1> q;              // number of calibration parameters t

  vector[n] y;                 // field observations
  vector[m] eta;               // simulator outputs

  matrix[n, p] xf;             // field inputs
  matrix[m, p] xc;             // simulator x inputs
  matrix[m, q] tc;             // simulator t inputs
}

transformed data {
  int<lower=1> N_eta;
  real jitter;

  N_eta = n + m;
  jitter = 1e-8;
}

parameters {
  row_vector<lower=1e-6, upper=1>[q] tf;
  row_vector<lower=1e-6, upper=1>[p+q] rho_eta;

  real<lower=1e-6> lambda_eta;
  real<lower=1e-6> lambda_e;
  real<lower=1e-4> sigma_sim;

  vector[N_eta] eta_std;
}

transformed parameters {
  row_vector[p+q] beta_eta;
  vector[N_eta] f_eta;

  beta_eta = -4.0 * log(rho_eta);

  {
    matrix[N_eta, p+q] xt_eta;
    matrix[N_eta, N_eta] K_eta;

    // build emulator inputs: [(xf, tf), (xc, tc)]
    xt_eta[1:n, 1:p] = xf;
    xt_eta[1:n, (p+1):(p+q)] = rep_matrix(tf, n);

    xt_eta[(n+1):(n+m), 1:p] = xc;
    xt_eta[(n+1):(n+m), (p+1):(p+q)] = tc;

    // GP covariance
    K_eta = diag_matrix(rep_vector(1 / lambda_eta, N_eta));

    for (i in 1:(N_eta - 1)) {
      for (j in (i + 1):N_eta) {
        row_vector[p+q] d = xt_eta[i] - xt_eta[j];
        real sq = dot_product(beta_eta .* d, d);
        real v = exp(-sq) / lambda_eta;
        K_eta[i, j] = v;
        K_eta[j, i] = v;
      }
    }

    K_eta += diag_matrix(rep_vector(jitter, N_eta));

    f_eta = cholesky_decompose(K_eta) * eta_std;
  }
}

model {
  // priors
  rho_eta ~ beta(1, 0.3);
  lambda_eta ~ gamma(10, 10);
  lambda_e ~ gamma(10, 0.03);
  sigma_sim ~ normal(0, 0.05);
  eta_std ~ std_normal();

  // field data
  y ~ normal(
    f_eta[1:n],
    sqrt(1 / lambda_e)
  );

  // simulator data
  eta ~ normal(
    f_eta[(n+1):(n+m)],
    sigma_sim
  );
}

generated quantities {
  vector[n] mu_field;
  vector[m] mu_sim;

  mu_field = f_eta[1:n];
  mu_sim = f_eta[(n+1):(n+m)];
}