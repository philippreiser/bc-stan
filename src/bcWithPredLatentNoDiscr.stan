data {
  int<lower=1> n; // number of field data
  int<lower=1> m; // number of computer simulation
  int<lower=1> n_pred; // number of predictions
  int<lower=1> p; // number of observable inputs x
  int<lower=1> q; // number of calibration parameters t
  vector[n] y; // field observations
  vector[m] eta; // output of computer simulations
  matrix[n, p] xf; // observable inputs corresponding to y
  // (xc, tc): design points corresponding to eta
  matrix[m, p] xc; 
  matrix[m, q] tc; 
  // x_pred: new design points for predictions
  matrix[n_pred, p] x_pred; 
}

transformed data {
  int<lower = 1> N;
  int<lower = 1> N_eta = n + m + n_pred;      // emulator GP size

  // int<lower = 1> N_delta = n + n_pred;        // discrepancy GP size
  real jitter = 1e-8;
  N = n + m + n_pred;
}

parameters {
  // tf: calibration parameters
  // rho_eta: reparameterization of beta_eta
  // rho_delta: reparameterization of beta_delta
  // lambda_eta: precision parameter for eta
  // lambda_delta: precision parameter for bias term
  // lambda_e: precision parameter of observation error
  // y_pred: predictions
  row_vector<lower=1e-6, upper=1>[q] tf; 
  row_vector<lower=1e-6, upper=1>[p+q] rho_eta; 
  // row_vector<lower=0, upper=1>[p] rho_delta; 
  real<lower=1e-6> lambda_eta; 
  // real<lower=1e-6> lambda_delta;
  real<lower=1e-6> lambda_e; 
  real<lower=1e-4> sigma_sim;
  
  vector[N_eta] eta_std;      // non-centered emulator GP
  // vector[N_delta] delta_std;  // non-centered discrepancy GP
}

transformed parameters {
  // beta_delta: correlation parameter for bias term
  // beta_e: correlation parameter of observation error
  row_vector[p+q] beta_eta;
  // row_vector[p] beta_delta;
  beta_eta = -4.0 * log(rho_eta);
  // beta_delta = -4.0 * log(rho_delta);
  
  vector[N_eta] f_eta;
  // vector[N_delta] f_delta;
  {
    matrix[N_eta, p+q] xt_eta;
    // matrix[N_delta, p] xt_delta;
    
    // Build emulator input
    xt_eta[1:n, 1:p] = xf;
    xt_eta[1:n, (p+1):(p+q)] = rep_matrix(tf, n);

    xt_eta[(n+1):(n+m), 1:p] = xc;
    xt_eta[(n+1):(n+m), (p+1):(p+q)] = tc;

    xt_eta[(n+m+1):(n+m+n_pred), 1:p] = x_pred;
    xt_eta[(n+m+1):(n+m+n_pred), (p+1):(p+q)] = rep_matrix(tf, n_pred);

    // Build discrepancy input
    // xt_delta[1:n,] = xf;
    // xt_delta[(n+1):(n+n_pred),] = x_pred;
    
    // --- Emulator GP ---
    matrix[N_eta, N_eta] K_eta =
      diag_matrix(rep_vector(1/lambda_eta, N_eta));

    for (i in 1:(N_eta-1))
      for (j in (i+1):N_eta) {
        row_vector[p+q] d = xt_eta[i] - xt_eta[j];
        real sq = dot_product(beta_eta .* d, d);
        real v = exp(-sq) / lambda_eta;
        K_eta[i,j] = v;
        K_eta[j,i] = v;
      }

    K_eta += diag_matrix(rep_vector(jitter, N_eta));
    matrix[N_eta,N_eta] L_eta = cholesky_decompose(K_eta);
    f_eta = L_eta * eta_std;
    
    // --- Discrepancy GP ---
    // matrix[N_delta, N_delta] K_delta =
    //   diag_matrix(rep_vector(1/lambda_delta, N_delta));
    // 
    // for (i in 1:(N_delta-1))
    //   for (j in (i+1):N_delta) {
    //     row_vector[p] d = xt_delta[i] - xt_delta[j];
    //     real sq = dot_product(beta_delta .* d, d);
    //     real v = exp(-sq) / lambda_delta;
    //     K_delta[i,j] = v;
    //     K_delta[j,i] = v;
    //   }
    // 
    // K_delta += diag_matrix(rep_vector(jitter, N_delta));
    // matrix[N_delta,N_delta] L_delta = cholesky_decompose(K_delta);
    // f_delta = L_delta * delta_std;
  }
}

model {
  // Specify priors here
  rho_eta[1:(p+q)] ~ beta(1.0, 0.3);
  // rho_delta[1:p] ~ beta(1.0, 0.3);
  lambda_eta ~ gamma(10, 10); // gamma (shape, rate)
  // lambda_delta ~ gamma(10, 0.3); 
  lambda_e ~ gamma(10, 0.03);
  sigma_sim ~ normal(0, 0.05);
  
  // new priors because of latent GP formultion?
  eta_std ~ std_normal();
  // delta_std ~ std_normal();
  
  // likelihood

  // field data
  y ~ normal(
    f_eta[1:n], // + f_delta[1:n],
    sqrt(1/lambda_e)
  );

  // simulator data
  eta ~ normal(
    f_eta[(n+1):(n+m)],
    sigma_sim   // tiny nugget for numerical stability
  );
}

generated quantities {
  vector[n_pred] mu_pred;
  for (j in 1:n_pred) {
    mu_pred[j] = f_eta[n+m+j]; //+ f_delta[n+j];
  }
  vector[n_pred] y_pred;
  for (j in 1:n_pred) {
    y_pred[j] = normal_rng(mu_pred[j], sqrt(1/lambda_e));
  }
}
