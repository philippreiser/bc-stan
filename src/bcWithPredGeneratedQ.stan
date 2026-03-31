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
  int<lower = 1> N_train;
  vector[n+m] y_eta;
  vector[n+m] mu; // mean vector
  matrix[n+n_pred, p] X; // X=[xf, x_pred]
  
  N = n + m + n_pred;
  N_train = n + m;
  // set mean vector to zero
  for (i in 1:N_train) {
    mu[i] = 0;
  }
  X = append_row(xf, x_pred);
  y_eta = append_row(y, eta); // y_eta = [y, eta]
}

parameters {
  // tf: calibration parameters
  // rho_eta: reparameterization of beta_eta
  // rho_delta: reparameterization of beta_delta
  // lambda_eta: precision parameter for eta
  // lambda_delta: precision parameter for bias term
  // lambda_e: precision parameter of observation error
  // y_pred: predictions
  row_vector<lower=0, upper=1>[q] tf; 
  row_vector<lower=0, upper=1>[p+q] rho_eta; 
  row_vector<lower=0, upper=1>[p] rho_delta; 
  real<lower=1e-6> lambda_eta; 
  real<lower=1e-6> lambda_delta;
  real<lower=1e-6> lambda_e; 
  // vector[n_pred] y_pred; 
}

transformed parameters {
  // beta_delta: correlation parameter for bias term
  // beta_e: correlation parameter of observation error
  row_vector[p+q] beta_eta;
  row_vector[p] beta_delta;
  beta_eta = -4.0 * log(rho_eta);
  beta_delta = -4.0 * log(rho_delta);
  
    // declare variables
  matrix[N_train, p+q] xt_train;
  matrix[N_train, N_train] sigma_eta; // simulator covarinace
  matrix[n, n] sigma_delta; // bias term covariance
  matrix[N_train, N_train] sigma_TT; // covariance matrix
  matrix[N_train, N_train] L_TT; // cholesky decomposition of covariance matrix 
  vector[N_train] z_train; // z_train = [y, eta]
  row_vector[p] temp_delta;
  row_vector[p+q] temp_eta;
  
  z_train = y_eta; // z_train = [y, eta]
  
  // xt_train = [[xf,tf],[xc,tc]]
  xt_train[1:n, 1:p] = xf;
  xt_train[1:n, (p+1):(p+q)] = rep_matrix(tf, n);
  xt_train[(n+1):(n+m), 1:p] = xc;
  xt_train[(n+1):(n+m), (p+1):(p+q)] = tc;
  
  // diagonal elements of sigma_eta
  sigma_eta = diag_matrix(rep_vector((1 / lambda_eta), N_train));

  // off-diagonal elements of sigma_eta
  for (i in 1:(N_train-1)) {
    for (j in (i+1):N_train) {
      temp_eta = xt_train[i] - xt_train[j];
      sigma_eta[i, j] = beta_eta .* temp_eta * temp_eta';
      sigma_eta[i, j] = exp(-sigma_eta[i, j]) / lambda_eta;
      sigma_eta[j, i] = sigma_eta[i, j];
    }
  }

  // diagonal elements of sigma_delta
  sigma_delta = diag_matrix(rep_vector((1 / lambda_delta), 
    n));
  
  // off-diagonal elements of sigma_delta
  for (i in 1:(n-1)) {
    for (j in (i+1):(n)) {
      temp_delta = xf[i] - xf[j];
      sigma_delta[i, j] = beta_delta .* temp_delta * temp_delta';
      sigma_delta[i, j] = exp(-sigma_delta[i, j]) / lambda_delta;
      sigma_delta[j, i] = sigma_delta[i, j];
    }   
  }

  // computation of covariance matrix sigma_z 
  sigma_TT = sigma_eta;
  sigma_TT[1:n, 1:n] = sigma_eta[1:n, 1:n] + 
    sigma_delta[1:n, 1:n];

  // add observation errors
  for (i in 1:n) {
    sigma_TT[i, i] += (1.0 / lambda_e);
  }  
  // make numerically stable
  sigma_TT += diag_matrix(rep_vector(1e-8, N_train));

  L_TT = cholesky_decompose(sigma_TT); // cholesky decomposition 
}

model {

  // Specify priors here
  rho_eta[1:(p+q)] ~ beta(1.0, 0.3);
  rho_delta[1:p] ~ beta(1.0, 0.3);
  lambda_eta ~ gamma(10, 10); // gamma (shape, rate)
  lambda_delta ~ gamma(10, 0.3); 
  lambda_e ~ gamma(10, 0.03); 
  
  z_train ~ multi_normal_cholesky(mu, L_TT);
}

generated quantities {
  vector[n_pred] y_pred;
  
  matrix[N_train, n_pred] Sigma_TP;
  matrix[n_pred, n_pred] Sigma_PP;

  // -----------------------------
  // 1. Emulator cross-covariance
  // -----------------------------

  for (i in 1:N_train) {
    for (j in 1:n_pred) {

      row_vector[p+q] x_pred_full;

      x_pred_full[1:p] = x_pred[j];
      x_pred_full[(p+1):(p+q)] = tf;

      row_vector[p+q] diff = xt_train[i] - x_pred_full;

      real sqdist = dot_product(beta_eta .* diff, diff);

      Sigma_TP[i,j] = exp(-sqdist) / lambda_eta;
    }
  }

  // -----------------------------
  // 2. Emulator predictive block
  // -----------------------------

  Sigma_PP = diag_matrix(rep_vector(1 / lambda_eta, n_pred));

  for (i in 1:(n_pred-1)) {
    for (j in (i+1):n_pred) {

      row_vector[p+q] xi;
      row_vector[p+q] xj;

      xi[1:p] = x_pred[i];
      xi[(p+1):(p+q)] = tf;

      xj[1:p] = x_pred[j];
      xj[(p+1):(p+q)] = tf;

      row_vector[p+q] diff = xi - xj;
      real sqdist = dot_product(beta_eta .* diff, diff);
      real val = exp(-sqdist) / lambda_eta;

      Sigma_PP[i,j] = val;
      Sigma_PP[j,i] = val;
    }
  }

  // ---------------------------------
  // 3. Add discrepancy cross-covariance
  //    (only for field rows)
  // ---------------------------------

  for (i in 1:n) {
    for (j in 1:n_pred) {

      row_vector[p] diff = xf[i] - x_pred[j];
      real sqdist = dot_product(beta_delta .* diff, diff);

      Sigma_TP[i,j] += exp(-sqdist) / lambda_delta;
    }
  }

  // ---------------------------------
  // 4. Add discrepancy predictive block
  // ---------------------------------

  for (i in 1:(n_pred-1)) {
    for (j in (i+1):n_pred) {

      row_vector[p] diff = x_pred[i] - x_pred[j];
      real sqdist = dot_product(beta_delta .* diff, diff);
      real val = exp(-sqdist) / lambda_delta;

      Sigma_PP[i,j] += val;
      Sigma_PP[j,i] += val;
    }
  }

  // ---------------------------------
  // 5. Conditional Gaussian formula
  // ---------------------------------

  matrix[N_train, n_pred] v =
    mdivide_left_tri_low(L_TT, Sigma_TP);

  matrix[n_pred, n_pred] Sigma_cond =
    Sigma_PP - v' * v;

  vector[n_pred] mu_cond =
    Sigma_TP'
    * mdivide_left_tri_low(L_TT,
        mdivide_left_tri_low(L_TT, z_train));

  // add observation noise
  for (i in 1:n_pred)
    Sigma_cond[i,i] += 1 / lambda_e;

  Sigma_cond += diag_matrix(rep_vector(1e-8, n_pred));

  matrix[n_pred, n_pred] L_cond =
    cholesky_decompose(Sigma_cond);

  y_pred =
    multi_normal_cholesky_rng(mu_cond, L_cond);
}
