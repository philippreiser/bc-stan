data {
  int<lower=1> N_sim;              // number of simulator data
  int<lower=1> p;                  // dimension of x
  int<lower=1> q;                  // dimension of calibration parameter w
  vector[N_sim] y_sim;             // simulator outputs
  array[N_sim] vector [p] x_sim;          // simulator x inputs
  array[N_sim] vector [q] w_sim;          // simulator w inputs
  int<lower=1> N_real;                  // number of field observations
  array[N_real] vector[p] x_real;       // field x inputs (known)
  vector[N_real] y_real;                // field observations
  real<lower=0> alpha_sim;              // likelihood weight for sim data
  real<lower=0> alpha_real;            // likelihood weight for real data
  vector[q] w_prior_mean;           // prior mean for w_real
  vector<lower=0>[q] w_prior_sigma; // prior sd for w_real

}
transformed data {
  real jitter = 1e-8;
  int N_total = N_sim + N_real;
}
parameters {
  array[p+q] real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=1e-6> sigma;
  vector[N_total] eta_std;               // latent GP in non-centered form
  vector<lower=0, upper=1>[q] w_real;
}
transformed parameters {
  vector[N_total] f;
  {
    array[N_total] vector[p+q] xw_sim_real;
    matrix[N_total, N_total] K;
    xw_sim_real[1:N_sim, 1:p] = x_sim;
    xw_sim_real[1:N_sim, (p+1):(p+q)] = w_sim;
    xw_sim_real[(N_sim+1):N_total, 1:p] = x_real;
    xw_sim_real[(N_sim+1):N_total, (p+1):(p+q)] = rep_array(w_real, N_real);
    K = gp_exp_quad_cov(xw_sim_real, alpha, rho);
    K += diag_matrix(rep_vector(jitter, N_total));
    f = cholesky_decompose(K) * eta_std;
  }
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ normal(0, 0.5);
  w_real ~ normal(w_prior_mean, w_prior_sigma);
  eta_std ~ std_normal();
  
  target += alpha_sim * normal_lpdf(y_sim | f[1:N_sim], sigma);
  target += alpha_real * normal_lpdf(y_real | f[(N_sim + 1):N_total], sigma);
}
