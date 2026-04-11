data {
  int<lower=1> N_sim;              // number of simulator data
  int<lower=1> N_pred;             // number of prediction points
  int<lower=1> p;                  // dimension of x
  int<lower=1> q;                  // dimension of calibration parameter w
  vector[N_sim] y_sim;             // simulator outputs
  array[N_sim] vector [p] x_sim;          // simulator x inputs
  array[N_sim] vector [q] w_sim;          // simulator w inputs
}
transformed data {
  real jitter = 1e-8;
}
parameters {
  array[p+q] real<lower=0> rho;
  real<lower=0> alpha;
  real<lower=1e-6> sigma;
  vector[N_sim] eta_std;               // latent GP in non-centered form
}
transformed parameters {
  vector[N_sim] f_sim;
  {
    array[N_sim] vector[p+q] xw_sim;
    matrix[N_sim, N_sim] K;
    xw_sim[1:N_sim, 1:p] = x_sim;
    xw_sim[1:N_sim, (p+1):(p+q)] = w_sim;
    K = gp_exp_quad_cov(xw_sim, alpha, rho);
    K += diag_matrix(rep_vector(jitter, N_sim));
    f_sim = cholesky_decompose(K) * eta_std;
  }
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ normal(0, 0.5);
  eta_std ~ std_normal();
  
  target += normal_lpdf(y_sim | f_sim, sigma);
}
