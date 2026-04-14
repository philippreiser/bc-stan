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
  vector[N_sim] eta_sim_std;               // latent GP in non-centered form
  vector<lower=0, upper=1>[q] w_real;
}
transformed parameters {
  // Latent GP values at simulation input locations
  vector[N_sim] f_sim;
  // GP conditional mean at real input locations, given f_sim
  vector[N_real] mu_real;
  {
    // Build joint (x, w) input matrix for simulator points
    array[N_sim] vector[p + q] xw_sim;
    xw_sim[1:N_sim, 1:p] = x_sim;
    xw_sim[1:N_sim, (p+1):(p+q)] = w_sim;
    
    // Real data joint inputs: x_real paired with the (shared) w_real
    array[N_real] vector[p + q] xw_real;
    xw_real[1:N_real, 1:p] = x_real;
    xw_real[1:N_real, (p+1):(p+q)] = rep_array(w_real, N_real);

    matrix[N_sim, N_sim] K_sim = gp_exp_quad_cov(xw_sim, alpha, rho);
    K_sim += diag_matrix(rep_vector(jitter, N_sim));
    matrix[N_sim, N_sim] L = cholesky_decompose(K_sim);
    f_sim = L * eta_sim_std;

    matrix[N_sim, N_real] K_cross = gp_exp_quad_cov(xw_sim, xw_real, alpha, rho);
    // Conditional mean: mu = K_cross' * K^{-1} * f_sim
    vector[N_sim] alpha_vec = mdivide_right_tri_low(
      mdivide_left_tri_low(L, f_sim)', L)';
    mu_real = K_cross' * alpha_vec;
  }
}
model {
  rho ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ normal(0, 0.5);
  w_real ~ normal(w_prior_mean, w_prior_sigma);
  eta_sim_std ~ std_normal();
  
  // --- Power-scaled likelihoods ---
  // Simulation data: full likelihood scaled by alpha_sim
  target += alpha_sim  * normal_lpdf(y_sim  | f_sim, sigma);

  // Real data: likelihood scaled by alpha_real (conditional GP mean as mu)
  target += alpha_real * normal_lpdf(y_real | mu_real, sigma);
}
