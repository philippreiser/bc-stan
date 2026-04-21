/**
 * GP_KOH.stan
 *
 * Kennedy–O'Hagan (KOH) calibration framework, adapted to match the
 * GP power-scaling models in kernel, priors, and parameterisation.
 *
 * Key design: both the emulator GP and the discrepancy GP use analytic
 * conditioning for predictions, so the sampled parameter space depends
 * only on the number of *training* points, never on N_pred:
 *
 *   eta_std   ~ N(0,I)   size N_sim          → f_eta_sim  (emulator at sim)
 *   delta_std ~ N(0,I)   size N_real         → f_delta_real (discrepancy at real)
 *
 * Predictions f_eta_pred and f_delta_pred are both computed in
 * generated quantities via analytic GP conditioning, adding zero
 * parameters and zero cost per leapfrog step.
 */

functions {

  /**
   * Analytic GP conditional mean at new locations given latent values at
   * training locations.  Works for any SE kernel dimension.
   */
  vector gp_conditional_mean(
    array[] vector xw_new,     // (N_new)  new input vectors
    array[] vector xw_obs,     // (N_obs)  training input vectors
    vector         f_obs,      // (N_obs)  latent GP values at training locs
    real           alpha_gp,   // GP marginal std
    array[] real   rho,        // ARD length scales
    real           jitter
  ) {
    int N_obs = size(xw_obs);
    int N_new = size(xw_new);
    matrix[N_obs, N_obs] K = gp_exp_quad_cov(xw_obs, alpha_gp, rho)
                              + diag_matrix(rep_vector(jitter, N_obs));
    matrix[N_obs, N_obs] L = cholesky_decompose(K);
    vector[N_obs] a         = mdivide_right_tri_low(
                                mdivide_left_tri_low(L, f_obs)', L)';
    return gp_exp_quad_cov(xw_obs, xw_new, alpha_gp, rho)' * a;
  }

  /**
   * Draw from the GP posterior at new locations, conditioned on latent
   * values at training locations.
   */
  vector gp_conditional_rng(
    array[] vector xw_new,
    array[] vector xw_obs,
    vector         f_obs,
    real           alpha_gp,
    array[] real   rho,
    real           jitter
  ) {
    int N_obs = size(xw_obs);
    int N_new = size(xw_new);
    matrix[N_obs, N_obs] K = gp_exp_quad_cov(xw_obs, alpha_gp, rho)
                              + diag_matrix(rep_vector(jitter, N_obs));
    matrix[N_obs, N_obs] L  = cholesky_decompose(K);
    vector[N_obs] a          = mdivide_right_tri_low(
                                 mdivide_left_tri_low(L, f_obs)', L)';
    matrix[N_obs, N_new] K_cross = gp_exp_quad_cov(xw_obs, xw_new, alpha_gp, rho);
    vector[N_new] mu_cond        = K_cross' * a;
    matrix[N_obs, N_new] V       = mdivide_left_tri_low(L, K_cross);
    matrix[N_new, N_new] K_new   = gp_exp_quad_cov(xw_new, alpha_gp, rho)
                                    + diag_matrix(rep_vector(jitter, N_new));
    return multi_normal_rng(mu_cond, K_new - V' * V);
  }

}

data {
  int<lower=1> N_sim;                    // number of simulator runs
  int<lower=1> N_real;                   // number of field observations
  int<lower=1> N_pred;                   // number of prediction locations
  int<lower=1> p;                        // dimension of observable inputs x
  int<lower=1> q;                        // dimension of calibration params w

  array[N_sim]  vector[p] x_sim;         // simulator x inputs
  array[N_sim]  vector[q] w_sim;         // simulator w inputs (known)
  vector[N_sim] y_sim;                   // simulator outputs

  array[N_real] vector[p] x_real;        // field x inputs
  vector[N_real] y_real;                 // field observations

  array[N_pred] vector[p] x_pred;        // prediction x inputs

  real<lower=0, upper=1> w_prior_mean;   // prior mean of w_real (in [0,1])
  real<lower=0>          w_prior_sigma;  // prior sd   of w_real
}

transformed data {
  real jitter = 1e-8;

  // Simulator joint (x, w) inputs — fixed, built once
  array[N_sim] vector[p+q] xw_sim;
  for (i in 1:N_sim) {
    xw_sim[i, 1:p]         = x_sim[i];
    xw_sim[i, (p+1):(p+q)] = w_sim[i];
  }

  // Discrepancy training inputs: x_real only (x-space, no w)
  // These are fixed — only x_real, not x_pred.
  // x_pred enters only in generated quantities.
  array[N_real] vector[p] x_delta_obs;
  for (j in 1:N_real) x_delta_obs[j] = x_real[j];
}

parameters {
  // Calibration parameter
  real<lower=0, upper=1> w_real;

  // Emulator GP hyperparameters (x+w space)
  array[p+q] real<lower=0> rho_eta;
  real<lower=0>            alpha_eta;

  // Discrepancy GP hyperparameters (x space only)
  array[p] real<lower=0> rho_delta;
  real<lower=0>          alpha_delta;

  // Noise
  real<lower=0> sigma;      // field observation noise
  real<lower=0> sigma_sim;  // simulator nugget

  // Non-centred latent variables
  // Size N_sim  : emulator at simulator design points
  // Size N_real : discrepancy at real-data locations ONLY (not N_real + N_pred)
  vector[N_sim]  eta_std;
  vector[N_real] delta_std;
}

transformed parameters {

  // ── Emulator GP at simulator locations (non-centred) ──────────────────────
  vector[N_sim] f_eta_sim;
  {
    matrix[N_sim, N_sim] K_eta = gp_exp_quad_cov(xw_sim, alpha_eta, rho_eta)
                                  + diag_matrix(rep_vector(jitter, N_sim));
    f_eta_sim = cholesky_decompose(K_eta) * eta_std;
  }

  // ── Emulator conditional mean at real locations (analytic, no new params) ──
  // w_real enters here through xw_real; xw_sim and f_eta_sim are fixed anchors.
  array[N_real] vector[p+q] xw_real;
  for (j in 1:N_real) {
    xw_real[j, 1:p]         = x_real[j];
    xw_real[j, (p+1):(p+q)] = rep_vector(w_real, q);
  }
  vector[N_real] mu_eta_real = gp_conditional_mean(
    xw_real, xw_sim, f_eta_sim, alpha_eta, rho_eta, jitter);

  // ── Discrepancy GP at real locations only (non-centred) ───────────────────
  // delta_std has size N_real — the only discrepancy parameters in the sampler.
  // f_delta_pred at x_pred is deferred entirely to generated quantities.
  vector[N_real] f_delta_real;
  {
    matrix[N_real, N_real] K_delta =
      gp_exp_quad_cov(x_delta_obs, alpha_delta, rho_delta)
      + diag_matrix(rep_vector(jitter, N_real));
    f_delta_real = cholesky_decompose(K_delta) * delta_std;
  }
}

model {
  // Priors — identical to GP power-scaling models
  rho_eta     ~ inv_gamma(5, 5);
  alpha_eta   ~ std_normal();
  rho_delta   ~ inv_gamma(5, 5);
  alpha_delta ~ std_normal();
  sigma       ~ normal(0, 0.5);
  sigma_sim   ~ normal(0, 0.5);
  eta_std     ~ std_normal();
  delta_std   ~ std_normal();
  w_real      ~ normal(w_prior_mean, w_prior_sigma);

  // Simulator likelihood: emulator reproduces simulator outputs
  target += normal_lpdf(y_sim | f_eta_sim, sigma_sim);

  // Field likelihood: emulator mean + discrepancy + noise
  target += normal_lpdf(y_real | mu_eta_real + f_delta_real, sigma);
}

generated quantities {

  // ── Emulator at prediction locations ──────────────────────────────────────
  // Analytic GP conditional draw, conditioned on f_eta_sim at xw_sim.
  array[N_pred] vector[p+q] xw_pred;
  for (j in 1:N_pred) {
    xw_pred[j, 1:p]         = x_pred[j];
    xw_pred[j, (p+1):(p+q)] = rep_vector(w_real, q);
  }
  vector[N_pred] f_eta_pred = gp_conditional_rng(
    xw_pred, xw_sim, f_eta_sim, alpha_eta, rho_eta, jitter);

  // ── Discrepancy at prediction locations ───────────────────────────────────
  // Analytic GP conditional draw, conditioned on f_delta_real at x_delta_obs.
  // The discrepancy GP lives in x-space only, so no w dimension here.
  array[N_pred] vector[p] x_delta_pred;
  for (j in 1:N_pred) x_delta_pred[j] = x_pred[j];

  vector[N_pred] f_delta_pred = gp_conditional_rng(
    x_delta_pred, x_delta_obs, f_delta_real, alpha_delta, rho_delta, jitter);

  // ── Posterior predictive ──────────────────────────────────────────────────
  vector[N_pred] mu_pred = f_eta_pred + f_delta_pred;

  vector[N_pred] y_pred;
  for (j in 1:N_pred)
    y_pred[j] = normal_rng(mu_pred[j], sigma);
}
