//
// This Stan program defines a simple model, with a
// vector of values 'y' modeled as normally distributed
// with mean 'mu' and standard deviation 'sigma'.
//
// Learn more about model development with Stan at:
//
//    http://mc-stan.org/users/interfaces/rstan.html
//    https://github.com/stan-dev/rstan/wiki/RStan-Getting-Started
//

// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  int<lower=0> H;
  vector[N] y;
  vector[N] x;
  real<lower=0> sigma_weight;
  real<lower=0> sigma_bias;
  real<lower=0> sigma_beta;
  real<lower=0> sigma_alpha;
}

// The parameters accepted by the model. Our model
// accepts two parameters 'mu' and 'sigma'.
parameters {
  vector[H] bias;
  vector[H] weights;
  vector[H] beta;
  real alpha;
  real<lower=0> sigma;

  real<lower=0> s_weight;
  real<lower=0> s_bias;
  real<lower=0> s_beta;
}

transformed parameters {
  matrix[N, H] hidden_units;
  vector[N] mu;

  for(n in 1:N) {
    mu[n] = alpha;
    for(h in 1:H) {
      hidden_units[n,h] = bias[h] + weights[h] * x[n];
      // hidden_units[n,h] = tanh(hidden_units[n,h]);
      hidden_units[n,h] = fmax(0, hidden_units[n,h]);
      mu[n] = mu[n] + beta[h] * hidden_units[n,h];
    }
  }
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'mu'
// and standard deviation 'sigma'.
model {
  y ~ normal(mu, sigma);
  bias ~ normal(0, s_bias);
  weights ~ normal(0, s_weight);
  beta ~ normal(0, s_beta);
  alpha ~ normal(0,sigma_alpha);

  s_bias ~ cauchy(0, sigma_bias);
  s_weight ~ cauchy(0, sigma_weight);
  s_beta ~ cauchy(0, sigma_beta);
}

