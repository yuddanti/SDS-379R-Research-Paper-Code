## Load

library(tidyverse)
library(rstan)

## Generate some data

generate_linear_data <- function(N = 200, noise_std = 0.2) {
  x <- runif(N)
  y <- 2 * x + 1 + rnorm(N, sd = noise_std)
  list(x = x, y = y)
}

generate_nonlinear_data <- function(N = 200, noise_std = 0.2) {
  x <- runif(N)
  y <- sin(2 * pi * x) + rnorm(N, sd = noise_std)
  list(x = x, y = y)
}

## Choose and generate data

data <- generate_nonlinear_data(N = 200, noise_std = 0.2)
x <- data$x
y <- data$y

plot(x, y, main = "Generated Data")

## Compile the model

my_stan_model <- stan_model("neural_network.stan")

H <- 30
my_data <- list(
  x = as.numeric(scale(x)),
  y = y,
  N = length(y),
  H = H,
  sigma_bias = 1,
  sigma_weight = 1,
  sigma_beta = 1 / H,
  sigma_alpha = 1
)

fitted_nn <- rstan::vb(my_stan_model, data = my_data, iter = 10000,
                       grad_samples = 10, adapt_iter = 1000,
                       algorithm = "fullrank")


output <- as.matrix(fitted_nn, "mu")

xs <- scale(x) %>% as.numeric()

mu_hat <- colMeans(output)
LCL <- apply(output, 2, \(x) quantile(x, 0.025))
UCL <- apply(output, 2, \(x) quantile(x, 0.975))

plot(xs, y)
o <- order(x)
lines(xs[o], mu_hat[o])
lines(xs[o], LCL[o], lty = 2)
lines(xs[o], UCL[o], lty = 2)

## Evaluation metrics

# Mean Absolute Error
mae <- mean(abs(mu_hat - y))

# Normalized accuracy
accuracy <- 1 - mae / (max(y) - min(y))

# Coverage (percentage of true y inside [LCL, UCL])
coverage <- mean(y >= LCL & y <= UCL)

# Average width of prediction interval
interval_width <- mean(UCL - LCL)

## Print metrics ----

cat("Evaluation Metrics:\n")
cat("MAE:", round(mae, 4), "\n")
cat("Accuracy:", round(accuracy * 100, 2), "%\n")
cat("Coverage:", round(coverage * 100, 2), "%\n")
cat("Interval Width:", round(interval_width, 4), "\n")


