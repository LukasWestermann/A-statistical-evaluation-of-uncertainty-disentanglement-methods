# install.packages("bamlss")   # if not installed
install.packages("bamlss")
install.packages("ggplot2")
library(bamlss)
library(ggplot2)

set.seed(0)

# 1) Simulate toy data (train x in [0, 10])
n_train <- 1000
x <- runif(n_train, min = 0, max = 10)
eps1 <- rnorm(n_train, mean = 0, sd = 0.3)
eps2 <- rnorm(n_train, mean = 0, sd = 0.3)
y <- x * sin(x) + eps1 * x + eps2

dat <- data.frame(x = x, y = y)

# Evaluation grid across ID + OOD (to 15)
x_grid <- seq(0, 15, length.out = 600)
grid <- data.frame(x = x_grid)
y_clean <- x_grid * sin(x_grid)

# 2) Bayesian GAMLSS with smooth μ(x) and σ(x)
#    Using P-splines (ps) for both mu and sigma
#    Note: You can adjust k (basis dimension) if you want more/less flexibility.
form <- list(
  y ~ s(x, bs = "ps", k = 20),
  sigma ~ s(x, bs = "ps", k = 15)
)

# MCMC controls (adjust to your compute budget)
ctrl <- list(n.iter = 12000, burnin = 2000, thin = 10)

fit <- bamlss(
  formula = form,
  family = "gaussian",
  data = dat,
  sampler = TRUE,
  n.iter = ctrl$n.iter, burnin = ctrl$burnin, thin = ctrl$thin
)

# 3) Posterior sampling of parameters on the grid
#    nsamples controls how many posterior draws to use for uncertainty decomposition.
nsamples <- 1000

# Predict posterior samples of mu(x) and sigma(x) on the grid.
# predict(..., type="parameter", model="mu"|"sigma", nsamples=...) returns
# a matrix-like object with one draw per column (or row, depending on version).
mu_samps <- predict(fit, newdata = grid, model = "mu", type = "parameter", nsamples = nsamples)
sg_samps <- predict(fit, newdata = grid, model = "sigma", type = "parameter", nsamples = nsamples)

# Ensure matrices are [N x S] (N points on grid, S posterior draws)
to_mat <- function(a) {
  a <- as.matrix(a)
  # Heuristic: if we have more rows than columns and rows == length(x_grid), assume it's [N x S].
  # Otherwise transpose.
  if(nrow(a) == length(x_grid)) {
    a
  } else if(ncol(a) == length(x_grid)) {
    t(a)
  } else {
    # Fallback: force [N x S]
    if(nrow(a) < ncol(a)) t(a) else a
  }
}
mu_samps <- to_mat(mu_samps)      # [N x S]
sg_samps <- to_mat(sg_samps)      # [N x S]

# 4) Uncertainty decomposition (per x on the grid)
mu_mean   <- rowMeans(mu_samps)                      # E[mu]
ale_var   <- rowMeans(sg_samps^2)                    # E[sigma^2] = aleatoric
epi_var   <- apply(mu_samps, 1, var)                 # Var(mu)   = epistemic
tot_var   <- ale_var + epi_var
sigma_mean <- rowMeans(sg_samps)

res <- data.frame(
  x = x_grid,
  mu = mu_mean,
  y_clean = y_clean,
  ale_sd = sqrt(ale_var),
  epi_sd = sqrt(epi_var),
  tot_sd = sqrt(tot_var),
  sigma_mean = sigma_mean
)

# 5) Plots
# Predictive mean with total ±1σ band
p1 <- ggplot(res, aes(x = x)) +
  geom_ribbon(aes(ymin = mu - tot_sd, ymax = mu + tot_sd), fill = "steelblue", alpha = 0.2) +
  geom_line(aes(y = mu), color = "steelblue", linewidth = 0.8) +
  geom_line(aes(y = y_clean), color = "black", linetype = "dashed") +
  geom_vline(xintercept = 10, linetype = "dotted") +
  labs(title = "Predictive mean and total uncertainty",
       y = "y", x = "x") +
  theme_bw()

p2 <- ggplot(res, aes(x = x, y = ale_sd)) +
  geom_line(color = "darkgreen") +
  geom_vline(xintercept = 10, linetype = "dotted") +
  labs(title = "Aleatoric sigma", y = "Aleatoric σ", x = "x") +
  theme_bw()

p3 <- ggplot(res, aes(x = x, y = epi_sd)) +
  geom_line(color = "firebrick") +
  geom_vline(xintercept = 10, linetype = "dotted") +
  labs(title = "Epistemic sigma", y = "Epistemic σ", x = "x") +
  theme_bw()

# Optional: mean sigma(x)
p4 <- ggplot(res, aes(x = x, y = sigma_mean)) +
  geom_line(color = "purple") +
  geom_vline(xintercept = 10, linetype = "dotted") +
  labs(title = "Posterior mean of sigma(x)", y = "E[σ(x)]", x = "x") +
  theme_bw()

print(p1); print(p2); print(p3); print(p4)

# 6) Notes:
# - Increase n.iter and nsamples for more stable results.
# - You can replace s(x, bs="ps") by s(x, bs="tp") or adjust 'k' to change smoothness.
# - For heavier tails, swap 'family' to a t-distribution family if available in your bamlss version
#   (e.g., "StudentT" or similar) and add a model for 'nu' if supported.
