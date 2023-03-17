# Examples

```@example 1
using Distributions
using ExpectationMaximization
using StatsPlots
```

## Univariate continuous  

### Exponential + Laplace + Uniform

```@example 1
# Parameters
N = 5_000
θ₁ = 5
θ₂ = 0.8
α = 0.5
β = 0.3
μ = -1
a = 2

mix_true = MixtureModel([Laplace(μ, θ₁), Normal(α, θ₂), Exponential(a)], [β/2, 1 - β, β/2])

# Components of the mixtures
plot(mix_true, label = ["Laplace" "Normal" "Exponential"])
ylabel!("Log PDF", yaxis = :log10)
```

```@example 1
# Sampling
y = rand(mix_true, N)

# Initial Condition
mix_guess = MixtureModel([Laplace(-1, 1), Normal(2, 1), Exponential(3)], [1/3, 1/3, 1/3])

# Fit Classic EM
mix_mle_C, hist_C = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true, method = ClassicEM())

# Fit Stochastic EM
mix_mle_S, hist_S = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true, method = StochasticEM())

x = -20:0.1:20
pmix = plot(x, pdf.(mix_true, x), label = "True", ylabel = "PDF")
plot!(pmix, x, pdf.(mix_guess, x), label = "Initial guess")
plot!(pmix, x, pdf.(mix_mle_C, x), label = "fit EM")
plot!(pmix, x, pdf.(mix_mle_S, x), label = "fit sEM")

ploss = plot(hist_C["logtots"], label = "ClassicEM with $(hist_C["iterations"]) iterations", c = 3, xlabel = "EM iterations", ylabel = "Log Likelihood")
plot!(ploss, hist_S["logtots"], label = "StochasticEM  with $(hist_S["iterations"]) iterations", c = 4, s = :dot)

plot(pmix, ploss)
```

### Mixture of Mixture and univariate

```@example 1
θ₁ = -5
θ₂ = 2
σ₁ = 1
σ₂ = 1.5
θ₀ = 0.1
σ₀ = 0.1

α = 1 / 2
β = 0.3

d1 = MixtureModel([Laplace(θ₁, σ₁), Normal(θ₂, σ₂)], [α, 1 - α])
d2 = Normal(θ₀, σ₀)
mix_true = MixtureModel([d1, d2], [β, 1 - β])
y = rand(mix_true, N)

# We choose initial guess very close to the true solution just to show the EM algorithm convergence.
# This particular choice of mixture of mixture Gaussian with another Gaussian is non identifiable hence we execpt other solution far away from the true solution
d1_guess = MixtureModel([Laplace(θ₁ - 2, σ₁ + 1), Normal(θ₂ + 1, σ₂ + 1)], [α + 0.1, 1 - α - 0.1])
d2_guess = Normal(θ₀ - 1, σ₀ - 0.05)

mix_guess = MixtureModel([d1_guess, d2_guess], [β + 0.1, 1 - β - 0.1])
mix_mle, hist_C = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true)
y_guess = rand(mix_mle, N)

x = -20:0.1:20
pmix = plot(x, pdf.(mix_true, x), label = "True", ylabel = "PDF")
plot!(pmix, x, pdf.(mix_guess, x), label = "Initial guess")
plot!(pmix, x, pdf.(mix_mle, x), label = "fit EM with $(hist_C["iterations"]) iterations")
```

## Multivariate mixtures

### Multivariate Gaussian Mixtures

```julia
θ₁ = [-1, 1]
θ₂ = [0, 2]
Σ₁ = [0.5 0.5
    0.5 1]
Σ₂ = [1 0.1
    0.1 1]
β = 0.3
D₁ = MvNormal(θ₁, Σ₁)
D₂ = MvNormal(θ₂, Σ₂)

mix_true = MixtureModel([D₁, D₂], [β, 1 - β])

# Generate samples from the true distribution
y = rand(mix_true, N)

# Initial Condition
D₁guess = MvNormal([0.2, 1], [1 0.6; 0.6 1])
D₂guess = MvNormal([1, 0.5], [1 0.2; 0.2 1])
mix_guess = MixtureModel([D₁guess, D₂guess], [0.4, 0.6])

# Fit MLE
mix_mle = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)
```

### Bernoulli Mixture

```julia
S = 10
K = 3
θ = zeros(S, K)
θ[:, 1] = (1:S) / S .- 0.05 # Bernoulli parameters
θ[:, 2] = (S:-1:1) / 2S # Bernoulli parameters
θ[:, 3] = ones(S) + 0.1 * [isodd(i) ? -1 : 1 for i in 1:S] .- 0.4# Bernoulli parameters
β = 0.3

mix_true = MixtureModel([product_distribution(Bernoulli.(θ[:, i])) for i in 1:K], [β / 2, 1 - β, β / 2])

# Generate samples from the true distribution
y = rand(mix_true, N)

# Initial Condition
mix_guess = MixtureModel([product_distribution(Bernoulli.(2θ[:, i] / 3)) for i in 1:K], [0.25, 0.55, 0.2])

# Fit MLE
mix_mle = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)
```
