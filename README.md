# ExpectationMaximization

[![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://dmetivie.github.io/ExpectationMaximization.jl/dev)

This package provides a simple implementation of the Expectation Maximization (EM) algorithm used to fit mixture models.
Due to [Julia](https://julialang.org/) amazing [dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY) systems, generic and reusable code spirit, and the [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) package, the code while being very generic is also very powerful! That means that it work on a lot of mixture:

- Mixture of Univariate continuous distributions
- Mixture of Univariate discrete distributions
- Mixture of Multivariate distributions (continuous or discrete).
- Mixture of mixtures (univariate or multivariate and continuous or discrete).

Note that [Distributions](https://juliastats.org/Distributions.jl/stable/) *currently* does not allow `MixtureModel` to both have discrete and continuous components (but who does that? Rain).

**Have a look at the docs Examples sections**.

To work, the only requirements are that the `dist<:Distribution` considered has implanted

1. `logpdf(dist, y)` (used in the E-step)
2. `fit_mle(dist, y, weigths)` (used in the M-step)

In general 1. is easy, while 2. is only known explicitly for a few common distributions.
In case 2. is not explicit known, you can always implement a numerical scheme, if it exists, for `fit_mle(dist, y)` see [`Gamma` distribution example](https://github.com/JuliaStats/Distributions.jl/blob/34a05d8a1671052624e7fa246b58484acc32cfe5/src/univariate/continuous/gamma.jl#L171).
Or, when possible, represent your “difficult” distribution as a mixture of simple terms.
(I had [this](https://stats.stackexchange.com/questions/63647/estimating-parameters-of-students-t-distribution) in mind, but it is not directly a mixture model.)

## TODO (feel free to contribute)

- Add variant to the EM algo (so far there are the classic and stochastic version).
- Add examples e.g., MNIST dataset and Bernoulli mixtures.
<!-- - Add Docs -->
- Benchmark against other EM implementations
- Speed up code (always!). So far I focused on readable code.

## Example

```julia
using Distributions
using ExpectationMaximization
```

### Model

```julia
N = 50_000
θ₁ = 10
θ₂ = 5
α = 0.2
β = 0.3
# Mixture Model here one can put any classical distributions
mix_true = MixtureModel([Exponential(θ₁), Gamma(α, θ₂)], [β, 1 - β]) 

# Generate N samples from the mixture
y = rand(mix_true, N) 
```

### Inference

```julia
# Initial guess
mix_guess = MixtureModel([Exponential(1), Gamma(0.5, 1)], [0.5, 1 - 0.5])

# Fit the MLE with the EM algorithm
mix_mle = fit_mle(mix_guess, y; display = :iter, atol = 1e-3, robust = false, infos = false)
```

### Verify results

```julia
rtol = 5e-2
p = params(mix_mle)[1] # (θ₁, (α, θ₂))
isapprox(β, probs(mix_mle)[1]; rtol = rtol)
isapprox(θ₁, p[1]...; rtol = rtol)
isapprox(α, p[2][1]; rtol = rtol)
isapprox(θ₂, p[2][2]; rtol = rtol)
```

![EM_mixture_example.svg](img/EM_mixture_example.svg)
