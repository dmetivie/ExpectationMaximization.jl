# ExpectationMaximization

This package provides a simple implementation of the Expectation Maximization (EM) algorithm used to fit mixture models.
Due to [Julia](https://julialang.org/) amazing [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY) systems and the [Distributions](https://juliastats.org/Distributions.jl/stable/) package, the code is very generic i.e., a lot of mixture should work:

- Univariate continuous distributions
- Univariate discrete distributions
- Multivariate distributions (continuous or discrete).
- Mixture of mixture (univariate or multivariate and continuous or discrete). Note that [Distributions](https://juliastats.org/Distributions.jl/stable/) currently does not allow `MixtureModel` to have discrete and continuous components (but who does that? Rain).

**Have a look at the tests sections for examples**.

To work, the only requirements are that the `dist<:Distribution` considered has implanted

1. `logpdf(dist, y)` (used in the E-step)
2. `fit_mle(dist, y, weigths)` (used in the M-step)

In general 1. is easy, while 2. is only known explicitly for a few common distributions.
In case 2. is not explicit known, you can always implement a numerical scheme, if it exists, for `fit_mle(dist, y)` see [`Gamma` distribution example](https://github.com/JuliaStats/Distributions.jl/blob/34a05d8a1671052624e7fa246b58484acc32cfe5/src/univariate/continuous/gamma.jl#L171).
Or, when possible, represent your “difficult” distribution as a mixture of simple terms.
(I had [this](https://stats.stackexchange.com/questions/63647/estimating-parameters-of-students-t-distribution) in mind, but it is not directly a mixture model.)

## TODO (feel free to contribute)

- Add different variants for E-step and M-steps like stochastic EM and others.
- Add examples e.g., MNIST dataset and Bernoulli mixtures.
- Add Docs
- Benchmark against popular EM implementations + Speed up code

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
mix_mle = fit_mle(mix_guess, y; display = :iter, tol = 1e-3, robust = false, infos = false)
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

## Bibliography

The EM algorithm was introduced by [A. P. Dempster, N. M. Laird, D. B. Rubin](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1977.tb01600.x). This is a very generic algorithm working for almost any distributions.
However, to my knowledge all coding implementations are specific to some mixtures class (mostly Gaussian mixtures, sometime double exponential or Bernoulli).

In this package thanks to Julia generic code spirit one can just code the algorithm without having to specify the distribution.

I was inspired by [Florian Oswald page](https://floswald.github.io/post/em-benchmarks/) and [Maxime Mouchet `HMMBase.jl` package](https://github.com/maxmouchet/HMMBase.jl).

## `Distributions.jl` compatibility

This package relies heavily on the types and methods (`fit_mle` and `logpdf`) defined `Distributions.jl`. However, it differs slightly by one point: it uses an instance version of `fit_mle(dist::Distribution, x, args...)`.
For the package to work and be generic, the input `dist` must be an "instance" i.e. an actual distribution like `Normal(0,1)` not just the type `Normal`.
For `MixtureModels` (and `ProductDistributions`) this is critical to extract the various distributions inside while the `Type` version does not (currently) contain such information.
Plus, for `MixtureModels` the `dist` serves as the initial point of EM algorithm.
I opened a [PR#1670](https://github.com/JuliaStats/Distributions.jl/pull/1670) regarding that, but it might not be accepted.
<!-- This is fully comatible with `Distributions.jl`, it just add  -->

I believe some packages dealing with complex distributions like [Copulas.jl](https://github.com/lrnv/Copulas.jl/blob/08aca27dc0a28e4932e5e2c0dd04482bb3b04f48/test/some_tests.jl#L8) or [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) could benefit from this formulation.
