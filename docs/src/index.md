```@raw html
---
layout: home

hero:
    name: ExpectationMaximization.jl
    text: Expectation Maximization algorithm for Julia mixture models
    tagline: One algorithm -readable and fast- working for all mixture models.
    actions:
      - theme: brand
        text: Explore the examples
        link: /examples/examples_multivariate/
      - theme: brand
        text: Explore the algorithms
        link: /algo/
      - theme: alt
        text: View on GitHub
        link: https://github.com/dmetivie/ExpectationMaximization.jl
    image:
      src: /logo.svg
      alt: ExpectationMaximization.jl

features:
  - icon: 🔄
    title: Classic and Stochastic EM
    details: Choose between the classical or stochastic version of the EM, with convergence controls and optional robust likelihood handling. 
  - icon: 🧩
    title: Broad mixture support
    details: Fit univariate, multivariate, discrete, continuous, nested, and user-defined distributions.
  - icon: ⚡
    title: Generic Julia design
    details: Build on `Distributions.jl` and multiple dispatch for concise and readable code that remains flexible and fast.
---
```

This package provides a simple implementation of the Expectation Maximization (EM) algorithm used to fit mixture models.
Due to [Julia](https://julialang.org/)'s amazing [dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY) system, generic and reusable code spirit, and the [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) package, the code while being very generic is both very expressive and fast! Take a look at the [Benchmark section](@ref Benchmarks).

## What type of mixtures?

In particular, it works on a lot of mixtures:

- Mixture of Univariate continuous distributions
- Mixture of Univariate discrete distributions
- Mixture of Multivariate distributions (continuous or discrete)
- Mixture of mixtures (univariate or multivariate and continuous or discrete)
- User defined mixtures (e.g. custom distributions)
- More?

## How?

Just define a [`mix::MixtureModel`](https://juliastats.org/Distributions.jl/stable/mixture/) and do `fit_mle(mix, y)` where `y` is your observation array (vector or matrix). That's it! For Stochastic EM, just do `fit_mle(mix, y, method = StochasticEM())`.
**Take a look at the [Examples](https://dmetivie.github.io/ExpectationMaximization.jl/dev/examples/#Examples) section**.

For a description of the implemented methods, see the [Algorithms and Methods](@ref AlgoMeth) page.

To work, the only requirements are that the components of the mixture `dist ∈ dists = components(mix)` considered (custom or coming from an existing package)

1. Are a subtype of `Distribution` i.e. `dist<:Distribution`.
2. The `logpdf(dist, y)` is defined (it is used in the E-step)
3. The `fit_mle(dist, y, weights)` returns the distribution with the updated parameters maximizing the likelihood. This is used in the M-step of the `ClassicalEM` algorithm. For the `StochasticEM` version, only `fit_mle(dist, y)` is needed. Type or instance version of `fit_mle` for your `dist` are accepted thanks to this [conversion line](https://github.com/dmetivie/ExpectationMaximization.jl/blob/60e833236a122cb5ef58150b1a445e2941ace5d1/src/that_should_be_in_Distributions.jl#L16).

In general, step 2. is easy, while step 3. is only known explicitly for a few common distributions.
In step 3., if the `fit_mle` is not explicitly known, you can always implement a numerical scheme, if it exists, for `fit_mle(dist, y)` see [`Gamma` distribution example](https://github.com/JuliaStats/Distributions.jl/blob/34a05d8a1671052624e7fa246b58484acc32cfe5/src/univariate/continuous/gamma.jl#L171) or use tools like [Optimizations.jl](https://docs.sciml.ai/Optimization/stable/).
Or, when possible, represent your “difficult” distribution as a mixture of simple terms.
(I had [this](https://stats.stackexchange.com/questions/63647/estimating-parameters-of-students-t-distribution) in mind, but it is not directly a mixture model.)

!!! note
    [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) *currently* does not allow `MixtureModel` to both have discrete and continuous components[^2].

[^2]: Rain is a good example of a mixture having both a discrete (`Delta` distribution in `0`) and continuous (`Exponential`, `Gamma`, ...) component.

### Getting started

```julia
using Distributions
using ExpectationMaximization

mix = MixtureModel([Exponential(10.0), Gamma(2.0, 5.0)], [0.3, 0.7])
y = rand(mix, 1_000)

mix_guess = MixtureModel([Exponential(1.0), Gamma(1.0, 1.0)], [0.5, 0.5])
mix_fit = fit_mle(mix_guess, y)
```

That's it! The `mix_fit` is now the fitted mixture model.

See the [Examples](https://dmetivie.github.io/ExpectationMaximization.jl/dev/examples/#Examples) section for more examples.
