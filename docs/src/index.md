# ExpectationMaximization.jl

This package provides a simple implementation of the Expectation Maximization (EM) algorithm used to fit mixture models.
Due to [Julia](https://julialang.org/)'s amazing [dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY) system, generic and reusable code spirit, and the [Distributions.jl](https://juliastats.org/Distributions.jl/stable/) package, the code while being very generic is both very expressive and fast! Take a look at the [Benchmark section](https://dmetivie.github.io/ExpectationMaximization.jl/dev/benchmarks/).

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

[^1]: Have a look at the [Benchmark section](@ref Benchmarks).
[^2]: Rain is a good example of a mixture having both a discrete (`Delta` distribution in `0`) and continuous (`Exponential`, `Gamma`, ...) component.

## Algorithms

So far, the classic EM algorithm and the Stochastic EM are implemented. Look at the [Bibliography section](https://dmetivie.github.io/ExpectationMaximization.jl/dev/biblio) for references.

```@docs
ClassicEM
```

```@docs
StochasticEM
```

## Main function

!!! warning
    To fit the mixture, use the “instance” version of `fit_mle(mix::MixtureModel, ...)` as described below and **NOT** the “Type” version, i.e., `fit_mle(Type{MixtureModel}, ...)`.
    The provided `mix` is used as the starting point of the EM algorithm.
    See [Instance vs Type version](@ref InstanceVType) section for more context.

```@docs
fit_mle(mix::MixtureModel, y::AbstractVecOrMat, weights...; kwargs...)
fit_mle(mix::AbstractArray{<:MixtureModel}, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, robust=false, infos=false)
```

## Utilities

```@docs
predict
predict_proba
```

## `fit_mle` methods that should be in `Distribution.jl`

I opened two PRs, [PR#1670](https://github.com/JuliaStats/Distributions.jl/pull/1670) and [PR#1676](https://github.com/JuliaStats/Distributions.jl/pull/1676) to add these methods.

```@docs
fit_mle(g::Product, x::AbstractMatrix, args...)
```

```@docs
fit_mle(::Type{<:Dirac}, x::AbstractArray{T}, w::AbstractArray{Float64}) where {T<:Real}
fit_mle(::Type{<:Laplace}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
```

## Index

```@index
```
