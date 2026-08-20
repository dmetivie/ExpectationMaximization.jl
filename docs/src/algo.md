# [Algorithms & Methods](@id AlgoMeth)

## Expectation Maximization (EM) algorithms

Currently, only the classic EM algorithm and the Stochastic EM are implemented for `Distributions.MixtureModel`.
Look at the [Bibliography section](https://dmetivie.github.io/ExpectationMaximization.jl/dev/biblio) for references.

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
fit_mle(mix::Distributions.MixtureModel, y::AbstractVecOrMat, weights...; kwargs...)
fit_mle(mix::AbstractArray{<:Distributions.MixtureModel}, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, robust=false, infos=false)
```

## Utilities

```@docs
predict
predict_proba
```

## `fit_mle` methods that should be in `Distribution.jl`

I opened two PRs, [PR#1670](https://github.com/JuliaStats/Distributions.jl/pull/1670) and [PR#1676](https://github.com/JuliaStats/Distributions.jl/pull/1676) to add these methods.

The "instance" version of `fit_mle` allows passing a distribution instance (e.g., `Normal(0,1)`) instead of a type (e.g., `Normal`). This is required for `MixtureModel` and `ProductDistribution` support.

```@docs
fit_mle(g::D, args...) where {D<:Distribution}
```

```@docs
fit_mle(g::Product, x::AbstractMatrix, args...)
```

```@docs
fit_mle(dists::Distributions.VectorOfUnivariateDistribution, x::AbstractMatrix{<:Real}, args...)
```

```@docs
fit_mle(::Type{<:Dirac}, x::AbstractArray{T}, w::AbstractArray{Float64}) where {T<:Real}
fit_mle(::Type{<:Laplace}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
fit_mle(::Type{<:Uniform}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
```

## Low-level API

The following functions implement the inner loop of the EM algorithms. They can be extended to support custom behavior.

```@docs
fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, method::ClassicEM)
fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, method::StochasticEM)
```

```@docs
ExpectationMaximization.M_step!
```

## Index

```@index
```
