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
fit_mle(mix::AbstractArray{<:Distributions.MixtureModel}, y::AbstractVecOrMat, weights...; kwargs...)
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

## How the implementation is organised

### The math

A mixture of `K` components with weights `α` has density `p(y) = Σₖ αₖ fₖ(y)`. Introducing the latent label
`zₙ ∈ 1:K` of observation `n`, EM alternates between the posterior of that label (E-step) and a refit of
every component with those posteriors as weights (M-step):

```math
\gamma_{nk} = \mathbb{P}(z_n = k \mid y_n) = \frac{\alpha_k f_k(y_n)}{\sum_j \alpha_j f_j(y_n)},
\qquad
\alpha_k \leftarrow \frac{1}{N}\sum_n \gamma_{nk},
\qquad
\theta_k \leftarrow \arg\max_{\theta} \sum_n \gamma_{nk} \log f(y_n; \theta),
```

each iteration increasing the loglikelihood

```math
\ell = \sum_n \log p(y_n) = \sum_n \log \sum_k \alpha_k f_k(y_n) = \sum_n c_n .
```

`StochasticEM` inserts an S-step: instead of the soft weights `γₙ.` it draws one hard label
`ẑₙ ∼ Categorical(γₙ.)` per observation, and refits component `k` on the observations that drew it — an
*unweighted* `fit_mle` on a subsample.

### The same thing in code

Nothing here maximises anything by itself: the update of `θₖ` is delegated to `Distributions.jl`, where
`fit_mle(dists[k], y, γₖ)` *is* the weighted maximum-likelihood estimate of component `k`. Everything is
computed in the log domain.

```julia
N, K = size_sample(y), length(dists)
LL = zeros(N, K)
c  = zeros(N)
γ  = LL   # in practice γ aliases LL: the posteriors overwrite the log-likelihoods
ℓ  = -Inf

for it in 1:maxiter
    # E-step
    for k in 1:K
        LL[:, k] .= log(α[k]) .+ logpdf.(dists[k], y)  # log αₖ + log fₖ(yₙ); for a matrix sample
    end                                                # each column is one observation
    c .= logsumexp.(eachrow(LL))                       # cₙ = log p(yₙ)
    γ .= exp.(LL .- c)                                 # in practice these two lines are a single fused,
                                                       # allocation-free, column-major kernel that
                                                       # subtracts the row maximum before `exp`

    # Convergence: ℓ is free, cₙ is already the log density of yₙ
    ℓ_new = sum(c)                                     # `sum(w .* c)` for a weighted fit
    abs(ℓ_new - ℓ) < atol && break                     # `rtol` adds a relative test
    ℓ = ℓ_new

    # M-step
    for k in 1:K
        α[k] = sum(γ[:, k]) / N                        # length(cat[k])/N for StochasticEM
        dists[k] = fit_mle(dists[k], y, γ[:, k])       # StochasticEM instead fits each component on a
    end                                                # view of the observations that drew it
end
```

!!! warning "`γ` aliases `LL`"
    The `N × K` matrix is allocated once and holds log-likelihoods, then posteriors: `LL` is dead the
    moment a row is normalised. Nothing after the E-step may read it expecting log-likelihoods, which is
    why the low-level functions take `LL` and `γ` as separate arguments — pass a distinct `γ` if you need
    both at once.

### Extension points

- A faster per-component likelihood: add a [`ExpectationMaximization.loglikelihoods!`](@ref) method. The
  only contract is the value of `LL[n, k]` above.
- A different parameter update: add a [`ExpectationMaximization.M_step!`](@ref) method for your method type
  (both the weighted and the unweighted signature).
- A different algorithm: `struct MyEM <: AbstractEM end` plus a `fit_mle!(α, dists, y, w, ::MyEM; kwargs...)`
  returning the `Dict{String,Any}` with `"converged"`, `"iterations"` and `"logtots"`.

## Low-level API

The following functions implement the inner loop of the EM algorithms. They can be extended to support custom behavior.

### EM loop entry points

```@docs
fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, method::ClassicEM)
fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, method::StochasticEM)
```

### E-step

```@docs
ExpectationMaximization.E_step!
ExpectationMaximization.loglikelihoods!
```

### M-step

```@docs
ExpectationMaximization.M_step!
```

### Internals

!!! warning "Not public API"
    This helper is an implementation detail: it starts with an underscore, it is not exported, and its
    signature can change in any patch release. It is documented here only because its in-place contract
    matters when extending the E-step.

```@docs
ExpectationMaximization._softmax_rows!
```

## Index

```@index
```
