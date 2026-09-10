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

A mixture of $K$ components with weights $\alpha$ has density $p(y) = \sum_k \alpha_k f_k(y)$. Introducing the latent label
$z_n \in \{1, \cdots,K\}$ of observation $n$, the EM algorithm alternates between the posterior of that label (E-step) and a refit of
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

`StochasticEM` inserts an S-step: instead of the soft weights $\gamma_n$ it draws one hard label
$\hat{z}_n \sim \mathrm{Categorical}(\gamma_n)$ per observation, and refits component `k` on the observations that drew it — an
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
    One `N × K` matrix is allocated per fit. It first holds the log-likelihoods `log αₖ + log fₖ(yₙ)`,
    and the normalisation then overwrites every entry with the corresponding posterior `γₙₖ`. So once
    the E-step has run the log-likelihoods are gone, and anything added after it must not read that
    matrix expecting to find them. The low-level functions still take `LL` and `γ` as two separate
    arguments, so pass two different matrices if you need the log-likelihoods and the posteriors at
    the same time.

### Extension points

- A faster per-component likelihood: for a matrix sample, implementing `Distributions._logpdf!` for your
  component is already enough, since that is what the E-step calls. Otherwise add a
  [`ExpectationMaximization.loglikelihoods!`](@ref) method. All
  it has to do is fill `LL[n, k]` with `log αₖ + log fₖ(yₙ)`, as the E-step above does; how it computes
  that value is up to you.
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
    These helpers are implementation details: they start with an underscore, they are not exported, and
    their signatures can change in any patch release.

```@docs
ExpectationMaximization._loglikelihood_col!
ExpectationMaximization._softmax_rows!
```

## Specialized components

For a matrix sample the generic E-step hands each component to `Distributions.logpdf!`, the batched entry
point of `Distributions.jl`. Its own fallback is one `logpdf` call per observation — what this package used
to do by hand — but a component that implements `Distributions._logpdf!` scores the whole sample in a
single call, at no cost in the generic code. A nested `MixtureModel` component gets its speed from exactly
that: 2.8× on the E-step of a mixture of multivariate mixtures and 1.5× on the whole fit, with a
bit-identical loglikelihood. Product distributions have no batched method, so for them the delegation is a
measured no-op.

`MvNormal` is the family where that is not enough. `logpdf!` reaches `Distributions.sqmahal!`, which
materialises a `D × N` centred copy of the sample and then still solves one triangular system per
observation; for an isotropic covariance it is even *slower* than the per-observation path (measured 1.8×
at `D = 50`, which is why the delegation and these kernels belong together). `src/specialized.jl` adds
fast paths for `MvNormal` as extra [`ExpectationMaximization._loglikelihood_col!`](@ref) methods, selected
by dispatch on the component type, so nothing in the generic path changes and any component they do not
match keeps the delegated `logpdf!` path; deleting the file would only make the package slower.

Two kernels cover the three covariance shapes:

- **isotropic or diagonal** `Σ`: the Mahalanobis form is a plain weighted sum of squares, evaluated in one
  pass with no temporary at all.
- **full** `Σ`: the sample is processed in blocks of `MVNORMAL_BLOCKSIZE` observations, each block centred
  into a cache-resident `D × MVNORMAL_BLOCKSIZE` buffer and whitened with one in-place `PDMats.whiten!` —
  a BLAS-3 `trsm` — instead of the one BLAS-2 `trsv` per observation that `logpdf` performs.

Measured speedup over the delegated `logpdf!` path, for one component, at `N = 10⁵`, single-threaded:

| `D` | 2 | 10 | 50 | 100 |
|:--|--:|--:|--:|--:|
| `FullNormal` | 11.4× | 5.6× | 4.2× | 3.5× |
| `DiagNormal` | 3.9× | 5.2× | 7.7× | 7.4× |
| `IsoNormal` | 4.4× | 4.0× | 5.6× | 5.4× |

Over that same range the allocation of one component's E-step goes from 1.5 MB (`D = 2`) to 76.3 MB
(`D = 100`) down to 206 KB for a full covariance and under 2 KB for the other two — and, unlike the
generic path, it does not grow with `N`. The results agree with the generic path to a few units in the
last place, and the `ZeroMean*` variants are covered for free because the
`IsoNormal`/`DiagNormal`/`FullNormal` aliases only constrain the covariance and element types.

The same file also adds a `Distributions.fit_mle(::FullNormal, y, w)` method for the M-step. It computes
exactly the same weighted maximum-likelihood estimate as `Distributions.jl` — the mean is bit-identical and
the covariance agrees to about `1e-14`, the difference being the order of accumulation — but builds the
scatter matrix blockwise with `syrk!` into a reused buffer instead of allocating a fresh `D × N` array on
every call. `DiagNormal` and `IsoNormal` deliberately keep their own fits, which are already cheap and, more
importantly, preserve the covariance type of the component.

!!! note "Adding your own"
    If your component can score a whole sample at once, implementing `Distributions._logpdf!(out, d, y)`
    is enough and nothing here needs to change. Write a `_loglikelihood_col!` method (and, if the
    maximum-likelihood estimate can reuse a buffer, a `fit_mle` method) only when you also want what
    `logpdf!` cannot express: folding `log α` into the same pass, or reusing a scratch buffer across
    calls, as the kernels above do. The contract is then only the value of `LL[n, k]` given in
    [How the implementation is organised](@ref); everything else, including the `γ`-aliases-`LL`
    convention, is handled by the generic E-step.

## Index

```@index
```
