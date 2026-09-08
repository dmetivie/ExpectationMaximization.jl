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

`ClassicEM` and `StochasticEM` share one driver, `fit_mle!`. It allocates the buffers, runs a first E-step to get a starting loglikelihood, then loops *M-step → E-step → loglikelihood → convergence test*. Only what happens between two E-steps differs: `M_step!` for `ClassicEM`, an S-step followed by `M_step!` for `StochasticEM`.

### The driver loop

```julia
N, K = size_sample(y), length(dists)
LL = zeros(N, K)   # LL[n, k] = log α[k] + logpdf(dists[k], yₙ)
c  = zeros(N)      # cₙ = logsumexp(LL[n, :]) = log ℙ(yₙ)
s  = zeros(N)      # scratch for `_softmax_rows!`
γ  = LL            # γ aliases LL: the posteriors overwrite the log-likelihoods

E_step!(LL, c, γ, s, dists, α, y; robust=robust)          # LL := log-likelihoods, then γ := posteriors
logtot = isnothing(w) ? sum(c) : _weighted_sum(w, c)      # ℓ⁽⁰⁾

for it in 1:maxiter
    M_step!(α, dists, y, γ, ...)                          # reads γ, updates α and dists
    E_step!(LL, c, γ, s, dists, α, y; robust=robust)      # refills the same buffers
    logtotp = isnothing(w) ? sum(c) : _weighted_sum(w, c) # ℓ⁽ⁱ⁺¹⁾
    push!(logtots, logtotp)
    if abs(logtotp - logtot) < atol ||
       (rtol !== nothing && abs(logtotp - logtot) < rtol * (abs(logtot) + abs(logtotp)) / 2)
        converged = true; break
    end
    logtot = logtotp
end
```

The loglikelihood is never computed by a separate pass over the data: `cₙ = log ℙ(yₙ)` is the normalising constant the E-step produces anyway, so `ℓ = Σₙ cₙ`, or `ℓ = Σₙ wₙ cₙ` for a weighted fit. `atol` compares the absolute change of `ℓ` between two consecutive iterations; `rtol`, when it is not `nothing`, compares that same change to `rtol` times the mean magnitude `(|ℓ⁽ⁱ⁾| + |ℓ⁽ⁱ⁺¹⁾|)/2`. Either one firing stops the loop and sets `converged`. With `maxiter = 0` the loop is skipped and `history["logtots"]` is empty.

### Buffers: `γ` aliases `LL`

`LL` is the only matrix (`N × K`), `c` and `s` the only vectors (length `N`), and `γ` is not allocated at all: every caller passes `γ = LL`. This is sound because `LL` is *dead* the moment the posteriors are formed. `_softmax_rows!` first reads all of `LL` to put the row maxima in `c`; its next pass reads `LL[n, k]` and writes `γ[n, k]` at the same index; the remaining passes touch only `γ`, `c` and `s`. Downstream, `M_step!` wants `γ` and nothing else, and the next E-step refills `LL` from scratch. (`predict_proba` allocates the same buffers and simply returns `γ`.)

!!! warning "The invariant to preserve"
    Once `_softmax_rows!` has started writing, `LL` holds **posteriors, not log-likelihoods**. A new pass inside `_softmax_rows!` may not re-read `LL`, element updates must stay same-index, and no code between an M-step and the next E-step may read `LL` expecting log-likelihoods. Anything that genuinely needs both at once must pass a distinct `γ` matrix — which is why the low-level functions take `LL` and `γ` as separate arguments. Note also that the weighted `M_step!` of `ClassicEM` mutates `γ` in place, so `γ` is only meaningful until it runs.

### The E-step in two stages

1. `loglikelihoods!(LL, dists, α, y)` fills `LL[n, k] = log(α[k]) + logpdf(dists[k], yₙ)`. For a vector sample this is one fused broadcast per component; for a matrix sample each component goes through a function barrier that takes the component as an argument, so its type stays concrete in the loop — one dynamic dispatch per component instead of one per observation. With `robust = true`, `-Inf` is then mapped to `nextfloat(-Inf)` and `Inf` to `log(prevfloat(Inf))`.
2. `_softmax_rows!(c, γ, LL, s)` normalises each row in one set of column-major passes: `c` gets the row maximum, then `γ[n, k] = exp(LL[n, k] - cₙ)` with `sₙ` accumulating the row sum, then `cₙ += log(sₙ)` and the row is scaled by `1/sₙ`. The result is `γ[n, k] = ℙ(zₙ = k ∣ yₙ)` and `cₙ = log ℙ(yₙ)`. Subtracting the maximum first is what keeps this in range: `exp` only ever sees non-positive arguments, the largest term is exactly `1` and `sₙ ≥ 1`, so a row of very negative log-likelihoods cannot underflow to an all-zero row and a division by `0`. Rows whose maximum is not finite are deliberately left unnormalised (`cₙ` stays the maximum); `robust = true` is how you avoid them.

### The M-step

`ClassicEM` walks the columns of `γ` in place: `α[k] = sum(γₖ)/N` and `dists[k] = fit_mle(dists[k], y, γₖ)`. The weighted variant folds the observation weights into the buffer once (`γ .*= w`) and divides by `sum(w)`, instead of materialising `w .* γₖ` for each of the `K` components — legitimate precisely because `γ` is scratch that the next E-step overwrites.

`StochasticEM` inserts an S-step: for each observation it draws a hard label `ẑₙ` from row `n` of `γ` by an inlined inverse-CDF scan, one `rand(method.rng)` and a forward walk of the cumulative probabilities. It consumes **exactly one random number per observation** (and compares with `<=`, matching `rand(rng, ::DiscreteNonParametric)`), which is what keeps seeded runs reproducible: drawing more or fewer numbers per observation shifts every later draw. `cat[k]` then collects the observations labelled `k`, and `M_step!` sets `α[k] = length(cat[k])/N` and refits each component on a *view* of its subsample, so only the unweighted `fit_mle(dist, y)` is required. The loop also throws a `DomainError` on a non-finite `ℓ`, because the corresponding row of `γ` is `NaN` and the scan would otherwise silently assign it to component `1`.

### Extension points

- A faster per-component likelihood: add a [`ExpectationMaximization.loglikelihoods!`](@ref) method. The only contract is the value of `LL[n, k]` given above.
- A different parameter update: add a [`ExpectationMaximization.M_step!`](@ref) method for your method type (both the weighted and the unweighted signature).
- A different algorithm: `struct MyEM <: AbstractEM end` plus a `fit_mle!(α, dists, y, w, ::MyEM; kwargs...)` returning the `Dict{String,Any}` with `"converged"`, `"iterations"` and `"logtots"`.

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
    their signatures can change in any patch release. They are documented here only because their
    in-place contracts matter when extending the E-step.

```@docs
ExpectationMaximization._loglikelihood_col!
ExpectationMaximization._softmax_rows!
```

## Specialized components

The generic E-step calls `logpdf` once per observation, which is the right thing to do for an arbitrary
component but leaves a lot on the table for the distributions whose density can be evaluated for a whole
sample at once. `src/specialized.jl` adds such fast paths for `MvNormal` as extra
[`ExpectationMaximization._loglikelihood_col!`](@ref) methods. They are selected by dispatch on the
component type, so nothing in the generic path changes and any component they do not match keeps the
per-observation fallback; deleting the file would only make the package slower.

Two kernels cover the three covariance shapes:

- **isotropic or diagonal** `Σ`: the Mahalanobis form is a plain weighted sum of squares, evaluated in one
  pass with no temporary at all.
- **full** `Σ`: the sample is processed in blocks of `MVNORMAL_BLOCKSIZE` observations, each block centred
  into a cache-resident `D × MVNORMAL_BLOCKSIZE` buffer and whitened with one in-place `PDMats.whiten!` —
  a BLAS-3 `trsm` — instead of the one BLAS-2 `trsv` per observation that `logpdf` performs.

Measured speedup over the generic fallback at `N = 10⁵`, `K = 2`, single-threaded:

| `D` | 2 | 10 | 50 | 100 |
|:--|--:|--:|--:|--:|
| `FullNormal` | 16.1× | 7.8× | 5.5× | 4.7× |
| `DiagNormal` | 17.6× | 12.8× | 15.3× | 17.0× |
| `IsoNormal` | 17.0× | 6.3× | 3.2× | 6.6× |

Allocation at `D = 100`, `N = 10⁵` drops from 185.6 MB to 211 KB for a full covariance, and from 92.8 MB to
under 2 KB for the other two. The results agree with the generic fallback to one or two units in the last
place, and the `ZeroMean*` variants are covered for free because the `IsoNormal`/`DiagNormal`/`FullNormal`
aliases only constrain the covariance and element types.

The same file also adds a `Distributions.fit_mle(::FullNormal, y, w)` method for the M-step. It computes
exactly the same weighted maximum-likelihood estimate as `Distributions.jl` — the mean is bit-identical and
the covariance agrees to about `1e-14`, the difference being the order of accumulation — but builds the
scatter matrix blockwise with `syrk!` into a reused buffer instead of allocating a fresh `D × N` array on
every call. `DiagNormal` and `IsoNormal` deliberately keep their own fits, which are already cheap and, more
importantly, preserve the covariance type of the component.

!!! note "Adding your own"
    This is the intended way to make a particular component fast: add a `_loglikelihood_col!` method (and, if
    the maximum-likelihood estimate can reuse a buffer, a `fit_mle` method) for your type. The contract is
    only the value of `LL[n, k]` given in [How the implementation is organised](@ref);
    everything else, including the `γ`-aliases-`LL` convention, is handled by the generic E-step.

## Index

```@index
```
