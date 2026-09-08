"""
    fit_mle(mix::MixtureModel, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, rtol=nothing, robust=false, infos=false)
Use an Expectation Maximization (EM) algorithm to maximize the Loglikelihood (fit) the mixture with an i.i.d sample `y`.
The `mix` input is a mixture that is used to initialize the EM algorithm.
When `y` is an `AbstractMatrix`, each **column** is one observation, i.e. there are `size(y, 2)` observations of dimension `size(y, 1)`.
- `weights` at most one positional weight vector `w`, of length `size_sample(y)`, may be given; it then computes a weighted version of the EM. (Useful for fitting mixture of mixtures)
- `method` determines the algorithm used.
- `infos = true` returns the tuple `(mix_fitted, history)` instead of just `mix_fitted`, where `history::Dict{String,Any}` holds `"converged"::Bool`, `"iterations"::Int` (number of EM iterations actually performed) and `"logtots"::Vector` (the loglikelihood after each of those iterations). The iteration-0 loglikelihood is **not** stored, so `length(history["logtots"]) == history["iterations"]`, and it is empty when `maxiter = 0`.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
- `atol` criteria determining the convergence of the algorithm. If the Loglikelihood difference between two iteration `i` and `i+1` is smaller than `atol` i.e. `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<atol`, the algorithm stops.
- `rtol` relative tolerance for convergence, `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<rtol*(|ℓ⁽ⁱ⁺¹⁾| + |ℓ⁽ⁱ⁾|)/2` (does not check if `rtol` is `nothing`)
- `display` value can be `:none`, `:iter`, `:final` to display Loglikelihood evolution at each iterations `:iter` or just the final one `:final`
"""
function fit_mle(
    mix::MixtureModel,
    y::AbstractVecOrMat,
    weights...;
    method=ClassicEM(),
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
    infos=false,
)

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

    # Splatting an empty `weights` resolves to the unweighted `fit_mle!` overload and a
    # one-element one to the weighted overload, so a single call site covers both cases.
    # (The `history` `Dict` is one allocation per fit, so it is not worth skipping when `infos = false`.)
    history = fit_mle!(
        α,
        dists,
        y,
        weights...,
        method;
        display=display,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
        robust=robust,
    )

    return infos ? (MixtureModel(dists, α), history) : MixtureModel(dists, α)
end

# `logtots` is empty when `maxiter = 0`; guard the index rather than throw a `BoundsError`
# that the surrounding `catch` would silently swallow.
_last_loglikelihood(history) = isempty(history["logtots"]) ? -Inf : history["logtots"][end]

"""
    fit_mle(mix::AbstractArray{<:MixtureModel}, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, rtol=nothing, robust=false, infos=false)

Do the same as `fit_mle` for each (initial) mixture in the `mix` array, then keep the fit with the largest
final loglikelihood `history["logtots"][end]` (taken as `-Inf` when `logtots` is empty, e.g. `maxiter = 0`);
ties keep the earliest initial condition.

Every initial condition runs inside a `try`/`catch`, so one singular solution does not abort the whole sweep
(using `robust = true` should be enough to avoid most errors in the first place). A failing initial condition
is reported with `@debug` and skipped.
- An `InterruptException` is never swallowed: it is rethrown immediately, so `Ctrl-C` still stops the sweep.
- If *every* initial condition fails, the error raised by the **first** failing one is rethrown.
- All keywords are forwarded unchanged to `fit_mle(mix[j], y, weights...)`; see its docstring for their meaning.
- `infos = true` returns `(mix_best, history_best)` for the selected fit instead of just `mix_best`.
"""
function fit_mle(
    mix::AbstractArray{<:MixtureModel},
    y::AbstractVecOrMat,
    weights...;
    method=ClassicEM(),
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
    infos=false,
)
    # Single source of truth for the keywords: forwarding them by hand is what used to drop
    # `rtol` on the first initial condition, leaving that fit with no convergence criterion.
    kwargs = (;
        method=method,
        display=display,
        maxiter=maxiter,
        atol=atol,
        rtol=rtol,
        robust=robust,
        infos=true,
    )

    mx_max, history_max, failure = nothing, nothing, nothing
    for j in eachindex(mix)
        try
            mx_new, history_new = fit_mle(mix[j], y, weights...; kwargs...)
            if isnothing(history_max) ||
               _last_loglikelihood(history_max) < _last_loglikelihood(history_new)
                mx_max, history_max = mx_new, history_new
            end
        catch e
            e isa InterruptException && rethrow()
            isnothing(failure) && (failure = e)
            @debug "fit_mle: initial condition $j failed" exception = e
        end
    end
    isnothing(mx_max) && throw(failure)

    return infos ? (mx_max, history_max) : mx_max
end

# E-step methods

"""
    loglikelihoods!(LL::AbstractMatrix, dists, α, y::AbstractVector)
    loglikelihoods!(LL::AbstractMatrix, dists, α, y::AbstractMatrix)
Fill `LL[n, k] = log(α[k]) + logpdf(dists[k], y[n])` and return `LL`. For the `AbstractMatrix` method each
**column** of `y` is one observation, so the entry is `log(α[k]) + logpdf(dists[k], y[:, n])`.

This is the extension hook of the E-step: add a method for your component or sample type if it can score a
whole sample at once, the only contract being the value of `LL[n, k]` above.
"""
function loglikelihoods!(LL::AbstractMatrix, dists, α, y::AbstractVector)
    # `logpdf.(dists[k], y)` is already a fused allocation-free broadcast, and the `Broadcasted`
    # object acts as a function barrier, so an abstract `eltype(dists)` costs one dynamic
    # dispatch per component rather than one per observation.
    for k in eachindex(dists)
        LL[:, k] .= log(α[k]) .+ logpdf.(dists[k], y)
    end
    return LL
end

function loglikelihoods!(LL::AbstractMatrix, dists, α, y::AbstractMatrix)
    for k in eachindex(dists)
        _loglikelihood_col!(view(LL, :, k), dists[k], log(α[k]), y)
    end
    return LL
end

# `d` is an argument instead of `dists[k]` indexed inside the loop: that keeps its type concrete
# in this body, so an abstract `eltype(dists)` costs one dynamic dispatch per component rather
# than one per observation.
function _loglikelihood_col!(LLₖ, d, logα, y::AbstractMatrix)
    @inbounds @views for n in axes(y, 2)
        LLₖ[n] = logα + logpdf(d, y[:, n])
    end
    return LLₖ
end

# `sum(w[n] * c[n] for n in eachindex(c))` is a bounds-checked scalar loop that does not vectorize.
# This does, without pulling in `LinearAlgebra` just for `dot`.
function _weighted_sum(w::AbstractVector, c::AbstractVector)
    t = zero(promote_type(eltype(w), eltype(c)))
    @inbounds @simd for n in eachindex(w, c)
        t += w[n] * c[n]
    end
    return t
end

# Same semantics as `replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))`, `NaN`
# included, but ~3x faster: `replace!` builds and tests a tuple of `Pair`s for every element.
function _clamp_inf!(LL)
    lo = nextfloat(-Inf)
    hi = log(prevfloat(Inf))
    @inbounds for i in eachindex(LL)
        v = LL[i]
        LL[i] = v == -Inf ? lo : (v == Inf ? hi : v)
    end
    return LL
end

"""
    _softmax_rows!(c, γ, LL, s)
Fused allocation-free row-wise log-sum-exp and softmax: writes `c[n] = logsumexp(LL[n, :])` and
`γ[n, :] = softmax(LL[n, :])` in a single set of column-major passes, where `logsumexp!` needed a
`16N`-byte temporary plus a second pass over `LL`.
- `γ` may alias `LL`: every element is a same-index read-then-write and no later pass re-reads `LL`.
- `s` is a length-`N` scratch vector.
- Rows whose maximum is not finite are left unnormalized so that `c` and `γ` reproduce the previous
  `logsumexp!` based implementation bit for bit (this is what `robust = true` is for).
"""
function _softmax_rows!(
    c::AbstractVector{T}, γ::AbstractMatrix, LL::AbstractMatrix, s::AbstractVector
) where {T}
    N, K = size(LL)
    copyto!(c, view(LL, :, 1))                                  # c := row-wise maximum
    @inbounds for k = 2:K
        @simd for n = 1:N
            c[n] = max(c[n], LL[n, k])
        end
    end
    fill!(s, zero(T))
    @inbounds for k = 1:K                                       # γ := exp(LL - max), s := Σₖ γ
        @simd for n = 1:N
            e = exp(LL[n, k] - c[n])
            γ[n, k] = e
            s[n] += e
        end
    end
    @inbounds @simd for n = 1:N
        m = c[n]
        finite = isfinite(m)                                    # degenerate row: leave it alone
        c[n] = ifelse(finite, m + log(s[n]), m)
        s[n] = ifelse(finite, inv(s[n]), one(T))
    end
    @inbounds for k = 1:K
        @simd for n = 1:N
            γ[n, k] *= s[n]
        end
    end
    return γ
end

"""
    E_step!(LL, c, γ, s, dists, α, y; robust=false)
E-step, in two stages: [`loglikelihoods!`](@ref) fills `LL[n, k] = log(α[k]) + logpdf(dists[k], y[n])`
(`y[:, n]` for multivariate samples), then [`_softmax_rows!`](@ref) turns every row into a posterior.
Returns `γ`.
- `LL` the `N × K` log-likelihood matrix. Entirely overwritten, first with the log-likelihoods and then
  with the posteriors.
- `c` a length-`N` vector filled with `c[n] = logsumexp(LL[n, :]) = log ℙ(y[n])`. The `fit_mle!` drivers
  sum (or weight-sum) it to get the loglikelihood, so no extra pass over `y` is needed.
- `γ` the `N × K` posterior matrix, `γ[n, k] = ℙ(zₙ = k ∣ yₙ)`. It **may alias** `LL`, and every caller in
  this package passes `γ === LL`, since `LL` is dead once the posteriors are formed. The contract is that
  nothing afterwards reads `LL` expecting log-likelihoods; pass a distinct `γ` if you need both.
- `s` a length-`N` scratch vector. Its contents are meaningless on entry and on exit.
- `dists`, `α` the current components and mixing weights; `y` the sample (a vector, or a `D × N` matrix).
- `robust = true` clamps `±Inf` log-likelihoods before normalizing, which is what prevents a degenerate,
  unnormalized row of `γ`.
"""
function E_step!(
    LL::AbstractMatrix,
    c::AbstractVector,
    γ::AbstractMatrix,
    s::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    α::AbstractVector,
    y::AbstractVecOrMat;
    robust=false,
)
    # evaluate likelihood for each type k
    loglikelihoods!(LL, dists, α, y)
    robust && _clamp_inf!(LL)
    # get posterior of each category (γ is allowed to alias LL)
    _softmax_rows!(c, γ, LL, s)
end
