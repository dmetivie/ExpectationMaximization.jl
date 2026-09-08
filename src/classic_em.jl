"""
    ClassicEM<:AbstractEM
The EM algorithm was introduced by A. P. Dempster, N. M. Laird and D. B. Rubin in 1977 in the reference paper [*Maximum Likelihood from Incomplete Data Via the EM Algorithm*](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1977.tb01600.x).
"""
struct ClassicEM <: AbstractEM end

"""
    fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, method::ClassicEM; display=:none, maxiter=1000, atol=1e-3, rtol=nothing, robust=false)
Use the EM algorithm to update the Distribution `dists` and weights `α` composing a mixture distribution.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
- `atol` criteria determining the convergence of the algorithm. If the Loglikelihood difference between two iteration `i` and `i+1` is smaller than `atol` i.e. `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<atol`, the algorithm stops.
- `rtol` relative tolerance for convergence, `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<rtol*(|ℓ⁽ⁱ⁺¹⁾| + |ℓ⁽ⁱ⁾|)/2` (does not check if `rtol` is `nothing`)
- `display` value can be `:none`, `:iter`, `:final` to display Loglikelihood evolution at each iterations `:iter` or just the final one `:final`
"""
function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    method::ClassicEM;
    kwargs...,
)
    fit_mle!(α, dists, y, nothing, method; kwargs...)
end

function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    w::Union{Nothing,AbstractVector},
    method::ClassicEM;
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size_sample(y), length(dists)
    # Allocate memory for in-place updates
    LL = zeros(N, K)
    c = zeros(N)
    s = zeros(N)
    γ = LL   # γ aliases LL: the posteriors overwrite the log-likelihoods in place

    !isnothing(w) && @argcheck length(w) == N

    converged = false
    iterations = 0
    logtots = eltype(c)[]

    # E-step
    E_step!(LL, c, γ, s, dists, α, y; robust=robust)

    # Loglikelihood
    logtot = isnothing(w) ? sum(c) : _weighted_sum(w, c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter
        # M-step
        isnothing(w) ? M_step!(α, dists, y, γ, method) : M_step!(α, dists, y, γ, w, method)

        # E-step
        E_step!(LL, c, γ, s, dists, α, y; robust=robust)

        # Loglikelihood
        logtotp = isnothing(w) ? sum(c) : _weighted_sum(w, c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(logtots, logtotp)
        iterations += 1

        if abs(logtotp - logtot) < atol || (rtol !== nothing && abs(logtotp - logtot) < rtol * (abs(logtot) + abs(logtotp)) / 2)
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            converged = true
            break
        end

        logtot = logtotp
    end

    if !converged
        if display in [:iter, :final]
            println(
                "EM has not converged after $iterations iterations, final loglikelihood = $logtot",
            )
        end
    end

    return Dict{String,Any}(
        "converged" => converged, "iterations" => iterations, "logtots" => logtots
    )
end

"""
    M_step!(α, dists, y, γ, method::ClassicEM)
For the `ClassicEM` the weigths `γ` computed at E-step for each observation in `y` are used to update `α` and `dists`.
"""
function M_step!(α, dists, y::AbstractVecOrMat, γ, method::ClassicEM)
    N = size(γ, 1)
    for (k, γₖ) in enumerate(eachcol(γ))
        α[k] = sum(γₖ) / N
        dists[k] = fit_mle(dists[k], y, γₖ)
    end
end

function M_step!(α, dists, y::AbstractVecOrMat, γ, w, method::ClassicEM)
    # `γ` is scratch memory that the next E-step overwrites entirely, so the weights can be folded
    # into it once instead of materializing `w .* γₖ` for each of the K components.
    γ .*= w
    sw = sum(w)
    for (k, γₖ) in enumerate(eachcol(γ))
        α[k] = sum(γₖ) / sw   # == mean(γ, weights(w), dims=1)[k]
        dists[k] = fit_mle(dists[k], y, γₖ)
    end
end
