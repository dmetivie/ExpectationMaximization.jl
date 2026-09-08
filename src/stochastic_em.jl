"""
    Base.@kwdef struct StochasticEM<:AbstractEM
        rng::AbstractRNG = Random.GLOBAL_RNG
    end
The Stochastic EM algorithm was introduced by G. Celeux, and J. Diebolt. in 1985 in [*The SEM Algorithm: A probabilistic teacher algorithm derived from the EM algorithm for the mixture problem*](https://cir.nii.ac.jp/crid/1574231874553755008).

The default random number generator is `Random.GLOBAL_RNG`. Pass another one with `StochasticEM(rng)` or
`StochasticEM(; rng = rng)`, e.g. `StochasticEM(MersenneTwister(0))`, to make a run reproducible. The
argument must be an `AbstractRNG`: an integer seed such as `StochasticEM(0)` is a `MethodError`.
"""
Base.@kwdef struct StochasticEM <: AbstractEM
    rng::AbstractRNG = Random.GLOBAL_RNG
end

"""
    fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, method::StochasticEM; display=:none, maxiter=1000, atol=1e-3, rtol=nothing, robust=false)
    fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, w::Union{Nothing,AbstractVector}, method::StochasticEM; display=:none, maxiter=1000, atol=1e-3, rtol=nothing, robust=false)
Use the stochastic EM algorithm to update **in place** the Distribution `dists` and weights `α` composing a mixture distribution.
When `y` is an `AbstractMatrix`, each column is one observation. `w`, when given and not `nothing`, is a weight vector of length `size_sample(y)`.
Returns `history::Dict{String,Any}` with keys `"converged"::Bool`, `"iterations"::Int` and `"logtots"::Vector` (loglikelihood after each performed iteration, empty when `maxiter = 0`).
- Throws a `DomainError` if the loglikelihood is not finite: some observation then has zero density under every component, its posterior row is `NaN`, and the S-step would silently assign it to component `1`. Try `robust = true` or another initial condition.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
- `atol` criteria determining the convergence of the algorithm. If the Loglikelihood difference between two iteration `i` and `i+1` is smaller than `atol` i.e. `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<atol`, the algorithm stops.
- `rtol` relative tolerance for convergence, `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<rtol*(|ℓ⁽ⁱ⁺¹⁾| + |ℓ⁽ⁱ⁾|)/2` (does not check if `rtol` is `nothing`)
- `display` value can be `:none`, `:iter`, `:final` to display Loglikelihood evolution at each iterations `:iter` or just the final one `:final`
"""
function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    method::StochasticEM;
    kwargs...,
)
    fit_mle!(α, dists, y, nothing, method; kwargs...)
end

function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    w::Union{Nothing,AbstractVector},
    method::StochasticEM;
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
    ẑ = zeros(Int, N)

    !isnothing(w) && @argcheck length(w) == N

    converged = false
    iterations = 0
    logtots = eltype(c)[]

    # E-step
    E_step!(LL, c, γ, s, dists, α, y; robust=robust)

    # Loglikelihood
    logtot = isnothing(w) ? sum(c) : _weighted_sum(w, c)
    (display == :iter) && println("Method = $(method)\nIteration 0: loglikelihood = ", logtot)

    for it = 1:maxiter
        # A non-finite loglikelihood means some observation has zero density under every component,
        # so its posterior row is `NaN` and the S-step would silently assign it to component 1.
        isfinite(logtot) || throw(
            DomainError(
                logtot,
                "StochasticEM: non-finite loglikelihood, some observation has zero density under every component. Try `robust = true` or another initial condition.",
            ),
        )

        # S-step: inlined inverse-CDF draw. `<=` (not `<`) reproduces
        # `rand(rng, ::DiscreteNonParametric)` bit for bit, i.e. one `rand(rng, Float64)` and a
        # forward scan of the cumulative probabilities, so a seeded run is unchanged.
        @inbounds for n in axes(γ, 1)
            u = rand(method.rng)
            k, acc = 1, γ[n, 1]
            while acc <= u && k < K
                k += 1
                acc += γ[n, k]
            end
            ẑ[n] = k
        end
        cat = [findall(ẑ .== k) for k = 1:K]

        # M-step
        isnothing(w) ? M_step!(α, dists, y, cat, method) : M_step!(α, dists, y, cat, w, method)

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
    M_step!(α, dists, y, cat, method::StochasticEM)
    M_step!(α, dists, y, cat, w, method::StochasticEM)
For the `StochasticEM` the `cat` drawn at S-step for each observation in `y` is used to update `α` and
`dists` in place. `cat[k]` indexes the observations assigned to component `k`, so the subsample is passed as
a **view** rather than copied — component `fit_mle` methods must therefore accept `SubArray`s.
The weighted variant sets `α[k] = sum(w[cat[k]]) / sum(w)` and forwards `view(w, cat[k])` to each component fit.
"""
function M_step!(α, dists, y::AbstractVector, cat, method::StochasticEM)
    N = size_sample(y)
    for (k, cₖ) in enumerate(cat)
        α[k] = length(cₖ) / N
        dists[k] = fit_mle(dists[k], view(y, cₖ))
    end
end

function M_step!(α, dists, y::AbstractMatrix, cat, method::StochasticEM)
    N = size_sample(y)
    for (k, cₖ) in enumerate(cat)
        α[k] = length(cₖ) / N
        dists[k] = fit_mle(dists[k], view(y, :, cₖ))
    end
end

function M_step!(α, dists, y::AbstractVector, cat, w, method::StochasticEM)
    sw = sum(w)
    for (k, cₖ) in enumerate(cat)
        α[k] = sum(view(w, cₖ)) / sw
        dists[k] = fit_mle(dists[k], view(y, cₖ), view(w, cₖ))
    end
end

function M_step!(α, dists, y::AbstractMatrix, cat, w, method::StochasticEM)
    sw = sum(w)
    for (k, cₖ) in enumerate(cat)
        α[k] = sum(view(w, cₖ)) / sw
        dists[k] = fit_mle(dists[k], view(y, :, cₖ), view(w, cₖ))
    end
end
