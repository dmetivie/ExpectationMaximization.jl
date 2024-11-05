"""
    Base.@kwdef struct StochasticEM<:AbstractEM
        rng::AbstractRNG = Random.GLOBAL_RNG
    end
The Stochastic EM algorithm was introduced by G. Celeux, and J. Diebolt. in 1985 in [*The SEM Algorithm: A probabilistic teacher algorithm derived from the EM algorithm for the mixture problem*](https://cir.nii.ac.jp/crid/1574231874553755008).

The default random seed is `Random.GLOBAL_RNG` but it can be changed via `StochasticEM(seed)`.
"""
Base.@kwdef struct StochasticEM <: AbstractEM
    rng::AbstractRNG = Random.GLOBAL_RNG
end

"""
    fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, method::StochasticEM; display=:none, maxiter=1000, atol=1e-3, robust=false)
Use the stochastic EM algorithm to update the Distribution `dists` and weights `α` composing a mixture distribution.
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
    display=:none,
    maxiter=1000,
    atol=1e-3,
    rtol=nothing,
    robust=false,
)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size_sample(y), length(dists)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)
    ẑ = zeros(Int, N)
    # E-step
    E_step!(LL, c, γ, dists, α, y; robust=robust)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Method = $(method)\nIteration 0: loglikelihood = ", logtot)

    for it = 1:maxiter
        # S-step
        ẑ[:] .= [rand(method.rng, Categorical(ℙ...)) for ℙ in eachrow(γ)]
        cat = [findall(ẑ .== k) for k = 1:K]

        # M-step
        M_step!(α, dists, y, cat, method)

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust=robust)

        # Loglikelihood
        logtotp = sum(c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < atol || (rtol !== nothing && abs(logtotp - logtot) < rtol * (abs(logtot) + abs(logtotp)) / 2)
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println(
                "EM has not converged after $(history["iterations"]) iterations, final loglikelihood = $logtot",
            )
        end
    end

    return history
end

"""
    M_step!(α, dists, y, cat, method::StochasticEM)
For the `StochasticEM` the `cat` drawn at S-step for each observation in `y` is used to update `α` and `dists`.
"""
function M_step!(α, dists, y::AbstractVector, cat, method::StochasticEM)
    # 
    α[:] = length.(cat) / size_sample(y)
    dists[:] = [fit_mle(dists[k], y[cₖ]) for (k, cₖ) in enumerate(cat)]
end

function M_step!(α, dists, y::AbstractMatrix, cat, method::StochasticEM)
    α[:] = length.(cat) / size_sample(y)
    dists[:] = [fit_mle(dists[k], y[:, cₖ]) for (k, cₖ) in enumerate(cat)]
end

function M_step!(α, dists, y::AbstractVector, cat, w, method::StochasticEM)
    α[:] = [sum(w[cₖ]) for cₖ in cat] / sum(w)
    dists[:] = [fit_mle(dists[k], y[cₖ], w[cₖ]) for (k, cₖ) in enumerate(cat)]
end

function M_step!(α, dists, y::AbstractMatrix, cat, w, method::StochasticEM)
    α[:] = [sum(w[cat[k]]) for k in 1:K] / sum(w)
    dists[:] = [fit_mle(dists[k], y[:, cₖ], w[cₖ]) for (k, cₖ) in enumerate(cat)]
end

# univariate version with weights
function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    w::AbstractVector,
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
    @argcheck length(w) == N
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # E-step
    E_step!(LL, c, γ, dists, α, y; robust=robust)

    # Loglikelihood
    logtot = sum(w[n] * c[n] for n = 1:N) #dot(w, c)
    (display == :iter) && println("Method = $(method)\nIteration 0: loglikelihood = ", logtot)

    for it = 1:maxiter
        # S-step
        ẑ = [rand(method.rng, Categorical(ℙ...)) for ℙ in eachrow(γ)]
        cat = [findall(ẑ .== k) for k = 1:K]

        # M-step
        M_step!(α, dists, y, cat, w, method)

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust=robust)

        # Loglikelihood
        logtotp = sum(w[n] * c[n] for n in eachindex(c)) #dot(w, c)
        (display == :iter) && println("Iteration $(it): loglikelihood = ", logtotp)

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < atol || (rtol !== nothing && abs(logtotp - logtot) < rtol * (abs(logtot) + abs(logtotp)) / 2)
            (display in [:iter, :final]) &&
                println("EM converged in ", it, " iterations, final loglikelihood = ", logtotp)
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println(
                "EM has not converged after $(history["iterations"]) iterations, final loglikelihood = $logtot",
            )
        end
    end

    return history
end
