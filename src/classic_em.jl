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
    display = :none,
    maxiter = 1000,
    atol = 1e-3,
    rtol = nothing,
    robust = false,
)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size_sample(y), length(dists)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # E-step
    E_step!(LL, c, γ, dists, α, y; robust = robust)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter
        # M-step
        M_step!(α, dists, y, γ, method)

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust = robust)

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
For the `ClassicEM` the weigths `γ` computed at E-step for each observation in `y` are used to update `α` and `dists`.
"""
function M_step!(α, dists, y::AbstractVecOrMat, γ, method::ClassicEM)
    α[:] = mean(γ, dims = 1)
    dists[:] = [fit_mle(dists[k], y, γₖ) for (k, γₖ) in enumerate(eachcol(γ))]
end

#TODO: could probably replace γ, w by γ*w,
function M_step!(α, dists, y::AbstractVecOrMat, γ, w, method::ClassicEM)
    α[:] = mean(γ, weights(w), dims = 1)
    dists[:] = [fit_mle(dists[k], y, w[:] .* γₖ) for (k, γₖ) in enumerate(eachcol(γ))]
end

function fit_mle!(
    α::AbstractVector,
    dists::AbstractVector{F} where {F<:Distribution},
    y::AbstractVecOrMat,
    w::AbstractVector,
    method::ClassicEM;
    display = :none,
    maxiter = 1000,
    atol = 1e-3,
    rtol = nothing,
    robust = false,
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
    E_step!(LL, c, γ, dists, α, y; robust = robust)

    # Loglikelihood
    logtot = sum(w[n] * c[n] for n = 1:N) #dot(w, c)
    (display == :iter) && println("Method = $(method)\nIteration 0: Loglikelihood = ", logtot)

    for it = 1:maxiter
        # M-step
        M_step!(α, dists, y, γ, w, method)

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust = robust)

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
