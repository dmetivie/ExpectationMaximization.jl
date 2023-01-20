"""
    fit_mle(mix::MixtureModel, y::AbstractVecOrMat, [w::AbstractVector]; display = :none, maxiter = 1000, tol = 1e-3, robust = false)

Use Expectation Maximization (EM) algorithm to maximize the Loglikelihood (fit) the mixture to an i.i.d sample `y`.
The `mix` agrument is a mixture that is used to initilize the EM algorithm.
"""
function fit_mle(mix::MixtureModel, y::AbstractVecOrMat, weights...; display=:none, maxiter=1000, tol=1e-3, robust=false, infos=false)

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

    #TODO is there a better way to avoid when infos = false allocating history?
    history = fit_mle!(α, dists, y, weights...; display=display, maxiter=maxiter, tol=tol, robust=robust)

    return infos ? (MixtureModel(dists, α), history) : MixtureModel(dists, α)
end

"""
    fit_mle(mix::AbstractArray{<:MixtureModel}, y::AbstractVecOrMat, weights...; display=:none, maxiter=1000, tol=1e-3, robust=false, infos=false)

Use `fit_mle` (EM) algorithm for all the different initial conditions in the mix array and select the one with the largest likelihood.
"""
function fit_mle(mix::AbstractArray{<:MixtureModel}, y::AbstractVecOrMat, weights...; display=:none, maxiter=1000, tol=1e-3, robust=false, infos=false)

    mx_max, history_max = fit_mle(mix[1], y, weights...; display=display, maxiter=maxiter, tol=tol, robust=robust, infos=true)
    for j in eachindex(mix)[2:end]
        mx_new, history_new = fit_mle(mix[j], y, weights...; display=display, maxiter=maxiter, tol=tol, robust=robust, infos=true)
        if history_max["logtots"][end] < history_new["logtots"][end]
            mx_max = mx_new
            history_max = copy(history_new)
        end
    end
    return infos ? (mx_max, history_max) : mx_max
end

function E_step!(LL::AbstractMatrix, c::AbstractVector, γ::AbstractMatrix, dists::AbstractVector{F} where {F<:Distribution}, α::AbstractVector, y::AbstractVector; robust=false)
    # evaluate likelihood for each type k
    for k = eachindex(dists)
        LL[:, k] = log(α[k]) .+ logpdf.(dists[k], y)
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)
end

function E_step!(LL::AbstractMatrix, c::AbstractVector, γ::AbstractMatrix, dists::AbstractVector{F} where {F<:Distribution}, α::AbstractVector, y::AbstractMatrix; robust=false)
    # evaluate likelihood for each type k
    for k = eachindex(dists)
        LL[:, k] .= log(α[k])
        for n in axes(y, 2)
            LL[n, k] += logpdf(dists[k], y[:, n])
        end
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)
end

size_sample(y::AbstractMatrix) = size(y, 2)
size_sample(y::AbstractVector) = length(y)

function fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat;
    display=:none, maxiter=1000, tol=1e-3, robust=false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size_sample(y), length(dists)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # E-step
    E_step!(LL, c, γ, dists, α, y; robust=robust)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter

        # M-step
        # with γ in hand, maximize (update) the parameters
        α[:] = mean(γ, dims=1)
        dists[:] = [fit_mle(dists[k], y, γ[:, k]) for k = 1:K]

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust=robust)

        # Loglikelihood
        logtotp = sum(c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println("EM has not converged after $(history["iterations"]) iterations, logtot = $logtot")
        end
    end

    return history
end

function fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVecOrMat, w::AbstractVector;
    display=:none, maxiter=1000, tol=1e-3, robust=false)

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
    logtot = sum(w[n] * c[n] for n in 1:N) #dot(w, c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter

        # M-step
        # with γ in hand, maximize (update) the parameters
        α[:] = mean(γ, weights(w), dims=1)
        dists[:] = [fit_mle(dists[k], y, w[:] .* γ[:, k]) for k = 1:K]

        # E-step
        # evaluate likelihood for each type k
        E_step!(LL, c, γ, dists, α, y; robust=robust)

        # Loglikelihood
        logtotp = sum(w[n] * c[n] for n in eachindex(c)) #dot(w, c)
        (display == :iter) && println("Iteration $it: logtot = $logtotp")

        push!(history["logtots"], logtotp)
        history["iterations"] += 1

        if abs(logtotp - logtot) < tol
            (display in [:iter, :final]) &&
                println("EM converged in $it iterations, logtot = $logtotp")
            history["converged"] = true
            break
        end

        logtot = logtotp
    end

    if !history["converged"]
        if display in [:iter, :final]
            println("EM has not converged after $(history["iterations"]) iterations, logtot = $logtot")
        end
    end

    return history
end