"""
fit_mle(mix::MixtureModel, y; display = :none, maxiter = 1000, tol = 1e-3, robust = false)

fit_em use Expectation Maximization (EM) algorithm to maximize the Loglikelihood (fit) the mixture to an i.i.d sample `y`.
The `mix` agrument is a mixture that is used to initilize the EM algorithm.
"""
#TODO redundant between y::Vector or y::Matrix -> because N = size(y,1)=length(y) or N = size(y,2) because of column convention
function fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractVector;
    display=:none, maxiter=1000, tol=1e-3, robust=false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = length(y), length(dists)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # E-step
    # evaluate likelihood for each type k
    for k = 1:K
        LL[:, k] = log(α[k]) .+ logpdf.(dists[k], y)
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter

        # M-step
        # with γ, maximize (update) the parameters
        α[:] = mean(γ, dims=1)
        dists[:] = [fit_mle(dists[k], y, γ[:, k]) for k = 1:K]

        # E-step
        # evaluate likelihood for each type k
        for k = 1:K
            LL[:, k] = log(α[k]) .+ logpdf.(dists[k], y)
        end
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        # get posterior of each category
        c[:] = logsumexp(LL, dims=2)
        γ[:, :] = exp.(LL .- c)

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

function fit_mle!(α::AbstractVector, dists::AbstractVector{F} where {F<:Distribution}, y::AbstractMatrix;
    display=:none, maxiter=1000, tol=1e-3, robust=false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size(y, 2), length(dists)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)

    # E-step
    # evaluate likelihood for each type k
    for k = 1:K
        LL[:, k] = log(α[k]) .+ logpdf.(dists[k], y)
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims=2)
    γ[:, :] = exp.(LL .- c)

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
        for k = 1:K
            LL[:, k] = log(α[k]) .+ logpdf.(dists[k], y)
        end
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        # get posterior of each category
        c[:] = logsumexp(LL, dims=2)
        γ[:, :] = exp.(LL .- c)

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

function fit_mle(mix::MixtureModel, y::AbstractVecOrMat; display=:none, maxiter=1000, tol=1e-3, robust=false, infos=false)

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

    #TODO is there a better way to avoid when infos = false allocating history?
    history = fit_mle!(α, dists, y; display=display, maxiter=maxiter, tol=tol, robust=robust)

    return infos ? (MixtureModel(dists, α), history) : MixtureModel(dists, α)
end