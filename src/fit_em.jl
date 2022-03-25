"""
fit_mle(mix::MixtureModel, y; display = :none, maxiter = 100, tol = 1e-3, robust = false)

fit_em use Expectation Maximization (EM) algorithm to maximize the Loglikelihood (fit) the mixture to an i.i.d sample `y`.
The `mix` agrument is a mixture that is used to initilize the EM algorithm.
"""
function fit_mle(mix::MixtureModel, y::AbstractVector; display = :none, maxiter = 1000, tol = 1e-3, robust = false, infos = false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = length(y), ncomponents(mix)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)
    # types = typeof.(components(mix))

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

    # E-step
    # evaluate likelihood for each type k
    for k = 1:K
        LL[:, k] = log(α[k]) .+ logpdf(dists[k], y)
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims = 2)
    γ[:, :] = exp.(LL .- c)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter

        # M-step
        # with γ, maximize (update) the parameters
        α[:] = mean(γ, dims = 1)
        dists[:] = [fit_mle(dists[k], y, γ[:, k]) for k = 1:K]

        # E-step
        # evaluate likelihood for each type k
        for k = 1:K
            LL[:, k] = log(α[k]) .+ logpdf(dists[k], y)
        end
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        # get posterior of each category
        c[:] = logsumexp(LL, dims = 2)
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

    return infos ? (MixtureModel(dists, α), history) : MixtureModel(dists, α)
end

function fit_mle(mix::MixtureModel, y::AbstractMatrix; display = :none, maxiter = 1000, tol = 1e-3, robust = false, infos = false)

    @argcheck display in [:none, :iter, :final]
    @argcheck maxiter >= 0

    N, K = size(y, 2), ncomponents(mix)
    history = Dict("converged" => false, "iterations" => 0, "logtots" => zeros(0))

    # Allocate memory for in-place updates

    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)
    # types = typeof.(components(mix))

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

    # E-step
    # evaluate likelihood for each type k
    for k = 1:K
        LL[:, k] = log(α[k]) .+ logpdf(dists[k], y)
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    c[:] = logsumexp(LL, dims = 2)
    γ[:, :] = exp.(LL .- c)

    # Loglikelihood
    logtot = sum(c)
    (display == :iter) && println("Iteration 0: logtot = $logtot")

    for it = 1:maxiter

        # M-step
        # with γ in hand, maximize (update) the parameters
        α[:] = mean(γ, dims = 1)
        dists[:] = [fit_mle(dists[k], y, γ[:, k]) for k = 1:K]

        # E-step
        # evaluate likelihood for each type k
        for k = 1:K
            LL[:, k] = log(α[k]) .+ logpdf(dists[k], y)
        end
        robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
        # get posterior of each category
        c[:] = logsumexp(LL, dims = 2)
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

    return infos ? (MixtureModel(dists, α), history) : MixtureModel(dists, α)
end