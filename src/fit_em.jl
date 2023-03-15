"""
    fit_mle(mix::MixtureModel, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, robust=false, infos=false)
Use the an Expectation Maximization (EM) algorithm to maximize the Loglikelihood (fit) the mixture with an i.i.d sample `y`.
The `mix` input is a mixture that is used to initilize the EM algorithm.
- `method` determines the algorithm used.
- `infos = true` returns a `Dict` with informations on the algorithm.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
- `atol` criteria determining the convergence of the algorithm. If the Loglikelihood difference between two iteration `i` and `i+1` is smaller than `atol` i.e. `|ℓ⁽ⁱ⁺¹⁾ - ℓ⁽ⁱ⁾|<atol`, the algorithm stops. 
- `display` value can be `:none`, `:iter`, `:final` to display Loglikelihood evolution at each iterations `:iter` or just the final one `:final`
"""
function fit_mle(mix::MixtureModel, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, robust=false, infos=false)

    # Initial parameters
    α = copy(probs(mix))
    dists = copy(components(mix))

    #TODO is there a better way to avoid when infos = false allocating history?
    if isempty(weights)
        history = fit_mle!(α, dists, y, method; display=display, maxiter=maxiter, atol=atol, robust=robust)
    else
        history = fit_mle!(α, dists, y, weights..., method; display=display, maxiter=maxiter, atol=atol, robust=robust)
    end

    return infos ? (MixtureModel(dists, α), history) : MixtureModel(dists, α)
end

"""
    fit_mle(mix::AbstractArray{<:MixtureModel}, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, robust=false, infos=false)

Do the same as `fit_mle` for each (initial) mixtures in the mix array. Then it selects the one with the largest loglikelihood.
It uses try and catch to avoid errors messages in case EM converges toward a singular solution (probably using robust should be enough in most case to avoid errors). 
"""
function fit_mle(mix::AbstractArray{<:MixtureModel}, y::AbstractVecOrMat, weights...; method = ClassicEM(), display=:none, maxiter=1000, atol=1e-3, robust=false, infos=false)

    mx_max, history_max = fit_mle(mix[1], y, weights...; method = method, display=display, maxiter=maxiter, atol=atol, robust=robust, infos=true)
    for j in eachindex(mix)[2:end]
        try
            mx_new, history_new = fit_mle(mix[j], y, weights...; method = method, display=display, maxiter=maxiter, atol=atol, robust=robust, infos=true)
            if history_max["logtots"][end] < history_new["logtots"][end]
                mx_max = mx_new
                history_max = copy(history_new)
            end
        catch
            continue
        end
    end
    return infos ? (mx_max, history_max) : mx_max
end

function E_step!(LL::AbstractMatrix{T}, c::AbstractVector{T}, γ::AbstractMatrix{T}, dists::AbstractVector{F} where {F<:Distribution}, α::AbstractVector, y::AbstractVector{<:Real}; robust=false) where {T<:AbstractFloat}
    # evaluate likelihood for each type k
    for k = eachindex(dists)
        LL[:, k] .= log(α[k]) .+ logpdf.(dists[k], y)
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    logsumexp!(c, LL) # c[:] = logsumexp(LL, dims=2)
    γ[:, :] .= exp.(LL .- c)
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

# Utilities 

size_sample(y::AbstractMatrix) = size(y, 2)
size_sample(y::AbstractVector) = length(y)

argmaxrow(M) = [argmax(r) for r in eachrow(M)]

"""
    most_likely_cat(mix::MixtureModel, y::AbstractVector; robust=false)
Evaluate the most likely category of each observations.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
"""
function most_likely_cat(mix::MixtureModel, y::AbstractVector; robust=false)
    return argmaxrow(likelihood_per_cat(mix, y; robust=robust))
end

"""
    likelihood_per_cat(mix::MixtureModel, y::AbstractVector; robust=false)
Evaluate the the probability for each observations to belong to a category.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.

"""
function likelihood_per_cat(mix::MixtureModel, y::AbstractVector; robust=false)
    # evaluate likelihood for each components k
    dists = mix.components
    α = probs(mix)
    K = length(dists)
    N = length(y)
    LL = zeros(N, K)
    c = zeros(N)
    for k = eachindex(dists)
        LL[:, k] .= log(α[k]) .+ logpdf.(dists[k], y)
    end
    robust && replace!(LL, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
    # get posterior of each category
    logsumexp!(c, LL) # c[:] = logsumexp(LL, dims=2)
    return exp.(LL .- c)
end