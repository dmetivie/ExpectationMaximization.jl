module ExpectationMaximization

using ArgCheck
using Distributions
using Distributions: ArrayOfUnivariateDistribution, VectorOfUnivariateDistribution # for product distributions
using LogExpFunctions: logsumexp! # kept as the reference implementation of `_softmax_rows!`
using StatsBase: weights
using Random # to add @kwdef

# Extended functions
import Distributions: fit_mle, params

export fit_mle, fit_mle!

abstract type AbstractEM end

# Utilities

size_sample(y::AbstractMatrix) = size(y, 2)
size_sample(y::AbstractVector) = length(y)

# `argmax` over `eachrow` of a column-major matrix walks each row with stride `N`; scanning column by
# column instead is ~5x faster. Strict `>` keeps `argmax`'s "first maximum wins" tie breaking.
function argmaxrow(M)
    z = Vector{Int}(undef, size(M, 1))
    @inbounds for n in axes(M, 1)
        best, j = M[n, firstindex(M, 2)], firstindex(M, 2)
        for k in axes(M, 2)
            M[n, k] > best && ((best, j) = (M[n, k], k))
        end
        z[n] = j
    end
    return z
end

"""
    predict(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
Evaluate the most likely category for each observation given a `MixtureModel`, i.e. the `argmax` over the
row of [`predict_proba`](@ref) belonging to that observation.
Returns a length-`N` `Vector{Int}` of component indices in `1:ncomponents(mix)`; ties go to the lowest index.
When `y` is an `AbstractMatrix`, each **column** is one observation, so `N = size(y, 2)`.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
"""
function predict(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
    return argmaxrow(predict_proba(mix, y; robust=robust))
end

"""
    predict_proba(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
Evaluate the probability for each observation to belong to a category given a `MixtureModel`.
Returns a fresh `N × K` matrix whose row `n` is the posterior distribution of the component label of
observation `n`, with `K = ncomponents(mix)` and `N = size_sample(y)` (`length(y)` for a vector,
`size(y, 2)` for a matrix, where each **column** is one observation). Every row sums to `1` unless it is
degenerate, i.e. the observation has zero density under every component (see `robust`).
- `robust = true` will prevent the (log)likelihood to under(overflow)flow to `-∞` (or `∞`).
"""
function predict_proba(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
    # evaluate likelihood for each components k
    dists = components(mix)
    α = probs(mix)
    K = length(dists)
    N = size_sample(y)
    LL = zeros(N, K)
    c = zeros(N)
    s = zeros(N)
    γ = LL   # γ aliases LL: the posteriors overwrite the log-likelihoods in place
    E_step!(LL, c, γ, s, dists, α, y; robust=robust)
    return γ
end

include("that_should_be_in_Distributions.jl")
include("fit_em.jl")
include("classic_em.jl")
include("stochastic_em.jl")


export ClassicEM, StochasticEM
export predict_proba, predict
end
