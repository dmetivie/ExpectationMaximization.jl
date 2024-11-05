module ExpectationMaximization

using ArgCheck
using Distributions
using Distributions: ArrayOfUnivariateDistribution, VectorOfUnivariateDistribution # for product distributions
using LogExpFunctions: logsumexp!, logsumexp
using StatsBase: weights
using Random # to add @kwdef 

# Extended functions
import Distributions: fit_mle, params

export fit_mle, fit_mle!

abstract type AbstractEM end

# Utilities

size_sample(y::AbstractMatrix) = size(y, 2)
size_sample(y::AbstractVector) = length(y)

argmaxrow(M) = [argmax(r) for r in eachrow(M)]

"""
    predict(mix::MixtureModel, y::AbstractVector; robust=false)
Evaluate the most likely category for each observations given a `MixtureModel`.
- `robust = true` will prevent the (log)likelihood to overflow to `-∞` or `∞`.
"""
function predict(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
    return argmaxrow(predict_proba(mix, y; robust=robust))
end

"""
    predict_proba(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
Evaluate the probability for each observations to belong to a category given a `MixtureModel`..
- `robust = true` will prevent the (log)likelihood to under(overflow)flow to `-∞` (or `∞`).
"""
function predict_proba(mix::MixtureModel, y::AbstractVecOrMat; robust=false)
    # evaluate likelihood for each components k
    dists = mix.components
    α = probs(mix)
    K = length(dists)
    N = size_sample(y)
    LL = zeros(N, K)
    γ = similar(LL)
    c = zeros(N)
    E_step!(LL, c, γ, dists, α, y; robust=robust)
    return γ
end

include("that_should_be_in_Distributions.jl")
include("fit_em.jl")
include("classic_em.jl")
include("stochastic_em.jl")


export ClassicEM, StochasticEM
export predict_proba, predict
end
