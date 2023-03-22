module ExpectationMaximization

using ArgCheck
using Distributions
using LogExpFunctions: logsumexp!, logsumexp
using StatsBase: weights
using Random # to add @kwdef 

# Extended functions
import Distributions: fit_mle, params

export fit_mle, fit_mle!

abstract type AbstractEM end

include("classic_em.jl")
include("stochastic_em.jl")
include("fit_em.jl")
include("that_should_be_in_Distributions.jl")

export ClassicEM, StochasticEM
export predict_proba, predict
end
