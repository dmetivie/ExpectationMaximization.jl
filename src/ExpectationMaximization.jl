module ExpectationMaximization

using ArgCheck
using Distributions
using LogExpFunctions: logsumexp
using StatsBase: weights

# Extended functions
import Distributions: fit_mle, params

export fit_mle, fit_mle!

include("fit_em.jl")
include("that_should_be_in_Distributions.jl")
end
