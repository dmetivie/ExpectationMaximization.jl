module ExpectationMaximization

using ArgCheck
using Distributions
using LogExpFunctions: logsumexp

# Extended functions
import Distributions: fit_mle, params

export fit_mle, fit_mle!

include("fit_em.jl")
include("fit_mle_product_distributions.jl")
end
