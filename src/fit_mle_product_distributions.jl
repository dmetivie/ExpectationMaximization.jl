#! What is the convention and or the most efficient way to store these multidim arrays: samples from column/row correspond to one distributions?
#! Here we choose row correspond to one distributions and number of row is the number of realization
#! It seems to be the convention
#TODO Only work for distributions with known fit_mle (cannot be product_distribution of mixture because of the typeof)
"""
Simply extend the `fit_mle` function to multivariate Product distributions.
"""
function fit_mle(g::Product, x::AbstractMatrix)
    S = size(x, 1) # Distributions convention
    vec_g = g.v
    @argcheck S == length(vec_g)
    return product_distribution([fit_mle(typeof(vec_g[s]), y) for (s, y) in enumerate(eachrow(x))])
end

function fit_mle(g::Product, x::AbstractMatrix, γ::AbstractVector)
    S = size(x, 1) # Distributions convention
    vec_g = g.v
    @argcheck S == length(vec_g)
    return product_distribution([fit_mle(typeof(vec_g[s]), y, γ) for (s, y) in enumerate(eachrow(x))])
end

params(g::Product) = params.(g.v)
"""
Now `fit_mle(Bernoulli(0.2), x)` is accepted in addition of `fit_mle(Bernoulli, x)` this allows compatibility with how `fit_mle(g::Product)` and `fit_mle(g::MixtureModel)` are written.
#! Not 100% sure it will not cause any issuses or conflic!
#! There might be another way to do with the type something like:
#! https://discourse.julialang.org/t/ann-copulas-jl-a-fully-distributions-jl-compliant-copula-package/76544/12?u=dmetivie
#! MyMarginals = Tuple{LogNormal,Pareto,Gamma,Normal};
#! fitted_model = fit(SklarDist{MyCop,MyMarginals},data)
#! and initial parameters as kwargs?
"""
function fit_mle(g::Distribution{Univariate,S}, args...) where {S}
    fit_mle(typeof(g), args...)
end

function fit_mle(g::Distribution{Multivariate,S}, args...) where {S}
    fit_mle(typeof(g), args...)
end