"""
    fit_mle(g::Product, x::AbstractMatrix)
    fit_mle(g::Product, x::AbstractMatrix, γ::AbstractVector)

The `fit_mle` for multivariate Product distributions `g` is the `product_distribution` of `fit_mle` of each components of `g`.
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

# * `fit_mle` (weighted or not) of some distribution

fit_mle(::Type{<:Dirac}, x::AbstractArray{T}) where {T<:Real} = length(unique(x)) == 1 ? Dirac(first(x)) : Dirac(NaN)
function fit_mle(::Type{<:Dirac}, x::AbstractArray{T}, w::AbstractArray{Float64}) where {T<:Real}
    n = length(x)
    if n != length(w)
        throw(DimensionMismatch("Inconsistent array lengths."))
    end
    return length(unique(x[findall(!iszero, w)])) == 1 ? Dirac(first(x)) : Dirac(NaN)
end

function fit_mle(::Type{<:Laplace}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
    xc = similar(x)
    copyto!(xc, x)
    m = median(xc, weights(w))
    xc .= abs.(x .- m)
    return Laplace(m, mean(xc, weights(w)))
end