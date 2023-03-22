#! In this file I implement some IMO cool features that should be in Distributions.jl, see my [PR#1670](https://github.com/JuliaStats/Distributions.jl/pull/1670). 
#! Currently it is not accepted and may never be.

## * Instance version of `fit_mle` * ##

"""
Now "instance" version of `fit_mle` is supported (in addition of the current "type" version). 
Example: `fit_mle(Bernoulli(0.2), x)` is accepted in addition of `fit_mle(Bernoulli, x)` this allows compatibility with how `fit_mle(g::Product)` and `fit_mle(g::MixtureModel)` are written.
#! Not 100% sure it will not cause any issuses or conflic!
#! There might be another way to do with the type something like:
#! https://discourse.julialang.org/t/ann-copulas-jl-a-fully-distributions-jl-compliant-copula-package/76544/12
#! MyMarginals = Tuple{LogNormal,Pareto,Gamma,Normal};
#! fitted_model = fit(SklarDist{MyCop,MyMarginals},data)
#! and initial parameters as kwargs?
"""
function fit_mle(g::D, args...) where {D<:Distribution}
    fit_mle(typeof(g), args...)
end

fit_mle(d::T, x::AbstractArray{<:Integer}) where {T<:Binomial} =
    fit_mle(T, suffstats(T, ntrials(d), x))
fit_mle(d::T, x::AbstractArray{<:Integer}) where {T<:Categorical} =
    fit_mle(T, ncategories(d), x)

## * `fit_mle` for `product_distribution`

"""
    fit_mle(g::Product, x::AbstractMatrix)
    fit_mle(g::Product, x::AbstractMatrix, γ::AbstractVector)

The `fit_mle` for multivariate Product distributions `g` is the `product_distribution` of `fit_mle` of each components of `g`.
Product is meant to be depreacated in next version of `Distribution.jl`. Use the analog `VectorOfUnivariateDistribution` type instead.
"""
function fit_mle(g::Product, x::AbstractMatrix, args...)
    d = size(x, 1)
    length(g) == d ||
        throw(DimensionMismatch("The dimensions of g and x are inconsistent."))
    return product_distribution([
        fit_mle(g.v[s], y, args...) for (s, y) in enumerate(eachrow(x))
    ])
end

params(g::Product) = params.(g.v)

#! `ArrayOfUnivariateDistribution` is not released yet
# params(d::ArrayOfUnivariateDistribution) = params.(d.dists)

# #### Fitting
# promote_sample(::Type{dT}, x::AbstractArray{T}) where {T<:Real, dT<:Real} = T <: dT ? x : convert.(dT, x)

# """
#     fit_mle(dists::ArrayOfUnivariateDistribution, x::AbstractArray)
#     fit_mle(dists::ArrayOfUnivariateDistribution, x::AbstractArray, γ::AbstractVector)

# The `fit_mle` for a `ArrayOfUnivariateDistribution` distributions `dists` is the `product_distribution` of `fit_mle` of each components of `dists`.
# """
# function fit_mle(dists::VectorOfUnivariateDistribution, x::AbstractMatrix{<:Real}, args...)
#     length(dists) == size(x, 1) || throw(DimensionMismatch("The dimensions of dists and x are inconsistent."))
#     return product_distribution([fit_mle(d, promote_sample(eltype(d), x[s, :]), args...) for (s, d) in enumerate(dists.dists)])
# end

# function fit_mle(dists::ArrayOfUnivariateDistribution, x::AbstractArray, args...)
#     size(dists) == size(first(x)) || throw(DimensionMismatch("The dimensions of dists and x are inconsistent."))
#     return product_distribution([fit_mle(d, promote_sample(eltype(d), [x[i][s] for i in eachindex(x)]), args...) for (s, d) in enumerate(dists.dists)])
# end


## * New `fit_mle` * ##
#! `fit_mle` (weighted or not) of Dirac and Laplace distribution. I also would prefer that in `Distribution.jl`
#! See [PR#1676](https://github.com/JuliaStats/Distributions.jl/pull/1676) and a following for Dirac?

fit_mle(::Type{<:Dirac}, x::AbstractArray{T}) where {T<:Real} =
    length(unique(x)) == 1 ? Dirac(first(x)) : Dirac(NaN)

"""
    fit_mle(::Type{<:Dirac}, x::AbstractArray{<:Real}[, w::AbstractArray{<:Real}])
`fit_mle` for `Dirac` distribution (weighted or not) data sets.
"""
function fit_mle(
    ::Type{<:Dirac},
    x::AbstractArray{T},
    w::AbstractArray{Float64},
) where {T<:Real}
    n = length(x)
    if n != length(w)
        throw(DimensionMismatch("Inconsistent array lengths."))
    end
    return length(unique(x[findall(!iszero, w)])) == 1 ? Dirac(first(x)) : Dirac(NaN)
end

"""
    fit_mle(::Type{<:Laplace}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
`fit_mle` for `Laplace` distribution weighted data sets.
"""
function fit_mle(::Type{<:Laplace}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
    xc = similar(x)
    copyto!(xc, x)
    m = median(xc, weights(w))
    xc .= abs.(x .- m)
    return Laplace(m, mean(xc, weights(w)))
end

"""
    fit_mle(::Type{<:Uniform}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
`fit_mle` for `Uniform` distribution weighted data sets. It is just the same as unweigted (removing zero weighted data).
"""
function fit_mle(::Type{<:Uniform}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
    size(x) == size(w) || throw(DimensionMismatch("Inconsistent array lengths."))
    return fit_mle(Uniform, x[findall(w .!= 0)])
end