#! In this file I implement some IMO cool features that should be in Distributions.jl, see my [PR#1670](https://github.com/JuliaStats/Distributions.jl/pull/1670). 
#! Currently it is not accepted and may never be.

## * Instance version of `fit_mle` * ##

"""
    fit_mle(g::D, args...) where {D<:Distribution}
In `ExpectationMaximization.jl` the "instance" version of `fit_mle` is supported (in addition of the current "type" version).
Note this is not supported in `Distributions.jl`.
Example: `fit_mle(Bernoulli(0.2), x)` is accepted in addition of `fit_mle(Bernoulli, x)` this allows compatibility with how `fit_mle(g::Product)` and `fit_mle(g::MixtureModel)` are written.

By default the instance is simply dropped in favour of `typeof(g).name.wrapper`. More specific methods are
provided wherever the instance carries information the type does not: `DiagNormal`/`IsoNormal`/`FullNormal`
(the covariance structure), `Binomial` (`ntrials`) and `Categorical` (`ncategories`).
"""
function fit_mle(g::D, args...) where {D<:Distribution}
    fit_mle(typeof(g).name.wrapper, args...)
end

# specific dispatch for MvNormal as we need to keep the type of the covariance matrix
fit_mle(g::DiagNormal, args...) = fit_mle(DiagNormal, args...)
fit_mle(g::IsoNormal, args...) = fit_mle(IsoNormal, args...)
fit_mle(g::FullNormal, args...) = fit_mle(FullNormal, args...)

fit_mle(d::T, x::AbstractArray{<:Integer}) where {T<:Binomial} = fit_mle(T, suffstats(T, ntrials(d), x))
fit_mle(d::T, x::AbstractArray{<:Integer}) where {T<:Categorical} =
    Categorical(probs(fit_mle(T, ncategories(d), x)))
fit_mle(d::T, x::AbstractArray{<:Integer}, w::AbstractArray{<:Real}) where {T<:Categorical} =
    Categorical(probs(fit_mle(T, ncategories(d), x, w)))

## * `fit_mle` for `product_distribution`

#TODO: add deprecation notice!
#TODO! but currently still have `product_distribution([d1, d2]) ≠ product_distribution(d1, d2)` (first is still `Product` while second is `Distributions.ProductDistribution`)
#TODO! open issue in `Distributions.jl`

# Both `fit_mle` methods below fit marginal `s` on row `s` of `x`, and `Base.reindex` copies a
# vector-of-indices column index once per row slice. For a gather view -- `view(y, :, cat[k])`, what
# the `StochasticEM` M-step passes -- row slicing therefore costs `size(x, 1) * size(x, 2) * 8` bytes
# instead of `sizeof(x)`: 60 MiB rather than 1 MiB per M-step on the 784 x 10000 `Matrix{Bool}` MNIST
# case. Gather the columns once instead. Note this dispatches on the *sample*, not on the component:
# any component reaching these two methods is covered, nested mixtures of products included.
_gather_rows(x::AbstractMatrix) = x
_gather_rows(x::SubArray{<:Any,2,<:Any,<:Tuple{Any,AbstractVector{<:Integer}}}) = copy(x)
# A range column index is not copied by `reindex` (`cols[Base.Slice]` is again a range), so a strided
# view such as `view(y, :, a:b)` is passed through untouched, as is a `Matrix`, an `Adjoint`, a `BitMatrix`.
_gather_rows(x::SubArray{<:Any,2,<:Any,<:Tuple{Any,AbstractRange{<:Integer}}}) = x

"""
    fit_mle(g::Product, x::AbstractMatrix, args...)

The `fit_mle` for a multivariate `Product` distribution `g` is the `product_distribution` of the `fit_mle`
of each of its components, marginal `s` being fitted on row `s` of `x` — i.e. each **column** of `x` is one
observation and `length(g) == size(x, 1)` is required.
`args...` is forwarded to every marginal `fit_mle`, so it is either empty or a single weight vector `γ` of
length `size(x, 2)`.
`Product` is meant to be deprecated in the next versions of `Distributions.jl`. Use the analog
`VectorOfUnivariateDistribution` type instead.
"""
function fit_mle(g::Product, x::AbstractMatrix, args...)
    d = size(x, 1)
    length(g) == d || throw(DimensionMismatch("The dimensions of g and x are inconsistent."))
    xr = _gather_rows(x)
    return product_distribution([fit_mle(g.v[s], y, args...) for (s, y) in enumerate(eachrow(xr))])
end

params(g::Product) = params.(g.v)

params(d::ArrayOfUnivariateDistribution) = params.(d.dists)

#### Fitting
promote_sample(::Type{dT}, x::AbstractArray{T}) where {T<:Real,dT<:Real} = T <: dT ? x : convert.(dT, x)

"""
    fit_mle(dists::VectorOfUnivariateDistribution, x::AbstractMatrix{<:Real}, args...)

The `fit_mle` for a `VectorOfUnivariateDistribution` `dists` is the `product_distribution` of the `fit_mle`
of each of its components, marginal `s` being fitted on `view(x, s, :)` — i.e. each **column** of `x` is one
observation and `size(x, 1) == length(dists)` is required. Because the row is passed as a view, component
`fit_mle` methods must accept `AbstractVector` rather than `Vector`.
`args...` is forwarded to every marginal `fit_mle`, so it is either empty or a single weight vector `γ` of
length `size(x, 2)`.
`VectorOfUnivariateDistribution` should act like the old `Product`, while the sibling
`fit_mle(dists::ArrayOfUnivariateDistribution, x::AbstractArray, args...)` (same idea, but `x` an array of
arrays) is not really tested yet and is deliberately left undocumented.
"""
function fit_mle(dists::VectorOfUnivariateDistribution, x::AbstractMatrix{<:Real}, args...)
    length(dists) == size(x, 1) || throw(DimensionMismatch("The dimensions of dists and x are inconsistent."))
    # `view` rather than `x[s, :]`: the latter gathers a fresh N-vector for each of the D marginals,
    # on every iteration. Same as the `Product` method above, which already uses `eachrow`.
    xr = _gather_rows(x)
    return product_distribution([fit_mle(d, promote_sample(eltype(d), view(xr, s, :)), args...) for (s, d) in enumerate(dists.dists)]...)
end

function fit_mle(dists::ArrayOfUnivariateDistribution, x::AbstractArray, args...)
    size(dists) == size(first(x)) || throw(DimensionMismatch("The dimensions of dists and x are inconsistent."))
    return product_distribution([fit_mle(d, promote_sample(eltype(d), [x[i][s] for i in eachindex(x)]), args...) for (s, d) in enumerate(dists.dists)]...)
end


## * New `fit_mle` * ##
#! `fit_mle` (weighted or not) of Dirac and Laplace distribution. I also would prefer that in `Distribution.jl`
#! See [PR#1676](https://github.com/JuliaStats/Distributions.jl/pull/1676) and a following for Dirac?

fit_mle(::Type{<:Dirac}, x::AbstractArray{T}) where {T<:Real} =
    length(unique(x)) == 1 ? Dirac(first(x)) : Dirac(NaN)

"""
    fit_mle(::Type{<:Dirac}, x::AbstractArray{<:Real})
    fit_mle(::Type{<:Dirac}, x::AbstractArray{<:Real}, w::AbstractArray{Float64})
`fit_mle` for `Dirac` distribution (weighted or not) data sets. Returns `Dirac(first(x))` when all the
observations carrying a non-zero weight are equal, and `Dirac(NaN)` otherwise.
Note that the weighted method requires `w::AbstractArray{Float64}` exactly: another real element type
(e.g. `Vector{Float32}`) is a `MethodError`, since there is no other weighted `Dirac` method to fall back on.
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
    m = median(x, weights(w))   # StatsBase's weighted quantile does not mutate `x`, so no copy is needed
    s = zero(float(eltype(x)))
    sw = zero(float(eltype(w)))
    @inbounds for i in eachindex(x, w)
        s += w[i] * abs(x[i] - m)
        sw += w[i]
    end
    return Laplace(m, s / sw)
end

"""
    fit_mle(::Type{<:Uniform}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
`fit_mle` for `Uniform` distribution weighted data sets. It is the same as the unweighted fit applied to the
observations carrying a non-zero weight: the MLE only depends on the extrema of the support, so the non-zero
weight values themselves are irrelevant. Requires `size(x) == size(w)`.
"""
function fit_mle(::Type{<:Uniform}, x::AbstractArray{<:Real}, w::AbstractArray{<:Real})
    size(x) == size(w) || throw(DimensionMismatch("Inconsistent array lengths."))
    return fit_mle(Uniform, x[findall(w .!= 0)])
end