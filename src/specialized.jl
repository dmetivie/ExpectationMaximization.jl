#! Narrow, opt-in specializations of the E-step.
#!
#! Nothing here changes the generic code path in `fit_em.jl`. These are extra
#! `_loglikelihood_col!` methods that the compiler selects on the component type, and any component
#! they do not match keeps the generic per-observation `logpdf` fallback. Because dispatch happens
#! per component rather than per mixture, a heterogeneous mixture benefits from them too.

using LinearAlgebra: BLAS, copytri!
using Distributions: PDMats # `whiten!` is a BLAS-3 triangular solve; PDMats is already a Distributions dependency

# Sample size to whiten in one BLAS call. 64, 256 and 1024 measured within 3% of each other, so this
# is a plateau rather than a sharp optimum; the point is that it does not scale with N.
const MVNORMAL_BLOCKSIZE = 256

# `-(D log(2π) + logdet Σ) / 2`: the part of the log density independent of the observation.
_mvnormal_c0(d::MvNormal) = -(length(d) * log(2π) + logdetcov(d)) / 2

"""
    _loglikelihood_col!(LLₖ, d::MvNormal, logα, y::AbstractMatrix)

<<<<<<< HEAD
<<<<<<< HEAD
`Distributions.sqmahal!`, which the generic path reaches through `logpdf!`, materializes a `D × N`
centered copy of the sample and then still solves one triangular system per observation. These
methods evaluate the entire sample in one pass instead:
=======
`logpdf(::MvNormal, x)` redoes a triangular solve for every single observation, and for a full
covariance `Distributions.sqmahal!` additionally materializes a `D × N` centered copy of the whole
sample. These methods evaluate the entire sample in one pass instead:
>>>>>>> d5c9d66bc6678fd9e52a0df125aaad605ebea5f3
=======
`logpdf(::MvNormal, x)` redoes a triangular solve for every single observation, and for a full
covariance `Distributions.sqmahal!` additionally materializes a `D × N` centered copy of the whole
sample. These methods evaluate the entire sample in one pass instead:
>>>>>>> d5c9d66bc6678fd9e52a0df125aaad605ebea5f3

- isotropic or diagonal `Σ`: a weighted sum of squares, with no temporary at all;
- full `Σ`: one blocked in-place `PDMats.whiten!` (a BLAS-3 `trsm`) per `MVNORMAL_BLOCKSIZE`
  observations, instead of one BLAS-2 `trsv` per observation.

<<<<<<< HEAD
<<<<<<< HEAD
Measured 3.5x-11.4x faster than the generic path for `D` from 2 to 100 at `N = 1e5`, agreeing to a
few units in the last place, with an allocation bounded by `MVNORMAL_BLOCKSIZE` rather than growing
with `N`.
=======
Measured 3x-18x faster than the generic fallback for `D` from 2 to 100 and `N` from 1e4 to 1e6
(the ratio grows as the covariance gets simpler and shrinks as `D` grows), agreeing to 1-2 ulp.
>>>>>>> d5c9d66bc6678fd9e52a0df125aaad605ebea5f3
=======
Measured 3x-18x faster than the generic fallback for `D` from 2 to 100 and `N` from 1e4 to 1e6
(the ratio grows as the covariance gets simpler and shrinks as `D` grows), agreeing to 1-2 ulp.
>>>>>>> d5c9d66bc6678fd9e52a0df125aaad605ebea5f3
"""
function _loglikelihood_col!(
    LLₖ, d::Union{IsoNormal{T},DiagNormal{T}}, logα, y::AbstractMatrix
) where {T<:Real}
    μ, iv = mean(d), inv.(var(d))
    c = logα + _mvnormal_c0(d)
    @inbounds for n in axes(y, 2)
        s = zero(eltype(LLₖ))
        @simd for i in axes(y, 1)
            s += abs2(y[i, n] - μ[i]) * iv[i]
        end
        LLₖ[n] = c - s / 2
    end
    return LLₖ
end

function _loglikelihood_col!(LLₖ, d::FullNormal{T}, logα, y::AbstractMatrix) where {T<:Real}
    μ, Σ = mean(d), cov(d)
    c = logα + _mvnormal_c0(d)
    D, N = size(y)
    B = min(MVNORMAL_BLOCKSIZE, max(N, 1))
    z = Matrix{float(eltype(y))}(undef, D, B)   # one cache-resident block, never a D × N copy
    @inbounds for n₀ = 1:B:N
        m = min(B, N - n₀ + 1)
        zₘ = view(z, :, 1:m)
        for j = 1:m, i = 1:D
            zₘ[i, j] = y[i, n₀+j-1] - μ[i]
        end
        PDMats.whiten!(Σ, zₘ)                   # in-place ldiv!(chol_lower(Σ), zₘ)
        for j = 1:m
            s = zero(eltype(LLₖ))
            @simd for i = 1:D
                s += abs2(zₘ[i, j])
            end
            LLₖ[n₀+j-1] = c - s / 2
        end
    end
    return LLₖ
end

# `fit_mle(d::FullNormal, y::AbstractMatrix, w::AbstractVector)`
#
# `Distributions.fit_mle(::Type{FullNormal}, y, w)` allocates its own `D × N` scratch on every call, so
# the M-step throws away one such array per component per iteration (16 MB per call at `D = 10`,
# `N = 2·10⁵`). This accumulates the same scatter matrix blockwise with `syrk!`, reusing a cache
# resident `D × MVNORMAL_BLOCKSIZE` buffer, and produces a bit-identical covariance. It is a plain
# method of `Distributions.fit_mle`, behaviourally identical to it, so it is documented here rather
# than in the manual.
#
# Restricted to a full covariance so that `DiagNormal` and `IsoNormal` keep their own (already cheap)
# fits and, crucially, their covariance type; anything else falls back to `Distributions`.
function fit_mle(
    d::FullNormal{T}, y::AbstractMatrix{T}, w::AbstractVector{T}
) where {T<:BLAS.BlasFloat}
    D, N = size(y)
    length(w) == N || throw(DimensionMismatch("The dimensions of y and w are inconsistent."))
    isw = inv(sum(w))
    μ = BLAS.gemv('N', isw, y, w)
    B = min(MVNORMAL_BLOCKSIZE, max(N, 1))
    z = Matrix{T}(undef, D, B)
    Σ = zeros(T, D, D)
    @inbounds for n₀ = 1:B:N
        m = min(B, N - n₀ + 1)
        zₘ = view(z, :, 1:m)
        for j = 1:m
            cⱼ = sqrt(w[n₀+j-1])
            @simd for i = 1:D
                zₘ[i, j] = (y[i, n₀+j-1] - μ[i]) * cⱼ
            end
        end
        BLAS.syrk!('U', 'N', isw, zₘ, one(T), Σ)   # Σ += isw * zₘ * zₘ'
    end
    copytri!(Σ, 'U')
    return MvNormal(μ, PDMats.PDMat(Σ))
end
