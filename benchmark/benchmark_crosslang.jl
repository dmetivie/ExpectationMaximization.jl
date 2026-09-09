# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Language EM Benchmark
# ExpectationMaximization.jl vs GaussianMixtures.jl vs mixtools (R) vs sklearn (Python)
# ═══════════════════════════════════════════════════════════════════════════════
#
# TODO: check number of iter for each run and check values consistencies between languages
# Benchmarks Gaussian mixture EM across four backends with fixed initial
# conditions and iteration count for fair comparison.
#
# Cases:
#   K2_D1    K=2,  D=1   Univariate Normal   (with cross-checks)
#   K5_D1    K=5,  D=1   Univariate Normal
#   K2_D2    K=2,  D=2   Multivariate Normal (distinct, correlated covariances)
#   K2_D10   K=2,  D=10  Multivariate Normal (distinct, correlated covariances)
#   K5_D15   K=5,  D=15  Multivariate Normal (distinct, correlated covariances)
#
# Usage:
#   julia --threads=1 --project=benchmark benchmark/benchmark_crosslang.jl
#
# Output (rewritten after every case, so an interrupted run still leaves data):
#   benchmark/results/benchmark_timings_<date>.csv
#   benchmark/results/benchmark_timings_latest.csv
#   benchmark/results/system_info_<date>.txt
#
# Plot them with: julia --project=benchmark benchmark/plot_crosslang.jl

cd(@__DIR__)
import Pkg; Pkg.activate(".")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Thread control — single thread everywhere for fair comparison
# ═══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra: BLAS, Diagonal, I, inv
BLAS.set_num_threads(1)

# Must be set before Python loads its BLAS
ENV["MKL_NUM_THREADS"] = "1"
ENV["NUMEXPR_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["VECLIB_MAXIMUM_THREADS"] = "1"

# @assert Threads.nthreads() == 1 "Start Julia with --threads=1 for fair benchmarking"

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Imports
# ═══════════════════════════════════════════════════════════════════════════════

using BenchmarkTools
using Dates
using Distributions
using ExpectationMaximization
using GaussianMixtures
using InteractiveUtils
using PythonCall

# Pkg.build("RCall")  # ensure RCall is built
using RCall
using StableRNGs
using Suppressor

# ── Python setup ──────────────────────────────────────────────────────────────
np = pyimport("numpy")
sklearn_mixture = pyimport("sklearn.mixture")
py_GaussianMixture = sklearn_mixture.GaussianMixture
os = pyimport("os")
os.environ["OMP_NUM_THREADS"] = "1"

# ── R setup ───────────────────────────────────────────────────────────────────
@rimport mixtools
R_normalmixEM = mixtools.normalmixEM

# ═══════════════════════════════════════════════════════════════════════════════
# 2.1 Helpers — non-trivial covariance structures
# ═══════════════════════════════════════════════════════════════════════════════
#
# AR(1)-structured covariance: Σ[i,j] = σ² * ρ^|i-j|. Positive-definite for
# |ρ| < 1, and gives each component a distinct, correlated (non-diagonal)
# covariance rather than a trivial identity/scalar one.
ar1_cov(D, σ, ρ) = Float64[σ^2 * ρ^abs(i - j) for i in 1:D, j in 1:D]

# Cross-checks between the backends are *recorded*, not thrown. A failing `@test` here would abort
# the script, and with it every remaining case — while the run is only interesting if it produces
# timings. Mismatches are warned about as they happen and re-raised at the very end, after the CSV
# is on disk, so an unattended (CI) run still goes red without losing its data.
const CROSSCHECK_FAILURES = String[]

function crosscheck(name, actual, expected; rtol)
    isapprox(actual, expected; rtol=rtol) && return true
    msg = "$name: got $actual, expected $expected (rtol = $rtol)"
    push!(CROSSCHECK_FAILURES, msg)
    @warn "cross-check failed — $msg"
    return false
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Backend wrappers — Univariate
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each function returns (time_seconds, fitted_result).
# We set very tight convergence tolerances (atol=-Inf, epsilon=1e-100,
# tol=1e-100) so that exactly `iters` EM steps are executed, regardless
# of convergence, for a fair timing comparison.

const BACKENDS = [
    "ExpectationMaximization.jl",
    "GaussianMixtures.jl",
    "mixtools.R",
    "sklearn.py",
]

function bench_EM_univ(y, mu, sigma, alpha; iters)
    K = length(mu)
    mix0 = MixtureModel([Normal(mu[i], sigma[i]) for i in 1:K], alpha)
    fit_mle(mix0, y; maxiter=iters, atol=-Inf)                       # warmup
    t = @belapsed fit_mle($mix0, $y; maxiter=$iters, atol=$(-Inf))
    res = fit_mle(mix0, y; maxiter=iters, atol=-Inf)
    return t, res
end

function bench_GMM_univ(y, mu, sigma, alpha; iters)
    K = length(mu)
    gmm0 = GMM(K, 1)
    gmm0.μ[:, 1] .= mu
    gmm0.Σ[:, 1] .= sigma        # sigma=[1,…] so σ²=σ here
    gmm0.w[:, 1] .= alpha
    Y = y[:, :]                   # N×1 matrix
    em!(copy(gmm0), Y, nIter=iters)                                  # warmup
    t = @belapsed em!(g, $Y, nIter=$iters) setup=(g = copy($gmm0))
    res = copy(gmm0); em!(res, Y, nIter=iters)
    return t, res
end

function bench_mixtools_univ(y, mu, sigma, alpha; iters)
    K = length(mu)
    @suppress R_normalmixEM(y, k=K, lambda=alpha, mu=mu, sigma=sigma,
                            maxit=iters, epsilon=1e-100)              # warmup
    t = @suppress @belapsed $R_normalmixEM($y, k=$K, lambda=$alpha,
                                           mu=$mu, sigma=$sigma,
                                           maxit=$iters, epsilon=$(1e-100))
    res = @suppress R_normalmixEM(y, k=K, lambda=alpha, mu=mu, sigma=sigma,
                                  maxit=iters, epsilon=1e-100)
    return t, res
end

function bench_sklearn_univ(y, mu, sigma, alpha; iters)
    K = length(mu)
    N = length(y)
    precisions_init = [1.0 / sigma[i]^2 for i in 1:K]
    Y    = reshape(y,               (N, 1))
    MU   = reshape(Float64.(mu),    (K, 1))
    prec = reshape(precisions_init, (K, 1))
    g0 = py_GaussianMixture(
        n_components    = K,
        covariance_type = "diag",
        weights_init    = Float64.(alpha),
        means_init      = MU,
        precisions_init = prec,
        max_iter        = iters,
        warm_start      = false,
        tol             = 1e-100,
    ).fit
    @suppress g0(Y)                                                   # warmup
    t = @suppress @belapsed g($Y) setup=(g = $g0)
    res = @suppress g0(Y)
    return t, res
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Backend wrappers — Multivariate
# ═══════════════════════════════════════════════════════════════════════════════
#
# Note on covariance estimation:
#   ExpectationMaximization.jl  — full covariance (MvNormal)
#   mixtools (R)                — full covariance (mvnormalmixEM default)
#   GaussianMixtures.jl         — diagonal covariance
#   sklearn                     — diagonal covariance (covariance_type="diag")
# All are initialized with identity covariance (diagonal=1), even though the
# *true* generating mixture below uses distinct, correlated covariances per
# component (see `ar1_cov`).

function bench_EM_mv(Y_DxN, mus, alpha; D, iters)
    K = length(mus)
    mix0 = MixtureModel([MvNormal(mus[k], Float64.(I(D))) for k in 1:K], alpha)
    fit_mle(mix0, Y_DxN; maxiter=iters, atol=-Inf)                   # warmup
    t = @belapsed fit_mle($mix0, $Y_DxN; maxiter=$iters, atol=$(-Inf))
    return t
end

function bench_GMM_mv(Y_NxD, mus_KxD, alpha; iters)
    K, D = size(mus_KxD)
    gmm0 = GMM(K, D)
    gmm0.μ .= mus_KxD
    # Set diagonal variances to 1 (identity covariance)
    for d in 1:D
        gmm0.Σ[:, d] .= 1.0
    end
    gmm0.w .= alpha
    em!(copy(gmm0), Y_NxD, nIter=iters)                              # warmup
    t = @belapsed em!(g, $Y_NxD, nIter=$iters) setup=(g = copy($gmm0))
    return t
end

function bench_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D, iters)
    # Set up data and initial parameters in R's global environment
    @rput Y_NxD mus_KxD K D alpha iters
    R"""
    r_mu  <- lapply(1:K, function(i) mus_KxD[i, ])
    r_sig <- lapply(1:K, function(i) diag(D))
    r_run_mvem <- function() {
        suppressWarnings(mixtools::mvnormalmixEM(
            Y_NxD, k = K, lambda = alpha,
            mu = r_mu, sigma = r_sig,
            maxit = iters, epsilon = 1e-100
        ))
    }
    """
    run_r() = R"r_run_mvem()"
    @suppress run_r()                                                 # warmup
    t = @suppress @belapsed $run_r()
    return t
end

function bench_sklearn_mv(Y_NxD, mus_KxD, alpha; K, D, iters)
    prec = ones(K, D)             # diagonal precision = 1 (variance = 1)
    g0 = py_GaussianMixture(
        n_components    = K,
        covariance_type = "diag",
        weights_init    = Float64.(alpha),
        means_init      = Float64.(mus_KxD),
        precisions_init = prec,
        max_iter        = iters,
        warm_start      = false,
        tol             = 1e-100,
    ).fit
    @suppress g0(Y_NxD)                                              # warmup
    t = @suppress @belapsed g($Y_NxD) setup=(g = $g0)
    return t
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Results collection
# ═══════════════════════════════════════════════════════════════════════════════

const Row = @NamedTuple{case::String, backend::String, K::Int, D::Int, N::Int, time_s::Float64}
results = Row[]

today = Dates.today()
mkpath("results")
const CSV_PATH = joinpath("results", "benchmark_timings_$(today).csv")
const CSV_LATEST = joinpath("results", "benchmark_timings_latest.csv")

# Called after every case rather than once at the end: the whole run takes hours, so a CI timeout
# or a failure in a later case would otherwise throw away every timing measured before it.
function save_results()
    open(CSV_PATH, "w") do f
        println(f, "case,backend,K,D,N,time_s")
        for r in results
            println(f, "$(r.case),$(r.backend),$(r.K),$(r.D),$(r.N),$(r.time_s)")
        end
    end
    cp(CSV_PATH, CSV_LATEST, force=true)   # fixed name, for CI and for plot_crosslang.jl
    return CSV_PATH
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Case 1: K=2, D=1 — Univariate Normal (with cross-checks)
# ═══════════════════════════════════════════════════════════════════════════════

println("=" ^ 70)
println("Case K2_D1 — K=2, D=1, Univariate Normal mixture")
println("=" ^ 70)

let
    # True parameters
    μ_true = [-4.0, 10.0]
    σ_true = [2.0, 10.0]
    α_true = [0.8, 0.2]
    mix_true = MixtureModel([Normal(μ_true[i], σ_true[i]) for i in 1:2], α_true)

    # Initial guess (shared across all backends)
    mu    = [-1.0, 1.0]
    sigma = [1.0, 1.0]
    alpha = [0.5, 0.5]
    iters = 8

    NN = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]

    @time "K2_D1" for N in NN
        @show N
        y = rand(StableRNG(123), mix_true, N)

        t_em,  res_em  = bench_EM_univ(y, mu, sigma, alpha; iters)
        t_gmm, res_gmm = bench_GMM_univ(y, mu, sigma, alpha; iters)
        t_R,   res_R   = bench_mixtools_univ(y, mu, sigma, alpha; iters)
        t_sk,  res_sk  = bench_sklearn_univ(y, mu, sigma, alpha; iters)

        push!(results, (case="K2_D1", backend=BACKENDS[1], K=2, D=1, N=N, time_s=t_em))
        push!(results, (case="K2_D1", backend=BACKENDS[2], K=2, D=1, N=N, time_s=t_gmm))
        push!(results, (case="K2_D1", backend=BACKENDS[3], K=2, D=1, N=N, time_s=t_R))
        push!(results, (case="K2_D1", backend=BACKENDS[4], K=2, D=1, N=N, time_s=t_sk))

        # ── Cross-checks: all backends should agree (mixtools is the reference) ───
        K = 2; rtol = 2e-2
        crosscheck("N=$N EM.jl μ", mean.(res_em.components), res_R[3][1:K]; rtol)
        crosscheck("N=$N EM.jl σ", std.(res_em.components), res_R[4][1:K]; rtol)
        crosscheck("N=$N EM.jl α", probs(res_em), res_R[2][1:K]; rtol)

        crosscheck("N=$N sklearn μ", vec(pyconvert(Matrix, res_sk.means_)), res_R[3][1:K]; rtol)
        crosscheck("N=$N sklearn σ", sqrt.(vec(pyconvert(Matrix, res_sk.covariances_))), res_R[4][1:K]; rtol)
        crosscheck("N=$N sklearn α", pyconvert(Vector, res_sk.weights_), res_R[2][1:K]; rtol)

        crosscheck("N=$N GMM.jl μ", vec(res_gmm.μ), res_R[3][1:K]; rtol)
        crosscheck("N=$N GMM.jl σ", sqrt.(vec(res_gmm.Σ)), res_R[4][1:K]; rtol)
        crosscheck("N=$N GMM.jl α", res_gmm.w, res_R[2][1:K]; rtol)

        println("  EM.jl=$(round(t_em, sigdigits=3))s  " *
                "GMM.jl=$(round(t_gmm, sigdigits=3))s  " *
                "R=$(round(t_R, sigdigits=3))s  " *
                "sklearn=$(round(t_sk, sigdigits=3))s")
    end
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Case 2: K=5, D=1 — Univariate Normal
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("Case K5_D1 — K=5, D=1, Univariate Normal mixture")
println("=" ^ 70)

let
    K = 5
    μ_true = [-8.0, -2.0, 3.0, 8.0, 15.0]
    σ_true = [2.0, 1.0, 3.0, 2.0, 4.0]
    α_true = [0.15, 0.25, 0.2, 0.25, 0.15]
    mix_true = MixtureModel([Normal(μ_true[i], σ_true[i]) for i in 1:K], α_true)

    mu    = [-5.0, 0.0, 5.0, 10.0, 18.0]
    sigma = [1.0, 1.0, 1.0, 1.0, 1.0]
    alpha = fill(1.0 / K, K)
    iters = 20

    NN = [500, 1_000, 5_000, 10_000, 50_000, 100_000]

    @time "K5_D1" for N in NN
        @show N
        y = rand(StableRNG(123), mix_true, N)

        t_em,  _ = bench_EM_univ(y, mu, sigma, alpha; iters)
        t_gmm, _ = bench_GMM_univ(y, mu, sigma, alpha; iters)
        t_R,   _ = bench_mixtools_univ(y, mu, sigma, alpha; iters)
        t_sk,  _ = bench_sklearn_univ(y, mu, sigma, alpha; iters)

        push!(results, (case="K5_D1", backend=BACKENDS[1], K=K, D=1, N=N, time_s=t_em))
        push!(results, (case="K5_D1", backend=BACKENDS[2], K=K, D=1, N=N, time_s=t_gmm))
        push!(results, (case="K5_D1", backend=BACKENDS[3], K=K, D=1, N=N, time_s=t_R))
        push!(results, (case="K5_D1", backend=BACKENDS[4], K=K, D=1, N=N, time_s=t_sk))

        println("  EM.jl=$(round(t_em, sigdigits=3))s  " *
                "GMM.jl=$(round(t_gmm, sigdigits=3))s  " *
                "R=$(round(t_R, sigdigits=3))s  " *
                "sklearn=$(round(t_sk, sigdigits=3))s")
    end
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Case 3: K=2, D=2 — Multivariate Normal
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("Case K2_D2 — K=2, D=2, Multivariate Normal mixture")
println("=" ^ 70)

let
    K = 2; D = 2
    μ₁ = [-3.0, 2.5]
    μ₂ = [4.0, -1.5]
    Σ₁ = ar1_cov(D, 1.8, 0.6)     # higher variance, positively correlated
    Σ₂ = ar1_cov(D, 1.0, -0.4)    # lower variance, negatively correlated
    α_true = [0.65, 0.35]
    mix_true = MixtureModel([MvNormal(μ₁, Σ₁), MvNormal(μ₂, Σ₂)], α_true)

    mus_guess = [[-1.0, 0.0], [1.0, 0.0]]
    mus_KxD   = vcat([m' for m in mus_guess]...)   # K×D matrix
    alpha     = [0.5, 0.5]
    iters     = 20

    NN = [500, 1_000, 5_000, 10_000, 50_000, 100_000]

    @time "K2_D2" for N in NN
        @show N
        Y_DxN = rand(StableRNG(123), mix_true, N)   # D×N
        Y_NxD = collect(Y_DxN')                      # N×D

        t_em  = bench_EM_mv(Y_DxN, mus_guess, alpha; D, iters)
        t_gmm = bench_GMM_mv(Y_NxD, mus_KxD, alpha; iters)
        t_R   = bench_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D, iters)
        t_sk  = bench_sklearn_mv(Y_NxD, mus_KxD, alpha; K, D, iters)

        push!(results, (case="K2_D2", backend=BACKENDS[1], K=K, D=D, N=N, time_s=t_em))
        push!(results, (case="K2_D2", backend=BACKENDS[2], K=K, D=D, N=N, time_s=t_gmm))
        push!(results, (case="K2_D2", backend=BACKENDS[3], K=K, D=D, N=N, time_s=t_R))
        push!(results, (case="K2_D2", backend=BACKENDS[4], K=K, D=D, N=N, time_s=t_sk))

        println("  EM.jl=$(round(t_em, sigdigits=3))s  " *
                "GMM.jl=$(round(t_gmm, sigdigits=3))s  " *
                "R=$(round(t_R, sigdigits=3))s  " *
                "sklearn=$(round(t_sk, sigdigits=3))s")
    end
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Case 4: K=2, D=10 — Multivariate Normal
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("Case K2_D10 — K=2, D=10, Multivariate Normal mixture")
println("=" ^ 70)

let
    K = 2; D = 10
    μ₁ = [-2.0 + 0.15d for d in 1:D]
    μ₂ = [ 3.0 - 0.10d for d in 1:D]
    Σ₁ = ar1_cov(D, 1.3, 0.5)     # higher variance, positively correlated
    Σ₂ = ar1_cov(D, 0.7, -0.3)    # lower variance, negatively correlated
    α_true = [0.6, 0.4]
    mix_true = MixtureModel([MvNormal(μ₁, Σ₁), MvNormal(μ₂, Σ₂)], α_true)

    mus_guess = [fill(-1.0, D), fill(1.0, D)]
    mus_KxD   = vcat([m' for m in mus_guess]...)
    alpha     = [0.5, 0.5]
    iters     = 20

    NN = [500, 1_000, 5_000, 10_000, 50_000]

    @time "K2_D10" for N in NN
        @show N
        Y_DxN = rand(StableRNG(123), mix_true, N)
        Y_NxD = collect(Y_DxN')

        t_em  = bench_EM_mv(Y_DxN, mus_guess, alpha; D, iters)
        t_gmm = bench_GMM_mv(Y_NxD, mus_KxD, alpha; iters)
        t_R   = bench_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D, iters)
        t_sk  = bench_sklearn_mv(Y_NxD, mus_KxD, alpha; K, D, iters)

        push!(results, (case="K2_D10", backend=BACKENDS[1], K=K, D=D, N=N, time_s=t_em))
        push!(results, (case="K2_D10", backend=BACKENDS[2], K=K, D=D, N=N, time_s=t_gmm))
        push!(results, (case="K2_D10", backend=BACKENDS[3], K=K, D=D, N=N, time_s=t_R))
        push!(results, (case="K2_D10", backend=BACKENDS[4], K=K, D=D, N=N, time_s=t_sk))

        println("  EM.jl=$(round(t_em, sigdigits=3))s  " *
                "GMM.jl=$(round(t_gmm, sigdigits=3))s  " *
                "R=$(round(t_R, sigdigits=3))s  " *
                "sklearn=$(round(t_sk, sigdigits=3))s")
    end
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 10. Case 5: K=5, D=15 — Multivariate Normal
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("Case K5_D15 — K=5, D=15, Multivariate Normal mixture")
println("=" ^ 70)

let
    K = 5; D = 15

    # Distinct means: separated cluster centers with a per-dimension gradient
    μs_true = [[3.0 * (k - (K + 1) / 2) + 0.1d for d in 1:D] for k in 1:K]
    # Distinct, correlated covariances per component
    Σs_true = [ar1_cov(D, 0.6 + 0.05k, (-1)^k * (0.2 + 0.03k)) for k in 1:K]
    alpha_raw = [1.0 + 0.05k for k in 1:K]
    α_true = alpha_raw ./ sum(alpha_raw)
    mix_true = MixtureModel([MvNormal(μs_true[k], Σs_true[k]) for k in 1:K], α_true)

    # Initial guess: shrunk toward zero, deliberately off from the truth
    mus_guess = [0.5 .* μs_true[k] for k in 1:K]
    mus_KxD   = vcat([m' for m in mus_guess]...)
    alpha     = fill(1.0 / K, K)
    iters     = 15

    NN = [5_000, 10_000, 50_000]

    case = "K$(K)_D$(D)"   # derived, so the label cannot drift from the actual K and D again

    @time case for N in NN
        @show N
        Y_DxN = rand(StableRNG(123), mix_true, N)
        Y_NxD = collect(Y_DxN')

        t_em  = bench_EM_mv(Y_DxN, mus_guess, alpha; D, iters)
        t_gmm = bench_GMM_mv(Y_NxD, mus_KxD, alpha; iters)
        t_R   = bench_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D, iters)
        t_sk  = bench_sklearn_mv(Y_NxD, mus_KxD, alpha; K, D, iters)

        push!(results, (case=case, backend=BACKENDS[1], K=K, D=D, N=N, time_s=t_em))
        push!(results, (case=case, backend=BACKENDS[2], K=K, D=D, N=N, time_s=t_gmm))
        push!(results, (case=case, backend=BACKENDS[3], K=K, D=D, N=N, time_s=t_R))
        push!(results, (case=case, backend=BACKENDS[4], K=K, D=D, N=N, time_s=t_sk))

        println("  EM.jl=$(round(t_em, sigdigits=3))s  " *
                "GMM.jl=$(round(t_gmm, sigdigits=3))s  " *
                "R=$(round(t_R, sigdigits=3))s  " *
                "sklearn=$(round(t_sk, sigdigits=3))s")
    end
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Save results to CSV
# ═══════════════════════════════════════════════════════════════════════════════

csv_path = save_results()   # already written after each case; this is the final, complete one
println("\n✓ $(length(results)) timings saved to $csv_path and $CSV_LATEST")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. Print summary table
# ═══════════════════════════════════════════════════════════════════════════════

println("\n" * "=" ^ 70)
println("Summary — EM.jl speed-up ratio  (backend_time / EM.jl_time)")
println("=" ^ 70)
for case in unique(r.case for r in results)
    println("\n  [$case]")
    rows = filter(r -> r.case == case, results)
    Ns = unique(r.N for r in rows)
    for N in Ns
        em_t = only(r.time_s for r in rows if r.N == N && r.backend == BACKENDS[1])
        ratios = join([
            begin
                bt = only(r.time_s for r in rows if r.N == N && r.backend == b)
                "$(rpad(b, 28)) $(round(bt / em_t, digits=1))×"
            end
            for b in BACKENDS[2:end]
        ], "   ")
        println("    N=$N:  $ratios")
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# 13. System information (reproducibility)
# ═══════════════════════════════════════════════════════════════════════════════

info_path = joinpath("results", "system_info_$(today).txt")
open(info_path, "w") do f
    redirect_stdout(f) do
        println("# Julia\n")
        InteractiveUtils.versioninfo()
        println("\nJulia threads      = $(Threads.nthreads())")
        println("OpenBLAS threads   = $(BLAS.get_num_threads())")

        println("\n# Julia packages\n")
        Pkg.status()

        println("\n# Python\n")
        sys = pyimport("sys")
        println("Python version         : $(pyconvert(String, sys.version))")
        sk = pyimport("sklearn")
        println("scikit-learn version   : $(pyconvert(String, sk.__version__))")
        np_mod = pyimport("numpy")
        println("numpy version          : $(pyconvert(String, np_mod.__version__))")

        println("\n# R\n")
        println(R"sessionInfo()")
    end
end
println("✓ System info saved to $info_path")

# ═══════════════════════════════════════════════════════════════════════════════
# 14. Re-raise cross-check failures
# ═══════════════════════════════════════════════════════════════════════════════
#
# Deliberately last: the timings and the system info are on disk by now, so a run that disagrees
# between backends still uploads its data, and still fails loudly.

if !isempty(CROSSCHECK_FAILURES)
    println("\n" * "=" ^ 70)
    error("$(length(CROSSCHECK_FAILURES)) cross-check(s) failed:\n  " *
          join(CROSSCHECK_FAILURES, "\n  "))
end
println("\n✓ All cross-checks passed")
