# ═══════════════════════════════════════════════════════════════════════════════
# Cross-Language EM Benchmark
# ExpectationMaximization.jl vs GaussianMixtures.jl vs mixtools (R) vs sklearn (Python)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Benchmarks Gaussian mixture EM across four backends with fixed initial
# conditions and iteration count for fair comparison.
#
# Every case cross-checks the backends against each other, on both halves of what makes a timing
# comparable (see §2.1):
#   - the fitted values  — same weights, means and variances, so the backends really did run the
#                          same algorithm on the same data from the same initial condition;
#   - the iteration count — same number of EM steps, so the times cover the same amount of work.
#
# EM.jl, GaussianMixtures.jl and sklearn are compared after every timed fit. mixtools is compared once
# per case, at the smallest N, with all of them run to convergence: its M-step differs from the
# textbook one, so its intermediate iterates are not comparable even though its fixed point is.
#
# Cases:
#   K2_D1    K=2,  D=1   Univariate Normal
#   K5_D1    K=5,  D=1   Univariate Normal
#   K2_D2    K=2,  D=2   Multivariate Normal (distinct, correlated covariances)
#   K2_D10   K=2,  D=10  Multivariate Normal (distinct, correlated covariances)
#
# Usage:
#   julia --threads=1 --project=benchmark benchmark/benchmark_crosslang.jl
#
# Output (rewritten after every case, so an interrupted run still leaves data):
#   benchmark/results/benchmark_timings_<date>.csv   case,backend,K,D,N,time_s,iters_asked,iters_run
#   benchmark/results/benchmark_timings_latest.csv
#   benchmark/results/benchmark_log_<date>.txt      everything printed below, including the
#   benchmark/results/benchmark_log_latest.txt      cross-checks — linked from docs/src/benchmarks.md
#   benchmark/results/system_info_<date>.txt
#
# Plot them with: julia --project=benchmark benchmark/plot_crosslang.jl

cd(@__DIR__)
import Pkg; Pkg.activate(".")

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Thread control — single thread everywhere for fair comparison
# ═══════════════════════════════════════════════════════════════════════════════

using LinearAlgebra: BLAS, Diagonal, I, diag, inv, UpperTriangular
BLAS.set_num_threads(1)

# Must be set before Python loads its BLAS
ENV["MKL_NUM_THREADS"] = "1"
ENV["NUMEXPR_NUM_THREADS"] = "1"
ENV["OMP_NUM_THREADS"] = "1"
ENV["OPENBLAS_NUM_THREADS"] = "1"
ENV["VECLIB_MAXIMUM_THREADS"] = "1"

# Checked rather than assumed, and checked *here* so a run started the wrong way fails in a second
# instead of after the hours it takes to find out from the numbers.
Threads.nthreads() == 1 ||
    error("start Julia with --threads=1: this benchmark compares single-threaded implementations, " *
          "and it is running with $(Threads.nthreads()) threads")

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

# The thread limits are the process-level `ENV` above, set before PythonCall starts the interpreter;
# Python inherits them. (Setting `os.environ` here instead would be too late — the libraries read
# their limits when they are imported, and `numpy` is already loaded by this line.) Read it back
# through Python rather than trusting that, since a silently multi-threaded numpy would make the
# whole comparison meaningless.
let omp = pyconvert(String, pyimport("os").environ.get("OMP_NUM_THREADS", ""))
    omp == "1" || error("Python sees OMP_NUM_THREADS = $(repr(omp)), expected \"1\"")
end

# sklearn only skips its KMeans initialisation when `weights_init`, `means_init` and
# `precisions_init` are *all* given, and that short-circuit lives in an override of
# `_initialize_parameters` on `GaussianMixture` itself. Without it every timed fit would first run a
# full KMeans over the data — several times the cost of the eight EM iterations of K2_D1, and
# invisible to every check in this script, because its result is then discarded.
let has_override = try
        pyconvert(Bool, py_GaussianMixture.__dict__.__contains__("_initialize_parameters"))
    catch err
        @warn "could not check sklearn's initialisation path" exception = err
        missing
    end
    has_override === false && error(
        "this scikit-learn ($(pyconvert(String, pyimport("sklearn").__version__))) does not " *
        "short-circuit its KMeans initialisation; the sklearn timings would include a KMeans fit"
    )
end

# `tol = 0` (see §3) means sklearn never sets `converged_`, so it raises a `ConvergenceWarning` on
# every single fit. That warning is expected here and says nothing we do not already record in
# `iters_run`, and Python writes it to its own stderr, which `@suppress` does not capture.
pyimport("warnings").filterwarnings(
    "ignore", category=pyimport("sklearn.exceptions").ConvergenceWarning
)

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

# `isapprox` on two arrays compares their *norms*: `norm(a - b) ≤ rtol * max(norm(a), norm(b))`. That
# gets weaker as the case gets bigger — with the 20 mean coordinates of K2_D10 flattened into one
# vector, a single coordinate could be off by about 10% and still pass at `rtol = 2e-2`, and the biggest
# cases are exactly the ones nobody inspects by hand. Compare element by element instead, each against
# its own magnitude or (for entries near zero, such as a mean coordinate that lands on the origin)
# against the largest entry of the same parameter.
_agree(a, b; rtol) = isapprox(a, b; rtol=rtol)
function _agree(a::AbstractArray, b::AbstractArray; rtol)
    length(a) == length(b) || return false
    atol = rtol * maximum(abs, b; init=zero(eltype(b)))
    return all(isapprox(x, y; rtol=rtol, atol=atol) for (x, y) in zip(a, b))
end

function crosscheck(name, actual, expected; rtol)
    _agree(actual, expected; rtol=rtol) && return true
    msg = "$name: got $actual, expected $expected (rtol = $rtol)"
    push!(CROSSCHECK_FAILURES, msg)
    say("  ⚠ cross-check failed — $msg")
    return false
end

# The tolerance every value cross-check uses, inherited from the earlier by-hand script
# (`benchmark_v2_K2_unidim.jl`). It is far looser than the agreement actually observed between the
# backends that run the same update — those match to about 1e-10 — so a failure at 2% means a real
# difference in what was computed, not accumulated rounding.
const RTOL = 2e-2

# ── A canonical fit summary ───────────────────────────────────────────────────
#
# Every backend spells its output differently — `mu`/`means_`/`μ`, standard deviations in R against
# variances everywhere else, a K×D matrix here against a list of D-vectors there. `Fit` is the one
# shape the cross-checks below actually compare, so they do not have to know any of that.
#
# Only the *diagonal* of each covariance is kept. All four fit a full covariance (§4), so the
# off-diagonal entries could be compared too; the diagonal is what the univariate and the
# multivariate cases have in common, and a disagreement large enough to matter shows up there.
const Fit = @NamedTuple{
    α::Vector{Float64},           # component weights
    μ::Vector{Vector{Float64}},   # one mean per component (a 1-vector in the univariate cases)
    σ²::Vector{Vector{Float64}},  # diagonal of each component covariance
    iters::Int,                   # EM iterations actually performed
}

_mean_vec(d::UnivariateDistribution) = [mean(d)]
_var_vec(d::UnivariateDistribution) = [var(d)]
_mean_vec(d::MultivariateDistribution) = collect(mean(d))
_var_vec(d::MultivariateDistribution) = collect(diag(cov(d)))

# ExpectationMaximization.jl — `infos = true` also returns the history, whose "iterations" is the
# number of EM steps performed. `atol = -Inf` means it can never stop early, but read the count
# rather than assume it: that is the whole point of the check.
function fit_EM(mix, history)::Fit
    dists = components(mix)
    return (α=collect(probs(mix)), μ=_mean_vec.(dists), σ²=_var_vec.(dists),
            iters=history["iterations"])
end

# GaussianMixtures.jl — `μ` is K×D. For a `:diag` GMM `Σ` is K×D and holds variances; for a `:full`
# one it is a vector of K `UpperTriangular`s holding the *inverse* Cholesky factor, which
# `GaussianMixtures.covar` turns back into a covariance matrix. `em!` has no convergence test at all
# and returns one average loglikelihood per iteration, so it always runs the `nIter` steps asked for.
#
# One asymmetry that `iters` cannot express, and that is left as it is rather than papered over:
# `em!` loops E, M with no trailing E-step, so `nIter` iterations are `nIter` E-steps and `nIter`
# M-steps. The other three all evaluate the loglikelihood of the parameters they end on, which costs
# one further E-step: EM.jl runs an E-step before its loop and then M, E per iteration
# (`src/classic_em.jl`), sklearn ends with `_e_step`, mixtools computes its initial `obsloglik`
# first. So at the same `iters`, GaussianMixtures.jl does one E-step less than everyone else —
# roughly 6% less work at `iters = 8`, 2.5% at `iters = 20`.
#
# Neither cross-check can see it: the trailing E-step updates no parameters, so the fitted values are
# identical either way, and `length(ll)` is `nIter` by construction. Asking `em!` for `nIter + 1`
# would buy an extra M-step along with the E-step and leave it fitting a different model, so the
# honest option is to report the difference rather than to hide it. It works against EM.jl, which is
# the harmless direction for a benchmark published by EM.jl.
function fit_GMM(gmm, ll)::Fit
    K = size(gmm.μ, 1)
    σ² = kind(gmm) == :diag ? [collect(gmm.Σ[k, :]) for k in 1:K] :
         [collect(diag(GaussianMixtures.covar(gmm.Σ[k]))) for k in 1:K]
    return (α=collect(vec(gmm.w)), μ=[collect(gmm.μ[k, :]) for k in 1:K],
            σ²=σ², iters=length(ll))
end

# mixtools — `all.loglik` holds the loglikelihood before the first iteration as well, hence the -1.
_mixtools_iters(res) = length(rcopy(res[Symbol("all.loglik")])) - 1

# `normalmixEM` returns `sigma` as standard deviations, one scalar per component.
function fit_mixtools_univ(res)::Fit
    return (α=collect(Float64, rcopy(res[:lambda])),
            μ=[[m] for m in rcopy(res[:mu])],
            σ²=[[s^2] for s in rcopy(res[:sigma])],
            iters=_mixtools_iters(res))
end

# `mvnormalmixEM` returns `mu` as a list of D-vectors and `sigma` as a list of D×D covariance
# matrices; `rcopy` gives those back as a `Vector{Any}`, hence the element-wise conversions.
function fit_mixtools_mv(res)::Fit
    return (α=collect(Float64, rcopy(res[:lambda])),
            μ=[collect(Float64, m) for m in rcopy(res[:mu])],
            σ²=[collect(Float64, diag(S)) for S in rcopy(res[:sigma])],
            iters=_mixtools_iters(res))
end

# sklearn — `means_` is K×D. `covariances_` is K×D holding variances under
# `covariance_type = "diag"`, and K×D×D holding covariance matrices under `"full"`. `n_iter_` is the
# number of EM steps performed.
function fit_sklearn(res)::Fit
    M = pyconvert(Matrix{Float64}, res.means_)
    C = pyconvert(Array{Float64}, res.covariances_)
    K = size(M, 1)
    σ² = ndims(C) == 2 ? [collect(C[k, :]) for k in 1:K] :
         [collect(diag(C[k, :, :])) for k in 1:K]
    return (α=pyconvert(Vector{Float64}, res.weights_),
            μ=[collect(M[k, :]) for k in 1:K], σ²=σ²,
            iters=pyconvert(Int, res.n_iter_))
end

# Nothing forces two backends to return the components in the same order, and a permuted mixture is
# the same mixture — so compare them in a canonical order instead. Sorting on the first coordinate of
# the mean is enough: every case below has components that differ there.
function canonical(f::Fit)::Fit
    p = sortperm(f.μ; by=first)
    return (α=f.α[p], μ=f.μ[p], σ²=f.σ²[p], iters=f.iters)
end

# The means and the variances are flattened over the components, so one failure message shows the
# whole parameter rather than the first component that disagrees.
function compare_fits(label, f::Fit, ref::Fit; rtol, what=(:α, :μ, :σ²))
    a, b = canonical(f), canonical(ref)
    :α in what && crosscheck("$label α", a.α, b.α; rtol=rtol)
    :μ in what && crosscheck("$label μ", reduce(vcat, a.μ), reduce(vcat, b.μ); rtol=rtol)
    :σ² in what && crosscheck("$label σ²", reduce(vcat, a.σ²), reduce(vcat, b.σ²); rtol=rtol)
    return nothing
end

# ── What can be compared where ────────────────────────────────────────────────
#
# EM.jl, GaussianMixtures.jl and sklearn perform the textbook EM update and agree with each other to
# about 1e-10 at every iteration, so they are compared after each timed fit, at `RTOL`.
#
# mixtools cannot be compared that way. Its M-step is not the textbook one: given identical inputs
# and a *single* EM step it returns the same means but different weights and variances (measured — at
# K=5, D=1, one step from the shared initial condition, the other three agree to 1e-10 among
# themselves and mixtools' σ differs in the second digit). The difference is not an off-by-one in the
# iteration count and not a convergence tolerance; and it is not a bug in mixtools either, because at
# its *fixed point* everything is consistent again — there `lambda` does equal `colMeans(posterior)`
# and the four backends agree.
#
# So mixtools is checked at the fixed point instead, once per case, with everyone run to convergence
# (`crosscheck_converged` below). That is also the only regime in which the old `epsilon = 1e-100`
# made the comparison pass: it let mixtools converge, at the cost of the timings being taken over a
# different number of iterations.
# A backend that did not run at this N (too slow, or no like-for-like fit in this case) is `nothing`
# and simply has nothing to compare.
function crosscheck_fixed_iters(case, N, (f_em, f_gmm, f_R, f_sk))
    isnothing(f_em) && return nothing
    isnothing(f_gmm) || compare_fits("$case N=$N GMM.jl vs EM.jl", f_gmm, f_em; rtol=RTOL)
    isnothing(f_sk) || compare_fits("$case N=$N sklearn vs EM.jl", f_sk, f_em; rtol=RTOL)
    return nothing
end

# Convergence settings for that fixed-point check. `maxiter` is a safety net, not a target: K5_D1
# needs a few thousand iterations because its components overlap, the other cases a few dozen.
const CONVERGED_EPSILON = 1e-8
const CONVERGED_MAXITER = 10_000

# `mix0` is the same initial mixture the timed fits start from, and `f_R` a converged mixtools fit of
# the same data. Both are run to their own convergence, so this compares fixed points, not iterates.
function crosscheck_converged(case, mix0, y, f_R::Fit)
    mix, history = fit_mle(mix0, y; maxiter=CONVERGED_MAXITER, atol=CONVERGED_EPSILON, infos=true)
    if !history["converged"]
        msg = "$case: EM.jl did not converge in $CONVERGED_MAXITER iterations, so this is not a " *
              "fixed-point comparison"
        push!(CROSSCHECK_FAILURES, msg)
        say("  ⚠ ", msg)
    end
    compare_fits("$case converged EM.jl vs mixtools", fit_EM(mix, history), f_R; rtol=RTOL)
    say("  fixed-point check: EM.jl $(history["iterations"]) iterations, " *
            "mixtools $(f_R.iters)")
    return nothing
end

# The converged mixtools fits themselves. Not timed, and run once per case rather than once per N.
function converged_mixtools_univ(y, mu, sigma, alpha)
    res = @suppress R_normalmixEM(y, k=length(mu), lambda=alpha, mu=mu, sigma=sigma,
                                  maxit=CONVERGED_MAXITER, epsilon=CONVERGED_EPSILON)
    return fit_mixtools_univ(res)
end

function converged_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D)
    @rput Y_NxD mus_KxD K D alpha
    res = @suppress R"""
    suppressWarnings(mixtools::mvnormalmixEM(
        Y_NxD, k = K, lambda = alpha,
        mu = lapply(1:K, function(i) mus_KxD[i, ]),
        sigma = lapply(1:K, function(i) diag(D)),
        maxit = $CONVERGED_MAXITER, epsilon = $CONVERGED_EPSILON))
    """
    return fit_mixtools_mv(res)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Backend wrappers — Univariate
# ═══════════════════════════════════════════════════════════════════════════════
#
# Each function returns (time_seconds, fit::Fit).
#
# Every backend is given a convergence criterion it can never meet, so that exactly `iters` EM steps
# are executed whether or not the fit has converged:
#
#   ExpectationMaximization.jl   atol = -Inf    `|Δℓ| < -Inf` is never true
#   GaussianMixtures.jl          (nothing)      `em!` has no convergence test at all
#   mixtools                     epsilon = -1   the loop is `while (diff > epsilon)` on the *signed*
#                                               increase, so a negative epsilon is out of reach
#   sklearn                      tol = 0.0      `abs(change) < 0` is never true
#
# A merely *tight* tolerance does not do it, which is the trap this code was in: both mixtools and
# sklearn stop as soon as the loglikelihood increase reaches exactly 0, and a fit that has converged
# to machine precision reaches it. With `epsilon = tol = 1e-100` and 40 requested iterations on a
# well-separated bivariate sample, `mvnormalmixEM` ran 5 and sklearn 7.
#
# Two traps in the values themselves: `epsilon = -Inf` does not work for `normalmixEM`, which starts
# from `diff <- epsilon + 1` and would compare `-Inf > -Inf`, and a negative `tol` is refused by
# sklearn's parameter validation. Hence a finite negative number on one side and exactly 0 on the
# other. `note_iterations` in §5 checks the outcome at every point rather than trusting it.
#
# The `Fit` the cross-checks use is the *warm-up* fit, not an extra one: every backend is
# deterministic here (same data, same initial condition, fixed iteration count), so the warm-up call
# produces exactly the fit a later call would, and at the largest N a spare `mvnormalmixEM` would
# cost minutes on its own.

const BACKENDS = [
    "ExpectationMaximization.jl",
    "GaussianMixtures.jl",
    "mixtools.R",
    "sklearn.py",
]

const MIXTOOLS_EPSILON = -1.0   # negative, and finite: see the note above
const SKLEARN_TOL = 0.0         # exactly zero: sklearn refuses a negative tolerance

function bench_EM_univ(y, mu, sigma, alpha; iters)
    K = length(mu)
    mix0 = MixtureModel([Normal(mu[i], sigma[i]) for i in 1:K], alpha)
    mix, history = fit_mle(mix0, y; maxiter=iters, atol=-Inf, infos=true)   # warmup
    t = @belapsed fit_mle($mix0, $y; maxiter=$iters, atol=$(-Inf))
    return t, fit_EM(mix, history)
end

function bench_GMM_univ(y, mu, sigma, alpha; iters)
    K = length(mu)
    gmm0 = GMM(K, 1)
    gmm0.μ[:, 1] .= mu
    gmm0.Σ[:, 1] .= sigma        # sigma=[1,…] so σ²=σ here
    gmm0.w[:, 1] .= alpha
    Y = y[:, :]                   # N×1 matrix
    res = copy(gmm0); ll = em!(res, Y, nIter=iters)                   # warmup
    t = @belapsed em!(g, $Y, nIter=$iters) setup=(g = copy($gmm0))
    return t, fit_GMM(res, ll)
end

function bench_mixtools_univ(y, mu, sigma, alpha; iters)
    K = length(mu)
    # The data is pushed into R *once*, outside the timed region, and the timed expression only calls
    # an R function that is already holding it — the same shape as `bench_mixtools_mv` below. Calling
    # `R_normalmixEM($y, ...)` directly instead would copy the whole sample into a fresh `SEXP` on
    # every repetition (4 MB at N = 5·10⁵), work no other backend is charged for.
    @rput y mu sigma alpha K iters
    R"""
    r_run_em <- function() {
        suppressWarnings(mixtools::normalmixEM(y, k = K, lambda = alpha, mu = mu, sigma = sigma,
                                               maxit = iters, epsilon = $MIXTOOLS_EPSILON))
    }
    """
    run_r() = R"r_run_em()"
    res = @suppress run_r()                                           # warmup
    t = @suppress @belapsed $run_r()
    return t, fit_mixtools_univ(res)
end

function bench_sklearn_univ(y, mu, sigma, alpha; iters)
    K = length(mu)
    N = length(y)
    precisions_init = [1.0 / sigma[i]^2 for i in 1:K]
    # A *numpy* array, built once here rather than per repetition. Handing the Julia array straight
    # to `fit` passes a `juliacall.ArrayValue`, which sklearn`s `check_array` has to run through the
    # buffer protocol on every call — and a Julia matrix reaches numpy in Fortran order, which is not
    # the layout sklearn`s code paths are written for. Neither cost is paid by the other backends,
    # which are handed an array they can use as-is.
    Y    = np.ascontiguousarray(np.asarray(reshape(y, (N, 1))))
    MU   = reshape(Float64.(mu),    (K, 1))
    prec = reshape(precisions_init, (K, 1))
    g0 = py_GaussianMixture(
        n_components    = K,
        covariance_type = "diag",
        weights_init    = Float64.(alpha),
        means_init      = MU,
        precisions_init = prec,
        max_iter        = iters,
        n_init          = 1,
        warm_start      = false,
        tol             = SKLEARN_TOL,
    ).fit
    res = @suppress g0(Y)                                             # warmup
    t = @suppress @belapsed g($Y) setup=(g = $g0)
    return t, fit_sklearn(res)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Backend wrappers — Multivariate
# ═══════════════════════════════════════════════════════════════════════════════
#
# All four estimate a **full** covariance per component:
#   ExpectationMaximization.jl  — a `FullNormal` component (see `initial_mixture_mv`)
#   mixtools (R)                — `mvnormalmixEM`, its only mode
#   GaussianMixtures.jl         — `GMM(K, D; kind = :full)`
#   sklearn                     — `covariance_type = "full"`
# Three of them had to be told: GaussianMixtures.jl and sklearn default to a diagonal covariance, and
# EM.jl fitted one too because its components were built over `Float64.(I(D))`, a `Diagonal`. Only
# mixtools was fitting what this benchmark claimed to compare. A diagonal fit is a different (here
# misspecified — the true covariances are correlated, see `ar1_cov`) problem with `D` free parameters
# per component instead of `D(D+1)/2`, and it is much cheaper per iteration, O(D) against O(D²) per
# observation, so mixing the two silently flatters whichever backends were on the cheap side.
#
# Only the multivariate cases are affected; in the univariate ones every covariance structure is the
# same single number, which is why `fit_GMM` and `fit_sklearn` still read both layouts.
#
# All are initialized with identity covariance, even though the *true* generating mixture below uses
# distinct, correlated covariances per component.

# The initial mixture every EM.jl fit of a multivariate case starts from — shared with the
# fixed-point check, so the two always start from the same place.
function initial_mixture_mv(mus, alpha; D)
    K = length(mus)
    # `Matrix(1.0I, D, D)`, not `Float64.(I(D))`: the latter is a `Diagonal`, which makes the
    # component a `DiagNormal`, and EM.jl's `fit_mle(g::DiagNormal, args...)` then estimates a
    # *diagonal* covariance — a different, cheaper model than the one the other backends fit, and not
    # the `fit_mle(d::FullNormal, y, w)` method in `src/specialized.jl` this case is meant to measure.
    return MixtureModel([MvNormal(mus[k], Matrix(1.0I, D, D)) for k in 1:K], alpha)
end

function bench_EM_mv(Y_DxN, mus, alpha; D, iters)
    mix0 = initial_mixture_mv(mus, alpha; D)
    mix, history = fit_mle(mix0, Y_DxN; maxiter=iters, atol=-Inf, infos=true)   # warmup
    t = @belapsed fit_mle($mix0, $Y_DxN; maxiter=$iters, atol=$(-Inf))
    return t, fit_EM(mix, history)
end

function bench_GMM_mv(Y_NxD, mus_KxD, alpha; iters)
    K, D = size(mus_KxD)
    gmm0 = GMM(K, D; kind=:full)
    gmm0.μ .= mus_KxD
    # A `:full` GMM stores the inverse Cholesky factor of each covariance, and the inverse Cholesky
    # factor of the identity is the identity — so this is the same identity start as the others.
    gmm0.Σ = [UpperTriangular(Matrix(1.0I, D, D)) for _ in 1:K]
    gmm0.w .= alpha
    res = copy(gmm0); ll = em!(res, Y_NxD, nIter=iters)               # warmup
    t = @belapsed em!(g, $Y_NxD, nIter=$iters) setup=(g = copy($gmm0))
    return t, fit_GMM(res, ll)
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
            maxit = iters, epsilon = $MIXTOOLS_EPSILON
        ))
    }
    """
    run_r() = R"r_run_mvem()"
    res = @suppress run_r()                                           # warmup
    t = @suppress @belapsed $run_r()
    return t, fit_mixtools_mv(res)
end

function bench_sklearn_mv(Y_NxD, mus_KxD, alpha; K, D, iters)
    # One D×D precision matrix per component, the identity, i.e. identity covariance.
    prec = zeros(K, D, D)
    for k in 1:K, d in 1:D
        prec[k, d, d] = 1.0
    end
    g0 = py_GaussianMixture(
        n_components    = K,
        covariance_type = "full",
        weights_init    = Float64.(alpha),
        means_init      = Float64.(mus_KxD),
        precisions_init = prec,
        max_iter        = iters,
        n_init          = 1,
        warm_start      = false,
        tol             = SKLEARN_TOL,
    ).fit
    Ypy = np.ascontiguousarray(np.asarray(Y_NxD))   # see `bench_sklearn_univ`
    res = @suppress g0(Ypy)                                          # warmup
    t = @suppress @belapsed g($Ypy) setup=(g = $g0)
    return t, fit_sklearn(res)
end

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Results collection
# ═══════════════════════════════════════════════════════════════════════════════

const Row = @NamedTuple{case::String, backend::String, K::Int, D::Int, N::Int, time_s::Float64,
                        iters_asked::Int, iters_run::Int}
results = Row[]

# Not a `crosscheck`: a backend that stops early is not *wrong*, it has simply done less work than
# the others, which is a statement about the timing rather than about the fit.
#
# With the criteria §3 gives them, none of the four can stop before `iters`, so this should never
# fire. It stays because that guarantee lives in four other projects: a future mixtools or sklearn
# may grow another way out of its EM loop (a maximum-restart rule, a relative criterion, a check on
# the parameters rather than on the loglikelihood), and a benchmark that silently compares 20
# iterations against 5 is worse than one that fails. The count goes into the CSV next to the timing,
# is warned about as it happens, and is listed again at the end of the run.
const ITERATION_NOTES = String[]

function note_iterations(case, N, backend, run, asked)
    run == asked && return run
    msg = "$case N=$N: $backend ran $run of the $asked requested EM iterations"
    push!(ITERATION_NOTES, msg)
    say("  ⚠ ", msg)
    return run
end

# One row per backend. `times` and `fits` are in `BACKENDS` order, and a backend that did not run at
# this N contributes `nothing` and no row.
function record!(case, K, D, N, iters, times, fits)
    for (backend, t, f) in zip(BACKENDS, times, fits)
        isnothing(t) && continue
        note_iterations(case, N, backend, f.iters, iters)
        push!(results, (case=case, backend=backend, K=K, D=D, N=N, time_s=t,
                        iters_asked=iters, iters_run=f.iters))
    end
    return nothing
end

# ── Dropping a backend that has become too slow ───────────────────────────────
#
# A single fit above this bound means the next N — larger by a factor of 2 to 10 — costs minutes or
# hours, and `@belapsed` pays for it more than once. Only the offending backend stops; the others
# keep their full curve, and both figures already handle a backend that is missing some points.
# mixtools is the one that hits this in practice: at K=2, D=2 its fit takes about 90 s at N = 5000.
const SLOW_BACKEND_LIMIT = 100.0   # seconds

# `fit` is a closure returning `(time, ::Fit)`; `slow` is the set of backends already dropped, which
# each case starts empty (a backend too slow at D = 2 may be perfectly fast in the next case).
function run_backend(fit, backend, slow)
    backend in slow && return (nothing, nothing)
    t, f = fit()
    if t > SLOW_BACKEND_LIMIT
        push!(slow, backend)
        say("  ↓ $backend took $(round(t, sigdigits=3))s, over the $(SLOW_BACKEND_LIMIT)s limit; " *
            "skipping it for the larger N of this case")
    end
    return t, f
end

_fmt_time(t) = isnothing(t) ? "skipped" : "$(round(t, sigdigits=3))s"

today = Dates.today()
mkpath("results")
const CSV_PATH = joinpath("results", "benchmark_timings_$(today).csv")
const CSV_LATEST = joinpath("results", "benchmark_timings_latest.csv")

# ── The run transcript ────────────────────────────────────────────────────────
#
# Everything this script reports goes through `say`, to the terminal and to two log files at once:
# one stamped with the date, one at a fixed name for the documentation to link to. The CSV holds the
# timings, but the cross-check results, the iteration counts and the per-case totals live only in the
# printed output, and those are what tell a reader whether a set of timings can be trusted.
#
# Warnings go through `say` too rather than through `@warn`, whose output goes to stderr and would be
# missing from the file. Each line is flushed: the run takes hours and a crash must not take the
# transcript with it.
const LOG_PATH = joinpath("results", "benchmark_log_$(today).txt")
const LOG_LATEST = joinpath("results", "benchmark_log_latest.txt")
const LOG_IOS = [open(LOG_PATH, "w"), open(LOG_LATEST, "w")]

function say(args...)
    println(stdout, args...)
    for io in LOG_IOS
        println(io, args...)
        flush(io)
    end
    return nothing
end

say("Cross-language EM benchmark — ", Dates.format(Dates.now(), "yyyy-mm-dd HH:MM"))
say("Julia threads $(Threads.nthreads()), BLAS threads $(BLAS.get_num_threads()); ",
    "full version details in $(joinpath("results", "system_info_$(today).txt"))")

# Called after every case rather than once at the end: the whole run takes hours, so a CI timeout
# or a failure in a later case would otherwise throw away every timing measured before it.
function save_results()
    open(CSV_PATH, "w") do f
        println(f, "case,backend,K,D,N,time_s,iters_asked,iters_run")
        for r in results
            println(f, "$(r.case),$(r.backend),$(r.K),$(r.D),$(r.N),$(r.time_s)," *
                       "$(r.iters_asked),$(r.iters_run)")
        end
    end
    cp(CSV_PATH, CSV_LATEST, force=true)   # fixed name, for CI and for plot_crosslang.jl
    return CSV_PATH
end

# ═══════════════════════════════════════════════════════════════════════════════
# 6. Case 1: K=2, D=1 — Univariate Normal (with cross-checks)
# ═══════════════════════════════════════════════════════════════════════════════

say("=" ^ 70)
say("Case K2_D1 — K=2, D=1, Univariate Normal mixture")
say("=" ^ 70)

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

    NN = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]

    slow = Set{String}()   # backends dropped for this case once they pass the time limit
    stats = @timed for N in NN
        say("N = $N")
        y = rand(StableRNG(123), mix_true, N)

        t_em,  f_em  = run_backend(() -> bench_EM_univ(y, mu, sigma, alpha; iters), BACKENDS[1], slow)
        t_gmm, f_gmm = run_backend(() -> bench_GMM_univ(y, mu, sigma, alpha; iters), BACKENDS[2], slow)
        t_R,   f_R   = run_backend(() -> bench_mixtools_univ(y, mu, sigma, alpha; iters), BACKENDS[3], slow)
        t_sk,  f_sk  = run_backend(() -> bench_sklearn_univ(y, mu, sigma, alpha; iters), BACKENDS[4], slow)

        fits = (f_em, f_gmm, f_R, f_sk)
        record!("K2_D1", 2, 1, N, iters, (t_em, t_gmm, t_R, t_sk), fits)
        crosscheck_fixed_iters("K2_D1", N, fits)
        if N == first(NN)   # once per case: the only regime in which mixtools is comparable
            mix0 = MixtureModel([Normal(mu[i], sigma[i]) for i in 1:2], alpha)
            crosscheck_converged("K2_D1", mix0, y, converged_mixtools_univ(y, mu, sigma, alpha))
        end

        say("  EM.jl=$(_fmt_time(t_em))  GMM.jl=$(_fmt_time(t_gmm))  " *
                "R=$(_fmt_time(t_R))  sklearn=$(_fmt_time(t_sk))")
    end
    say("  [", "K2_D1", "] ", round(stats.time, digits=1), " s, ", Base.format_bytes(stats.bytes), " allocated")
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 7. Case 2: K=5, D=1 — Univariate Normal
# ═══════════════════════════════════════════════════════════════════════════════

say("\n" * "=" ^ 70)
say("Case K5_D1 — K=5, D=1, Univariate Normal mixture")
say("=" ^ 70)

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

    slow = Set{String}()   # backends dropped for this case once they pass the time limit
    stats = @timed for N in NN
        say("N = $N")
        y = rand(StableRNG(123), mix_true, N)

        t_em,  f_em  = run_backend(() -> bench_EM_univ(y, mu, sigma, alpha; iters), BACKENDS[1], slow)
        t_gmm, f_gmm = run_backend(() -> bench_GMM_univ(y, mu, sigma, alpha; iters), BACKENDS[2], slow)
        t_R,   f_R   = run_backend(() -> bench_mixtools_univ(y, mu, sigma, alpha; iters), BACKENDS[3], slow)
        t_sk,  f_sk  = run_backend(() -> bench_sklearn_univ(y, mu, sigma, alpha; iters), BACKENDS[4], slow)

        fits = (f_em, f_gmm, f_R, f_sk)
        record!("K5_D1", K, 1, N, iters, (t_em, t_gmm, t_R, t_sk), fits)
        crosscheck_fixed_iters("K5_D1", N, fits)
        if N == first(NN)   # once per case: the only regime in which mixtools is comparable
            mix0 = MixtureModel([Normal(mu[i], sigma[i]) for i in 1:K], alpha)
            crosscheck_converged("K5_D1", mix0, y, converged_mixtools_univ(y, mu, sigma, alpha))
        end

        say("  EM.jl=$(_fmt_time(t_em))  GMM.jl=$(_fmt_time(t_gmm))  " *
                "R=$(_fmt_time(t_R))  sklearn=$(_fmt_time(t_sk))")
    end
    say("  [", "K5_D1", "] ", round(stats.time, digits=1), " s, ", Base.format_bytes(stats.bytes), " allocated")
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 8. Case 3: K=2, D=2 — Multivariate Normal
# ═══════════════════════════════════════════════════════════════════════════════

say("\n" * "=" ^ 70)
say("Case K2_D2 — K=2, D=2, Multivariate Normal mixture")
say("=" ^ 70)

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

    NN = [500, 1_000, 5_000, 10_000, 25_000]

    slow = Set{String}()   # backends dropped for this case once they pass the time limit
    stats = @timed for N in NN
        say("N = $N")
        Y_DxN = rand(StableRNG(123), mix_true, N)   # D×N
        Y_NxD = collect(Y_DxN')                      # N×D

        t_em,  f_em  = run_backend(() -> bench_EM_mv(Y_DxN, mus_guess, alpha; D, iters), BACKENDS[1], slow)
        t_gmm, f_gmm = run_backend(() -> bench_GMM_mv(Y_NxD, mus_KxD, alpha; iters), BACKENDS[2], slow)
        t_R,   f_R   = run_backend(() -> bench_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D, iters), BACKENDS[3], slow)
        t_sk,  f_sk  = run_backend(() -> bench_sklearn_mv(Y_NxD, mus_KxD, alpha; K, D, iters), BACKENDS[4], slow)

        fits = (f_em, f_gmm, f_R, f_sk)
        record!("K2_D2", K, D, N, iters, (t_em, t_gmm, t_R, t_sk), fits)
        crosscheck_fixed_iters("K2_D2", N, fits)
        if N == first(NN)
            mix0 = initial_mixture_mv(mus_guess, alpha; D)
            crosscheck_converged("K2_D2", mix0, Y_DxN,
                                 converged_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D))
        end

        say("  EM.jl=$(_fmt_time(t_em))  GMM.jl=$(_fmt_time(t_gmm))  " *
                "R=$(_fmt_time(t_R))  sklearn=$(_fmt_time(t_sk))")
    end
    say("  [", "K2_D2", "] ", round(stats.time, digits=1), " s, ", Base.format_bytes(stats.bytes), " allocated")
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 9. Case 4: K=2, D=10 — Multivariate Normal
# ═══════════════════════════════════════════════════════════════════════════════

say("\n" * "=" ^ 70)
say("Case K2_D10 — K=2, D=10, Multivariate Normal mixture")
say("=" ^ 70)

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

    NN = [500, 1_000, 5_000, 10_000, 20_000]

    slow = Set{String}()   # backends dropped for this case once they pass the time limit
    stats = @timed for N in NN
        say("N = $N")
        Y_DxN = rand(StableRNG(123), mix_true, N)
        Y_NxD = collect(Y_DxN')

        t_em,  f_em  = run_backend(() -> bench_EM_mv(Y_DxN, mus_guess, alpha; D, iters), BACKENDS[1], slow)
        t_gmm, f_gmm = run_backend(() -> bench_GMM_mv(Y_NxD, mus_KxD, alpha; iters), BACKENDS[2], slow)
        t_R,   f_R   = run_backend(() -> bench_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D, iters), BACKENDS[3], slow)
        t_sk,  f_sk  = run_backend(() -> bench_sklearn_mv(Y_NxD, mus_KxD, alpha; K, D, iters), BACKENDS[4], slow)

        fits = (f_em, f_gmm, f_R, f_sk)
        record!("K2_D10", K, D, N, iters, (t_em, t_gmm, t_R, t_sk), fits)
        crosscheck_fixed_iters("K2_D10", N, fits)
        if N == first(NN)
            mix0 = initial_mixture_mv(mus_guess, alpha; D)
            crosscheck_converged("K2_D10", mix0, Y_DxN,
                                 converged_mixtools_mv(Y_NxD, mus_KxD, alpha; K, D))
        end

        say("  EM.jl=$(_fmt_time(t_em))  GMM.jl=$(_fmt_time(t_gmm))  " *
                "R=$(_fmt_time(t_R))  sklearn=$(_fmt_time(t_sk))")
    end
    say("  [", "K2_D10", "] ", round(stats.time, digits=1), " s, ", Base.format_bytes(stats.bytes), " allocated")
    save_results()   # so an interrupted run still leaves the cases that did finish
end

# ═══════════════════════════════════════════════════════════════════════════════
# 11. Save results to CSV
# ═══════════════════════════════════════════════════════════════════════════════

csv_path = save_results()   # already written after each case; this is the final, complete one
say("\n✓ $(length(results)) timings saved to $csv_path and $CSV_LATEST")
say("✓ this transcript is saved to $LOG_PATH and $LOG_LATEST")

# ═══════════════════════════════════════════════════════════════════════════════
# 12. Print summary table
# ═══════════════════════════════════════════════════════════════════════════════

say("\n" * "=" ^ 70)
say("Summary — EM.jl speed-up ratio  (backend_time / EM.jl_time)")
say("=" ^ 70)
for case in unique(r.case for r in results)
    say("\n  [$case]")
    rows = filter(r -> r.case == case, results)
    Ns = unique(r.N for r in rows)
    # A backend may be missing a row: dropped for being too slow at a smaller N, or sitting the case
    # out entirely. `at` returns `nothing` there rather than throwing.
    at(b, N) = (m = filter(r -> r.N == N && r.backend == b, rows); isempty(m) ? nothing : only(m))
    for N in Ns
        em = at(BACKENDS[1], N)
        isnothing(em) && continue
        # A backend that stopped early is flagged inline: its ratio then compares a shorter run
        # against EM.jl's full one, so it is not the like-for-like number the column suggests.
        ratios = join([
            begin
                short = row.iters_run == row.iters_asked ? "" :
                        " [$(row.iters_run)/$(row.iters_asked) iters]"
                "$(rpad(row.backend, 28)) $(round(row.time_s / em.time_s, digits=1))×$short"
            end
            for row in filter(!isnothing, [at(b, N) for b in BACKENDS[2:end]])
        ], "   ")
        say("    N=$N:  $ratios")
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
say("✓ System info saved to $info_path")

# ═══════════════════════════════════════════════════════════════════════════════
# 14. Iteration-count report, then re-raise cross-check failures
# ═══════════════════════════════════════════════════════════════════════════════
#
# Deliberately last: the timings and the system info are on disk by now, so a run that disagrees
# between backends still uploads its data, and still fails loudly.
#
# The iteration counts are reported but do not fail the run — unlike a disagreement on the fitted
# values, a backend stopping early is a property of its convergence test, not a bug to fix here.
# Where it happens, `time_s / iters_run` is the comparable quantity rather than `time_s`.

if isempty(ITERATION_NOTES)
    say("\n✓ Every backend ran exactly the requested number of EM iterations")
else
    say("\n" * "=" ^ 70)
    say("Backends that stopped before the requested number of iterations")
    say("=" ^ 70)
    for note in ITERATION_NOTES
        say("  ", note)
    end
end

if !isempty(CROSSCHECK_FAILURES)
    say("\n" * "=" ^ 70)
    error("$(length(CROSSCHECK_FAILURES)) cross-check(s) failed:\n  " *
          join(CROSSCHECK_FAILURES, "\n  "))
end
say("\n✓ All cross-checks passed")
