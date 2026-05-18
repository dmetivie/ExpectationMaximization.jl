using BenchmarkTools
using Distributions
using ExpectationMaximization
using MLDatasets: MNIST
using StableRNGs
using LinearAlgebra: BLAS, I

BLAS.set_num_threads(1)

const SUITE = BenchmarkGroup()

# ── Univariate K=2: Laplace + Normal, varying N and method ───────────────────
# Inspired by examples_univariate.jl (Normal + Laplace + Exponential example)
SUITE["univariate"] = BenchmarkGroup()

let
    mix_true = MixtureModel([Laplace(-1.0, 10.0), Normal(0.5, 0.8)], [0.3, 0.7])
    mix_guess = MixtureModel([Laplace(-2, 5.0), Normal(1.0, 1.0)], [0.4, 0.6])

    for method_label in ["ClassicEM", "StochasticEM"]
        SUITE["univariate"][method_label] = BenchmarkGroup()
        method = method_label == "ClassicEM" ? ClassicEM() : StochasticEM(StableRNG(1))
        for N in [1_000, 10_000, 100_000]
            y = rand(StableRNG(1), mix_true, N)
            SUITE["univariate"][method_label][N] = @benchmarkable(
                fit_mle($mix_guess, $y; method=$method, maxiter=100, atol=1e-4),
                evals = 1
            )
        end
    end
end

# ── Univariate K=3: Laplace + Normal + Exponential, varying N ─────────────────
# Directly from examples_univariate.jl
SUITE["univariate_K3"] = BenchmarkGroup()

let
    mix_true = MixtureModel([Laplace(-1.0, 5.0), Normal(0.5, 0.8), Exponential(2.0)], [0.15, 0.7, 0.15])
    mix_guess = MixtureModel([Laplace(-1.0, 1.0), Normal(2.0, 1.0), Exponential(3.0)], [1 / 3, 1 / 3, 1 / 3])

    for method_label in ["StochasticEM"]
        SUITE["univariate_K3"][method_label] = BenchmarkGroup()
        method = method_label == "ClassicEM" ? ClassicEM() : StochasticEM(StableRNG(1))
        for N in [1_000, 10_000, 100_000]
            y = rand(StableRNG(1), mix_true, N)
            SUITE["univariate_K3"][method_label][N] = @benchmarkable(
                fit_mle($mix_guess, $y; method=$method, maxiter=100, atol=1e-4), 
                evals = 1
            )
        end
    end
end

# ── Univariate: ClassicEM, varying number of components K ────────────────────
SUITE["univariate_Kvar"] = BenchmarkGroup()

let
    N = 10_000
    for K in [2, 5, 10]
        μs = range(-2(K - 1), 2(K - 1), length=K)
        mix_true = MixtureModel([Normal(μ, 1.0) for μ in μs])
        mix_guess = MixtureModel([Normal(μ + 0.5, 1.2) for μ in μs])
        y = rand(StableRNG(1), mix_true, N)

        SUITE["univariate_Kvar"][K] = BenchmarkGroup()
        for method_label in ["ClassicEM", "StochasticEM"]
            method = method_label == "ClassicEM" ? ClassicEM() : StochasticEM(StableRNG(1))
            SUITE["univariate_Kvar"][K][method_label] = @benchmarkable(
                fit_mle($mix_guess, $y; method=$method, maxiter=100, atol=1e-4),
                evals = 1
            )
        end
    end
end

# ── Multivariate: MvNormal K=2, varying dimension D and sample size N ─────────
# Inspired by examples_multivariate.jl (2D Gaussian mixture and Old Faithful)
SUITE["multivariate"] = BenchmarkGroup()

for D in [2, 10, 50]
    SUITE["multivariate"][D] = BenchmarkGroup()
    μ₁ = fill(-1.0, D)
    μ₂ = fill(1.0, D)
    mix_true = MixtureModel([MvNormal(μ₁, I(D)), MvNormal(μ₂, I(D))], [0.4, 0.6])
    mix_guess = MixtureModel([MvNormal(fill(-0.5, D), I(D)), MvNormal(fill(0.5, D), I(D))], [0.5, 0.5])
    for N in [1_000, 10_000]
        y = rand(StableRNG(1), mix_true, N)
        SUITE["multivariate"][D][N] = @benchmarkable(
            fit_mle($mix_guess, $y; maxiter=100, atol=1e-4),
            evals = 1
        )
    end
end

# ── Weighted ClassicEM: univariate K=2, varying N ────────────────────────────
SUITE["weighted"] = BenchmarkGroup()

let
    mix_true = MixtureModel([Normal(0.0, 1.0), Normal(5.0, 1.0)], [0.4, 0.6])
    mix_guess = MixtureModel([Normal(0.5, 1.2), Normal(4.5, 0.9)], [0.5, 0.5])

    for N in [1_000, 10_000, 100_000]
        y = rand(StableRNG(1), mix_true, N)
        w = rand(StableRNG(2), N) .+ 0.5  # positive weights in [0.5, 1.5]
        SUITE["weighted"][N] = @benchmarkable(
            fit_mle($mix_guess, $y, $w; maxiter=100, atol=1e-4),
            evals = 1
        )
    end
end

# ── MNIST Bernoulli Mixture K=10, N=10_000, maxiter=20 ───────────────────────
# From examples_multivariate.jl (MNIST section)
SUITE["mnist_bernoulli"] = BenchmarkGroup()

let
    binarify(x) = x != 0 ? true : false
    dataset = MNIST(:train)
    X, y = dataset[1:10000]
    Xb = binarify.(reshape(X, (28^2, size(X, 3))))
    id = [findall(y .∈ i) for i in 0:9]
    dist_guess = [product_distribution(Bernoulli.(mean(Xb[:, l] for l in id[i]))) for i in eachindex(id)]
    α = fill(1 / 10, 10)
    mix_guess = MixtureModel(dist_guess, α)

    for method_label in ["ClassicEM", "StochasticEM"]
        method = method_label == "ClassicEM" ? ClassicEM() : StochasticEM(StableRNG(1))
        SUITE["mnist_bernoulli"][method_label] = @benchmarkable(
            fit_mle($mix_guess, $Xb; method=$method, robust=true, maxiter=20),
            evals = 1
        )
    end
end
