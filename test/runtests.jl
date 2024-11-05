using ExpectationMaximization
using Distributions
using Distributions: params
using Test
using StableRNGs, Random

@testset "Univariate continuous Mixture Exponential + Gamma" begin
    rng = StableRNG(123)
    N = 50_000
    θ₁ = 10
    θ₂ = 5
    α = 0.8
    β = 0.6
    rtol = 6e-2
    mix_true = MixtureModel([Exponential(θ₁), Gamma(α, θ₂)], [β, 1 - β])
    y = rand(rng, mix_true, N)
    mix_guess = MixtureModel([Exponential(1), Gamma(0.5, 1)], [0.5, 1 - 0.5])
    mix_mle =
        fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)

    p = params(mix_mle)[1]
    @test isapprox([β, 1 - β], probs(mix_mle); rtol=rtol)
    @test isapprox(θ₁, p[1]...; rtol=rtol)
    @test isapprox(α, p[2][1]; rtol=rtol)
    @test isapprox(θ₂, p[2][2]; rtol=2rtol) # harder to get high accuracy here apparently

    # Test rtol
    mix_mle2 =
        fit_mle(mix_guess, y; display=:none, rtol=1e-8, atol=0, robust=false, infos=false)
    p = params(mix_mle2)[1]
    @test isapprox([β, 1 - β], probs(mix_mle2); rtol=rtol)
    @test isapprox(θ₁, p[1]...; rtol=rtol)
    @test isapprox(α, p[2][1]; rtol=rtol)
    @test isapprox(θ₂, p[2][2]; rtol=2rtol) # harder to get high accuracy here apparently
end

@testset "Stochastic EM Univariate continuous Mixture Exponential + Laplace" begin
    rng = StableRNG(123)
    N = 50_000
    θ₁ = 10
    θ₂ = 0.8
    α = 0.5
    β = 0.3
    μ = -1
    rtol = 7e-2
    mix_true = MixtureModel([Laplace(μ, θ₁), Normal(α, θ₂)], [β, 1 - β])
    y = rand(rng, mix_true, N)
    mix_guess = MixtureModel([Laplace(1), Normal(0.5, 1)], [0.5, 1 - 0.5])
    mix_mle = fit_mle(
        mix_guess,
        y;
        display=:none,
        atol=1e-3,
        robust=false,
        infos=false,
        method=StochasticEM(),
    )

    p = params(mix_mle)[1]
    @test isapprox([β, 1 - β], probs(mix_mle); rtol=rtol)
    @test isapprox(θ₁, p[1][2]; rtol=rtol)
    @test isapprox(μ, p[1][1]; rtol=0.1)
    @test isapprox(α, p[2][1]; rtol=rtol)
    @test isapprox(θ₂, p[2][2]; rtol=rtol)

    mix_mle2 = fit_mle(
        mix_guess,
        y;
        display=:none,
        atol=0,
        rtol=1e-6,
        robust=false,
        infos=false,
        method=StochasticEM(),
    )
    p = params(mix_mle2)[1]
    @test isapprox([β, 1 - β], probs(mix_mle2); rtol=rtol)
    @test isapprox(θ₁, p[1][2]; rtol=rtol)
    @test isapprox(μ, p[1][1]; rtol=0.1)
    @test isapprox(α, p[2][1]; rtol=rtol)
    @test isapprox(θ₂, p[2][2]; rtol=rtol)
end

@testset "Multivariate Gaussian Mixture" begin
    rng = StableRNG(123)
    N = 50_000
    rtol = 5e-2
    θ₁ = [-1, 1]
    θ₂ = [0, 2]
    Σ₁ = [
        0.5 0.5
        0.5 1
    ]
    Σ₂ = [
        1 0.1
        0.1 1
    ]
    β = 0.3
    D₁ = MvNormal(θ₁, Σ₁)
    D₂ = MvNormal(θ₂, Σ₂)

    mix_true = MixtureModel([D₁, D₂], [β, 1 - β])

    # Generate samples from the true distribution
    y = rand(rng, mix_true, N)

    # Initial Condition
    D₁guess = MvNormal([0.2, 1], [1 0.6; 0.6 1])
    D₂guess = MvNormal([1, 0.5], [1 0.2; 0.2 1])
    mix_guess = MixtureModel([D₁guess, D₂guess], [0.4, 0.6])

    # Fit MLE
    mix_mle =
        fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)

    p = params(mix_mle)[1]
    @test isapprox([β, 1 - β], probs(mix_mle); rtol=rtol)
    @test isapprox(collect(p[1]), [θ₁, Σ₁], rtol=rtol)
    @test isapprox(collect(p[2]), [θ₂, Σ₂], rtol=rtol)
end

# Bernoulli Mixture i.e. Mixture of Bernoulli Product (S = 10 term and K = 3 mixture components).
@testset "Multivariate Product Bernoulli Mixture" begin
    rng = StableRNG(123)
    N = 50_000
    rtol = 5e-2

    S = 10
    K = 3
    θ = zeros(S, K)
    θ[:, 1] = (1:S) / S .- 0.05 # Bernoulli parameters
    θ[:, 2] = (S:-1:1) / 2S # Bernoulli parameters
    θ[:, 3] = ones(S) + 0.1 * [isodd(i) ? -1 : 1 for i = 1:S] .- 0.4# Bernoulli parameters
    β = 0.3

    mix_true = MixtureModel(
        [product_distribution(Bernoulli.(θ[:, i])) for i = 1:K],
        [β / 2, 1 - β, β / 2],
    )

    # Generate samples from the true distribution
    y = rand(rng, mix_true, N)

    # Initial Condition -> currently generate `Product` distributions depreacated 
    mix_guess = MixtureModel(
        [product_distribution(Bernoulli.(2θ[:, i] / 3)) for i = 1:K],
        [0.25, 0.55, 0.2],
    )

    # Fit MLE
    mix_mle =
        fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)

    p = params(mix_mle)[1]
    @test isapprox([β / 2, 1 - β, β / 2], probs(mix_mle); rtol=rtol)
    @test isapprox(first.(hcat(p...)), θ, rtol=rtol)

    # Initial Condition -> generate Distributions.ProductDistribution (only `...` difference)
    mix_guess = MixtureModel(
        [product_distribution(Bernoulli.(2θ[:, i] / 3)...) for i = 1:K],
        [0.25, 0.55, 0.2],
    )

    # Fit MLE
    mix_mle =
        fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)

    p = params(mix_mle)[1]
    @test isapprox([β / 2, 1 - β, β / 2], probs(mix_mle); rtol=rtol)
    @test isapprox(hcat([first.([pp...]) for pp in p]...), θ, rtol=rtol)
end

@testset "Univariate continuous Mixture of (mixture + Normal)" begin
    rng = StableRNG(123)
    N = 50_000
    θ₁ = -5
    θ₂ = 2
    σ₁ = 1
    σ₂ = 1.5
    θ₀ = 0.1
    σ₀ = 0.1

    α = 1 / 2
    β = 0.3

    rtol = 5e-2 #
    d1 = MixtureModel([Normal(θ₁, σ₁), Normal(θ₂, σ₂)], [α, 1 - α])
    d2 = Normal(θ₀, σ₀)
    mix_true = MixtureModel([d1, d2], [β, 1 - β])
    y = rand(rng, mix_true, N)

    # We choose initial guess very close to the true solution just to show the EM algorithm convergence.
    # This particular choice of mixture of mixture Gaussian with another Gaussian is non identifiable hence we execpt other solution far away from the true solution
    d1_guess = MixtureModel(
        [Normal(θ₁ - 0.1, σ₁ + 0.1), Normal(θ₂ + 0.1, σ₂ - 0.1)],
        [α + 0.1, 1 - α - 0.1],
    )
    d2_guess = Normal(θ₀ + 0.1, σ₀ - 0.01)

    mix_guess = MixtureModel([d1_guess, d2_guess], [β + 0.1, 1 - β - 0.1])
    mix_mle = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)
    y_guess = rand(rng, mix_mle, N)

    @test probs(mix_mle) ≈ [β, 1 - β] rtol = rtol
    p = params(mix_mle)[1]
    @test p[1][2] ≈ [α, 1 - α] rtol = rtol
    @test θ₁ ≈ p[1][1][1][1] rtol = rtol
    @test σ₁ ≈ p[1][1][1][2] rtol = rtol
    @test θ₂ ≈ p[1][1][2][1] rtol = rtol
    @test σ₂ ≈ p[1][1][2][2] rtol = rtol
    @test θ₀ ≈ p[2][1] rtol = rtol
    @test σ₀ ≈ p[2][2] rtol = rtol
end

@testset "Univariate continuous Mixture of (Laplace + Normal)" begin
    rng = StableRNG(123)
    N = 50_000
    θ₁ = -2
    θ₂ = 2
    σ₁ = 1
    σ₂ = 1.5
    θ₀ = 0.1
    σ₀ = 0.2

    α = 1 / 4
    β = 0.3

    rtol = 5e-2 #
    d1 = MixtureModel([Normal(θ₁, σ₁), Laplace(θ₂, σ₂)], [α, 1 - α])
    d2 = Normal(θ₀, σ₀)
    mix_true = MixtureModel([d1, d2], [β, 1 - β])
    y = rand(rng, mix_true, N)

    d1_guess = MixtureModel(
        [Normal(θ₁ - 4, σ₁ + 2), Laplace(θ₂ + 2, σ₂ - 1)],
        [α + 0.1, 1 - α - 0.1],
    )
    d2_guess = Normal(θ₀ + 2, 10σ₀)

    mix_guess = MixtureModel([d1_guess, d2_guess], [β + 0.1, 1 - β - 0.1])
    mix_mle =
        fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)
    # without print
    # 1.368 s (17002715 allocations: 1.48 GiB)
    #  1.485 s (17853393 allocations: 1.61 GiB)
    y_guess = rand(rng, mix_mle, N)

    @test probs(mix_mle) ≈ [β, 1 - β] rtol = rtol
    p = params(mix_mle)[1]
    @test p[1][2] ≈ [α, 1 - α] rtol = rtol
    @test θ₁ ≈ p[1][1][1][1] rtol = rtol
    @test σ₁ ≈ p[1][1][1][2] rtol = rtol
    @test θ₂ ≈ p[1][1][2][1] rtol = rtol
    @test σ₂ ≈ p[1][1][2][2] rtol = rtol
    @test θ₀ ≈ p[2][1] rtol = rtol
    @test σ₀ ≈ p[2][2] rtol = rtol
end

@testset "Univariate discrete Mixture of Mixture (Poisson + Geom)" begin
    rng = StableRNG(123)
    N = 50_000
    θ₁ = 5
    θ₂ = 1 / 2
    σ₁ = 10
    σ₂ = 1 / 5

    α = 1 / 4
    β = 0.3

    rtol = 8e-2 #
    d1 = MixtureModel([Poisson(θ₁), Geometric(θ₂)], [α, 1 - α])
    d2 = MixtureModel([Poisson(σ₁), Geometric(σ₂)], [α, 1 - α])
    mix_true = MixtureModel([d1, d2], [β, 1 - β])
    y = rand(rng, mix_true, N)

    d1_guess = MixtureModel(
        [Poisson(θ₁ + 2), Geometric(θ₂ + 0.2)],
        [α + 0.15, 1 - α - 0.15],
    )
    d2_guess = MixtureModel(
        [Poisson(σ₁ + 2), Geometric(σ₂ + 0.2)],
        [α + 0.15, 1 - α - 0.15],
    )

    mix_guess = MixtureModel([d1_guess, d2_guess], [β + 0.1, 1 - β - 0.1])

    for meth in [ClassicEM(), StochasticEM(rng)]
        mix_mle, hist =
            fit_mle(mix_guess, y; display=:none, atol=2e-4, robust=true, infos=true, method=meth, maxiter=100_000)

        @test hist["converged"]
        #note: atol seems more appropiate for [0,1] numbers
        @test probs(mix_mle)[1] ≈ β atol = rtol
        p = params(mix_mle)[1]
        @test p[1][2][1] ≈ α atol = rtol
        @test p[2][2][1] ≈ α atol = rtol

        @test θ₁ ≈ p[1][1][1][1] rtol = rtol
        @test θ₂ ≈ p[1][1][2][1] atol = rtol
        @test σ₁ ≈ p[2][1][1][1] rtol = rtol
        @test σ₂ ≈ p[2][1][2][1] atol = rtol
    end
end

@testset "Most likely category identification" begin
    rng = StableRNG(123)
    m = MixtureModel([Normal(), Laplace(2)], [0.2, 0.8])
    α = probs(m)
    dists = components(m)
    N = 1000
    z = zeros(Int, N)
    y = zeros(N)
    for i = 1:N
        z[i] = rand(rng, Categorical(α))
        y[i] = rand(rng, dists[z[i]])
    end
    ẑ = predict(m, y)
    @test count(ẑ .== z) / N > 0.85
end

@testset "LatentClassAnalysis.jl like test i.e. Mixture of Product Distribution of Categorical" begin
    rng = StableRNG(12)

    n_samples = 10000  # Increased sample size
    n_categoriesⱼ = [4, 2, 3, 5] # number of possible values for each element depending on the col
    n_items = length(n_categoriesⱼ)  # number of cols
    n_classes = 3 # latent class / hidden state

    # `Dirichlet` distribution generate random proba vector i.e. sum = 1
    prob_jck = [rand(rng, Dirichlet(ones(n_categoriesⱼ[j])), n_classes) for j in 1:n_items]

    prob_class = rand(rng, Dirichlet(ones(n_classes)))

    dist_true = MixtureModel([product_distribution([Categorical(prob_jck[j][:, k]) for j in 1:n_items]) for k in 1:n_classes], prob_class)
    data_with_mix = rand(rng, dist_true, n_samples)

    prob_jck_guess = [rand(rng, Dirichlet(ones(n_categoriesⱼ[j])), n_classes) for j in 1:n_items]
    prob_class_guess = prob_class + 0.02 * (rand(rng, Dirichlet(ones(n_classes))) .- 1 / n_classes) #

    dist_ini = MixtureModel([product_distribution([Categorical(prob_jck_guess[j][:, k]) for j in 1:n_items]) for k in 1:n_classes], prob_class_guess)

    dist_fit = fit_mle(dist_ini, data_with_mix, atol=1e-5, maxiter=10000) # 

    # with this seed indices of latent classes get inverted hence the reorder
    kk = [1, 3, 2]
    @test probs(dist_fit)[kk] ≈ probs(dist_true) rtol = 1e2
    for k in 1:n_classes
        @test all(isapprox.(probs.(components(dist_fit)[kk[k]].v), probs.(components(dist_true)[k].v), atol=10e-2))
    end

    dist_fit = fit_mle(dist_ini, data_with_mix, atol=1e-3, maxiter=100, method=StochasticEM(rng)) # just to check it runs
end
# @btime ExpectationMaximization.fit_mle(dist_ini, $(data_with_mix), atol=1e-3, maxiter=1000)
# 1.159 s (33147640 allocations: 1.73 GiB) # before @views
# 862.141 ms (27640 allocations: 254.45 MiB) # after some @views in Estep
# @profview [ExpectationMaximization.fit_mle(dist_ini, (data_with_mix), atol=1e-3, maxiter=1000) for i in 1:10]
