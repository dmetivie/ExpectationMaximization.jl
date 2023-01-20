using ExpectationMaximization
using Distributions
using Test
using Random

@testset "Univariate continuous Mixture" begin
    N = 50_000
    seed = MersenneTwister(0)
    θ₁ = 10
    θ₂ = 5
    α = 0.5
    β = 0.3
    rtol = 6e-2
    mix_true = MixtureModel([Exponential(θ₁), Gamma(α, θ₂)], [β, 1 - β])
    y = rand(seed, mix_true, N)
    mix_guess = MixtureModel([Exponential(1), Gamma(0.5, 1)], [0.5, 1 - 0.5])
    mix_mle = fit_mle(mix_guess, y; display=:iter, tol=1e-3, robust=false, infos=false)

    p = params(mix_mle)[1]
    @test isapprox([β, 1 - β], probs(mix_mle); rtol=rtol)
    @test isapprox(θ₁, p[1]...; rtol=rtol)
    @test isapprox(α, p[2][1]; rtol=rtol)
    @test isapprox(θ₂, p[2][2]; rtol=rtol)
end

@testset "Multivariate Gaussian Mixture" begin
    N = 50_000
    seed = MersenneTwister(0)
    rtol = 5e-2
    θ₁ = [-1, 1]
    θ₂ = [0, 2]
    Σ₁ = [0.5 0.5
        0.5 1]
    Σ₂ = [1 0.1
        0.1 1]
    β = 0.3
    D₁ = MvNormal(θ₁, Σ₁)
    D₂ = MvNormal(θ₂, Σ₂)

    mix_true = MixtureModel([D₁, D₂], [β, 1 - β])

    # Generate samples from the true distribution
    y = rand(seed, mix_true, N)

    # Initial Condition
    D₁guess = MvNormal([0.2, 1], [1 0.6; 0.6 1])
    D₂guess = MvNormal([1, 0.5], [1 0.2; 0.2 1])
    mix_guess = MixtureModel([D₁guess, D₂guess], [0.4, 0.6])

    # Fit MLE
    mix_mle = fit_mle(mix_guess, y; display=:iter, tol=1e-3, robust=false, infos=false)

    p = params(mix_mle)[1]
    @test isapprox([β, 1 - β], probs(mix_mle); rtol=rtol)
    @test isapprox(collect(p[1]), [θ₁, Σ₁], rtol=rtol)
    @test isapprox(collect(p[2]), [θ₂, Σ₂], rtol=rtol)
end

# Bernoulli Mixture i.e. Mixture of Bernoulli Product (S = 10 term and K = 3 mixture components).
@testset "Multivariate Product Bernoulli Mixture" begin
    N = 50_000
    seed = MersenneTwister(0)
    rtol = 5e-2

    S = 10
    K = 3
    θ = zeros(S, K)
    θ[:, 1] = (1:S) / S .- 0.05 # Bernoulli parameters
    θ[:, 2] = (S:-1:1) / 2S # Bernoulli parameters
    θ[:, 3] = ones(S) + 0.1 * [isodd(i) ? -1 : 1 for i in 1:S] .- 0.4# Bernoulli parameters
    β = 0.3

    mix_true = MixtureModel([product_distribution(Bernoulli.(θ[:, i])) for i in 1:K], [β / 2, 1 - β, β / 2])

    # Generate samples from the true distribution
    y = rand(seed, mix_true, N)

    # Initial Condition
    mix_guess = MixtureModel([product_distribution(Bernoulli.(2θ[:, i] / 3)) for i in 1:K], [0.25, 0.55, 0.2])

    # Fit MLE
    mix_mle = fit_mle(mix_guess, y; display=:iter, tol=1e-3, robust=false, infos=false)

    p = params(mix_mle)[1]
    @test isapprox([β / 2, 1 - β, β / 2], probs(mix_mle); rtol=rtol)
    @test isapprox(first.(hcat(p...)), θ, rtol=rtol)
end

@testset "Univariate continuous Mixture of (mixture + Normal)" begin
    N = 50_000
    seed = MersenneTwister(0)
    θ₁ = -5
    θ₂ = 2
    σ₁ = 1
    σ₂ = 1.5
    θ₀ = 0.1
    σ₀ = 0.1

    α = 1 / 2
    β = 0.3

    rtol = 2e-2 # 2% 
    d1 = MixtureModel([Normal(θ₁, σ₁), Normal(θ₂, σ₂)], [α, 1 - α])
    d2 = Normal(θ₀, σ₀)
    mix_true = MixtureModel([d1, d2], [β, 1 - β])
    y = rand(seed, mix_true, N)

    # We choose initial guess very close to the true solution just to show the EM algorithm convergence.
    # This particular choice of mixture of mixture Gaussian with another Gaussian is non identifiable hence we execpt other solution far away from the true solution
    d1_guess = MixtureModel([Normal(θ₁ - 0.1, σ₁ + 0.1), Normal(θ₂ + 0.1, σ₂ - 0.1)], [α + 0.1, 1 - α - 0.1])
    d2_guess = Normal(θ₀ + 0.1, σ₀ - 0.01)

    mix_guess = MixtureModel([d1_guess, d2_guess], [β + 0.1, 1 - β - 0.1])
    mix_mle = fit_mle(mix_guess, y; display=:iter, tol=1e-3, robust=false, infos=false)
    y_guess = rand(seed, mix_mle, N)

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