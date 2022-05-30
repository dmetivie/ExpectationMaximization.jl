using ExpectationMaximization
using Distributions
using Test
using Random

@testset "ExpectationMaximization.jl" begin
    N = 50_000
    seed = MersenneTwister(0)
    θ₁ = 10
    θ₂ = 5
    α = 0.5
    β = 0.3
    rtol = 5e-2
    mix_true = MixtureModel([Exponential(θ₁), Gamma(α, θ₂)], [β, 1 - β])
    y = rand(seed, mix_true, N)
    mix_guess = MixtureModel([Exponential(1), Gamma(0.5, 1)], [0.5, 1 - 0.5])
    mix_mle = fit_mle(mix_guess, y; display = :iter, tol = 1e-3, robust = false, infos = false)

    p = params(mix_mle)[1]
    @test isapprox(β, probs(mix_mle)[1]; rtol = rtol)
    @test isapprox(θ₁, p[1]...; rtol = rtol)
    @test isapprox(α, p[2][1]; rtol = rtol)
    @test isapprox(θ₂, p[2][2]; rtol = rtol)
end
