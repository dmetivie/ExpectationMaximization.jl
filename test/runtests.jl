using ExpectationMaximization
using Distributions
using Distributions: params
using Test
using StableRNGs, Random
using LinearAlgebra: I
using MLDatasets: MNIST
using LogExpFunctions: logsumexp!

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
    mix_mle = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)

    p = params(mix_mle)[1]
    @test isapprox([β, 1 - β], probs(mix_mle); rtol=rtol)
    @test isapprox(θ₁, p[1]...; rtol=rtol)
    @test isapprox(α, p[2][1]; rtol=rtol)
    @test isapprox(θ₂, p[2][2]; rtol=2rtol) # harder to get high accuracy here apparently

    # Test rtol
    mix_mle2 = fit_mle(mix_guess, y; display=:none, rtol=1e-8, atol=0, robust=false, infos=false)
    p = params(mix_mle2)[1]
    @test isapprox([β, 1 - β], probs(mix_mle2); rtol=rtol)
    @test isapprox(θ₁, p[1]...; rtol=rtol)
    @test isapprox(α, p[2][1]; rtol=rtol)
    @test isapprox(θ₂, p[2][2]; rtol=2rtol) # harder to get high accuracy here apparently
end

@testset "Stochastic EM Univariate continuous Mixture Exponential + Laplace" begin
    rng = StableRNG(1234) # these SEM tests are quite sensitive to seed/rtol
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
        method=StochasticEM(rng),
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
        method=StochasticEM(rng),
    )
    p = params(mix_mle2)[1]
    @test isapprox([β, 1 - β], probs(mix_mle2); rtol=rtol)
    @test isapprox(θ₁, p[1][2]; rtol=rtol)
    @test isapprox(μ, p[1][1]; rtol=rtol)
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
        fit_mle(mix_guess, y; display=:none, atol=1e-2, robust=false, infos=false)
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

        @test θ₁ ≈ p[1][1][1][1] rtol = 2.5 * rtol
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

@testset "Weighted ClassicEM equals repeated samples" begin
    rng = StableRNG(42)
    mix_true = MixtureModel([Normal(0.0, 1.0), Normal(5.0, 1.0)], [0.4, 0.6])
    y_base = rand(rng, mix_true, 1000)
    w = float.(rand(rng, 1:5, 1000))
    y_rep = vcat([fill(y_base[i], Int(w[i])) for i in eachindex(w)]...)

    mix_guess = MixtureModel([Normal(0.5, 1.2), Normal(4.5, 0.9)], [0.5, 0.5])
    mix_w = fit_mle(mix_guess, y_base, w; atol=1e-8, infos=false)
    mix_rep = fit_mle(mix_guess, y_rep; atol=1e-8, infos=false)

    @test probs(mix_w) ≈ probs(mix_rep) rtol = 1e-3
    for k in 1:2, j in 1:2
        @test params(mix_w)[1][k][j] ≈ params(mix_rep)[1][k][j] rtol = 1e-3
    end
end

@testset "Weighted StochasticEM recovers true parameters" begin
    rng = StableRNG(7)
    N = 5000
    mix_true = MixtureModel([Normal(0.0, 1.0), Normal(5.0, 1.0)], [0.4, 0.6])
    y_base = rand(rng, mix_true, N)
    # all-ones weights should give the same result as unweighted
    w = ones(N)
    mix_guess = MixtureModel([Normal(0.5, 1.2), Normal(4.5, 0.9)], [0.5, 0.5])
    mix_w = fit_mle(mix_guess, y_base, w; atol=1e-5, infos=false, method=StochasticEM(StableRNG(7)))
    mix_uw = fit_mle(mix_guess, y_base; atol=1e-5, infos=false, method=StochasticEM(StableRNG(7)))
    # identical weights → identical result
    @test probs(mix_w) ≈ probs(mix_uw) rtol = 1e-4
    for k in 1:2, j in 1:2
        @test params(mix_w)[1][k][j] ≈ params(mix_uw)[1][k][j] rtol = 1e-4
    end
end

@testset "ClassicEM loglikelihood is non-decreasing" begin
    rng = StableRNG(1)
    y = rand(rng, MixtureModel([Normal(-2.0, 1.0), Normal(2.0, 1.0)], [0.4, 0.6]), 5000)
    mix_guess = MixtureModel([Normal(-1.0, 1.0), Normal(1.0, 1.0)], [0.5, 0.5])
    _, hist = fit_mle(mix_guess, y; atol=0, maxiter=50, infos=true)
    ll = hist["logtots"]
    @test all(diff(ll) .>= -1e-8)
end

@testset "history structure and maxiter respected" begin
    rng = StableRNG(1)
    y = rand(rng, MixtureModel([Normal(-2.0, 1.0), Normal(2.0, 1.0)], [0.4, 0.6]), 5000)
    mix_guess = MixtureModel([Normal(-1.0, 1.0), Normal(1.0, 1.0)], [0.5, 0.5])

    for meth in [ClassicEM(), StochasticEM(StableRNG(2))]
        _, hist = fit_mle(mix_guess, y; atol=0, maxiter=4, infos=true, method=meth)
        @test hist["iterations"] == 4
        @test hist["converged"] == false
        @test length(hist["logtots"]) == hist["iterations"]
    end
end

@testset "Multivariate StochasticEM" begin
    rng = StableRNG(99)
    N = 20_000
    rtol = 8e-2
    D₁ = MvNormal([-2.0, 0.0], [1.0 0.0; 0.0 1.0])
    D₂ = MvNormal([2.0, 0.0], [1.0 0.0; 0.0 1.0])
    β = 0.4
    mix_true = MixtureModel([D₁, D₂], [β, 1 - β])
    y = rand(rng, mix_true, N)

    mix_guess = MixtureModel(
        [MvNormal([-1.0, 0.0], [1.2 0.0; 0.0 1.2]),
            MvNormal([1.0, 0.0], [1.2 0.0; 0.0 1.2])],
        [0.5, 0.5],
    )
    mix_mle = fit_mle(mix_guess, y; atol=1e-3, infos=false, method=StochasticEM(rng))

    @test probs(mix_mle)[1] ≈ β rtol = rtol
    p = params(mix_mle)[1]
    @test p[1][1] ≈ mean(D₁) rtol = rtol
    @test p[2][1] ≈ mean(D₂) rtol = rtol
end

@testset "predict on multivariate mixture" begin
    rng = StableRNG(5)
    N = 2000
    D₁ = MvNormal([-5.0, 0.0], I(2))
    D₂ = MvNormal([5.0, 0.0], I(2))
    mix = MixtureModel([D₁, D₂], [0.5, 0.5])
    y = rand(rng, mix, N)
    # generate true labels: component with higher likelihood
    z_true = [pdf(D₁, y[:, i]) >= pdf(D₂, y[:, i]) ? 1 : 2 for i in 1:N]
    ẑ = predict(mix, y)
    @test count(ẑ .== z_true) / N > 0.99  # well-separated clusters → near-perfect prediction
end

@testset "Fused softmax kernel matches the logsumexp! reference" begin
    EM = ExpectationMaximization
    function reference!(c, γ, LL)   # exactly what the E-step used to do
        logsumexp!(c, LL)
        @. γ = exp(LL - c)
        return c, γ
    end

    rows = Any[
        [-Inf -Inf], [Inf Inf], [Inf 0.0], [Inf -Inf], [-Inf 0.0], [NaN 0.0], [NaN NaN],
        [-Inf -Inf -Inf], [0.0 -1.0 -2.0], [-1e300 -1.1e300], [709.782712893384 0.0],
        [-1e308 -1e308],
    ]
    for LL0 in rows, robust in (false, true)
        N, K = size(LL0)
        LLr, cr, γr = copy(LL0), zeros(N), zeros(N, K)
        LLn, cn, γn, s = copy(LL0), zeros(N), zeros(N, K), zeros(N)
        if robust
            replace!(LLr, -Inf => nextfloat(-Inf), Inf => log(prevfloat(Inf)))
            EM._clamp_inf!(LLn)
        end
        @test all(isequal.(LLr, LLn))       # _clamp_inf! ≡ replace! with the two Pairs
        reference!(cr, γr, LLr)
        EM._softmax_rows!(cn, γn, LLn, s)

        if !isfinite(maximum(LLn))
            # degenerate row: reproduce the old output bit for bit (isequal, so NaN matches NaN)
            @test all(isequal.(cr, cn))
            @test all(isequal.(γr, γn))
        elseif isapprox(sum(γr), N)
            # ordinary row: the reference adds log1p(Σ_{k≠argmax}) where the fused kernel adds
            # log(Σ_k), so they may differ by an ulp
            @test all(isapprox.(cr, cn; atol=1e-12, rtol=1e-14))
            @test γr ≈ γn atol = 1e-12
        else
            # A row of huge but finite values: `c + log(s)` rounds back to `c`, so the reference
            # returned an UNNORMALISED γ (all ones, summing to K) which silently corrupts α and
            # every fit_mle in the M-step. Reachable with `robust = true` for an observation that
            # lies outside every component's support. The fused kernel normalises; that is a fix.
            @test !isapprox(sum(γr), N)     # the previous behaviour really was broken here
            @test sum(γn) ≈ N               # every row of the new posteriors sums to one
            @test cr == cn                  # the loglikelihood itself is unchanged
        end
    end
end

@testset "Fused softmax: γ may alias LL, and posteriors sum to one" begin
    EM = ExpectationMaximization
    rng = StableRNG(20240)
    for _ = 1:200
        N, K = rand(rng, 1:60), rand(rng, 1:6)
        LL = (rand(rng, N, K) .- 0.5) .* exp(20rand(rng))
        for _ = 1:rand(rng, 0:3)
            LL[rand(rng, 1:N), rand(rng, 1:K)] = rand(rng, (-Inf, Inf, NaN, -1e308, 0.0))
        end
        c1, γ1, s1 = zeros(N), zeros(N, K), zeros(N)
        c2, LL2, s2 = zeros(N), copy(LL), zeros(N)
        EM._softmax_rows!(c1, γ1, LL, s1)
        EM._softmax_rows!(c2, LL2, LL2, s2)     # γ === LL
        @test reinterpret(UInt64, c1) == reinterpret(UInt64, c2)
        @test reinterpret(UInt64, vec(γ1)) == reinterpret(UInt64, vec(LL2))
    end
    # A row of huge but finite log-likelihoods: `c + log(s)` rounds back to `c`, and the previous
    # implementation then returned γ = [1 1], which does not sum to one and silently corrupts α and
    # every fit_mle in the M-step. Reachable with `robust = true` for an observation that lies
    # outside every component's support.
    LL = [-1e308 -1e308]
    c, γ, s = zeros(1), zeros(1, 2), zeros(1)
    EM._softmax_rows!(c, γ, LL, s)
    @test sum(γ) ≈ 1
    @test γ ≈ [0.5 0.5]
end

@testset "E-step allocates nothing per iteration (univariate)" begin
    EM = ExpectationMaximization
    rng = StableRNG(4)
    dists = [Normal(-1.0, 1.0), Normal(1.0, 1.5)]
    α = [0.4, 0.6]
    N, K = 5_000, 2
    y = rand(rng, MixtureModel(dists, α), N)
    LL, c, s = zeros(N, K), zeros(N), zeros(N)
    EM.E_step!(LL, c, LL, s, dists, α, y)                    # compile
    @test (@allocated EM.E_step!(LL, c, LL, s, dists, α, y)) == 0
    @test all(isapprox.(sum(LL, dims=2), 1))                 # LL now holds the posteriors
end

@testset "Multi-start forwards rtol and does not silently drop initial conditions" begin
    rng = StableRNG(11)
    y = rand(rng, MixtureModel([Normal(-2.0, 1.0), Normal(2.0, 1.0)], [0.4, 0.6]), 5_000)
    g1 = MixtureModel([Normal(-1.0, 1.0), Normal(1.0, 1.0)], [0.5, 0.5])
    g2 = MixtureModel([Normal(-0.5, 2.0), Normal(0.5, 2.0)], [0.5, 0.5])

    # `rtol` used to be dropped for the FIRST initial condition, leaving it with no convergence
    # criterion at all (atol = 0), so it ran the full maxiter and returned a different model.
    _, h_single = fit_mle(g1, y; atol=0, rtol=1e-6, maxiter=10_000, infos=true)
    _, h_array = fit_mle([g1], y; atol=0, rtol=1e-6, maxiter=10_000, infos=true)
    @test h_array["converged"]
    @test h_array["iterations"] == h_single["iterations"]

    # `logtots` is empty when maxiter = 0. Indexing it used to throw a BoundsError that the bare
    # `catch` swallowed, silently discarding every initial condition after the first.
    m0, h0 = fit_mle([g1, g2], y; maxiter=0, infos=true)
    @test m0 isa MixtureModel
    @test h0["iterations"] == 0
    @test isempty(h0["logtots"])

    # the best initial condition is still the one returned
    _, h_best = fit_mle([g1, g2], y; atol=1e-8, infos=true)
    _, h_g1 = fit_mle(g1, y; atol=1e-8, infos=true)
    _, h_g2 = fit_mle(g2, y; atol=1e-8, infos=true)
    @test h_best["logtots"][end] ≈ max(h_g1["logtots"][end], h_g2["logtots"][end])
end

@testset "predict tie-breaking is first-maximum-wins" begin
    EM = ExpectationMaximization
    M = [0.5 0.5 0.0
         0.2 0.4 0.4
         1.0 0.0 0.0]
    @test EM.argmaxrow(M) == [1, 2, 1] == [argmax(r) for r in eachrow(M)]
    R = rand(StableRNG(8), 500, 4)
    @test EM.argmaxrow(R) == [argmax(r) for r in eachrow(R)]
end

@testset "Weighted Laplace fit is unchanged by dropping the sample copy" begin
    EM = ExpectationMaximization
    rng = StableRNG(31)
    x = rand(rng, Laplace(1.5, 2.0), 20_000)
    w = rand(rng, 20_000) .+ 0.1
    d = fit_mle(Laplace, x, w)
    m = median(x, EM.weights(w))
    @test params(d)[1] == m
    @test params(d)[2] ≈ mean(abs.(x .- m), EM.weights(w)) rtol = 1e-12
end

@testset "Vector-of-arrays sample layout is still unsupported" begin
    # The univariate E-step uses `logpdf.(dists[k], y)`, which only works because Distributions
    # makes univariate distributions broadcast-scalar. So the `ArrayOfUnivariateDistribution`
    # layout (a vector of arrays) cannot be reached through `fit_mle`. Pinned here so that a
    # future E-step rewrite starts supporting it deliberately rather than by accident. Note that
    # `Distributions.logpdf!` does accept this layout, which is one reason the univariate path keeps
    # the broadcast instead of delegating like the matrix path does.
    y = [rand(StableRNG(9 + i), 2) for i = 1:50]
    mix = MixtureModel(
        [product_distribution([Normal(), Gamma(2, 1)]),
            product_distribution([Normal(3), Gamma(3, 1)])], [0.5, 0.5])
    @test_throws MethodError fit_mle(mix, y)
end

@testset "MvNormal E-step kernels match the generic fallback" begin
    EM = ExpectationMaximization
    PD = Distributions.PDMats
    # the generic per-observation fallback, spelled out, as the reference
    function generic_col!(LLₖ, d, logα, y::AbstractMatrix)
        @inbounds @views for n in axes(y, 2)
            LLₖ[n] = logα + logpdf(d, y[:, n])
        end
        return LLₖ
    end
    mk(kind, D) = begin
        rng = StableRNG(D + (kind === :full ? 7 : 0))
        μ = D == 1 ? [0.0] : collect(range(-1, 1, length=D))
        kind === :full ? (A = randn(rng, D, D); MvNormal(μ, Matrix(A * A' / D + D * I))) :
        kind === :diag ? MvNormal(μ, PD.PDiagMat(rand(rng, D) .+ 0.5)) :
        kind === :iso ? MvNormal(μ, PD.ScalMat(D, 1.7)) :
        kind === :zmfull ? (A = randn(rng, D, D); MvNormal(PD.PDMat(A * A' / D + D * I))) :
        MvNormal(PD.ScalMat(D, 2.3))
    end

    for kind in (:full, :diag, :iso, :zmfull, :zmiso), D in (1, 2, 7, 33), N in (1, 255, 256, 257)
        d = mk(kind, D)
        y = rand(StableRNG(N + D), d, N)
        a, b = zeros(N), zeros(N)
        generic_col!(a, d, log(0.3), y)
        EM._loglikelihood_col!(b, d, log(0.3), y)
        @test a ≈ b rtol = 1e-12
    end
    # an empty sample must not error
    @test EM._loglikelihood_col!(Float64[], mk(:full, 3), 0.0, zeros(3, 0)) == Float64[]

    # component types without a kernel must keep the generic fallback
    generic = which(EM._loglikelihood_col!,
        Tuple{Vector{Float64},MvNormal{Float64,PD.PDMat{Float64,Matrix{Float64}},Vector{Float64}},
            Float64,Matrix{Float64}})
    for d in (product_distribution([Normal(), Gamma(2, 1)]),
        MvNormalCanon([1.0, 2.0], [1.0, 1.0]),
        MixtureModel([MvNormal([0.0, 0.0], I(2)), MvNormal([1.0, 1.0], I(2))]))
        m = which(EM._loglikelihood_col!, Tuple{Vector{Float64},typeof(d),Float64,Matrix{Float64}})
        @test m != generic                          # not the MvNormal kernel
        @test occursin("fit_em.jl", String(m.file)) # the generic hook
    end
end

@testset "Blocked weighted FullNormal fit matches Distributions" begin
    PD = Distributions.PDMats
    for D in (2, 7, 10, 33), N in (255, 256, 257, 2000)
        rng = StableRNG(D + N)
        A = randn(rng, D, D)
        d = MvNormal(collect(range(-1, 1, length=D)), Matrix(A * A' / D + D * I))
        y = rand(StableRNG(N + D), d, N)
        w = rand(StableRNG(D), N) .+ 0.1
        ref, new = fit_mle(FullNormal, y, w), fit_mle(d, y, w)
        @test mean(new) == mean(ref)                # bit identical
        @test cov(new) ≈ cov(ref) rtol = 1e-12      # blocked accumulation, so 1-2 ulp
        @test typeof(new) == typeof(ref)
    end
    # DiagNormal and IsoNormal must keep their own fit, and their covariance type
    y, w = rand(StableRNG(1), 2, 200), rand(StableRNG(2), 200) .+ 0.1
    @test fit_mle(MvNormal([0.0, 0.0], PD.PDiagMat([2.0, 3.0])), y, w) isa Distributions.DiagNormal
    @test fit_mle(MvNormal([0.0, 0.0], PD.ScalMat(2, 4.0)), y, w) isa Distributions.IsoNormal
    # A non-BLAS eltype must not be claimed by the blocked method; it has to fall through to
    # Distributions, which does not support BigFloat either, but that is its call and not ours.
    A = randn(StableRNG(3), 2, 2)
    db = MvNormal([0.0, 0.0], Matrix(A * A' + 2I))
    @test occursin("specialized.jl",
        String(which(fit_mle, Tuple{typeof(db),Matrix{Float64},Vector{Float64}}).file))
    @test !occursin("specialized.jl",
        String(which(fit_mle, Tuple{typeof(db),Matrix{BigFloat},Vector{BigFloat}}).file))
end

# A component implementing only the scalar `logpdf`, i.e. the minimum the genericity contract asks
# for. `Distributions.logpdf!` must fall back to one call per observation for it.
struct ScalarOnlyMv <: Distributions.ContinuousMultivariateDistribution
    μ::Vector{Float64}
end
Base.length(d::ScalarOnlyMv) = length(d.μ)
Distributions._logpdf(d::ScalarOnlyMv, x::AbstractVector) = -sum(abs2, x .- d.μ) / 2

@testset "Generic matrix E-step delegates to Distributions.logpdf!" begin
    EM = ExpectationMaximization
    # The generic `_loglikelihood_col!` calls `Distributions.logpdf!`, whose own fallback is one
    # `logpdf` per observation. A component with a batched `_logpdf!` (a nested `MixtureModel`,
    # `MvNormalCanon`, ...) is therefore scored in a single call, and everything else keeps the
    # per-observation behaviour. The values must be identical either way.
    function reference!(LLₖ, d, logα, y)
        @inbounds @views for n in axes(y, 2)
            LLₖ[n] = logα + logpdf(d, y[:, n])
        end
        return LLₖ
    end
    for D in (1, 2, 6), N in (0, 1, 137)
        rng = StableRNG(10D + N)
        for d in (
            MixtureModel([MvNormal(randn(rng, D), I(D)) for _ = 1:3], [0.25, 0.35, 0.4]),
            MvNormalCanon(randn(rng, D), rand(rng, D) .+ 0.5),
            product_distribution([Normal(randn(rng), rand(rng) + 0.5) for _ = 1:D]),
            ScalarOnlyMv(randn(rng, D)),
        )
            y = randn(StableRNG(D + N), D, N)
            a, b = zeros(N), zeros(N)
            reference!(a, d, log(0.3), y)
            EM._loglikelihood_col!(b, d, log(0.3), y)
            @test a ≈ b rtol = 1e-12
        end
    end
    # A weight of zero must still give `-Inf`, not `NaN`, now that `logα` is added in a second pass.
    d = MvNormal(zeros(3), I(3))
    @test all(==(-Inf), EM._loglikelihood_col!(zeros(4), d, -Inf, randn(StableRNG(5), 3, 4)))

    # A mixture of *multivariate* mixtures is documented as supported but had no test. It is also
    # the configuration this delegation speeds up (measured 1.5x end to end), so pin that it fits.
    D, N = 4, 1000
    rng = StableRNG(42)
    inner(c) = MixtureModel([MvNormal(c .+ randn(rng, D) ./ 4, I(D)) for _ = 1:2], [0.4, 0.6])
    mix_true = MixtureModel([inner(-2.5), inner(2.5)], [0.45, 0.55])
    y = rand(StableRNG(7), mix_true, N)
    mix_fit, hist = fit_mle(mix_true, y; maxiter=5, atol=0.0, infos=true)
    @test all(diff(hist["logtots"]) .>= -1e-8)
    @test mix_fit isa MixtureModel
    @test sum(probs(mix_fit)) ≈ 1
end
@testset "StochasticEM subsample is gathered before it is sliced by rows" begin
    # The S-step hands each component `view(y, :, cat[k])`, and `Base.reindex` copies that column
    # index once per row slice. Without `_gather_rows` a `Product` fit therefore costs
    # `D * length(cat[k]) * 8` bytes instead of one `D * length(cat[k]) * sizeof(eltype(y))` gather,
    # which is 60x more memory per M-step on the `Matrix{Bool}` MNIST case.
    EM = ExpectationMaximization
    D, N = 64, 2_000
    y = rand(StableRNG(11), Bool, D, N)
    idx = collect(2:2:N)                                  # the shape `findall(ẑ .== k)` returns

    @test EM._gather_rows(view(y, :, idx)) isa Matrix{Bool}   # gathered once
    @test EM._gather_rows(view(y, :, idx)) == y[:, idx]
    @test EM._gather_rows(view(y, :, 1:100)) isa SubArray      # strided: passed through
    @test EM._gather_rows(y) === y

    # both `product_distribution` spellings: `Product` and `VectorOfUnivariateDistribution`
    for g in (product_distribution(Bernoulli.(fill(0.5, D))),
        product_distribution(Bernoulli.(fill(0.5, D))...))
        @test fit_mle(g, view(y, :, idx)) == fit_mle(g, y[:, idx])
        @test (@allocated fit_mle(g, view(y, :, idx))) < 4 * D * length(idx)
    end

    mix = MixtureModel(
        [product_distribution(Bernoulli.(rand(StableRNG(12 + k), D))) for k = 1:2], [0.5, 0.5]
    )
    sem() = fit_mle(mix, y; method=StochasticEM(StableRNG(1)), robust=true, maxiter=2)
    sem()                                                 # compile
    # Each of the two M-steps gathers `sum(length, cat) == N` columns of `D` bytes, so the whole
    # fit allocates 0.42 MiB. Row-slicing the views instead costs `D * N * 8` per M-step and the
    # same fit allocates 2.16 MiB, so the bound below is a factor 2.3 away from either side.
    @test (@allocated sem()) < 8 * D * N
end

@testset "MNIST Bernoulli Mixture (ClassicEM and StochasticEM)" begin
    binarify(x) = x != 0 ? true : false
    dataset = MNIST(:train)
    X, y = dataset[1:10000]
    Xb = binarify.(reshape(X, (28^2, size(X, 3))))
    id = [findall(y .∈ i) for i in 0:9]
    dist_guess = [product_distribution(Bernoulli.(mean(Xb[:, l] for l in id[i]))) for i in eachindex(id)]
    α = fill(1 / 10, 10)
    mix_guess = MixtureModel(dist_guess, α)

    # ClassicEM: check it runs and loglikelihood is non-decreasing
    mix_mle, hist = fit_mle(mix_guess, Xb; infos=true, robust=true, maxiter=20, method=ClassicEM())
    @test hist["iterations"] <= 20
    @test all(diff(hist["logtots"]) .>= -1e-8)

    # StochasticEM: just check it runs
    mix_mle_s, hist_s = fit_mle(mix_guess, Xb; infos=true, robust=true, maxiter=20, method=StochasticEM(StableRNG(1)))
    @test hist_s["iterations"] <= 20
end
