# Benchmarking

using Distributions, Random
using BenchmarkTools, Test
using RCall
using GaussianMixtures # v0.3.8
using StatsPlots
using ExpectationMaximization # v0.2.2
using PyCall
using Conda
using LinearAlgebra
using LaTeXStrings
# @rimport microbenchmark # One could use that to benchmark in R to prevent a potential overhead time of using RCall.
# timeit = pyimport("timeit")  # One could use that to benchmark in Python to prevent a potential overhead time of using PyCall.

@rimport mixtools # https://www.rdocumentation.org/packages/mixtools/versions/2.0.0

os = pyimport("os")
os.environ["OMP_NUM_THREADS"] = "1"

np = pyimport("numpy")
sklearn = pyimport("sklearn.mixture") # scikit-learn 1.3.2
mixem = pyimport("mixem")

# GaussianMixtures.jl
function jl_GMM(y, mu, sigma, alpha, iters=100)
    gmm = GMM(length(mu), 1)  # initialize an empty GMM object
    # stick in our starting values
    gmm.μ[:, 1] .= mu
    gmm.Σ[:, 1] .= sigma
    gmm.w[:, 1] .= alpha

    # run em!
    # em!(gmm, y[:, :], nIter=iters)
    res = copy(gmm)
    em!(res, y[:,:], nIter=iters)
    time = @belapsed $em!(g, $(y[:,:]), nIter=iters) setup=(g = copy($gmm))
    return time, res
end

py_EM = sklearn.GaussianMixture
function py_sklearn(y, mu, sigma, alpha, iters=100; infos=0, verbose_interval=1, tol=1e-5)
    precisions_init = [inv(Diagonal(sigma .^ 2))[i, i] for i in eachindex(mu)]
    K = length(mu)
    Y = reshape(y, (length(y), 1))
    MU = reshape(mu, (length(mu), 1))
    invsigma2 = reshape(precisions_init, (length(precisions_init), 1))
    g₀ = py_EM(n_components=K, covariance_type="diag", weights_init=alpha, means_init=MU, precisions_init=invsigma2, max_iter=iters, warm_start=false, verbose=infos, verbose_interval=verbose_interval, tol=tol).fit
    g = g₀
    res = g(Y)
    time = @belapsed g($Y) setup=(g = $g₀)
    return time, res
end

# py_mixemEM = mixem.em
# function py_mixem(y, mu, sigma, alpha, iters=100)
#     K = length(mu)
#     dists = [mixem.distribution.NormalDistribution(mu=mu[i], sigma=sigma[i]) for i in 1:K]
#     return @belapsed $py_mixemEM($y, dists, initial_weights=alpha, max_iterations=iters, progress_callback = :none, tol=1e-5, tol_iters=1)
# end

function my_em(y, mu, sigma, alpha, iters=100)
    mix = MixtureModel([Normal(mu[i], sigma[i]) for i in eachindex(mu)], alpha)
    res = fit_mle(mix, y, maxiter=iters, atol=1e-5)
    time = @belapsed $fit_mle($mix, $y, maxiter=$iters, atol=$1e-5)
    return time, res
end

# function in question
R_EM = mixtools.normalmixEM

function R_mixtools(y, mu, sigma, alpha, iters=100)
    K = length(mu)
    res = R_EM(y, k=K, lambda=alpha, mu=mu, sigma=sigma, maxit=iters, epsilon=1e-5)
    time = @belapsed $R_EM($y, k=$K, lambda=$alpha, mu=$mu, sigma=$sigma, maxit=$iters, epsilon=$1e-5)
    return time, res
end

Names = ["ExpectationMaximization.jl", "GaussianMixtureModel.jl", "mixtools.R", "Sklearn.py", "mixem.py"]
# true values
μ = [-4, 10]
σ = [2, 10]
α = [0.8, 0.2]
mi = MixtureModel([Normal(μ[i], σ[i]) for i in eachindex(μ)], α)

# guess 
mu = [-1, 1]
sigma = [1, 1]
alpha = [0.5, 0.5]

iters = 8
NN = Int[500, 1000, 5000, 1e4, 5e4, 1e5, 5e5, 1e6]

KK = [2]

d = Dict{String,Matrix}()
for key in Names
    d[key] = ones(length(NN), length(KK))
end

for j in eachindex(KK)
    K = KK[j]
    for i in eachindex(NN)
        rtol = 1e-2
        Random.seed!(3333)
        n = NN[i]
        y = rand(mi, n)
        @show n

        # I found that @belapsed was suspect (sometimes asking to define y while it was already defined)
        d["ExpectationMaximization.jl"][i, j], res_my = my_em(y, mu, sigma, alpha, iters)
        d["GaussianMixtureModel.jl"][i, j], res_gmm = jl_GMM(y, mu, sigma, alpha, iters)

        # This is suspect since it triggers a warning regarding Kmean which I do not want to use. 
        # I asked a question [here](https://github.com/scikit-learn/scikit-learn/discussions/25916)
        # Plus there seem to have a long compilation time
        d["Sklearn.py"][i, j], res_sk = py_sklearn(y, mu, sigma, alpha, iters)

        # this one seems to overflow for n larger than ~500, I believe it is due to the way log(sum(exp))...
        # def logsumexp(X,axis=None,keepdims=1,log=1):
        # d["mixem.py"][i, j] = py_mixem(y, mu, sigma, alpha, iters)

        # this seems to work just fine (just print an annoying warning about convergence that I could not remove)
        d["mixtools.R"][i, j], res_R = R_mixtools(y, mu, sigma, alpha, iters)

        # mixtools vs EM.jl
        @test mean.(res_my.components) ≈  res_R[3][1:K] rtol = rtol
        @test std.(res_my.components) ≈ res_R[4][1:K] rtol = rtol
        @test probs(res_my) ≈ res_R[2][1:K] rtol = rtol

        # mixtools vs py_sklearn
        @test vec(res_sk.means_) ≈  res_R[3][1:K] rtol = rtol
        @test sqrt.(vec(res_sk.covariances_)) ≈ res_R[4][1:K] rtol = rtol
        @test res_sk.weights_ ≈ res_R[2][1:K] rtol = rtol

        # mixtools vs gmm.jl
        @test vec(res_gmm.μ) ≈  res_R[3][1:K] rtol = rtol
        @test sqrt.(vec(res_gmm.Σ)) ≈ res_R[4][1:K] rtol = rtol
        @test res_gmm.w ≈ res_R[2][1:K] rtol = rtol
    end
end

# Random.seed!(3333)
# n = 500
# i = 1
# j =1
# y = rand(mi, n)
# d["mixem.py"][i, j] = py_mixem(y, mu, sigma, alpha, iters)

xNN = 10 .^ (2:6)
yti = 10 .^ (-4.0:1)
default(fontfamily="Computer Modern", linewidth=1, label=nothing)
begin
    K = 2
    plot(legend=:topleft, size=(900, 800), legendfontsize = 16)
    [plot!(NN, d[key][:, 1], label=key, c=na) for (na, key) in collect(enumerate(Names))[1:end-1]]
    # scatter!([NN[1]], [d[Names[end]][1, 1]], label=Names[end], c=length(Names))
    ylabel!("Time (s)", ylabelfontsize = 16)
    yaxis!(:log10)
    xaxis!(:log10)
    xlabel!(L"n", xlabelfontsize = 18)
    xticks!(xNN, xtickfontsize = 16)
    yticks!(yti, ytickfontsize = 16, minorticks=9, minorgrid=true, gridalpha=0.3, minorgridalpha=0.15)
end
savefig("timing_K_2.svg")
