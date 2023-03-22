# Benchmarking

using Distributions, Random
using BenchmarkTools, DataFrames, Test
using RCall, GaussianMixtures
using StatsPlots
using ExpectationMaximization
using PyCall
using LinearAlgebra

R"""
library(microbenchmark)
library(mixtools)
"""

py"""
import os
os.environ["OMP_NUM_THREADS"] = '1'
import numpy as np
from sklearn import mixture
import timeit
import mixem
"""

# GaussianMixtures.jl
function jl_GMM(y, mu, sigma, alpha, iters=100)
    gmm = GMM(length(mu), 1)  # initialize an empty GMM object
    # stick in our starting values
    gmm.μ[:, 1] .= mu
    gmm.Σ[:, 1] .= sigma
    gmm.w[:, 1] .= alpha

    # run em!
    # em!(gmm, y[:, :], nIter=iters)
    return @belapsed em!($(copy(gmm)), $y[:, :], nIter=iters)
end

function py_sklearn(y, mu, sigma, alpha, iters=100; infos = 0, verbose_interval = 1, tol = 1e-5)
    precisions_init = [inv(Diagonal(sigma .^ 2))[i, i] for i in eachindex(mu)]
    py"""
    K = len($mu)
    y = $y
    y = y.reshape(-1, 1)
    mu = $mu
    mu = mu.reshape(-1, 1)
    invsigma2 = $precisions_init
    invsigma2 = invsigma2.reshape(-1,1)
    alpha = $alpha
    iters = $iters
    infos = $infos
    verbose_interval = $verbose_interval
    g = mixture.GaussianMixture(n_components=K, covariance_type="diag", weights_init = alpha, means_init = mu, precisions_init = invsigma2, max_iter = iters, warm_start = False, verbose = infos, verbose_interval = verbose_interval, tol = $tol)
    """
    return @belapsed py"g.fit(y)"
end

function py_mixem(y, mu, sigma, alpha, iters=100)
    py"""
    K = len($mu)
    y = $y
    mu = $mu
    sigma = $sigma
    alpha = $alpha
    iters = $iters
    dists = [mixem.distribution.NormalDistribution(mu=mu[i], sigma=sigma[i]) for i in range(K)]
    """
    return @belapsed py"mixem.em(y, dists, initial_weights=alpha, max_iterations=iters, progress_callback = None, tol=1e-5, tol_iters=1)"
end

function my_em(y, mu, sigma, alpha, iters=100)
    mix = MixtureModel([Normal(mu[i], sigma[i]) for i in eachindex(mu)], alpha)
    return @belapsed fit_mle($mix, $y, maxiter=iters, atol = 1e-5)
end

function R_mixtools(y, mu, sigma, alpha, iters=100)
    @rput mu
    @rput sigma
    @rput alpha
    @rput y
    @rput iters

    R"""
    N = length(y)
    K = length(mu)
    """
    return @belapsed R"normalmixEM(y, k = K,lambda = alpha, mu = mu, sigma = sigma, maxit = iters, epsilon = 1e-5)"
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
NN = Int.([500, 1000, 5000, 1e4, 5e4, 1e5, 5e5, 1e6])

KK = [2]

d = Dict{String,Matrix}()
for key in Names
    d[key] = ones(length(NN), length(KK))
end

for j in eachindex(KK)
    K = KK[j]
    for i in eachindex(NN)
        Random.seed!(3333)
        n = NN[i]
        y = rand(mi, n)
        @show n

        # I found that @belapsed was suspect (sometimes asking to define y while it was already defined)
        d["ExpectationMaximization.jl"][i, j] = my_em(y, mu, sigma, alpha, iters)
        d["GaussianMixtureModel.jl"][i, j] = jl_GMM(y, mu, sigma, alpha, iters)

        # This is suspect since it triggers a warning regarding Kmean which I do not want to use. 
        # I asked a question [here](https://github.com/scikit-learn/scikit-learn/discussions/25916)
        # Plus there seem to have a long compilation time
        d["Sklearn.py"][i, j] = py_sklearn(y, mu, sigma, alpha, iters)

        # this one seems to overflow for n larger than ~500, I believe it is due to the way log(sum(exp))...
        # def logsumexp(X,axis=None,keepdims=1,log=1):
        # d["mixem.py"][i, j] = py_mixem(y, mu, sigma, alpha, iters)

        # this seems to work just fine (just print an annoying warning about convergence that I could not remove)
        d["mixtools.R"][i, j] = R_mixtools(y, mu, sigma, alpha, iters)
    end
end

Random.seed!(3333)
n = 500
i = 1
j =1
y = rand(mi, n)
d["mixem.py"][i, j] = py_mixem(y, mu, sigma, alpha, iters)

xNN = 10 .^(2:6)
yti = 10 .^(-4.:1)
default(fontfamily="Computer Modern", linewidth=1, label=nothing, size=(1000, 800))
begin
    K = 2
    plot(legend = :topleft)
    [plot!(NN, d[key][:, 1], label=key, c=na) for (na, key) in collect(enumerate(Names))[1:end-1]]
    scatter!([NN[1]], [d[Names[end]][1, 1]], label=Names[end], c=length(Names))
    ylabel!("time (s)")
    yaxis!(:log10)
    xaxis!(:log10)
    xlabel!("n")
    xticks!(xNN)
    yticks!(yti, minorticks = 9, minorgrid = true, gridalpha = 0.2)
end
