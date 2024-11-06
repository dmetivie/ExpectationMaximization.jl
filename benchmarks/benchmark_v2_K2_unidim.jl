# # Benchmarking

# These are preliminaries benchmarks between different (Gaussian) Mixture packages. In fact `ExpectationMaximization.jl` deals with arbitrary mixture models (given the functions `log(dist)` and `fit_mle(dist, y[, w])`). `Sklearn` and `GaussianMixtures.jl` are specialized for Gaussian Mixtures. `mixtools` support several mixtures type.

# In principle I should benchmark larger hidden state/latent/class `K`. Here it is only `K = 2`. The dimension also should be larger.
# I tried to be as fair as possible with the benchmarks, but some packages do many things I am not certain to have completely avoided e.g. I want all benchmark to have same initial state, however `sklearn` tries to find good initial condition.

# ## Set up
cd(@__DIR__) #src
import Pkg; Pkg.activate(".") #src
using StableRNGs
using Distributions, LinearAlgebra
using BenchmarkTools, Test
using StatsPlots, LaTeXStrings

using ExpectationMaximization 
using GaussianMixtures 
using PythonCall, CondaPkg
using RCall

# One could use `@rimport microbenchmark` to benchmark in R and `timeit = pyimport("timeit")` in Python to prevent a potential overhead time of using `RCall.jl` and `PythonCall.jl`, however I find them very hard to manipulate compare to `BenchmarkTools.jl`.

# Set threads to 1 to be fair. (Note that `ExpectationMaximization.jl` is expected to have thread support at some point).

BLAS.set_num_threads(1)

Threads.nthreads()
# Python threads
os = pyimport("os")
os.environ["OMP_NUM_THREADS"] = "1"


# ### R

# For some reason this was not working from Julia.
# ```julia
# R"""install.packages("mixtools", repos='http://cran.us.r-project.org')"""
# ```
@rimport mixtools # https://www.rdocumentation.org/packages/mixtools/versions/2.0.0
R_EM = mixtools.normalmixEM

#-
function R_mixtools(y, mu, sigma, alpha, iters=100)
    K = length(mu)
    res = R_EM(y, k=K, lambda=alpha, mu=mu, sigma=sigma, maxit=iters, epsilon=1e-5)
    time = @belapsed $R_EM($y, k=$K, lambda=$alpha, mu=$mu, sigma=$sigma, maxit=$iters, epsilon=$1e-5)
    return time, res
end

# ### Python

# To install and import `sklearn`.

# Install scikit-learn
# ```julia
# CondaPkg.add("scikit-learn"; channel="conda-forge")
# ```

np = pyimport("numpy")
sklearn = pyimport("sklearn.mixture") # scikit-learn 1.3.2
py_EM = sklearn.GaussianMixture
#-
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

# `mixem` was removed from benchmark because to slow and numerically instable i.e. return `NaN`.
## mixem = pyimport("mixem")
# Here was the code for `mixem`
# py_mixemEM = mixem.

# ```julia
# function py_mixem(y, mu, sigma, alpha, iters=100)
#     K = length(mu)
#     dists = [mixem.distribution.NormalDistribution(mu=mu[i], sigma=sigma[i]) for i in 1:K]
#     return @belapsed $py_mixemEM($y, dists, initial_weights=alpha, max_iterations=iters, progress_callback = :none, tol=1e-5, tol_iters=1)
# end
# ```

# ### `GaussianMixtures.jl`
# Julia package specilized for Gaussian Mixtures. Not sure it is actively maintained (it works however).
function jl_GMM(y, mu, sigma, alpha, iters=100)
    gmm = GMM(length(mu), 1)  # initialize an empty GMM object
    ## stick in our starting values
    gmm.μ[:, 1] .= mu
    gmm.Σ[:, 1] .= sigma
    gmm.w[:, 1] .= alpha

    ## run em!
    res = copy(gmm)
    em!(res, y[:,:], nIter=iters)
    time = @belapsed $em!(g, $(y[:,:]), nIter=iters) setup=(g = copy($gmm))
    return time, res
end

# ### `ExpectationMaximization.jl`

function my_em(y, mu, sigma, alpha, iters=100)
    mix = MixtureModel([Normal(mu[i], sigma[i]) for i in eachindex(mu)], alpha)
    res = fit_mle(mix, y, maxiter=iters, atol=1e-5)
    time = @belapsed $fit_mle($mix, $y, maxiter=$iters, atol=$1e-5)
    return time, res
end

# ## Test 

Names = ["ExpectationMaximization.jl", "GaussianMixtureModel.jl", "mixtools.R", "Sklearn.py"] #, "mixem.py"]
# ### True values
μ = [-4, 10]
σ = [2, 10]
α = [0.8, 0.2]
mi = MixtureModel([Normal(μ[i], σ[i]) for i in eachindex(μ)], α)

# ### Guess 
mu = [-1, 1]
sigma = [1, 1]
alpha = [0.5, 0.5]

iters = 8
NN = Int[500, 1000, 5000, 1e4, 5e4, 1e5, 5e5, 1e6, 2e6]

KK = [2]

d = Dict{String,Matrix}()
for key in Names
    d[key] = ones(length(NN), length(KK))
end

# Note that the `Sklearn` is kinda supsect since it triggers(ed?) a warning regarding Kmean which I do not want to use. 
# I asked a question [here](https://github.com/scikit-learn/scikit-learn/discussions/25916)
# Plus there seem to have a long compilation time

for j in eachindex(KK)
    K = KK[j]
    for i in eachindex(NN)
        n = NN[i]
        rtol = 2e-2
        rng = StableRNG(123)
        y = rand(rng, mi, n)
        @show n

        d["ExpectationMaximization.jl"][i, j], res_my = my_em(y, mu, sigma, alpha, iters)
        d["GaussianMixtureModel.jl"][i, j], res_gmm = jl_GMM(y, mu, sigma, alpha, iters)

        d["Sklearn.py"][i, j], res_sk = py_sklearn(y, mu, sigma, alpha, iters)

        ## this seems to work just fine (just print an annoying warning about convergence that I could not remove)
        d["mixtools.R"][i, j], res_R = R_mixtools(y, mu, sigma, alpha, iters)

        ## mixtools vs EM.jl
        @test mean.(res_my.components) ≈ res_R[3][1:K] rtol = rtol
        @test std.(res_my.components) ≈ res_R[4][1:K] rtol = rtol
        @test probs(res_my) ≈ res_R[2][1:K] rtol = rtol

        ## mixtools vs py_sklearn
        @test vec(pyconvert(Matrix,res_sk.means_)) ≈ res_R[3][1:K] rtol = rtol
        @test sqrt.(vec(pyconvert(Matrix,res_sk.covariances_))) ≈ res_R[4][1:K] rtol = rtol
        @test pyconvert(Vector,res_sk.weights_) ≈ res_R[2][1:K] rtol = rtol

        ## mixtools vs gmm.jl
        @test vec(res_gmm.μ) ≈  res_R[3][1:K] rtol = rtol
        @test sqrt.(vec(res_gmm.Σ)) ≈ res_R[4][1:K] rtol = rtol
        @test res_gmm.w ≈ res_R[2][1:K] rtol = rtol
    end
end

# ## Results

xNN = 10 .^ (2:6)
yti = 10 .^ (-4.0:1)
default(fontfamily="Computer Modern", linewidth=1, label=nothing)
begin
    K = 2
    plot(legend=:topleft, size=(900, 800), legendfontsize = 16, bottom_margin = 10Plots.mm, tickfontsize = 16, xlabelfontsize = 18, ylabelfontsize = 18)
    [plot!(NN, d[key][:, 1], label=key, c=na) for (na, key) in collect(enumerate(Names))]
    ylabel!("Time (s)", ylabelfontsize = 16)
    yaxis!(:log10)
    xaxis!(:log10)
    xlabel!(L"N")
    xticks!(xNN)
    yticks!(yti, minorticks=9, minorgrid=true, gridalpha=0.3, minorgridalpha=0.15)
end
#-
savefig("timing_K_2.pdf") 
savefig("timing_K_2.svg") 

# Ratio view
begin
    K = 2
    plot(legend=:topleft, size=(900, 800), legendfontsize = 16, bottom_margin = 10Plots.mm, tickfontsize = 16, xlabelfontsize = 18, ylabelfontsize = 18)
    [plot!(NN, d[key][:, 1]./d["ExpectationMaximization.jl"][:, 1], label=key, c=na+1) for (na, key) in collect(enumerate(Names[2:end]))]
    ylabel!("Time (s)")
    hline!([1], c=:black, label = :none)
        xaxis!(:log10)
    ylims!(0,12.5)
    xlabel!(L"N")
    xticks!(xNN)
end
#-
savefig("timing_K_2_ratio.pdf")
savefig("timing_K_2_ratio.svg")

# ## Reproducibility

# Computer and Julia
using InteractiveUtils
InteractiveUtils.versioninfo()

# Julia packages
import Pkg; Pkg.status()

# Python
sys = pyimport("sys")
print("Python version:", sys.version)
# Sklearn
CondaPkg.status()
# R and packages
println(R"sessionInfo()")