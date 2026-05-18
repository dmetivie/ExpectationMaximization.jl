using Markdown#hide
import Pkg #hide
Pkg.activate(joinpath(@__DIR__, "..", "docs")) #src

using Distributions
using ExpectationMaximization
using StatsPlots

using Random
Random.seed!(1234)
md"""
# Univariate Examples
"""

md"""
## Normal + Laplace + Exponential

Here for fun we test a very unusual combination of mixtures (with distributions not necessarily having the same support, like `Normal` and `Exponential`).
"""

# Parameters
N = 5_000
θ₁ = 5
θ₂ = 0.8
α = 0.5
β = 0.3
μ = -1
a = 2

mix_true = MixtureModel([Laplace(μ, θ₁), Normal(α, θ₂), Exponential(a)], [β / 2, 1 - β, β / 2])

# Components of the mixture
begin 
    plot(mix_true, label=["Laplace" "Normal" "Exponential"])
    ylabel!("Log PDF", yaxis=:log10)
end

md"""
Now we generate some data from this mixture.
"""

y = rand(mix_true, N)

md"""
We specify an initial condition for the EM fit.
"""
mix_guess = MixtureModel([Laplace(-1, 1), Normal(2, 1), Exponential(3)], [1 / 3, 1 / 3, 1 / 3]);


# Fit Classic EM
mix_mle_C, hist_C = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true, method=ClassicEM())

# Fit Stochastic EM
mix_mle_S, hist_S = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true, method=StochasticEM())

# Plot the results
begin
    x = -20:0.1:20
    pmix = plot(x, pdf.(mix_true, x), label="True", ylabel="PDF")
    plot!(pmix, x, pdf.(mix_guess, x), label="Initial guess")
    plot!(pmix, x, pdf.(mix_mle_C, x), label="fit EM")
    plot!(pmix, x, pdf.(mix_mle_S, x), label="fit sEM")

    ploss = plot(hist_C["logtots"], label="ClassicEM with $(hist_C["iterations"]) iterations",
        c=3, xlabel="EM iterations", ylabel="Log Likelihood")
    plot!(ploss, hist_S["logtots"], label="StochasticEM with $(hist_S["iterations"]) iterations",
        c=4, s=:dot)

    plot(pmix, ploss)
end

md"""
## Mixture of Mixture and Univariate

To showcase the generality of the package, we show how training a `MixtureModel` composed of a univariate distribution + a `MixtureModel` works.
Note that, in practice, this can cause some identifiability issues (if everything is `Normal` for example).
"""

θ₁ = -5
θ₂ = 2
σ₁ = 1
σ₂ = 1.5
θ₀ = 0.1
σ₀ = 0.1

α = 1 / 2
β = 0.3

d1 = MixtureModel([Laplace(θ₁, σ₁), Normal(θ₂, σ₂)], [α, 1 - α])
d2 = Normal(θ₀, σ₀)
mix_true = MixtureModel([d1, d2], [β, 1 - β])

md"""
We generate some data from this mixture.
"""
y = rand(mix_true, N)

md"""
We choose initial guess very close to the true solution to show EM convergence.
This Gaussian mixture of mixture is non-identifiable: other solutions far from the truth exist.
"""

d1_guess = MixtureModel([Laplace(θ₁ - 2, σ₁ + 1), Normal(θ₂ + 1, σ₂ + 1)], [α + 0.1, 1 - α - 0.1])
d2_guess = Normal(θ₀ - 1, σ₀ - 0.05)

mix_guess = MixtureModel([d1_guess, d2_guess], [β + 0.1, 1 - β - 0.1])
mix_mle, hist_C = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true)

begin 
    x = -20:0.1:20
    pmix = plot(x, pdf.(mix_true, x), label="True", ylabel="PDF")
    plot!(pmix, x, pdf.(mix_guess, x), label="Initial guess")
    plot!(pmix, x, pdf.(mix_mle, x), label="fit EM with $(hist_C["iterations"]) iterations")
end
