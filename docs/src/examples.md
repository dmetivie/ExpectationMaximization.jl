
# Examples

```@example 1
using Distributions
using ExpectationMaximization
using StatsPlots
```

## Univariate continuous  

### Normal + Laplace + Exponential

```@example 1
# Parameters
N = 5_000
θ₁ = 5
θ₂ = 0.8
α = 0.5
β = 0.3
μ = -1
a = 2

mix_true = MixtureModel([Laplace(μ, θ₁), Normal(α, θ₂), Exponential(a)], [β/2, 1 - β, β/2])

# Components of the mixtures
plot(mix_true, label = ["Laplace" "Normal" "Exponential"])
ylabel!("Log PDF", yaxis = :log10)
```

```@example 1
# Sampling
y = rand(mix_true, N)

# Initial Condition
mix_guess = MixtureModel([Laplace(-1, 1), Normal(2, 1), Exponential(3)], [1/3, 1/3, 1/3])

# Fit Classic EM
mix_mle_C, hist_C = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true, method = ClassicEM())

# Fit Stochastic EM
mix_mle_S, hist_S = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true, method = StochasticEM())

x = -20:0.1:20
pmix = plot(x, pdf.(mix_true, x), label = "True", ylabel = "PDF")
plot!(pmix, x, pdf.(mix_guess, x), label = "Initial guess")
plot!(pmix, x, pdf.(mix_mle_C, x), label = "fit EM")
plot!(pmix, x, pdf.(mix_mle_S, x), label = "fit sEM")

ploss = plot(hist_C["logtots"], label = "ClassicEM with $(hist_C["iterations"]) iterations", c = 3, xlabel = "EM iterations", ylabel = "Log Likelihood")
plot!(ploss, hist_S["logtots"], label = "StochasticEM  with $(hist_S["iterations"]) iterations", c = 4, s = :dot)

plot(pmix, ploss)
```

### Mixture of Mixture and univariate

```@example 1
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
y = rand(mix_true, N)

# We choose initial guess very close to the true solution just to show the EM algorithm convergence.
# This particular choice of mixture of mixture Gaussian with another Gaussian is non identifiable hence we execpt other solution far away from the true solution
d1_guess = MixtureModel([Laplace(θ₁ - 2, σ₁ + 1), Normal(θ₂ + 1, σ₂ + 1)], [α + 0.1, 1 - α - 0.1])
d2_guess = Normal(θ₀ - 1, σ₀ - 0.05)

mix_guess = MixtureModel([d1_guess, d2_guess], [β + 0.1, 1 - β - 0.1])
mix_mle, hist_C = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=true)
y_guess = rand(mix_mle, N)

x = -20:0.1:20
pmix = plot(x, pdf.(mix_true, x), label = "True", ylabel = "PDF")
plot!(pmix, x, pdf.(mix_guess, x), label = "Initial guess")
plot!(pmix, x, pdf.(mix_mle, x), label = "fit EM with $(hist_C["iterations"]) iterations")
```

## Multivariate mixtures

### Old Faithful Geyser Data (Multivariate Normal)

This seems like a canonical example for Gaussian mixtures, so let's do it. Note the use of the amazing [`ClipData.jl`](https://github.com/pdeffebach/ClipData.jl).

Using [Clustering.jl](https://juliastats.org/Clustering.jl/dev/index.html) package, one could easily initilize the `mix_guess` using K-means algorithms (and others).

```julia
using ClipData, DataFrames, StatsPlots
using Distributions, ExpectationMaximization
# https://gist.githubusercontent.com/curran/4b59d1046d9e66f2787780ad51a1cd87/raw/9ec906b78a98cf300947a37b56cfe70d01183200/data.tsv
data = cliptable() |> DataFrame

@df data scatter(:eruptions, :waiting, label = "Observations", xlabel = "Duration of the eruption (min)", ylabel = " Duration until the next eruption (min)")

y = permutedims(Matrix(data))

D₁guess = MvNormal([22, 55], [1 0.6; 0.6 1])
D₂guess = MvNormal([4, 80], [1 0.2; 0.2 1])
mix_guess = MixtureModel([D₁guess, D₂guess], [1/2,1/2])

mix_mle, info = fit_mle(mix_guess, y, infos = true)

# mix_mleS, infoS = fit_mle(mix_guess, y, infos = true, method = StochasticEM())

xrange = 1:0.05:6
yrange = 40:0.1:100
zlevel = [pdf(mix_mle, [x, y]) for y in yrange, x in xrange]
contour!(xrange, yrange, zlevel)
```

![old_faithful](https://user-images.githubusercontent.com/46794064/227059681-53e08e6d-8a77-4f52-b1dc-50e3e3794763.svg)

### Another Multivariate Gaussian Mixtures

```julia
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
y = rand(mix_true, N)

# Initial Condition
D₁guess = MvNormal([0.2, 1], [1 0.6; 0.6 1])
D₂guess = MvNormal([1, 0.5], [1 0.2; 0.2 1])
mix_guess = MixtureModel([D₁guess, D₂guess], [0.4, 0.6])

# Fit MLE
mix_mle = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)
```

### MNIST dataset: Bernoulli Mixture

A classical example in clustering (pattern recognition) is the MNIST handwritten digits' data sets.
One of the simplest[^1] ways to address the problem is to fit a Bernoulli mixture with 10 components for the ten digits 0, 1, 2, ..., 9 (see [Pattern Recognition and Machine Learning by C. Bishop, Section 9.3.3.](https://d1wqtxts1xzle7.cloudfront.net/30428242/bg0137-libre.pdf?1390888009=&response-content-disposition=inline%3B+filename%3DPattern_recognition_and_machine_learning.pdf&Expires=1679414339&Signature=fEpdcg3ZXYvfcSTtQBe6pF2UqhlrEV2hG0~djNJrglRKQRmt3iYE1OmgoEO0byuCs5HNRLFXKqKNs7l5ry-1pLTzMU87W8QqU8zn0STVozwWL-T2Yd-dmEjw-f8bbrvoq5WOzcUfj25MxLCfJRx66Q~zJwNDJYYnFeAyYFJdWnfPBf3GsR7nR6GYCQH~qvLfzGh~zOYHa7Gmr3yvz9mkjWFWMM4pAikNTmmw6F~N1rqXra2ZIL4kQqvfG-WjU-j0G5TdItSYn2FfoLcXPHXvA1nLfTB2vY5sGY8YKgFqez-~eQKt72diTZZnKNBJKnnnbZ0iWJzuTqzsqi2C4hVpLQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) for more context).
Each of the components is a product distribution of $28\times 28$ independent Bernoulli. This simple (but rather big) model can be fitted via the EM algorithm.

[^1]: I am not sure if this was historically one of the first way to approach this problem. Anyway this is more like an academic application rather than a good method to solve the MNIST problem.

```julia
using MLDatasets: MNIST
using Distributions, ExpectationMaximization
using Plots

binarify(x) = x != 0 ? true : false

dataset = MNIST(:train)
X, y = dataset[:]
Xb = binarify.(reshape(X, (28^2, size(X, 3))))
id = [findall(y .∈ i) for i in 0:9]

# Very Informed guess (it is not true clustering since I use the label for the initial condition (IC). It also works good with other not too far IC )
dist_guess = [product_distribution(Bernoulli.(mean(Xb[:,l] for l in id[i]))) for i in eachindex(id)]
α = fill(1/10, 10)

mix_guess = MixtureModel(dist_guess, α)
pguess = [heatmap(reshape(succprob.(dist_guess[i].v), 28,28)', yflip = :true, cmap = :grays, clims = (0,1), ticks = :none) for i in eachindex(id)]
plot(pguess..., layout = (2,5), size = (900,300))

@time mix_mle, info = fit_mle(mix_guess, Xb, infos = true, display = :iter, robust = true)

# Plot the fitted mixture components
pmle = [heatmap(reshape(succprob.(components(mix_mle)[i].v), 28,28)', yflip = :true, cmap = :grays, clims = (0,1), ticks = :none) for i in eachindex(id)]
plot(pmle..., layout = (2,5), size = (900,300))

# Test results
test_data = MNIST(:test)
test_X, test_y = test_data[:]
test_Xb = binarify.(reshape(test_X, (28^2,size(test_X, 3))))

predict_y = predict(mix_mle, test_Xb, robust = true)

println("There are 28^2*10 + 9 = ", 28^2*10 + (10-1), " parameters in the model.")
println("Learning accuracy ", count(predict_y.-1 .== test_y)/length(test_y), "%.")
```

```jldoctest
There are 28^2*10 + 9 = 7849 parameters in the model.

Learning accuracy 0.6488%.
```

The accuracy is of course far from the current best models (though it has a relative number of parameters). For example, this model assumes conditional independence of each pixel given the components (which is far from being true) + I am not sure the EM found the global maxima (and not just a local one).

![fit_mle_Bernoulli_mixtures](https://user-images.githubusercontent.com/46794064/227059598-1ae0ec10-b802-40ef-bc85-ffbdafbf276e.svg)
