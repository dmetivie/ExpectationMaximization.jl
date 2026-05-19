```@meta
EditURL = "../../../examples/examples_multivariate.jl"
```

````@example examples_multivariate
import Pkg #hide

using Distributions
using ExpectationMaximization
using StatsPlots
using Random
Random.seed!(1234)
````

# Multivariate Examples

## Old Faithful Geyser Data (Multivariate Normal)

This seems like a canonical example for Gaussian mixtures, so let's do it.

Using [Clustering.jl](https://juliastats.org/Clustering.jl/dev/index.html) package, one could easily initialize the `mix_guess` using K-means algorithms (and others).

!!! note
    I like using [`ClipData.jl`](https://github.com/pdeffebach/ClipData.jl) to quickly copy-paste data from the web (or any table in a spreadsheet like Excel) into Julia.
    ```julia
    data = cliptable() |> DataFrame
    ```
    For continuous integration of this example I just download the data.

````@example examples_multivariate
using DataFrames, CSV

data = CSV.read(download("https://gist.githubusercontent.com/curran/4b59d1046d9e66f2787780ad51a1cd87/raw/9ec906b78a98cf300947a37b56cfe70d01183200/data.tsv"), DataFrame)
first(data, 10)
````

The data is now converted into a matrix `y` of size 2 x N (N is the number of observations) to be fitted with a Gaussian mixture model.

````@example examples_multivariate
y = permutedims(Matrix(data))
````

We choose an initial guess for the parameters of the Gaussian mixture model.

````@example examples_multivariate
D₁guess = MvNormal([22, 55], [1 0.6; 0.6 1])
D₂guess = MvNormal([4, 80], [1 0.2; 0.2 1])
mix_guess = MixtureModel([D₁guess, D₂guess], [1 / 2, 1 / 2])

mix_mle, info = fit_mle(mix_guess, y, infos=true)
````

We can now plot the fitted model.

````@example examples_multivariate
begin
    @df data scatter(:eruptions, :waiting, label="Observations",
        xlabel="Duration of the eruption (min)",
        ylabel="Duration until the next eruption (min)")
    xrange = 1:0.05:6
    yrange = 40:0.1:100
    zlevel = [pdf(mix_mle, [x, y]) for y in yrange, x in xrange]
    contour!(xrange, yrange, zlevel)
end
````

## MNIST Dataset: Bernoulli Mixture

A classical example in clustering (pattern recognition) is the MNIST handwritten digits' data sets.
One of the simplest[^1] ways to address the problem is to fit a Bernoulli mixture with 10 components for the ten digits 0, 1, 2, ..., 9 (see [Pattern Recognition and Machine Learning by C. Bishop, Section 9.3.3.](https://d1wqtxts1xzle7.cloudfront.net/30428242/bg0137-libre.pdf?1390888009=&response-content-disposition=inline%3B+filename%3DPattern_recognition_and_machine_learning.pdf&Expires=1679414339&Signature=fEpdcg3ZXYvfcSTtQBe6pF2UqhlrEV2hG0~djNJrglRKQRmt3iYE1OmgoEO0byuCs5HNRLFXKqKNs7l5ry-1pLTzMU87W8QqU8zn0STVozwWL-T2Yd-dmEjw-f8bbrvoq5WOzcUfj25MxLCfJRx66Q~zJwNDJYYnFeAyYFJdWnfPBf3GsR7nR6GYCQH~qvLfzGh~zOYHa7Gmr3yvz9mkjWFWMM4pAikNTmmw6F~N1rqXra2ZIL4kQqvfG-WjU-j0G5TdItSYn2FfoLcXPHXvA1nLfTB2vY5sGY8YKgFqez-~eQKt72diTZZnKNBJKnnnbZ0iWJzuTqzsqi2C4hVpLQ__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA) for more context).
Each of the components is a product distribution of $28\times 28$ independent Bernoulli. This simple (but rather big) model can be fitted via the EM algorithm.

!!! note
    Having a product distribution of Bernoulli means that the model assumes that each pixel is independent of the others given the component (digit) of the mixture. This is of course a very strong assumption and other models with more complex dependencies between pixels could be used for example see [SpatialBernoulli.jl](https://github.com/caroline-cognot/SpatialBernoulli.jl).

[^1]: I am not sure if this was historically one of the first way to approach this problem. Anyway, this is more like an academic application rather than a good method to solve the MNIST problem.

````@example examples_multivariate
using MLDatasets: MNIST

binarify(x) = x != 0 ? true : false

dataset = MNIST(:train)
````

````@example examples_multivariate
X, y = dataset[:]
Xb = binarify.(reshape(X, (28^2, size(X, 3))))
id = [findall(y .∈ i) for i in 0:9];
nothing #hide
````

As initial guess, we can use the mean of each class as the parameter of the Bernoulli distribution for each component of the mixture.
This is of course a very informed guess to help the EM algorithm to converge toward a good solution and avoid local maxima (but it also shows that EM can be used for clustering with a good initialization).

````@example examples_multivariate
dist_guess = [product_distribution(Bernoulli.(mean(Xb[:, l] for l in id[i]))) for i in eachindex(id)]
α = fill(1 / 10, 10)
mix_guess = MixtureModel(dist_guess, α);
nothing #hide
````

Now we can fit the model with the EM algorithm.

````@example examples_multivariate
@time mix_mle, info = fit_mle(mix_guess, Xb, infos=true, display=:iter, robust=true, maxiter=5);
info
````

We plot the resulting fitted model.

````@example examples_multivariate
begin
    pmle = [heatmap(reshape(succprob.(components(mix_mle)[i].v), 28, 28)', yflip=true,
    cmap=:grays, clims=(0, 1), ticks=:none) for i in eachindex(id)]
    plot(pmle..., layout=(2, 5), size=(900, 300))
end
````

We now test the model in a Machine Learning classification task.

````@example examples_multivariate
test_data = MNIST(:test)
test_X, test_y = test_data[:]
test_Xb = binarify.(reshape(test_X, (28^2, size(test_X, 3))))
predict_y = predict(mix_mle, test_Xb, robust=true)

println("There are 28^2*10 + 9 = ", 28^2 * 10 + (10 - 1), " parameters in the model.")
println("Learning accuracy ", count(predict_y .- 1 .== test_y) / length(test_y), "%.")
````

The accuracy is of course far from the current best models (though it has a relative number of parameters). For example, this model assumes conditional independence of each pixel given the components (which is far from being true), and the EM algorithm may have converged to a local maximum.

## Another Multivariate Gaussian Mixture

Here we show another example of fitting a Gaussian mixture with the EM algorithm. The data is generated from a mixture of two 2D Gaussians, and we fit a Gaussian mixture model to it.

````@example examples_multivariate
N = 2_000
θ₁ = [-1, 1]
θ₂ = [0, 2]
Σ₁ = [0.5 0.5; 0.5 1]
Σ₂ = [1 0.1; 0.1 1]
β = 0.3

D₁ = MvNormal(θ₁, Σ₁)
D₂ = MvNormal(θ₂, Σ₂)
mix_true = MixtureModel([D₁, D₂], [β, 1 - β])
````

````@example examples_multivariate
y = rand(mix_true, N)

D₁guess = MvNormal([0.2, 1], [1 0.6; 0.6 1])
D₂guess = MvNormal([1, 0.5], [1 0.2; 0.2 1])
mix_guess = MixtureModel([D₁guess, D₂guess], [0.4, 0.6]);
nothing #hide
````

Now we can fit the model with the EM algorithm.

````@example examples_multivariate
mix_mle = fit_mle(mix_guess, y; display=:none, atol=1e-3, robust=false, infos=false)
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

