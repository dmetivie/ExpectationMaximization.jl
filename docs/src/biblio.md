
# Bibliography

## Theory

The EM algorithm was introduced by A. P. Dempster, N. M. Laird and D. B. Rubin in 1977 in the reference paper [*Maximum Likelihood from Incomplete Data Via the EM Algorithm*](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1977.tb01600.x). This is a very generic algorithm, working for almost any distributions.
I also added the stochastic version introduced by G. Celeux, and J. Diebolt. in 1985 in [*The SEM Algorithm: A probabilistic teacher algorithm derived from the EM algorithm for the mixture problem*](https://cir.nii.ac.jp/crid/1574231874553755008).
Other versions can be added PR are welcomed.

## Implementations

Despite being generic, to my knowledge, almost all coding implementations are specific to some mixtures class (mostly Gaussian mixtures, sometime double exponential or Bernoulli mixtures).

In this package, thanks to Julia generic code spirit, one can just code the algorithm, and it works for all distributions.

I know of the Python [`mixem`](https://github.com/sseemayer/mixem) package doing also using a generic algorithm implementation. However, the available distribution choice is very limited as the authors have to define each distribution (Top-Down approach).
This package does not define distribution[^1], it simply uses the `Distribution` type and what is in `Distributions.jl`.

In Julia, there is the [`GaussianMixtures.jl`](https://github.com/davidavdav/GaussianMixtures.jl) package that also does EM. It seems a little faster than my implementation when used with Gaussian mixtures (I'd like to understand what is creating this difference, though, maybe the in-place allocation while `fit_mle` creates copy).
However, I am not sure if this is maintained anymore.

Have a look at the [benchmark](@ref Benchmarks) section for some comparisons.

I was inspired by **Florian Oswald** [page](https://floswald.github.io/post/em-benchmarks/) and **Maxime Mouchet** [`HMMBase.jl` package](https://github.com/maxmouchet/HMMBase.jl).

[^1]: I added `fit_mle` methods for Product distributions, weighted Laplace and Dirac. I am doing PR to merge that directly into the `Distributions.jl` package.
