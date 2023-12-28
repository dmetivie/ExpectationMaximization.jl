
# Benchmarks

I was inspired by [this benchmark](https://floswald.github.io/post/em-benchmarks/).
I am not too sure how to do 100% fair comparisons across languages[^1].
There is a small overhead for using `PyCall` and `RCall`. I checked that it was small in my experimentation (~ few milliseconds?).

I test only Gaussian Mixture since it is the most common type of mixture (remembering that this packages allow plenty of mixtures, and it should be fast in general).

In my code, I did not use fancy programming tricks, the speed only comes from Julia, `LogExpFunctions.jl` for `logsumexp!` function and `fit_mle` for each distribution's coming from `Distributions.jl` package.

## Univariate Gaussian mixture with 2 components

I compare with [Sklearn.py](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture)[^2], [mixtool.R](https://cran.r-project.org/web/packages/mixtools/index.html), [mixem.py](https://mixem.readthedocs.io/en/latest/index.html)[^3].
I wanted to try [mclust](https://cloud.r-project.org/web/packages/mclust/vignettes/mclust.html), but I did not manage to specify initial conditions

Overall, [mixtool.R](https://cran.r-project.org/web/packages/mixtools/index.html) and [mixem.py](https://mixem.readthedocs.io/en/latest/index.html) were constructed in a similar spirit as this package hence easy to use for me. [Sklearn.py](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) is build to fit the Sklearn format (all in one). `GaussianMixturesModel.jl` is build with a similar vibe.

If you have comments to improve these benchmarks, comments are welcomed.

You can find the benchmark code in [here](https://github.com/dmetivie/ExpectationMaximization.jl/tree/master/benchmarks/benchmark_v1_K2_unidim.jl).

**Conclusion: for Gaussian mixture, `ExpectationMaximization.jl` is about 2 to 10 times faster than Python or R implementations** and about as fast as the specialized Julia package `GaussianMixturesModel.jl`.

![timing_K_2_rudimentary_wo_memory_leak](https://user-images.githubusercontent.com/46794064/227195619-c75b9276-932b-4029-8b49-6cce919acc87.svg)

[^1]: Note that `@btime` with `RCall` and `PyCall` might produce a small-time overhead compare to the true R/Python time see [here for example](https://discourse.julialang.org/t/benchmarking-julia-vs-python-vs-r-with-pycall-and-rcall/37308).
I did compare with `R` `microbenchmark` and Python `timeit` and it produces very similar timing but in my experience `BenchmarkTools` is smarter and simpler to use, i.e. it will figure out alone the number of repetition to do in function of the run.

[^2]: There is a suspect triggers warning regarding K-means which I do not want to use here. I asked a question [here](https://github.com/scikit-learn/scikit-learn/discussions/25916). It lead to [this issue](https://github.com/scikit-learn/scikit-learn/issues/26015) and [that PR](https://github.com/scikit-learn/scikit-learn/pull/26021). Turns out even if intial condition were provided K-mean were still computed. However to this day 23-11-29 with `scikit-learn 1.3.2` it still get the warning. Maybe it will be in the next release? I also noted this recent [PR](https://github.com/scikit-learn/scikit-learn/pull/26416).
Last, the step by step likelihood of `Sklearn` is not the same as outputted by `ExpectationMaximization.jl` and [mixtool.R](https://cran.r-project.org/web/packages/mixtools/index.html) (both agree), so I am a bit suspicious.

[^3]: It overflows very quickly for $n>500$ or so. I think it is because of naive implementation of [`logsumexp`](https://github.com/sseemayer/mixem/blob/2ffd990b22a12d48313340b427feae73bcf6062d/mixem/em.py#L5). So I eventually did not include the result in the benchmark.
