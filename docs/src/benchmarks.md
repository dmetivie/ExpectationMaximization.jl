# Benchmarks

I was inspired by [this benchmark](https://floswald.github.io/post/em-benchmarks/).
I am not too sure how to do 100% fair comparisons across languages[^1].
There is a small overhead for using `PythonCall.jl` and `RCall.jl`. I checked that it was small in my experimentation (~ few milliseconds?).
Here is the [Jupyter notebook of the benchmark](https://github.com/dmetivie/ExpectationMaximization.jl/blob/714da3ee132984a0ce71263bcc20a70615fab454/benchmarks/benchmark_v2_K2_unidim.ipynb).

I test only the Gaussian Mixture case, since it is the most common type of mixture (remember that this package allows plenty of other mixtures).

In the code, I did not use (too much) fancy programming tricks, the speed only comes mostly from Julia usual performance tips:

- E-step: Pre-allocating memory, using `@views`, type-stable code (could be improved here) + the package `LogExpFunctions.jl` for `logsumexp!` function.
- M-step: `fit_mle` for each distribution's coming from `Distributions.jl` package. In principle, this should be quite fast. For example, look at the Multivariate Normal [code](https://github.com/JuliaStats/Distributions.jl/blob/aad64af36e83f9a191de34f497e584943ffa84e5/src/multivariate/mvnormal.jl#L419).

## Univariate Gaussian mixture with 2 components

I compare with [Sklearn.py](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture)[^2], [mixtool.R](https://cran.r-project.org/web/packages/mixtools/index.html), [mixem.py](https://mixem.readthedocs.io/en/latest/index.html)[^3].
I wanted to try [mclust](https://cloud.r-project.org/web/packages/mclust/vignettes/mclust.html), but I did not manage to specify initial conditions

Overall, [mixtool.R](https://cran.r-project.org/web/packages/mixtools/index.html) and [mixem.py](https://mixem.readthedocs.io/en/latest/index.html) were constructed in a similar spirit as this package, making them easy to use for me. [Sklearn.py](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html#sklearn.mixture.GaussianMixture) is built to match the Sklearn format (all in one). `GaussianMixturesModel.jl` is built with a similar vibe.

If you have comments to improve these benchmarks, they are welcome.

You can find the benchmark code [here](https://github.com/dmetivie/ExpectationMaximization.jl/tree/master/benchmarks/benchmark_v1_K2_unidim.jl).

![timing_K_2](https://github.com/user-attachments/assets/d0eed38a-a66e-4b7e-8e4d-de3782190343)
or the ratio view
![timing_K_2_ratio](https://github.com/user-attachments/assets/3d174899-c50c-4f71-8dbb-4a9a0a238713)
**Conclusion: for Gaussian mixtures, `ExpectationMaximization.jl` is about 4 times faster than `Python` `Sklearn` and 7 times faster than `R` `mixtools` implementations** and slightly slower than the specialized Julia package `GaussianMixturesModel.jl`.

[^1]: Note that `@btime` with `RCall.jl` and `PythonCall.jl` might produce a small-time overhead compared to the pure R/Python time; see [here for example](https://discourse.julialang.org/t/benchmarking-julia-vs-python-vs-r-with-pycall-and-rcall/37308).
I did compare with `R` `microbenchmark` and Python `timeit` and they produced very similar timing but in my experience `BenchmarkTools.jl` is smarter and simpler to use, i.e. it will figure out alone the number of repetition to do in function of the run.

[^2]: There is a suspect trigger warning regarding K-means which I do not want to use here. I asked a question [here](https://github.com/scikit-learn/scikit-learn/discussions/25916). It led to [this issue](https://github.com/scikit-learn/scikit-learn/issues/26015) and [that PR](https://github.com/scikit-learn/scikit-learn/pull/26021). It turns out that even if initial conditions were provided, the K-mean was still computed. However, to this day (23-11-29) with `scikit-learn 1.3.2` it still gets the warning. Maybe it will be in the next release? I also noted this recent [PR](https://github.com/scikit-learn/scikit-learn/pull/26416).
Last, the step-by-step likelihood of `Sklearn` is not the same as outputted by `ExpectationMaximization.jl` and [mixtool.R](https://cran.r-project.org/web/packages/mixtools/index.html) (both agree), so I am a bit suspicious.

[^3]: It overflows very quickly for $n>500$ or so. I think it is because of the implementation of [`logsumexp`](https://github.com/sseemayer/mixem/blob/2ffd990b22a12d48313340b427feae73bcf6062d/mixem/em.py#L5). So I eventually did not include the result in the benchmark.
