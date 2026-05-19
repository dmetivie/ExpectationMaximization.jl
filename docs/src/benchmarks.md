# [Comparison with other packages](@id Benchmarks) 

This benchmark was inspired by [this post](https://floswald.github.io/post/em-benchmarks/).
The full benchmark code is available as a [Jupyter notebook](https://github.com/dmetivie/Pluto_export/blob/main/jupyter/benchmark_EM/benchmark_v2_K2_unidim.ipynb) and [here](https://github.com/dmetivie/ExpectationMaximization.jl/tree/master/benchmark/benchmark_v2_K2_unidim.jl).

## Scope and limitations of competing packages

A key distinction of `ExpectationMaximization.jl` is its **genericity**: it works with any mixture of distributions supported by `Distributions.jl` (univariate, multivariate, continuous, discrete, or custom), without any modification to the core algorithm. The competing packages benchmarked here are, in contrast, largely restricted to Gaussian mixtures:

| Package | Language | Gaussian only? | Notes |
|---|---|---|---|
| [`Sklearn`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) | Python | **Yes** | Hardcoded Gaussian; opinionated API[^2] |
| [`mixtools`](https://cran.r-project.org/web/packages/mixtools/index.html) | R | Mostly | Supports some other families but not extensible |
| [`mixem`](https://mixem.readthedocs.io/en/latest/index.html) | Python | Mostly | Numerically fragile[^3]; not actively maintained |
| [`GaussianMixtures.jl`](https://github.com/davidavdav/GaussianMixtures.jl) | Julia | **Yes** | Highly optimized but Gaussian-specific |
| `ExpectationMaximization.jl` | Julia | **No** | Any `Distributions.jl` distribution |

The benchmark below only tests the **Gaussian mixture** case (the most common), which is deliberately the strongest case for the specialized packages. Despite this, `ExpectationMaximization.jl` remains highly competitive.

## Why is `ExpectationMaximization.jl` fast?

No heavy programming tricks are used. The performance comes from standard Julia best practices:

- **E-step**: pre-allocated memory, `@views`, type-stable code, and `logsumexp!` from `LogExpFunctions.jl`.
- **M-step**: delegates to `fit_mle` from `Distributions.jl`, which is well-optimized for each distribution (e.g., see the Multivariate Normal [implementation](https://github.com/JuliaStats/Distributions.jl/blob/aad64af36e83f9a191de34f497e584943ffa84e5/src/multivariate/mvnormal.jl#L419)).

!!! note "Clean Julia code"
    Many more optimizations are possible, however, I'd like to keep the code as simple and readable as possible.

## Results

![timing_K_2](https://raw.githubusercontent.com/dmetivie/ExpectationMaximization.jl/refs/heads/master/benchmark/timing_K_2.svg)

Or the ratio view:

![timing_K_2_ratio](https://raw.githubusercontent.com/dmetivie/ExpectationMaximization.jl/refs/heads/master/benchmark/timing_K_2_ratio.svg)

**Conclusion: for Gaussian mixtures, `ExpectationMaximization.jl` is about 4× faster than `Sklearn` (Python) and 7× faster than `mixtools` (R), while being only slightly slower than the Gaussian-specialized `GaussianMixtures.jl`. Crucially, unlike all competing packages, `ExpectationMaximization.jl` handles arbitrary mixture distributions out of the box.**

If you have comments to improve these benchmarks, they are welcome.

!!! note "Benchmarking methodology"
    Cross-language comparisons are inherently imperfect[^1]. `PythonCall.jl` and `RCall.jl` introduce a small overhead (~few milliseconds), which was verified to be negligible here.

[^1]: `@btime` with `RCall.jl` and `PythonCall.jl` may add a small overhead; see [this discussion](https://discourse.julialang.org/t/benchmarking-julia-vs-python-vs-r-with-pycall-and-rcall/37308). Timings were cross-checked against `R` `microbenchmark` and Python `timeit`, which gave consistent results. `BenchmarkTools.jl` automatically determines the number of repetitions needed for a reliable estimate.

[^2]: `Sklearn`'s `GaussianMixture` used to run K-means initialization even when initial conditions are explicitly provided — see [this discussion](https://github.com/scikit-learn/scikit-learn/discussions/25916), [issue](https://github.com/scikit-learn/scikit-learn/issues/26015), and [PR](https://github.com/scikit-learn/scikit-learn/pull/26021). I should be fix by now.

[^3]: `mixem` overflows for $n \gtrsim 500$ due to a fragile [`logsumexp` implementation](https://github.com/sseemayer/mixem/blob/2ffd990b22a12d48313340b427feae73bcf6062d/mixem/em.py#L5) and was excluded from the benchmark.
