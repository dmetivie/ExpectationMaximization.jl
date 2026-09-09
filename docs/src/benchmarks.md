# [Comparison with other packages](@id Benchmarks) 

This benchmark was inspired by [this post](https://floswald.github.io/post/em-benchmarks/).
The full benchmark code is [`benchmark/benchmark_crosslang.jl`](https://github.com/dmetivie/ExpectationMaximization.jl/tree/master/benchmark/benchmark_crosslang.jl), with the figures drawn by [`benchmark/plot_crosslang.jl`](https://github.com/dmetivie/ExpectationMaximization.jl/tree/master/benchmark/plot_crosslang.jl). The original univariate-only version is kept as a [Jupyter notebook](https://github.com/dmetivie/Pluto_export/blob/main/jupyter/benchmark_EM/benchmark_v2_K2_unidim.ipynb).

## Scope and limitations of competing packages

A key distinction of `ExpectationMaximization.jl` is its **genericity**: it works with any mixture of distributions supported by `Distributions.jl` (univariate, multivariate, continuous, discrete, or custom), without any modification to the core algorithm. The competing packages benchmarked here are, in contrast, largely restricted to Gaussian mixtures:

| Package | Language | Gaussian only? | Notes |
|---|---|---|---|
| [`Sklearn`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) | Python | **Yes** | Hardcoded Gaussian[^2] |
| [`mixtools`](https://cran.r-project.org/web/packages/mixtools/index.html) | R | Mostly | Supports some other families but not extensible |
| [`mixem`](https://mixem.readthedocs.io/en/latest/index.html) | Python | Mostly | Numerically fragile[^3] |
| [`GaussianMixtures.jl`](https://github.com/davidavdav/GaussianMixtures.jl) | Julia | **Yes** | Hardcoded Gaussian |
| `ExpectationMaximization.jl` | Julia | **No** | Any `Distributions.jl` distribution |

The benchmark below only tests the **Gaussian mixture** case (the most common), which is deliberately the strongest case for the specialized packages. Despite this, `ExpectationMaximization.jl` remains highly competitive.

## Why is `ExpectationMaximization.jl` fast?

No heavy programming tricks are used. The performance comes from standard Julia best practices:

- **E-step**: memory allocated once and reused at every iteration (the posteriors overwrite the log-likelihood matrix in place), `@views`, type-stable code behind a per-component function barrier, and a fused allocation-free row-wise log-sum-exp/softmax in place of `logsumexp!` followed by a second `exp.(LL .- c)` pass over the whole matrix.
- **M-step**: delegates to `fit_mle` from `Distributions.jl`, which is well-optimized for each distribution (e.g., see the Multivariate Normal [implementation](https://github.com/JuliaStats/Distributions.jl/blob/aad64af36e83f9a191de34f497e584943ffa84e5/src/multivariate/mvnormal.jl#L419)).

!!! note "Clean Julia code"
    Many more optimizations are possible, however, I'd like to keep the code as simple and readable as possible.

## Results

All benchmark cases are shown in a single figure: one panel per `(K, D)`, with the fit time against the sample size `N` and one line per backend. Every backend gets the same initial conditions and the same, fixed number of EM iterations, so the panels compare the cost of an iteration rather than convergence speed.

![timing_crosslang](https://raw.githubusercontent.com/dmetivie/ExpectationMaximization.jl/refs/heads/master/benchmark/timing_crosslang.svg)

Or the same panels as a ratio to `ExpectationMaximization.jl`. Above the dashed line means slower than `ExpectationMaximization.jl`; the vertical axis is logarithmic because `mixtools` is several orders of magnitude slower on the multivariate cases.

![timing_crosslang_ratio](https://raw.githubusercontent.com/dmetivie/ExpectationMaximization.jl/refs/heads/master/benchmark/timing_crosslang_ratio.svg)

**Conclusion: for Gaussian mixtures, `ExpectationMaximization.jl` is about 4× faster than `Sklearn` (Python) and 7× faster than `mixtools` (R) on the univariate `K = 2` case, while being only slightly slower than the Gaussian-specialized `GaussianMixtures.jl`. The multivariate panels widen that gap against `mixtools`, whose `mvnormalmixEM` is orders of magnitude slower. Crucially, unlike all competing packages, `ExpectationMaximization.jl` handles arbitrary mixture distributions out of the box.**

### Reproducing these figures

```sh
julia --threads=1 --project=benchmark benchmark/benchmark_crosslang.jl   # writes benchmark/results/*.csv
julia --project=benchmark benchmark/plot_crosslang.jl                    # writes benchmark/timing_crosslang*.svg
```

The benchmark and the plotting are separate scripts on purpose: the timings are written to `benchmark/results/benchmark_timings_<date>.csv` (columns `case,backend,K,D,N,time_s`), so a figure can be restyled, or an older run replotted, without paying for four backends again. `plot_crosslang.jl` takes an optional path to a specific CSV and derives the panels from whichever cases it finds in the file.

There is also a Julia-only suite in `benchmark/benchmarks.jl`, run on every pull request by [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl), which covers univariate, multivariate (isotropic, diagonal and full covariance), weighted and Bernoulli-product mixtures.

If you have comments to improve these benchmarks, they are welcome.

!!! note "Benchmarking methodology"
    Cross-language comparisons are inherently imperfect[^1]. `PythonCall.jl` and `RCall.jl` introduce a small overhead (~few milliseconds), which was verified to be negligible here.

[^1]: `@btime` with `RCall.jl` and `PythonCall.jl` may add a small overhead; see [this discussion](https://discourse.julialang.org/t/benchmarking-julia-vs-python-vs-r-with-pycall-and-rcall/37308). Timings were cross-checked against `R` `microbenchmark` and Python `timeit`, which gave consistent results. `BenchmarkTools.jl` automatically determines the number of repetitions needed for a reliable estimate.

[^2]: `Sklearn`'s `GaussianMixture` used to run K-means initialization even when initial conditions are explicitly provided — see [this discussion](https://github.com/scikit-learn/scikit-learn/discussions/25916), [issue](https://github.com/scikit-learn/scikit-learn/issues/26015), and [PR](https://github.com/scikit-learn/scikit-learn/pull/26021). I should be fix by now.

[^3]: `mixem` overflows for $n \gtrsim 500$ due to a fragile [`logsumexp` implementation](https://github.com/sseemayer/mixem/blob/2ffd990b22a12d48313340b427feae73bcf6062d/mixem/em.py#L5) and was excluded from the benchmark.
