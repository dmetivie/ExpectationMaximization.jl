# [Comparison with other packages](@id Benchmarks)

This benchmark was inspired by [this post](https://floswald.github.io/post/em-benchmarks/).
The full benchmark code is [`benchmark/benchmark_crosslang.jl`](https://github.com/dmetivie/ExpectationMaximization.jl/tree/master/benchmark/benchmark_crosslang.jl), with the figures drawn by [`benchmark/plot_crosslang.jl`](https://github.com/dmetivie/ExpectationMaximization.jl/tree/master/benchmark/plot_crosslang.jl). The original univariate-only version is kept as a [Jupyter notebook](https://github.com/dmetivie/Pluto_export/blob/main/jupyter/benchmark_EM/benchmark_v2_K2_unidim.ipynb).

## Scope and limitations of competing packages

A key distinction of `ExpectationMaximization.jl` is its **genericity**: it works with any mixture of distributions supported by `Distributions.jl` (univariate, multivariate, continuous, discrete, or custom), without any modification to the core algorithm. The competing packages benchmarked here are, in contrast, largely restricted to Gaussian mixtures:

| Package | Language | Gaussian only? | Notes |
| --- | --- | --- | --- |
| [`Sklearn`](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) | Python | **Yes** | Hardcoded Gaussian[^2] |
| [`mixtools`](https://cran.r-project.org/web/packages/mixtools/index.html) | R | Mostly | Supports some other families but not extensible |
| [`mixem`](https://mixem.readthedocs.io/en/latest/index.html) | Python | Mostly | Numerically fragile[^3] |
| [`GaussianMixtures.jl`](https://github.com/davidavdav/GaussianMixtures.jl) | Julia | **Yes** | Hardcoded Gaussian |
| `ExpectationMaximization.jl` | Julia | **No** | Any `Distributions.jl` distribution. Possibilities for custom distributions via the `Distributions.jl` interface. |

The benchmark below only tests the **Gaussian mixture** case (the most common), which is deliberately the strongest case for the specialized packages. Despite this, `ExpectationMaximization.jl` remains highly competitive.

## Why is `ExpectationMaximization.jl` fast?

No heavy programming tricks are used. The performance comes from standard Julia best practices:

- **E-step**: memory allocated once and reused at every iteration (the posteriors overwrite the log-likelihood matrix in place), `@views`, type-stable code, and a fused allocation-free row-wise log-sum-exp/softmax.
- **M-step**: delegates to `fit_mle` from `Distributions.jl`, which is well-optimized for each distribution (e.g., see the Multivariate Normal [implementation](https://github.com/JuliaStats/Distributions.jl/blob/aad64af36e83f9a191de34f497e584943ffa84e5/src/multivariate/mvnormal.jl#L419)).

!!! note "Clean Julia code"
    Many more optimizations are possible, however, I'd like to keep the code as simple and readable as possible. Note that as of v0.3.5, I am testing LLM suggestions to improve performance without sacrificing readability (too much). If you have suggestions, please open an issue or PR.

## Results

All benchmark cases are shown in a single figure: one panel per `(K, D)`, with the ratio of each backend's fit time to `ExpectationMaximization.jl`'s against the sample size `N`. Above the dashed line means slower than `ExpectationMaximization.jl`; the vertical axis is logarithmic because `mixtools` is several orders of magnitude slower on the multivariate cases.

Only the ratio is shown. Absolute times are not comparable across cases — each case fixes its own number of EM iterations — whereas a ratio between two backends at the same `N` of the same case always compares equal amounts of work.

![timing_crosslang_ratio](https://raw.githubusercontent.com/dmetivie/ExpectationMaximization.jl/refs/heads/master/benchmark/timing_crosslang_ratio.svg)

Every backend gets the same data, the same initial conditions, the same full-covariance model and the same, fixed number of EM iterations, and the benchmark verifies all of that as it runs rather than assuming it: after each timed fit it checks that the backends agree on the fitted weights, means and variances, and that each one really performed the number of iterations it was asked for. The full transcript of the published run — cross-check results, iteration counts, per-case totals — is kept alongside the figures:

[`benchmark/results/benchmark_log_latest.txt`](https://github.com/dmetivie/ExpectationMaximization.jl/blob/master/benchmark/results/benchmark_log_latest.txt)

Three caveats that transcript will show you.

- `mixtools` performs a *multicycle ECM* step — two E-steps per iteration, the second one after the means have been updated — so it does more work per iteration than the other three, and its intermediate iterates differ from theirs. It is therefore compared with them at its fixed point, with everything run to convergence. Part of its distance from the other backends is this extra work, not R being slow.
- `GaussianMixtures.jl` ends its loop on an M-step, while the other three evaluate the likelihood of the parameters they finish on and so pay one further E-step. At the same iteration count it therefore does about 6% less work at `iters = 8` and 2.5% at `iters = 20`. Asking it for one more iteration would buy an extra M-step too, so the difference is reported rather than papered over.
- A backend whose fit exceeds 100 s is dropped from the larger `N` of that case, so its curve stops early rather than costing hours.

!!! tip "Results"
    For univariate Gaussian mixtures, `ExpectationMaximization.jl` is about 7-10× faster than `Sklearn` (Python) and `mixtools` (R).
    For multivariate Gaussian mixtures that gap is  widen especially for `mixtools`, whose `mvnormalmixEM` is orders of magnitude slower. 
    It achieves a speed comparable to the Gaussian-specialized `GaussianMixtures.jl`. Crucially, unlike all other packages, `ExpectationMaximization.jl` handles arbitrary mixture distributions out of the box.

### Reproducing these figures

```sh
julia --threads=1 --project=benchmark benchmark/benchmark_crosslang.jl   # writes benchmark/results/*
julia --project=benchmark benchmark/plot_crosslang.jl                    # writes benchmark/timing_crosslang*.svg
```

The benchmark and the plotting are separate scripts. The benchmark writes, under `benchmark/results/`, both a dated copy and a `_latest` one of:

| file | contents |
|---|---|
| `benchmark_timings_<date>.csv` | one row per (case, backend, `N`): `case,backend,K,D,N,time_s,iters_asked,iters_run` |
| `benchmark_log_<date>.txt` | the run transcript — the file linked above |
| `system_info_<date>.txt` | Julia, R and Python versions, package versions, thread counts |

Both are rewritten after every case, so an interrupted run still leaves the cases that did finish.

If you have comments to improve these benchmarks, they are welcome.

!!! note "Benchmarking methodology"
    Cross-language comparisons are inherently imperfect[^1]. `PythonCall.jl` and `RCall.jl` introduce a small overhead (~few milliseconds), which was verified to be negligible here.

!!! note
    There is also a Julia-only suite in `benchmark/benchmarks.jl`, run on every pull request by [AirspeedVelocity.jl](https://github.com/MilesCranmer/AirspeedVelocity.jl), which covers univariate, multivariate, weighted and Classical and Stochastic EM. It allows tracking performance regressions and improvements over time. These results are displayed in each PR as a comment.

[^1]: `@btime` with `RCall.jl` and `PythonCall.jl` may add a small overhead; see [this discussion](https://discourse.julialang.org/t/benchmarking-julia-vs-python-vs-r-with-pycall-and-rcall/37308). Timings were cross-checked against `R` `microbenchmark` and Python `timeit`, which gave consistent results. `BenchmarkTools.jl` automatically determines the number of repetitions needed for a reliable estimate.

[^2]: `Sklearn`'s `GaussianMixture` used to run K-means initialization even when initial conditions are explicitly provided — see [this discussion](https://github.com/scikit-learn/scikit-learn/discussions/25916), [issue](https://github.com/scikit-learn/scikit-learn/issues/26015), and [PR](https://github.com/scikit-learn/scikit-learn/pull/26021). I should be fix by now.

[^3]: `mixem` overflows for $n \gtrsim 500$ due to a fragile [`logsumexp` implementation](https://github.com/sseemayer/mixem/blob/2ffd990b22a12d48313340b427feae73bcf6062d/mixem/em.py#L5) and was excluded from the benchmark.
