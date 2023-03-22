
# [Instance vs Type version](@id InstanceVType)

This package relies heavily on the types and methods defined in `Distributions.jl` e.g., `fit_mle` and `logpdf`.
However, it differs slightly by one point: it defines and uses an “instance version” of `fit_mle(dist::Distribution, x, args...)`.
For the package to work and be generic, the input `dist` must be an “instance” i.e., an actual distribution like `Normal(0,1)` not just the type `Normal`.
For `MixtureModels` (and `ProductDistributions`) this is critical to extract the various distributions inside, while the `Type` version does not (currently) contain such information.
Plus, for `MixtureModels` the `dist` serves as the initial point of the EM algorithm.
For classic distributions, it changes nothing, i.e., `fit_mle(Normal(0,1), y)` gives the same result as `fit_mle(Normal, y)`.

I opened a [PR#1670](https://github.com/JuliaStats/Distributions.jl/pull/1670) in `Distributions.jl` to include this instance version, but it might not be accepted.

!!! note "Compatibility"
    The new `fit_mle` methods defined in this package are fully compatible with `Distributions.jl` (it does not break any regular `Distributions.fit_mle`).

I believe that some packages dealing with complex distributions like [Copulas.jl](https://github.com/lrnv/Copulas.jl/blob/08aca27dc0a28e4932e5e2c0dd04482bb3b04f48/test/some_tests.jl#L8) or [HMMBase.jl](https://github.com/maxmouchet/HMMBase.jl) could also use this formulation.
