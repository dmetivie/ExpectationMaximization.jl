# ExpectationMaximization

This package provides a simple implementation of the Expectation Maximization algorithm used to fit mixture models.
Due to [Julia](https://julialang.org/) amazing [multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY) systems and the [Distributions](https://juliastats.org/Distributions.jl/stable/) package, the code is very generic i.e., mixture of all common distributions should be supported.

## Example

```julia
using Distributions
using ExpectationMaximization
```

### Model

```julia
N = 50000
θ₁ = 10
θ₂ = 5
α = 0.2
β = 0.3
# Mixture Model here one can put any classical distributions
mix_true = MixtureModel([Exponential(θ₁), Gamma(α, θ₂)], [β, 1 - β]) 

# Generate N samples from the mixture
y = rand(mix_true, N) 
```

### Inference

```julia
# Initial guess
mix_mle = fit_mle(mix_guess, y; display = :iter, tol = 1e-3, robust = false, infos = false)

# Fit the MLE with the EM algorithm
mix_mle = fit_mle(mix_guess, y; display = :iter, tol = 1e-3, robust = false, infos = false)
```

### Verify results

```julia
rtol = 5e-2
p = params(mix_mle)[1] # (θ₁, (α, θ₂))
isapprox(β, probs(mix_mle)[1]; rtol = rtol)
isapprox(θ₁, p[1]...; rtol = rtol)
isapprox(α, p[2][1]; rtol = rtol)
isapprox(θ₂, p[2][2]; rtol = rtol)
```
