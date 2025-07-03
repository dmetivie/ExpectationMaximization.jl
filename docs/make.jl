using Documenter, ExpectationMaximization
using Distributions, StatsPlots

include("pages.jl")

makedocs(
    sitename="ExpectationMaximization.jl",
    authors="David Métivier",
    modules=[ExpectationMaximization],
    clean=true,
    doctest=false,
    assets=["assets/favicon.ico"],
    format=Documenter.HTML(assets=["assets/favicon.ico"],
        prettyurls=true,
        repolink="https://github.com/dmetivie/ExpectationMaximization.jl",
        canonical="https://dmetivie.github.io/ExpectationMaximization.jl",),
    pages=pages,
)

deploydocs(repo="github.com/dmetivie/ExpectationMaximization.jl.git")
