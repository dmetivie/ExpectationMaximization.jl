using Documenter, ExpectationMaximization
using Distributions, StatsPlots

include("pages.jl")

makedocs(
    sitename = "ExpectationMaximization.jl",
    authors = "David Métivier",
    modules = [ExpectationMaximization],
    clean = true,
    doctest = false,
    pages = pages,
)

deploydocs(repo = "github.com/dmetivie/ExpectationMaximization.jl.git")
