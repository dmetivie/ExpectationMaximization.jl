using Documenter, Pkg, Literate
using ExpectationMaximization
using Distributions: VectorOfUnivariateDistribution

PROJECT_TOML = Pkg.TOML.parsefile(joinpath(@__DIR__, "..", "Project.toml"))
PkgVERSION = PROJECT_TOML["version"]
NAME = PROJECT_TOML["name"]
AUTHORS = join(PROJECT_TOML["authors"], ", ") * " and contributors"
GITHUB = "https://github.com/dmetivie/ExpectationMaximization.jl"

## Generate example pages from Julia source files with Literate.jl
examples_jl_path = joinpath(dirname(@__DIR__), "examples")
examples_md_path = joinpath(@__DIR__, "src", "examples")

mkpath(examples_md_path)
for file in readdir(examples_md_path)
    endswith(file, ".md") && rm(joinpath(examples_md_path, file))
end

for file in readdir(examples_jl_path)
    Literate.markdown(joinpath(examples_jl_path, file), examples_md_path, mdstrings=true)
end

include("pages.jl")

fmt = Documenter.HTML(
    prettyurls=true,
    repolink=GITHUB,
    canonical="https://dmetivie.github.io/ExpectationMaximization.jl",
    assets=["assets/favicon.ico"],
    footer="[$NAME.jl]($GITHUB) v$PkgVERSION docs powered by [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl)."
)

makedocs(
    sitename="ExpectationMaximization.jl",
    authors=AUTHORS,
    modules=[ExpectationMaximization],
    clean=true,
    doctest=false,
    format=fmt,
    pages=pages,
)

deploydocs(repo="github.com/dmetivie/ExpectationMaximization.jl.git", devbranch="master")
