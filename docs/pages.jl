# Put in a separate page in case it get famous enough to be used in a big project

SUBSECTION_EXAMPLES = [
    "Univariate" => joinpath("examples", "examples_univariate.md"),
    "Multivariate" => joinpath("examples", "examples_multivariate.md"),
]

pages = [
    "Home" => "index.md",
    "📖 Examples" => SUBSECTION_EXAMPLES,
    "biblio.md",
    "benchmarks.md",
    "fit_mle.md",
]
