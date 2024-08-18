using Documenter

makedocs(
    sitename = "MPMSolver.jl",
    modules = [MPMSolver],
    checkdocs = :all,
    clean = true,
    doctest = true,
    authors = "Zenan Huo",
    repo = "https://github.com/LandslideSIM/MPMSolver.jl",
    pages = [
        "Home" => "index.md",
        "changes.md",
        "method.md",
        "API Documentation" => [
            "system.md",
            "physics.md",
            "solutions.md",
            "solver.md",
            "post.md",
            "quantities.md",
            "misc.md",
            "internal.md",
            "allindex.md",
        ],
        "Tutorial Notebooks" => notebooks,
        "Examples" => generated_examples,
    ],
)
