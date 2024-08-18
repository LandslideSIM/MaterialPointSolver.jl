using Documenter, MaterialPointSolver, DocumenterTools

makedocs(
    modules=[MaterialPointSolver],
    sitename = "MaterialPointSolver.jl",
    authors = "Zenan Huo",
    pages = [
        "Home" => "index.md",
        "Workflow" => Any[
            "workflow/w1_2D_profile.md",
            "workflow/w2_2D_hetergeneous.md",
            "workflow/w3_3D_point_cloud.md",
            "workflow/w4_3D_hetergeneous.md",
            "workflow/others.md"
        ],
        "Example" => Any[
            "example/w1_case.md",
            "example/w2_case.md",
            "example/w3_case.md",
            "example/w4_case.md",
        ],
        "API" => "api.md"
    ],
    warnonly = [:missing_docs, :cross_references],
)

deploydocs(
    repo = "github.com/LandslideSIM/MaterialPointSolver.jl.git",
    target = "build",
    branch = "gh-pages",
    versions = ["stable" => "v^", "v#.#"],
)