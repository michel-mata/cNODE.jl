using Documenter
using cNODE

makedocs(
    sitename = "cNODE.jl",
    modules = [cNODE],
    pages = [
            "Home" => "index.md",
            "Generate Data" => "generator.md",
            "Load Data" => "loader.md",
            "Use cNODE" => "trainer.md"],
    format = Documenter.HTML()
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/michel-mata/cNODE.jl.git",
    devbranch = "main",
    target = "build",
    versions = ["stable" => "v^"]
)
