using Documenter
using cNODE

makedocs(
    sitename = "cNODE.jl",
    modules = [cNODE],
    pages = [
            "Home" => "index.md",
            ],
    format = Documenter.HTML(prettyurls = true)
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(
    repo = "github.com/michel-mata/cNODE.jl.git",
    devbranch = "main",
    target = "build",
    push_preview = false
)
