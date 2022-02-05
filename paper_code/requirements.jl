loaded_modules(m::Module = Main) = filter(x -> eval(x) isa Module && x ≠ Symbol(m), names(m, imported = true))

function load_modules(pkgs_list)
    # Install missing packages automatically
    installed = [ p.name for p in values(Pkg.dependencies()) if p.is_direct_dep ]
    to_install = filter(p-> p ∉ installed, pkgs_list)
    length(to_install) > 0 && Pkg.add(to_install)
    # Use all required packages
    [ @eval using $(p) for p in Symbol.(pkgs_list) ]
    # Report success
    loaded = loaded_modules()
    "Loaded Packages:" |> println
    for p in pkgs_list
        "\t$p" |> println
    end
end

pkgs_names = [
    "DelimitedFiles", "CSV", "DataFrames",
    "LinearAlgebra", "SharedArrays", "SparseArrays",
    "StatsBase", "Random", "Statistics",
    "DifferentialEquations", "Distances",
    "Flux", "DiffEqFlux",
    "MLDataPattern"]


load_modules(pkgs_names)
