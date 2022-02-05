__precompile__(true)
module cNODE

    # libraries
    using Distributed
    using Flux
    using Flux.Optimise: update!, apply!
    using DiffEqFlux
    using Zygote: gradient
    using MLDataPattern: eachbatch, splitobs, shuffleobs
    using DifferentialEquations: ODEProblem, solve, Tsit5
    using Distributions: Normal, Uniform
    using Statistics, LinearAlgebra
    using SparseArrays, SharedArrays
    using DelimitedFiles

    # functions
    include("./module/generator.jl")
    export getParameters, getIC, getGLV
    export getSteadyState, getRewiredNetwork
    export generate_data

    include("./module/trainer.jl")
    export FitnessLayer
    export getModel
    export predict
    export loss
    export train_reptile

    include("./module/loader.jl")
    export import_data
    export split_data

    # Parameters for module
    const synthetic_data = ["Strength","Connectivity","Universality","Noise","Rewiring"]
    const real_data = ["Drosophila_Gut","Soil_Vitro","Soil_Vivo","Human_Gut","Human_Oral","Ocean"]
    const S = [.1,.15,.2]
    const C = [0.5,0.75,1]
    const U = [0.1,1,2]
    const E = [.001,.01,.025]
    const R = [.01,.1,.25]
    export synthetic_data, real_data
    export S, C, U, E, R

end
