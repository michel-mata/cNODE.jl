begin
    println("Loading Path")
    if length( findall(x -> x==pwd(), LOAD_PATH) ) .== 0
        push!(LOAD_PATH, pwd())
    end
    println("Done, actual path: $(pwd())")
end
####################################################
# Libraries
begin
    println("Loading Modules")
    using Flux, DifferentialEquations, DiffEqFlux
    using Statistics, MultivariateStats, StatsPlots
    using LinearAlgebra, Distances, Random
    using Distributions: Normal, Uniform
    using StatsBase: sample,Weights
    using MLDataPattern
    using Base.Iterators: partition
    using Printf, Plots, DelimitedFiles
    println("Done")
end
####################################################
# Functions
begin
    function xavier_uniform(dims...)
        bound = sqrt(1 / dims[2])
        return Float64.(rand(Uniform(-bound, bound), dims...))
    end

    struct FitnessLayer
        W
    end

    FitnessLayer(N::Integer) = FitnessLayer( param(xavier_uniform(N,N)) )
    Flux.@treelike FitnessLayer

    function (L::FitnessLayer)(p)
        f = L.W *p
        ṗ = p .* (f - ones(size(p,1))*p'*f)
        return ( ṗ )
    end

    # NODE
    function predict(θ,z)
        node(x) = neural_ode(θ,x,(0.0,1.0),Tsit5(),saveat=1.0)[:, end ]
        q = hcat([ node(z[:,i]) for i in 1:size(z,2) ]...)
        return q
    end

    Loss(θ,z,p) = mean( [ braycurtis( p[:,i], predict(θ,z[:,i]) ) for i in 1:size(z,2)] )

    function clean_data(p,t)
        species = p[:,2]
        communities = split.(p[:,1],'-')

        dicS = Dict()
        [ dicS[s] = i for (i,s) in enumerate(unique(species)) ]
        inxS = [ (x->dicS[x]).(s) for s in species]

        dicC = Dict()
        [ dicC[c] = i for (i,c) in enumerate(unique(communities)) ]
        inxC = [ dicC[c] for c in communities]

        N = 5
        M = length(dicC)

        P = zeros(N,M)
        [ P[inxS[i],inxC[i]] = p[:,t][i] for i in 1:length(inxC)]

        P = hcat([ normalize(p,1) for p in eachcol(P)]...)
        return P
    end
end
####################################################
pars = 1:7
reps = 0:9

test_loss = zeros(length(pars),length(reps))
train_loss = zeros(length(pars),length(reps))

for i in pars, j in reps
    trn = readdlm("./Results/Maynard/trn$i$j.csv",',',Any,'\n')
    tst = readdlm("./Results/Maynard/tst$i$j.csv",',',Any,'\n')
    titles = trn[1,:]
    trn = trn[2:end,:]
    tst = tst[2:end,:]

    ptrn = clean_data(trn,3)
    qtrn = clean_data(trn,4)
    ptst = clean_data(tst,3)
    qtst = clean_data(tst,4)

    train_loss[i,j+1] = mean( [ braycurtis( ptrn[:,i], qtrn[:,i] ) for i in 1:size(ptrn,2)] )
    test_loss[i,j+1] = mean( [ braycurtis( ptst[:,i], qtst[:,i] ) for i in 1:size(ptst,2)] )
end

writedlm("./Results/Maynard/test_loss.csv",test_loss, ',')
writedlm("./Results/Maynard/train_loss.csv",train_loss, ',')


# trn = readdlm("./Data/Results/cNODE/train_loss.csv",',',Float64,'\n')
# tst = readdlm("./Data/Results/cNODE/test_loss.csv",',',Float64,'\n')
#
#
# p2 = plot(mean(test_loss,dims=2),label="Maynard")
# plot!(p2,mean(tst,dims=2),label="cNODE")
# p1 = plot(mean(train_loss,dims=2),label="Maynard")
# plot!(p1,mean(trn,dims=2),label="cNODE")
# display(plot(p1,p2))
