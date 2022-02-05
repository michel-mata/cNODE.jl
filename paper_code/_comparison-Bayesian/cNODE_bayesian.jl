####################################################
# Set directory
begin
    println("Loading Path")
    cd("../endpoints")
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
end
####################################################
begin
    Datasets = 1:7
    LR = [0.001,0.005]
    Epochs = 500
    MB = 5
    train_loss = zeros(7,10)
    test_loss = zeros(7,10)
end

for j in Datasets
    "="^50 |> println
    for k in 0:9

        Z = readdlm("./Data/Dissimilarity/5/Z$j$k.csv",',',Float64,'\n')'
        Z = hcat([ normalize(z,1) for z in eachcol(Z)]...)
        Z = [z for z in Z |> eachcol]
        coms = Z|> unique
        inx = [ findall(x->x==c,Z) for c in coms ]
        Z = hcat(coms...)

        P = readdlm("./Data/Dissimilarity/5/P$j$k.csv",',',Float64,'\n')'
        P = hcat([ normalize(p,1) for p in eachcol(P)]...)
        P = [ median(P[:,i],dims=2) for i in inx ]
        P = hcat(P...)

        N,M = size(Z);
        @show size(P)

        epochs = Epochs;
        mb_size = MB;
        report = 10;

        (ztrn,ptrn),(ztst,ptst)= splitobs(shuffleobs((Z,P)),at=0.7)
        @show size(ztrn)
        @show size(ztst)

        node = FitnessLayer(N);
        W = params(node);

        function cb(e)
            ltrn = Loss(node,ztrn,ptrn).data
            ltst = Loss(node,ztst,ptst).data
            @printf("%d ====\t%.4f\t%.4f\n",e, ltrn,ltst)
            #
            q1 = predict(node,ztrn).data
            q2 = predict(node,ztst).data

            p1 = plot(0:1,0:1,label="",aspect_ratio=1)
            scatter!(p1, ptrn[:], q1[:], title="trn $ltrn",label="",color=:black,markersize=2.5)
            p2 = plot(0:1,0:1,label="",aspect_ratio=1)
            scatter!(p2, ptst[:], q2[:], title="tst $ltst",label="",color=:red,markersize=2.5)

            display(plot(p1,p2))
            #
        end

        for e in 1:epochs
            l_trn = Loss(node,ztrn,ptrn).data
            l_tst = Loss(node,ztst,ptst).data

            # Weights
            V = deepcopy(Flux.data.(W))

            # Inner loop
            for mb in partition(randperm(size(ztrn,2)), mb_size)
                l = Loss(node,ztrn[:,mb],ptrn[:,mb])
                Flux.back!(l)
                Flux.Optimise._update_params!(ADAM(LR[1]), W)
            end

            # Reptile update
            for (w, v) in zip(W,V)
                dv = Flux.Optimise.apply!(ADAM(LR[2]), v, w.data-v)
                @. w.data = v + dv
            end

            (e%report == 0 || e==1) && cb(e)
        end

        writedlm("./Results/cNODE/real$j$k.csv",ptst, ',')
        writedlm("./Results/cNODE/pred$j$k.csv",predict(node,ztst).data, ',')

        train_loss[j,k+1] = Loss(node,ztrn,ptrn).data
        test_loss[j,k+1] = Loss(node,ztst,ptst).data
    end
end

writedlm("./Results/cNODE/test_loss.csv",test_loss, ',')
writedlm("./Results/cNODE/train_loss.csv",train_loss, ',')
test_loss
