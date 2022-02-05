####################################################
# Set directory
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
end
####################################################
begin
    Datasets = ["Drosophila_Gut","Soil_Vitro","Human_Gut","Soil_Vivo"]
    LR = [[0.001,0.01],[0.01,0.05],[0.01,0.05],[0.05,0.1]]
    Epochs = [200,100,150,50]
    MB = [5,23,5,25];
end

for (j,DATA) in enumerate(Datasets)
    DATA = Datasets[j]
    "="^50 |> println
    DATA  |> println

    P = readdlm("../Data/Experimental/$DATA/P.csv",',',Float64,'\n')
    P = hcat([ normalize(p,1) for p in eachcol(P)]...)
    Z = (x-> x>0 ? 1 : 0 ).(P)
    Z = hcat([ normalize(z,1) for z in eachcol(Z)]...)

    if DATA == "Soil_Vivo"
        (Z,P),_ = splitobs(shuffleobs((Z,P)),at=0.1)
    end

    N,M = size(Z);

    epochs = Epochs[j];
    mb_size = MB[j];
    report = 10;

    global train_loss = []
    global test_loss = []
    global Ptst = []
    global Qtst = []
    global TR = []
    global TS = []


    for (i,fold) in enumerate(kfolds((Z,P); k = M))

        ((Ztrn,Ptrn),(ztst,ptst)) = fold

        node = FitnessLayer(N);
        W = params(node);

        function cb()
            ltrn = Loss(node,Ztrn,Ptrn).data
            ltst = Loss(node,ztst,ptst).data
            @printf(" ====\t%.4f\t%.4f\n", ltrn,ltst)
            #=
            q1 = predict(node,ztrn).data
            q2 = predict(node,ztst).data

            p1 = plot(0:1,0:1,label="",aspect_ratio=1)
            scatter!(p1, ptrn[:], q1[:], title="trn $ltrn",label="",color=:black,markersize=2.5)
            p2 = plot(0:1,0:1,label="",aspect_ratio=1)
            scatter!(p2, ptst[:], q2[:], title="tst $ltst",label="",color=:red,markersize=2.5)

            display(plot(p1,p2))
            =#
        end

        l_trn = []
        l_tst = []

        for ((ztrn,ptrn),_) in kfolds((Ztrn,Ptrn); k = 7)
            for e in 1:epochs
                # Weights
                V = deepcopy(Flux.data.(W))

                # Inner loop
                for mb in partition(randperm(size(ztrn,2)), mb_size)
                    l = Loss(node,ztrn[:,mb],ptrn[:,mb])
                    Flux.back!(l)
                    Flux.Optimise._update_params!(ADAM(LR[j][1]), W)
                end

                # Reptile update
                for (w, v) in zip(W,V)
                    dv = Flux.Optimise.apply!(ADAM(LR[j][2]), v, w.data-v)
                    @. w.data = v + dv
                end

                (e%report == 0 ) && cb()

                push!(l_trn,Loss(node,Ztrn,Ptrn).data)
                push!(l_tst,Loss(node,ztst,ptst).data)
            end
        end

        push!(TR, l_trn)
        push!(TS, l_tst)

        push!(Qtst,predict(node,ztst).data)
        push!(Ptst,ptst)

        push!(train_loss,Loss(node,Ztrn,Ptrn).data)
        push!(test_loss,Loss(node,ztst,ptst).data)
        cb()
        println('_'^30)
    end

    writedlm("./Results/Experimental/Reptile/cNODE/$DATA/real_sample.csv",Ptst, ',')
    writedlm("./Results/Experimental/Reptile/cNODE/$DATA/pred_sample.csv",Qtst, ',')
    writedlm("./Results/Experimental/Reptile/cNODE/$DATA/test_loss.csv",test_loss, ',')
    writedlm("./Results/Experimental/Reptile/cNODE/$DATA/train_loss.csv",train_loss, ',')
end
