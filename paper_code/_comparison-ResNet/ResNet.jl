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

    struct ResNetLayer
        W1
        b1
        W2
        b2
        W3
        b3
    end

    ResNetLayer(N::Integer) = ResNetLayer( param(xavier_uniform(N,N)) , param(rand(N)) ,
                                             param(xavier_uniform(N,N)) , param(rand(N)) ,
                                             param(xavier_uniform(N,N)) , param(rand(N)) )
    Flux.@treelike ResNetLayer

    function (L::ResNetLayer)(z)
        h1 = relu.( L.W1*z + L.b1 )
        h2 = h1 + relu.( L.W2*h1 + L.b2 )
        h3 = h2 + relu.( L.W3*h2 + L.b3 )
        p = h3 .* z
        p = p ./ (x-> x<=0 ? 1 : x).(sum(p))
        return ( p )
    end

    predictR(node,z) = hcat([ node(z[:,i]) for i in 1:size(z,2) ]...)

    LossR(θ,z,p) =  [ braycurtis( p[:,i], predictR(θ,z[:,i]) ) for i in 1:size(z,2)] |> mean

end
####################################################
begin
    Datasets = ["DrosophilaGut","SoilInVitro","HumanGut","SoilInVivo"];
    LR = [0.01,0.05,0.05,0.1];
    Epochs = [200,100,50,50];
    MB = [5,23,5,25];
end

using Suppressor
@suppress_err for (j,DATA) in enumerate(Datasets)
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

    global train_loss = []
    global test_loss = []
    global Ptst = []
    global Qtst = []

    @show N
    @show M
    @show mb_size
    "="^50 |> println


    for (i,fold) in enumerate(kfolds((Z,P); k = M))

        "="^50 |> println
        ((ztrn,ptrn),(ztst,ptst)) = fold

        resnet = ResNetLayer(N);
        lossR(z,p) = LossR(resnet,z,p)

        # Callback function
        function cb()
            ltrn = LossR(resnet,ztrn,ptrn).data
            ltst = LossR(resnet,ztst,ptst).data
            @printf(" ====\t%.4f\t%.4f\n", ltrn,ltst)

            #=
            q1 = predictR(resnet,ztrn).data
            q2 = predictR(resnet,ztst).data

            p1 = plot(0:1,0:1,label="",aspect_ratio=1)
            scatter!(p1,ptrn[:], q1[:], title="trnR $ltrn",label="",color=:black,markersize=2.5)
            p2 = plot(0:1,0:1,label="",aspect_ratio=1)
            scatter!(p2,ptst[:], q2[:], title="tstR $ltst",label="",color=:red,markersize=2.5)

            display(plot(p1,p2))
            =#
        end

        i  |> println
        @show size(ztrn)

        Flux.@epochs epochs Flux.train!(lossR, params(resnet),
                eachbatch((ztrn,ptrn),size=mb_size),ADAM(LR[j]))

        push!(Qtst,predictR(resnet,ztst).data)
        push!(Ptst,ptst)

        push!(train_loss,LossR(resnet,ztrn,ptrn).data)
        push!(test_loss,LossR(resnet,ztst,ptst).data)

        cb()
        "="^50 |> println
    end

    writedlm("./Results/Experimental/ClassicTrain/ResNet/$DATA/real_sample.csv",Ptst, ',')
    writedlm("./Results/Experimental/ClassicTrain/ResNet/$DATA/pred_sample.csv",Qtst, ',')
    writedlm("./Results/Experimental/ClassicTrain/ResNet/$DATA/test_loss.csv",test_loss, ',')
    writedlm("./Results/Experimental/ClassicTrain/ResNet/$DATA/train_loss.csv",train_loss, ',')
end
