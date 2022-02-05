####################################################
"start..." |> println
using Distributed

"loading workers and module..." |> println
nprocs() < 63 && addprocs( 63 - nprocs() )

@everywhere begin
    "loading libraries..." |> println
    !(pwd() in LOAD_PATH) && push!(LOAD_PATH, pwd())
    include("../cNODE.jl")
    # Modules
    using DelimitedFiles: readdlm, writedlm
    using LinearAlgebra: normalize
    using Base.Iterators: enumerate, partition, product
    using MLDataPattern: splitobs, shuffleobs, kfolds
    using SharedArrays

    function import_data(DATA)
        P = readdlm("./Data/Experimental/$DATA/P.csv",',',Float64,'\n')
        P = hcat([ normalize(p,1) for p in eachcol(P)]...)
        Z = (x-> x>0 ? 1 : 0 ).(P)
        Z = hcat([ normalize(z,1) for z in eachcol(Z)]...)
        if DATA == "Soil_Vivo"
            (Z,P),_ = splitobs(shuffleobs((Z,P)),at=0.1)
        end
        return Z,P
    end

    "loading data..." |> println
    Datasets = ["Drosophila_Gut","Soil_Vitro","Soil_Vivo","Human_Gut","Human_Oral","Ocean"]
    LearningRates = [[0.001,0.0025],[0.001,0.005],[0.001,0.01],[0.01,0.025],[0.01,0.05],[0.01,0.1]]
    Minibatches = [1,5,10]
    max_epochs = 1000
    early_stoping = 100
end
inx = collect(product(enumerate(LearningRates),enumerate(Minibatches)))[:]


for (i,DATA) in enumerate(Datasets)
    Z,P = import_data(DATA)
    N,M = size(Z);
    report = 200;
    "training $DATA..." |> println

    for it in inx
        ((j,lr),(k,mb)) = it
        mb = Minibatches[k]
        "LR: $lr MB: $mb" |> println
        Qtst = SharedArray{Float64}(M,N)
        Ptst = SharedArray{Float64}(M,N)
        LossTrain = SharedArray{Float64}(M)
        LossTest = SharedArray{Float64}(M)

        LeaveOneOut = kfolds((Z,P); k = M) |> enumerate |> collect
        @sync @distributed for l in LeaveOneOut
            (l,((ztrn,ptrn),(ztst,ptst))) = l
            "training $l..."|>println
            # Get cDNN
            node = FitnessLayer(N)

            # Train cDNN
            W, loss_train, loss_test = train_reptile(node, max_epochs, mb, lr, ztrn, ptrn, ztst, ptst, report, early_stoping)

            Ptst[l,:] = ptst
            Qtst[l,:] = predict(node,ztst).data
            LossTrain[l] = loss_train[end]
            LossTest[l] = loss_test[end]

            println(l,'\t',loss_train[end],'\t',loss_test[end])
            println('#'^30)
        end
        writedlm("./Results/Experimental/$DATA/Hyperparameters/real_sample_$(j)$(k).csv",Ptst, ',')
        writedlm("./Results/Experimental/$DATA/Hyperparameters/pred_sample_$(j)$(k).csv",Qtst, ',')
        writedlm("./Results/Experimental/$DATA/Hyperparameters/test_loss_$(j)$(k).csv",LossTest, ',')
        writedlm("./Results/Experimental/$DATA/Hyperparameters/train_loss_$(j)$(k).csv",LossTrain, ',')
    end
end
