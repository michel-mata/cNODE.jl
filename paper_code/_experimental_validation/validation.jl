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
    using Suppressor

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
    max_epochs = 1000
    early_stoping = 100
    LearningRates = [[0.001,0.0025],[0.001,0.005],[0.001,0.01],[0.01,0.025],[0.01,0.05],[0.01,0.1]]
    Minibatches = [1,5,10]
end

for (i,DATA) in enumerate(Datasets)
    root = "./Results/Experimental/$DATA/Hyperparameters/"
    _mean = [ mean(readdlm(root*"test_loss_$i$j.csv",',',Float64,'\n')) for i in 1:6, j in 1:3]
    _mean = _mean|>argmin
    println(_mean)
    Z,P = import_data(DATA)
    N,M = size(Z);
    mb = Minibatches[_mean[2]]
    lr = LearningRates[_mean[1]]
    report = 10000;

    Qtst = SharedArray{Float64}(M,N)
    Ptst = SharedArray{Float64}(M,N)
    LossTrain = SharedArray{Float64}(M)
    LossTest = SharedArray{Float64}(M)


    LeaveOneOut = kfolds((Z,P); k = M) |> enumerate |> collect
    "training $DATA..." |> println
    @suppress_err @sync @distributed for i in LeaveOneOut
        i,fold = i
        ((ztrn,ptrn),(ztst,ptst)) = fold
        # Get cDNN
        node = FitnessLayer(N)

        # Train cDNN
        W, loss_train, loss_test, loss_val = train_reptile(node, max_epochs, mb, lr, ztrn, ptrn, ztst, ptst, report, early_stoping)

        Ptst[i,:] = ptst
        Qtst[i,:] = predict(node,ztst).data
        LossTrain[i] = loss_train[end]
        LossTest[i] = loss_test[end]

        writedlm("./Results/Experimental/$DATA/loo/epochs/train$i.csv",loss_train, ',')
        writedlm("./Results/Experimental/$DATA/loo/epochs/test$i.csv",loss_test, ',')
        writedlm("./Results/Experimental/$DATA/loo/epochs/val$i.csv",loss_val, ',')
        println(i,'\t',loss_train[end],'\t',loss_test[end])
    end
    writedlm("./Results/Experimental/$DATA/loo/real_sample.csv",Ptst, ',')
    writedlm("./Results/Experimental/$DATA/loo/pred_sample.csv",Qtst, ',')
    writedlm("./Results/Experimental/$DATA/loo/test_loss.csv",LossTest, ',')
    writedlm("./Results/Experimental/$DATA/loo/train_loss.csv",LossTrain, ',')
end
