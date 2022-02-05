####################################################
"start..." |> println
using Distributed

"loading workers and module..." |> println
nprocs() < 63 && addprocs( 63 - nprocs() )

@everywhere begin
    "loading libraries" |> println
    !(pwd() in LOAD_PATH) && push!(LOAD_PATH, pwd())
    include("../cNODE.jl")
    # Modules
    using DelimitedFiles: readdlm, writedlm
    using LinearAlgebra: normalize
    using Base.Iterators: enumerate, partition, product
    using MLDataPattern: splitobs, shuffleobs
    using SharedArrays

    function import_data(DATA,i,j)
        P = readdlm("./Data/Synthetic/$N/$DATA/P$(i)_$j.csv",',',Float64,'\n')
        Z = (x-> x>0 ? 1 : 0 ).(P)
        reps = unique([ Set(findall([ zz == z for z in eachcol(Z) ])) for zz in eachcol(Z) ])
        inx = [ sample([r...]) for r in reps]
        P = P[:,inx]
        Z = Z[:,inx]
        P = hcat([ normalize(p,1) for p in eachcol(P)]...)
        Z = hcat([ normalize(z,1) for z in eachcol(Z)]...)
        (Z,P),_ = splitobs(shuffleobs((Z,P)),at=0.350)
        return Z,P
    end

    function split_data(Z,P,p)
        (ztrn,ptrn),(ZT,PT) = splitobs(shuffleobs((Z,P)),at = 300/350)
        (ztrn,ptrn),_ = splitobs(shuffleobs((ztrn,ptrn)),at = p - 1e-5)
        (zval,pval),(ztst,ptst) = splitobs(shuffleobs((ZT,PT)),at = 1/2)
        return ztrn,ptrn,zval,pval,ztst,ptst
    end

    "loading data..." |> println
    Datasets = ["Strength","Connectivity","Universality","Noise","Rewiring"]
    LearningRates = [[0.001,0.0025],[0.001,0.005],[0.001,0.01],[0.01,0.025],[0.01,0.05],[0.01,0.1]]
    Minibatches = [1,5,10]
    max_epochs = 1500
    early_stoping = 100
    parameters = 0:2
    nets = 0:9
    N = 100
    report = 1000
    inx = collect(product(enumerate(LearningRates),enumerate(Minibatches),parameters,nets))[:]
end

@sync @distributed for it in inx
    ((j,lr),(k,mb),par,net) = it
    "\t LR: $lr MB: $mb Net: $net" |> println

    Z,P = import_data(Datasets[1],par,net)
    N,M = size(Z);

    # Load Data
    ztrn,ptrn,zval,pval,ztst,ptst = split_data(Z,P,1)
    # Get cDNN
    node = FitnessLayer(N)
    mb_size = mb
    if mb > size(ztrn,2)
        mb_size = size(ztrn,2)
    end
    # Train cDNN
    W, loss_train, loss_val, loss_test = train_reptile(node, max_epochs, mb_size, lr, ztrn, ptrn, zval, pval, ztst, ptst, report, early_stoping)

    println("$net $lr $mb\t",loss_train[end],'\t',loss_val[end],'\t',loss_test[end])

    writedlm("./Results/Synthetic/$N/Hyperparameters/train_loss_$(j)$(k)_$(par)$(net).csv",loss_train[end], ',')
    writedlm("./Results/Synthetic/$N/Hyperparameters/val_loss_$(j)$(k)_$(par)$(net).csv",loss_val[end], ',')
    writedlm("./Results/Synthetic/$N/Hyperparameters/test_loss_$(j)$(k)_$(par)$(net).csv",loss_test[end], ',')
end
