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
    using StatsBase: countmap

    function import_data(i,j)
        Pa = readdlm("./Data/Synthetic/100/Multistability/P$(j)a.csv",',',Float64,'\n')
        Pb = readdlm("./Data/Synthetic/100/Multistability/P$(j)b.csv",',',Float64,'\n')
        N,M = size(Pa);
        ps = [.05,.25,.5]
        P = hcat([ rand()>ps[i+1] ? Pa[:,k] : Pb[:,k] for k in 1:M ]...)
        Z = (x-> x>0 ? 1 : 0).(P)
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
    LearningRates = [[0.001,0.0025],[0.001,0.005],[0.001,0.01],[0.01,0.025],[0.01,0.05],[0.01,0.1]]
    Minibatches = [1,5,10]
    max_epochs = 1000
    early_stoping = 100
    percentage = [.001,.01,0.1,0.25,0.5,0.75,1]
    parameters = 0:2
    nets = 0:9
end

begin
    "select hyperparameters..." |> println
    root = "./Results/Synthetic/100/Hyperparameters/"
    Validation =  []
    for par in parameters
        val_hp = Array{Float64,2}(undef,6,3)
        for i in 1:6
            for j in 1:3
                val =  []
                for net in nets
                    vld = readdlm(root*"val_loss_$(i)$(j)_$(par)$(net).csv",',',Float64,'\n')
                    push!(val, vld[end])
                end
                val_hp[i,j] = mean(val)
            end
        end
        push!(Validation,val_hp)
    end

    HP = [ vcat(d...) for d in collect(product(LearningRates,Minibatches))]
    opt_hp = hcat(HP[ [ argmin(d) for d in Validation ]]...)
    vals = [ sort(collect(countmap(o)), by=x->x[2], rev=true)[1][1] for o in eachrow(opt_hp)]
    _opt = findall(x-> x == vals,HP)[1]
    println(_opt)
    "warming for training..." |> println
    inx = collect(product(parameters,nets))[:]
    mb = Minibatches[_opt[2]]
    lr = LearningRates[_opt[1]]
    report = 200;
end


"training!" |> println
@sync @distributed for it in inx
    (i,j) = it
    "Par: $i Net: $j" |> println

    Z,P = import_data(i,j)
    N,M = size(Z);
    LossTrain = SharedArray{Float64}(length(percentage))
    LossVal = SharedArray{Float64}(length(percentage))
    LossTest = SharedArray{Float64}(length(percentage))

    @time for p in 1:length(percentage)
        # Load Data
        ztrn,ptrn,zval,pval,ztst,ptst = split_data(Z,P,percentage[p])
        println("Sizes: trn ($(size(ztrn)))\ttst ($(size(ztst)))\tval ($(size(zval)))")
        # Get cDNN
        node = FitnessLayer(N)
        mb_size = mb
        if mb > size(ztrn,2)
            mb_size = size(ztrn,2)
        end
        # Train cDNN
        W, loss_train, loss_val, loss_test = train_reptile(node, max_epochs, mb_size, lr, ztrn, ptrn, zval, pval, ztst, ptst, report, early_stoping)

        LossTrain[p] = loss_train[end]
        LossVal[p] = loss_val[end]
        LossTest[p] = loss_test[end]

        # Save last training
        if percentage[p]==percentage[end]
            writedlm("./Results/Synthetic/100/Multistability/loss_epochs/train_$i$j.csv",loss_train, ',')
            writedlm("./Results/Synthetic/100/Multistability/loss_epochs/val_$i$j.csv",loss_val, ',')
            writedlm("./Results/Synthetic/100/Multistability/loss_epochs/test_$i$j.csv",loss_test, ',')
        end
        #
    end

    writedlm("./Results/Synthetic/100/Multistability/loss_percentage/test_$i$j.csv",LossTest, ',')
    writedlm("./Results/Synthetic/100/Multistability/loss_percentage/val_$i$j.csv",LossVal, ',')
    writedlm("./Results/Synthetic/100/Multistability/loss_percentage/train_$i$j.csv",LossTrain, ',')
end
