################################################################################
#       0. CONFIGURATION
################################################################################

# Setup initial configuration
begin
    num_workers = 4
    include("setup.jl")
end
# NOTE: first time running can take some minutes, packages installing

################################################################################
#       1. HYPERPARAMETER SEARCH
################################################################################

# Grid of values
@everywhere begin
    # Define potential hyperparameter values
    LearningRates = [[0.001,0.0025],[0.001,0.005],[0.001,0.01],[0.01,0.025],[0.01,0.05],[0.01,0.1]]
    Minibatches = [1,5,10]
    # Parameters for run
    max_epochs = 1000
    early_stoping = 100
    report = 50
    # Iterator over hyperparams, params and repetitions
    inx = collect(product(enumerate(LearningRates),enumerate(Minibatches)))[:]
    # Select "Drosophila_Gut" and "Soil_Vitro" as examples
    pars = real_data[1:2]
    # NOTE: variable `real_data` and values imported from module cNODE
    "search hyperparameters..." |> println
end


for (i,DATA) in enumerate(pars)
    # Load percentage of dataset
    path = "./data/real/$DATA/P.csv"
    Z,P = import_data(path,2/3)
    N,M = size(Z)
    # Explore hyperparameters in small dataset
    "training $DATA..." |> println
    for it in inx
        ((j,lr),(k,mb)) = it
        mb = Minibatches[k]
        "LR: $lr MB: $mb" |> println
        Qtst = SharedArray{Float64}(M,N)
        Ptst = SharedArray{Float64}(M,N)
        LossTrain = SharedArray{Float64}(M)
        LossTest = SharedArray{Float64}(M)
        # Use Leave-one-out cross validation
        LeaveOneOut = kfolds((Z,P); k = M) |> enumerate |> collect
        @sync @distributed for l in LeaveOneOut
            (l,((ztrn,ptrn),(ztst,ptst))) = l
            "training $l..."|>println
            # Get cNODE
            cnode = getModel(N)
            # Train cNODE
            W, loss_train, loss_val, loss_test = train_reptile(
                                                    cnode, max_epochs,
                                                    mb, lr,
                                                    ztrn, ptrn, ztst, ptst, ztst, ptst,
                                                    report, early_stoping
                                                )
            # Save
            Ptst[l,:] = ptst
            Qtst[l,:] = predict(cnode,ztst)
            LossTrain[l] = loss_train[end]
            LossTest[l] = loss_test[end]
            # Report
            println(l,'\t',loss_train[end],'\t',loss_test[end])
            println('#'^30)
        end
        # Save results
        results = "./results/real/$DATA/hyperparameters/"
        !ispath(results) && mkpath(results)
        writedlm(results*"real_sample_$(j)$(k).csv", Ptst, ',')
        writedlm(results*"pred_sample_$(j)$(k).csv", Qtst, ',')
        writedlm(results*"test_loss_$(j)$(k).csv",   LossTest, ',')
        writedlm(results*"train_loss_$(j)$(k).csv",  LossTrain, ',')
    end
end

################################################################################
#       2. Experimental Validation
################################################################################

for (i,DATA) in enumerate(pars)
    # Import full dataset
    path = "./data/real/$DATA/P.csv"
    Z,P = import_data(path)
    N,M = size(Z)
    # Select hyperparameters
    results = "./results/real/$DATA/hyperparameters/"
    _mean = [ mean(readdlm(results*"test_loss_$i$j.csv",',',Float64,'\n')) for i in 1:6, j in 1:3] |> argmin
    mb = Minibatches[_mean[2]]
    lr = LearningRates[_mean[1]]
    # Allocate variables
    Qtst = SharedArray{Float64}(M,N)
    Ptst = SharedArray{Float64}(M,N)
    LossTrain = SharedArray{Float64}(M)
    LossTest = SharedArray{Float64}(M)
    # Run validation
    results = "./results/real/$DATA/"
    LeaveOneOut = kfolds((Z,P); k = M) |> enumerate |> collect
    "training $DATA..." |> println
    @sync @distributed for l in LeaveOneOut
        (l,((ztrn,ptrn),(ztst,ptst))) = l
        "training $l..."|>println
        # Get cNODE
        cnode = getModel(N)
        # Train cNODE
        W, loss_train, loss_val, loss_test = train_reptile(
                                                cnode, max_epochs,
                                                mb, lr,
                                                ztrn, ptrn, ztst, ptst, ztst, ptst,
                                                report, early_stoping
                                            )
        # Save values
        Ptst[l,:] = ptst
        Qtst[l,:] = predict(cnode,ztst)
        LossTrain[l] = loss_train[end]
        LossTest[l] = loss_test[end]
        # Save realization
        !ispath(results*"loss_epochs/") && mkpath(results*"loss_epochs/")
        writedlm(results*"loss_epochs/train$l.csv",loss_train, ',')
        writedlm(results*"loss_epochs/test$l.csv",loss_test, ',')
        writedlm(results*"loss_epochs/val$l.csv",loss_val, ',')
        # Report
        println(i,'\t',loss_train[end],'\t',loss_test[end])
    end
    # Write full results
    writedlm(results*"real_sample.csv",Ptst, ',')
    writedlm(results*"pred_sample.csv",Qtst, ',')
    writedlm(results*"test_loss.csv",LossTest, ',')
    writedlm(results*"train_loss.csv",LossTrain, ',')
end
