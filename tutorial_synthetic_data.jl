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
#       1. DATA GENERATION
################################################################################

# Fixed parameters
@everywhere begin
    repetitions = 3         # number of ecological networks
    vals = 2                # number of values for sweeping parameter
    N = 8                   # number of species
    M = (2^N)-1             # number of communities
end

# Select "Strength", "Connectivity", and "Universality" as example
pars = synthetic_data[1:3]
# NOTE: variable `synthetic_data` and values imported from module cNODE

# Generate GLV data
begin
    "generating synthethic data..." |> println
    generate_data(N,M,repetitions,vals,pars)
end

################################################################################
#       2. HYPERPARAMETER SEARCH
################################################################################

# Grid of values
@everywhere begin
    # Define potential hyperparameter values
    LearningRates = [[0.01,0.01],[0.01,0.05],[0.01,0.1]]
    Minibatches = [5,10]
    # Parameters for run
    max_epochs = 1000
    early_stoping = 100
    report = 10
    # Iterator over hyperparams, params and repetitions
    inx = collect(product(
                enumerate(LearningRates),
                enumerate(Minibatches),
                0:repetitions-1
            ))[:]
    # Run hyperparameter search using
    # the low value:
    v = vals[1]
    # of "Strength":
    DATA = synthetic_data[1]
    "search hyperparameters..." |> println
end

@sync @distributed for it in inx
    # Unpack hyperparameters and report
    ((j,lr),(k,mb),rep) = it
    "\t LR: $lr MB: $mb Rep: $rep" |> println
    # Load Data
    path = "./data/synthetic/$N/$DATA/P$(v)_$rep.csv"
    Z,P = import_data(path)
    ztrn,ptrn,zval,pval,ztst,ptst = split_data(Z,P)
    # Get cNODE
    cnode = getModel(N)
    # Minibatch size
    mb_size = mb > size(ztrn,2) ? size(ztrn,2) : mb
    # Train cNODE
    W, loss_train, loss_val, loss_test = train_reptile(
                                            cnode, max_epochs,
                                            mb_size, lr,
                                            ztrn, ptrn,
                                            zval, pval,
                                            ztst, ptst,
                                            report, early_stoping
                                            )
    # Save
    results = "./results/synthetic/$N/hyperparameters/"
    !ispath(results) && mkpath(results)
    writedlm(results* "train_loss_$(j)$(k)_$val$rep.csv", loss_train[end], ',')
    writedlm(results* "val_loss_$(j)$(k)_$val$rep.csv",   loss_val[end],   ',')
    writedlm(results* "test_loss_$(j)$(k)_$val$rep.csv",  loss_test[end],  ',')
end

################################################################################
#       3. HYPERPARAMETER SELECTION
################################################################################

begin
    "select hyperparameters..." |> println
    # Path to hyperparameter results
    root = "./results/synthetic/$N/hyperparameters/val_loss_"
    # Read every combination and average over repetitions
    val_hp = Array{Float64,2}(undef,length(LearningRates),length(Minibatches))
    for i in 1:length(LearningRates), j in 1:length(Minibatches)
        val =  [ readdlm(root * "$(i)$(j)_0$r.csv", ',',Float64,'\n')[end]
                    for r in 0:repetitions-1]
        val_hp[i,j] = sum(val)/length(val)
    end
    # Find minimum validation error
    best_hp = findmin(val_hp)|>last
    # Select optimum hyperparams in search
    @everywhere lr,mb = collect(product(LearningRates,Minibatches))[findmin(val_hp)|>last]
end

################################################################################
#       4. VALIDATION OF ALL DATASETS
################################################################################

@everywhere begin
    # Relative size of training dataset
    percentage = [.001,.01,0.1,0.25,0.5,0.75,1]
    # Iterator over distinct parameters, different values and repetitions
    inx = collect(product(pars,0:vals-1,0:repetitions-1))[:]
end

@sync @distributed for it in inx
    # Unpack parameters and report
    (DATA,i,j) = it
    "$DATA - \t LR: $lr MB: $mb Par: $i Rep: $j" |> println
    # Load Data
    path = "./data/synthetic/$N/$DATA/P$(i)_$(j).csv"
    Z, P = import_data(path)
    # Allocate for sync workers
    LossTrain = SharedArray{Float64}(length(percentage))
    LossVal = SharedArray{Float64}(length(percentage))
    LossTest = SharedArray{Float64}(length(percentage))
    # Results folders
    results = "./results/synthetic/$N/$DATA/"
    !ispath(results) && mkpath(results)
    !ispath(results*"loss_epochs/") && mkpath(results*"loss_epochs/")
    !ispath(results*"loss_percentage/") && mkpath(results*"loss_percentage/")
    # For different training dataset sizes
    @time for p in 1:length(percentage)
        # Load Data
        ztrn,ptrn,zval,pval,ztst,ptst = split_data(Z,P,percentage[p])
        # Get cNODE
        cnode = getModel(N)
        # Minibatches
        mb_size = mb > size(ztrn,2) ? size(ztrn,2) : mb_size
        # Train cNODE
        W, loss_train, loss_val, loss_test = train_reptile(
                                                cnode, max_epochs,
                                                mb_size, lr,
                                                ztrn, ptrn,
                                                zval, pval,
                                                ztst, ptst,
                                                report, early_stoping
                                                )
        # Save last value
        LossTrain[p] = loss_train[end]
        LossVal[p] = loss_val[end]
        LossTest[p] = loss_test[end]
        # Save last training
        if percentage[p]==percentage[end]
            writedlm(results*"loss_epochs/train_$i$j.csv", loss_train, ',')
            writedlm(results*"loss_epochs/val_$i$j.csv",   loss_val,   ',')
            writedlm(results*"loss_epochs/test_$i$j.csv",  loss_test,  ',')
        end
        "--- $(percentage[p]) ---" |> println
    end
    # Save
    writedlm(results*"loss_percentage/train_$i$j.csv", LossTrain, ',')
    writedlm(results*"loss_percentage/val_$i$j.csv",   LossVal,   ',')
    writedlm(results*"loss_percentage/test_$i$j.csv",  LossTest,  ',')
end
