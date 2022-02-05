####################################################
"start..." |> println
begin
    "loading libraries" |> println
    !(pwd() in LOAD_PATH) && push!(LOAD_PATH, pwd())
    include("../cNODE.jl")
    # Modules
    using DelimitedFiles: readdlm, writedlm
    using LinearAlgebra: normalize
    using Base.Iterators: enumerate, partition, product
    using MLDataPattern: splitobs, shuffleobs, kfolds
    using SharedArrays
    using Distributions: Dirichlet

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
end

for (j,DATA) in enumerate(Datasets)
    Z,P = import_data(DATA)
    N,M = size(Z);
    M |> println
    LossTrain = Array{Float64}(undef,M)
    LossTest = Array{Float64}(undef,M)
    real_sample = Array{Float64}(undef,N,M)
    pred_sample = Array{Float64}(undef,N,M)

    LeaveOneOut = kfolds((Z,P); k = M) |> enumerate |> collect

    for (i,fold) in LeaveOneOut
        ((ztrn,ptrn),(ztst,ptst)) = fold

        # Save loss
        l,p,q = null_model(ztrn,ptrn,1000)
        LossTrain[i] = mean(l)
        l,p,q = null_model(ztst,ptst,1000)
        LossTest[i],real_sample[:,i],pred_sample[:,i] = mean(l),p,q[1]
        println(i,'\t',LossTrain[i],'\t',LossTest[i])
    end

    writedlm("./Results/Experimental/$DATA/null_model/test_loss.csv",LossTest, ',')
    writedlm("./Results/Experimental/$DATA/null_model/train_loss.csv",LossTrain, ',')
    writedlm("./Results/Experimental/$DATA/null_model/real_sample.csv",real_sample, ',')
    writedlm("./Results/Experimental/$DATA/null_model/pred_sample.csv",pred_sample, ',')
end
