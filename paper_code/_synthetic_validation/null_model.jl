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
    using MLDataPattern: splitobs, shuffleobs
    using SharedArrays

    N = 100
    function import_data(DATA,i,j)
        if DATA == "Multistability"
            P = readdlm("./Data/Synthetic/$N/Strength/P$(i)_$j.csv",',',Float64,'\n')
        else
            P = readdlm("./Data/Synthetic/$N/$DATA/P$(i)_$j.csv",',',Float64,'\n')
        end
        P = hcat([ normalize(p,1) for p in eachcol(P)]...)
        Z = (x-> x>0 ? 1 : 0 ).(P)
        (Z,P),_ = splitobs(shuffleobs((Z,P)),at=0.325)
        return Z,P
    end

    function split_data(Z,P,p)
        (ZT,PT),(ztst,ptst) = splitobs(shuffleobs((Z,P)),at = 300/325)
        (ztrn,ptrn),_ = splitobs(shuffleobs((ZT,PT)),at = p - 1e-3)
        return ztrn,ptrn,ztst,ptst
    end

    Datasets = ["Strength","Connectivity","Universality","Noise","Rewiring","Multistability"]
    percentage = [.001,.01,0.1,0.25,0.5,0.75,1]
    parameters = 0:2
    nets = 0:9
end
####################################################
println("Begining Validation... ")
inx = collect(product(parameters,nets))[:]

@time for (d,DATA) in enumerate(Datasets)
    for it in inx
        i,j = it
        println(DATA,'\t',i,'\t',j)

        # Data and variables
        Z,P = import_data(DATA,i,j)
        LossTrain = SharedArray{Float64}(length(percentage))
        LossTest = SharedArray{Float64}(length(percentage))

        # For percentages of split
        @time for (k,p) in enumerate(percentage)
            # Load Data
            ztrn,ptrn,ztst,ptst = split_data(Z,P,p)

            # Save loss
            LossTrain[k] = mean(null_model(ztrn,ptrn,10))
            LossTest[k] = mean(null_model(ztst,ptst,10))
            println(p,'\t',LossTrain[k],'\t',LossTest[k])

        end
        # Export
        writedlm("./Results/Synthetic/$N/$DATA/loss_percentage_null/train_$i$j.csv",LossTrain, ',')
        writedlm("./Results/Synthetic/$N/$DATA/loss_percentage_null/test_$i$j.csv",LossTest, ',')
    end
end
