"""
    FitnessLayer(
        N::Int64
        )

    Layer with `N × N` parameters with dynamics:

    ```
    function (L::FitnessLayer)(p)
        f = L.W *p
        ṗ = p .* (f - ones(size(p,1))*p'*f)
        return ( ṗ )
    end
    ```

    # Arguments:
    - `N::Int64` : layer size on declaration
    - `p::Array` : initial condition of dynamics on run
"""
struct FitnessLayer
    W
end

FitnessLayer(N::Int64) = FitnessLayer( zeros(N,N) )
Flux.@functor FitnessLayer

function (L::FitnessLayer)(p)
    f = L.W *p
    ṗ = p .* (f - ones(size(p,1))*p'*f)
    return ( ṗ )
end

"""
    getModel(
        N::Int64
        )

    Get compositional Neural ODE (cNODE) for system of size `N`.

    # Arguments:
    - `N::Int64` : system size
"""
getModel(N::Int64) = NeuralODE(FitnessLayer(N),(0.0,1.0),Tsit5(),saveat=1.0)

"""
    predict(
        cnode::NeuralODE,
        z
        )

    Predict composition from normalized collection `z`.

    # Arguments:
    - `cnode::NeuralODE` : compositional Neural ODE with `FitnessLayer`
    - `z::Array` : normalized collection of species with size `N×1`
"""
predict(cnode::NeuralODE,z) = Array(cnode(z).u[end])

"""
    loss(
        node::NeuralODE,
        z,
        p
    )

    Compute loss between predictions `q = node(z)` and true data `p`.

    # Arguments:
    - `node::NeuralODE` : neural ODE with `FitnessLayer`
    - `z::Array` : normalized collection of species with size `N×1`
    - `p::Array` : true relative abundances, size `N×1`
"""
_loss(cnode::NeuralODE,z,p) = sum(abs.(p .- predict(cnode,z))) / sum(abs.(p .+ predict(cnode,z)))
loss(cnode::NeuralODE,Z,P) = mean([ _loss(cnode,Z[:,i],P[:,i]) for i in 1:size(Z,2) ])

"""
    train_reptile(
        cnode::NeuralODE,
        epochs::Int64,
        mb::Int64,
        LR::Array{Float64},
        Ztrn, Ptrn,
        Zval, Pval,
        Ztst, Ptst,
        report::Int64,
        early_stoping::Int64
    )

    Train cNODE using the Reptile meta-learning algorithm.

    # Arguments:
    - cnode::NeuralODE : compositional Neural ODE to train
    - epochs::Int64 : number of epochs for training
    - mb::Int64 : minibatch size
    - LR::Array{Float64} : array with internal and external learning rates
    - Ztrn, Ptrn : training data
    - Zval, Pval : validation data
    - Ztst, Ptst : test data
    - report::Int64 : interval of epochs to report
    - early_stoping::Int64 : min epochs to start early stopping

"""
function train_reptile(
            cnode::NeuralODE,
            epochs::Int64,
            mb::Int64,
            LR::Array{Float64},
            Ztrn, Ptrn,
            Zval, Pval,
            Ztst, Ptst,
            report::Int64,
            early_stoping::Int64
        )
    loss_train = []
    loss_test = []
    loss_val = []
    stoping = []
    EarlyStoping = []
    es = 0

    W = Flux.params(cnode)
    opt_in = ADAM(LR[1])
    opt_out = ADAM(LR[2])

    for e in 1:epochs
        V = deepcopy(W)
        for (z,p) in eachbatch(shuffleobs((Ztrn,Ptrn)),mb)
            grads = gradient(()->loss(cnode,z,p),W)
            update!(opt_in, W, grads)
        end
        for (w, v) in zip(W,V)
            dv = apply!(opt_out, v, w .- v)
            @. w = v + dv
        end
        push!(loss_train,loss(cnode,Ztrn,Ptrn))
        push!(loss_test,loss(cnode,Ztst,Ptst))
        push!(loss_val,loss(cnode,Zval,Pval))

        push!(stoping,[loss(cnode,Ztrn[:,k],Ptrn[:,k]) for k in 1:size(Ztrn,2)])

        if e > early_stoping
            counts = count(mean(stoping[end-early_stoping:end-1]).<= stoping[end])/size(Ztrn,2)
            push!(EarlyStoping,counts)
            e%report==0 && println(e,
                            "\ttrain - ",round(loss_train[end];digits=3),
                            "\ttest - ",round(loss_test[end];digits=3),
                            '\t',round(counts;digits=3),
                            '\t',round(es;digits=3))
            if counts>0.5
                es +=1
                if es>10
                    "Early stoping at $e..." |> println
                    break
                end
            end
        elseif e>1
            e%report==0 && println(e,
                            "\ttrain - ",round(loss_train[end];digits=3),
                            "\ttest - ",round(loss_test[end];digits=3),'\t',"---")
        end
    end

    return W, loss_train, loss_val, loss_test
end
