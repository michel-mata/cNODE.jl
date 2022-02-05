begin
    Pkg.add("DiffEqFlux", v"0.7.0")
    Pkg.add("DifferentialEquations", v"6.9.0")
    Pkg.add("Distances", v"0.8.2")
    Pkg.add("Distributions", v"0.19.2")
    Pkg.add("Flux", v"0.9.0")
    Pkg.add("MLDataPattern", v"0.5.3")
    Pkg.add("StatsBase", v"0.33.0")

    pkgs_names = [
        "DelimitedFiles", "Random", "Statistics", "LinearAlgebra",
        "Distributed", "SharedArrays", "Combinatorics"
        ]

    Pkg.add.(pkgs_names)
end

using Statistics: mean
using Distances: braycurtis

using Distributions: Uniform, Dirichlet
using DifferentialEquations: Tsit5
using StatsBase: sample
using Random: rand, randperm, MersenneTwister
using Base.Iterators: partition
using Flux
using Flux.Tracker: gradient
using Flux.Optimise: update!, apply!
using DiffEqFlux: neural_ode
using MLDataPattern: eachbatch, splitobs, shuffleobs


# cNODE architecture
struct FitnessLayer
    W
end

FitnessLayer(N::Integer) = FitnessLayer( zeros(N,N) )
Flux.@functor FitnessLayer

function (L::FitnessLayer)(p)
    f = L.W *p
    ṗ = p .* (f - ones(size(p,1))*p'*f)
    return ( ṗ )
end

# Predict using neural ODE
function predict(θ,z)
    node(x) = neural_ode(θ,x,(0.0,1.0),Tsit5(),saveat=1.0)[:, end ]
    q = hcat([ node(z[:,i]) for i in 1:size(z,2) ]...)
    return q
end

# BrayCurtis loss function
Loss(θ,z,p) = mean( [ braycurtis( p[:,i], predict(θ,z[:,i]) ) for i in 1:size(z,2)] )

# Training Reptile loop
function train_reptile(node, epochs, mb, LR, Z, P, report)
    W = Flux.params(node)
    loss(z,p) = Loss(node,z,p)
    for e in 1:epochs
        V = deepcopy(Flux.data.(W))
        for (z,p) in eachbatch(shuffleobs((Z,P)),mb)
            grads = gradient(()->loss(z,p),W)
            update!(ADAM(LR[1]), W, grads)
        end
        for (w, v) in zip(W,V)
            dv = apply!(ADAM(LR[2]), v, w.data-v)
            @. w.data = v + dv
        end
        e%report==0 && println(e,'\t',Loss(node,Z,P))
    end
    W = node.W.data
    return W
end

# Training Reptile Loop with trace
function train_reptile(node, epochs, mb, LR, Ztrn, Ptrn, Ztst, Ptst, report)
    W = Flux.params(node)
    loss_train = []
    loss_test = []
    loss(z,p) = Loss(node,z,p)
    for e in 1:epochs
        V = deepcopy(Flux.data.(W))
        for (z,p) in eachbatch(shuffleobs((Ztrn,Ptrn)),mb)
            grads = gradient(()->loss(z,p),W)
            update!(ADAM(LR[1]), W, grads)
        end
        for (w, v) in zip(W,V)
            dv = apply!(ADAM(LR[2]), v, w.data-v)
            @. w.data = v + dv
        end
        push!(loss_train,Loss(node,Ztrn,Ptrn).data)
        push!(loss_test,Loss(node,Ztst,Ptst).data)
        e%report==0 && println(e,'\t',loss_train[end],'\t',loss_test[end])
    end
    W = node.W.data
    return W, loss_train, loss_test
end

# Training Reptile Loop with early stoping (test)
function train_reptile(node, epochs, mb, LR, Ztrn, Ptrn, Ztst, Ptst, report, early_stoping)
    W = Flux.params(node)
    loss_train = []
    loss_test = []
    stoping = []
    EarlyStoping = []
    loss(z,p) = Loss(node,z,p)
    es = 0
    for e in 1:epochs
        V = deepcopy(Flux.data.(W))
        for (z,p) in eachbatch(shuffleobs((Ztrn,Ptrn)),mb)
            grads = gradient(()->loss(z,p),W)
            update!(ADAM(LR[1]), W, grads)
        end
        for (w, v) in zip(W,V)
            dv = apply!(ADAM(LR[2]), v, w.data-v)
            @. w.data = v + dv
        end
        push!(loss_train,loss(Ztrn,Ptrn).data)
        push!(loss_test,loss(Ztst,Ptst).data)

        push!(stoping,[loss(Ztrn[:,k],Ptrn[:,k]).data for k in 1:size(Ztrn,2)])

        if e > early_stoping
            counts = count(mean(stoping[end-early_stoping:end-1]).<= stoping[end])/size(Ztrn,2)
            push!(EarlyStoping,counts)
            e%report==0 && println(e,'\t',loss_train[end],'\t',loss_test[end],'\t',counts,'\t',es)
            if counts>0.5
                es +=1
                if es>10
                    "Early stoping at $e..." |> println
                    break
                end
            end
        elseif e>1
            e%report==0 && println(e,'\t',loss_train[end],'\t',loss_test[end],'\t',"---")
        end
    end
    W = node.W.data
    return W, loss_train, loss_test
end

# Training Reptile Loop with early stoping (validation and test)
function train_reptile(node, epochs, mb, LR, Ztrn, Ptrn, Zval, Pval, Ztst, Ptst, report, early_stoping)
    W = Flux.params(node)
    loss_train = []
    loss_test = []
    loss_val = []
    stoping = []
    EarlyStoping = []
    loss(z,p) = Loss(node,z,p)
    es = 0
    for e in 1:epochs
        V = deepcopy(Flux.data.(W))
        for (z,p) in eachbatch(shuffleobs((Ztrn,Ptrn)),mb)
            grads = gradient(()->loss(z,p),W)
            update!(ADAM(LR[1]), W, grads)
        end
        for (w, v) in zip(W,V)
            dv = apply!(ADAM(LR[2]), v, w.data-v)
            @. w.data = v + dv
        end
        push!(loss_train,loss(Ztrn,Ptrn).data)
        push!(loss_test,loss(Ztst,Ptst).data)
        push!(loss_val,loss(Zval,Pval).data)

        push!(stoping,[loss(Ztrn[:,k],Ptrn[:,k]).data for k in 1:size(Ztrn,2)])

        if e > early_stoping
            counts = count(mean(stoping[end-early_stoping:end-1]).<= stoping[end])/size(Ztrn,2)
            push!(EarlyStoping,counts)
            e%report==0 && println(e,'\t',loss_train[end],'\t',loss_test[end],'\t',counts,'\t',es)
            if counts>0.5
                es +=1
                if es>10
                    "Early stoping at $e..." |> println
                    break
                end
            end
        elseif e>1
            e%report==0 && println(e,'\t',loss_train[end],'\t',loss_test[end],'\t',"---")
        end
    end
    W = node.W.data
    return W, loss_train, loss_val, loss_test
end

function null_model(Z,P,r)
    inx = [ findall(z.==1) for z in eachcol(Z) ]
    loss = []
    preds = []
    for _ in 1:r
        Q = zeros(size(Z))
        for (j,i) in enumerate(inx)
            Q[i,j] = rand(Dirichlet(length(i),1))
        end
        push!(preds,Q)
        push!(loss,braycurtis(P,Q))
    end
    return loss
end
