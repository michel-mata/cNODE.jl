begin
    using DifferentialEquations
    using Distributions, Distances
    using CSV,DataFrames

    function getParameters(N, C, σ)
        A = [ if i==j || rand()>C; 0 else rand(Normal(0,σ^2)) end for i=1:N,j=1:N ] - eye(N)
        r = rand(Uniform(0,1),N,1)
        return(A,r)
    end

    function getGLV(x, p, t)
        N = length(x)
        A = p[1:N, 1:N]
        r = p[1:N, N+1]
        pos = find(x .<= 0)
        x[pos] = 0
        dx = diagm(vec(x)) * ( A * x + r )
        dx[pos] = 0
        return(dx)
    end


    function getZ(N,M)
        z = zeros(N,M)
        inx = Any[]
        while size(inx,1)< M
            push!(inx,sort(unique(rand(1:N,rand(1:N)))))
            inx = unique(inx)
        end
        for i in 1:M
            z[inx[i],i]=1
        end
        return(z)
    end

    function getIC(N,M)
        z = getZ(N,M)
        x0 = z.*rand(Uniform(0,1),N,M)
        return(z,x0)
    end

    function RewireNetwork(A0, p)
        N = size(A0,1)
        A = sparse(A0 + 1.0*eye(N))
        row, col, s = findnz(A)
        if p > 0.0
            set = find(rand(nnz(A), 1) .< p)
            for i = 1:length(set)
                startnode = col[set[i]];
                index = find(col .== startnode)
                alreadyconnected = row[index]
                possibletargets = setdiff(1:N, alreadyconnected)
                if length(possibletargets) .>= 1
                    row[set[i]] = sample(possibletargets, 1)[1]
                end
            end
        end
        Arewired = sparse(row,col,s,N,N) - 1.0*eye(N)
        return Arewired
    end

    function getSteadyState(x0, τ, A, r)
        prob = ODEProblem(getGLV, x0, (0.0,τ), hcat(A, r))
        sol = solve(prob,force_dtmin=true)
        xf = sol.u[end]
        return(xf)
    end
    S = [.1,.15,.2]
    U = [0.1,1,2]
    E = [.001,.01,.025]
    R = [.01,.1,.25]
    C = [0.5,0.75,1]

    parameters = ["Strength","Connectivity","Universality","Noise","Rewiring"]
    println("DONE")
end



for (d,D) in enumerate(parameters)

    println("Validating: $D")

    # Fixed parameters
    N = 10
    M = 2^N-1
    T = 1000

    inx = [ (r,p) for r in 0:9,p in 1:3]

    #@sync @parallel
    for (e,p) in inx
        print(p-1,"_",e)

        # Changing parameters
        if d == 1
            σ,c,η,ϵ,ρ = S[p],C[1],0,0,0
        elseif d == 2
            σ,c,η,ϵ,ρ = S[1],C[p],0,0,0
        elseif d == 3
            σ,c,η,ϵ,ρ = S[1],C[1],U[p],0,0
        elseif d == 4
            σ,c,η,ϵ,ρ = S[1],C[1],0,E[p],0
        elseif d == 5
            σ,c,η,ϵ,ρ = S[1],C[1],0,0,R[p]
        end
        print("  $σ,  $c, $η,  $ϵ,  $ρ\t")

        # Allocate
        X = SharedArray{Float64,2}(N,M)
        A = SharedArray{Float64,2}(N,N)
        r = SharedArray{Float64,2}(N,1)

        # Get initial values
        A₀,r = getParameters(N,c,σ)
        Z,x₀ = getIC(N,M)

        conv = 0
        j = 1
        while j <= M
            # Rewiring
            A = RewireNetwork(A₀,ρ)

            # Non-Universality: noise in non-zero entries
            if η > 0
                A += η*( rand(Normal(0,σ^2),N,N) .* (x-> x != 0 ? 1:0 ).(A) )
            end

            # Solve
            x = getSteadyState(x₀[:,j], T, A, r)

            # Convergence
            conv += sum(diagm(vec(x)) * ( A * x + r))

            # Noise in non-zero entries
            x += ϵ*( rand(Uniform(0,mean(x[x.>0])),N) .* rand([-1,1],N) .* Z[:,j])
            x[x.<0]=0

            X[:,j] = x
            j += 1
            j%100==0&&print("··")
        end

        dist = pairwise(BrayCurtis(),X)
        dist = mean(dist[:][dist[:].!= 0])

        print("\t== $(conv/M) == $(dist) \n")
        mkpath("./test_data/$N/")
        CSV.write("./test_data/$N/$D/A$(p-1)_$e.csv",convert(DataFrame,hcat(A₀,r)); header=false)
        CSV.write("./test_data/$N/$D/Z$(p-1)_$e.csv",convert(DataFrame,Z); header=false)
        CSV.write("./test_data/$N/$D/P$(p-1)_$e.csv",convert(DataFrame,X); header=false)
    end
end
