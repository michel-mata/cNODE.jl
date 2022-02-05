
if length( find(x -> x==pwd(), LOAD_PATH) ) .== 0
    println("Adding current path to LOAD_PATH...")
    push!(LOAD_PATH, pwd())
end

begin
    using DifferentialEquations
    using Distributions
    using CSV,DataFrames
    using Combinatorics

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
        A = sparse(A0 + eye(N))
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

    function getZall(n,rep)
        m = 2^n-1;
        Z = zeros(rep*m,n)
        for (i,c) in enumerate(combinations(1:n))
            for j in 1:rep
                k = (i-1)*rep
                Z[k+j,c] = 1
            end
        end
        return Z
    end
end

N = 5
rep = 3
begin
    M = 2^N-1

    S = collect(0.1:0.05:0.4);
    noise = [1e-4,1e-3,5e-3,1e-2,5e-2];
    R = [0.01,0.05,0.1,0.25,0.5];
    parameters = ["Dissimilarity","Rewiring","Universality","Noise"];
    d=1
    D = parameters[d]


    E = 10
    C = 0.5
    T = 1000

    if d==1
        it = length(S)
    elseif d==2
        it = length(R)
    else
        it = length(noise)
    end
    inx = [ (e,p) for e in 0:9,p in 1:it]
end

for p in 1:it
    for pp in 0:9
        # Changing parameters
        if d == 1
            σ,ρ,η,ϵ = S[p],0,0,0
        elseif d == 2
            σ,ρ,η,ϵ = S[1],R[p],0,0
        elseif d == 3
            σ,ρ,η,ϵ = S[1],0,noise[p],0
        elseif d == 4
            σ,ρ,η,ϵ = S[1],0,0,noise[p]
        end
        print("  $σ,  $ρ,  $η,  $ϵ\t")

        # Allocate
        global X = SharedArray{Float64,2}(M*rep,N)
        global A = SharedArray{Float64,2}(N,N)
        global r = SharedArray{Float64,2}(N,1)

        # Get initial values
        A₀,r = getParameters(N,C,σ)
        global Z = getZall(N,rep)

        conv = 0
        j = 1
        while j <= M*rep
            # Rewiring
            A = RewireNetwork(A₀,ρ)

            # Non-Universality: noise in non-zero entries
            if rand()<0.5
                A += η*( rand(Normal(0,1/N),N,N) .* (x-> x != 0 ? 1:0 ).(A) )
            end

            # Solve
            x = getSteadyState(Z[j,:].*rand(Uniform(0,10),N), T, A, r)

            # Convergence
            conv += sum(diagm(vec(x)) * ( A * x + r))

            # Noise in non-zero entries
            x += ϵ*( rand(Uniform(0,mean(x[x.>0])),N) .* rand([-1,1],N) .* Z[j,:])

            if sum(x.<0)==0
                X[j,:] = x
                j += 1
                j%100==0&&print("··")
            end
        end
        print("\t== $(conv/M) \n")

        CSV.write("./Data/$D/$N/A$p$pp.csv",convert(DataFrame,hcat(A,r)); header=false)
        CSV.write("./Data/$D/$N/Z$p$pp.csv",convert(DataFrame,Z); header=false)
        CSV.write("./Data/$D/$N/P$p$pp.csv",convert(DataFrame,X); header=false)
    end
end
