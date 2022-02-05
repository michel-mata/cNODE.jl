
"""
    getParameters(
        N::Int64,
        C::Float64,
        σ::Float64
        )

Get matrix of interactions `A` and instrisic growth rates `r`.

# Arguments:
- `N::Int64` : number of species
- `C::Float64` : connectivity
- `σ::Float64` : interaction strength
"""
function getParameters(
            N::Int64,
            C::Float64,
            σ::Float64
            )
    A = [ i==j ? -1. : rand()<C ? rand(Normal(0.,σ^2)) : 0. for i in 1:N, j in 1:N ]
    r = rand(N,1)
    return A, r
end

"""
    getGLV(
        x,
        pars::Array,
        t
    )

Get GLV dynamics.

# Arguments:
- `pars::Array` : concatenation of matrix `A` and vector `r`
- `x` and `t` : placeholders for ODE solver
"""
function getGLV(
            x,
            pars::Array,
            t
            )
    A = pars[:, 1:end-1]
    r = pars[:, end]
    inx = x .< 0
    x[inx] .= 0.0
    dx = diagm(vec(x)) * ( A * x + r )
    dx[inx] .= 0.0
    return dx
end

"""
    getIC(
        N::Int64,
        M::Int64
        )

Get collection of species `z` and numerical initial condition `x₀`.

# Arguments:
- `N::Int64` : number of species
- `M::Int64` : number of communities
"""
function getIC(
            N::Int64,
            M::Int64
            )
    z = zeros(Int64,N,M)
    inx = Any[]
    while size(inx,1) < M
        push!(inx, sort(unique( rand(1:N,rand(1:N)) )))
        unique!(inx)
    end
    for i in 1:M
        z[inx[i],i] .= 1
    end
    x₀ = z .* rand(N,M)
    return z, x₀
end

"""
    getSteadyState(
        x₀::Vector,
        τ::Float64,
        A::Array{Float64},
        r::Array{Float64}
        )

Get steady-state of dynamics by integrating ODE.

# Arguments:
- `x₀::Vector` : numerical initial condition
- `τ::Float64` : integration time
- `A::Array{Float64}` : interaction matrix
- `r::Array{Float64}` : intrinsic growth rates
"""
function getSteadyState(
            x₀::Vector,
            τ::Float64,
            A::Array{Float64},
            r::Array{Float64}
            )
    prob = ODEProblem(getGLV, x₀, (0.0,τ), hcat(A, r))
    sol = solve(prob,force_dtmin=true)
    xf = sol.u[end]
    return xf
end

"""
    getRewiredNetwork(
        A::Array{Float64},
        p::Float64
        )

Get rewired interaction matrix.

# Arguments:
- `A::Array{Float64}` : interaction matrix
- `p::Float64` : rewiring probability
"""
function getRewiredNetwork(
            A::Array{Float64},
            p::Float64
            )
    N = size(A,1)
    sA = sparse(A + diagm(ones(N)))
    row, col, s = findnz(sA)
    if p > 0.0
        for i in findall(rand(nnz(sA)) .< p)
            alreadyconnected = row[col .== col[i]]
            possibletargets = setdiff(1:N, alreadyconnected)
            length(possibletargets) >= 1 && (row[i] = rand(possibletargets))
        end
    end
    return sparse(row,col,s,N,N) - diagm(ones(N))
end

"""
        generate_data(
            N::Int64,
            M::Int64,
            repetitions::Int64,
            values::Int64,
            params::Array
        )

Generate synthetic data for parameters in `params`.
The function generates a number of `repetitions` for every value of `values`.

# Arguments:
- `N::Int64` : number of species
- `M::Int64` : number of communitites
- `repetitions::Int64` : number of ecological networks
- `values::Int64` : number of values per parameter
- `params::Array` : sweeping parameters
"""
function generate_data(N,M,repetitions,values,params)
    reps = 0:repetitions-1
    num = 1:values
    for par in params
        "Generating data: $par" |> println
        inx = [ (r,p) for r in reps, p in num ]
        @sync @distributed for i in inx
            # Changing parameters
            rep,p = i
            print(p-1,"_",rep)
            if par == "Strength"
                σ,c,η,ϵ,ρ = S[p],C[1],0,0,0.
            elseif par == "Connectivity"
                σ,c,η,ϵ,ρ = S[1],C[p],0,0,0.
            elseif par == "Universality"
                σ,c,η,ϵ,ρ = S[1],C[1],U[p],0,0.
            elseif par == "Noise"
                σ,c,η,ϵ,ρ = S[1],C[1],0,E[p],0.
            elseif par == "Rewiring"
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
                A = getRewiredNetwork(A₀,ρ)
                # Non-Universality: noise in non-zero entries
                if η > 0
                    A += η*( rand(Normal(0,σ^2),N,N) .* (x-> x != 0 ? 1 : 0 ).(A) )
                end
                # Solve
                x = getSteadyState(x₀[:,j], 1000., A, r)
                # Convergence
                conv += sum(diagm(vec(x)) * ( A * x + r))
                # Noise in non-zero entries
                x += ϵ*( rand(Uniform(0,mean(x[x.>0])),N) .* rand([-1,1],N) .* Z[:,j])
                x[x.<0].=0
                X[:,j] = x
                j += 1
                j%100==0&&print("··")
            end
            print("\t== $(conv/M) \n")

            path = "./data/synthetic/$N/$par/"
            !ispath(path) && mkpath(path)

            writedlm(path*"A$(p-1)_$rep.csv", hcat(A₀,r), ',')
            writedlm(path*"Z$(p-1)_$rep.csv", Z, ',')
            writedlm(path*"P$(p-1)_$rep.csv", X, ',')
        end
    end
end
