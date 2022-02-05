"""
    import_data(
        path::String
        )

    Import species collections `Z` and compositions `P`.

    # Arguments:
    - `path::String` : location of data
    - `p::Float64` : optional partition
"""
function import_data(
            path::String,
            p::Float64=1.0
            )
    P = readdlm(path,',',Float64,'\n')
    Z = (x-> x>0 ? 1 : 0 ).(P)
    reps = unique([ Set(findall([ zz == z for z in eachcol(Z) ])) for zz in eachcol(Z) ])
    inx = [ rand([r...]) for r in reps]
    P = P[:,inx]
    Z = Z[:,inx]
    P = hcat([ normalize(p,1) for p in eachcol(P)]...)
    Z = hcat([ normalize(z,1) for z in eachcol(Z)]...)
    if p<1.0
        (Z,P),_ = splitobs(shuffleobs((Z,P)),at=p)
    end

    return Z, P
end

"""
    split_data(
        Z::Array{Float64},
        P::Array{Float64},
        p::Float64,
        q::Float64
    )

    Split data into train, validation and test sets.

    # Arguments:
    - `Z::Array{Float64}` : collections
    - `P::Array{Float64}` : compositions
    - `p::Float64` : percentage for training
    - `q::Float64` : partition into training (`q`%) and validation (`1-q`%) sets

"""
function split_data(
                Z::Array{Float64},
                P::Array{Float64},
                p::Float64=1.0,
                q::Float64=0.8
                )
    (ztrn,ptrn),(ZT,PT) = splitobs(shuffleobs((Z,P)),at = q)
    (ztrn,ptrn),_ = splitobs(shuffleobs((ztrn,ptrn)),at = p - 1e-5)
    (zval,pval),(ztst,ptst) = splitobs(shuffleobs((ZT,PT)),at = 1/2)
    return ztrn,ptrn,zval,pval,ztst,ptst
end
