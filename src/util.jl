import DataStructures: DefaultDict
import SparseArrays: sparsevec

function SparseArrays.sparsevec(d::DefaultDict{Int, T}, m::Int) where T
    SparseArrays.sparsevec(collect(keys(d)), collect(values(d)), m)
end

init_dictionary(n::Int, K::Int) = init_dictionary(Float64, n, K)
function init_dictionary(T::Type, n::Int, K::Int)
    # D must be a full-rank matrix
    D = rand(n, K)
    while rank(D, rtol=sqrt(min(n,K)*eps())) != min(n, K)
        D = rand(T, n, K)
    end
    D = convert(Matrix{T}, D)

    @inbounds for k in 1:K
        D[:, k] ./= norm(@view(D[:, k]))
    end
    return D
end

""" Redefine findmax for vector of floats to not do nan-checks.

By default,`findmax` uses `isless`, which does a nan-check before computing `<(lhs, rhs)`.
We roll basically the same logic as in `Julia/Base/reduce.jl:findmax` but we directly use `<`, which gives us about a 1.5x speedup.
"""
function findmax_fast(data::Vector{T}) where T
    cmp_tpl((fm, im), (fx, ix)) = (fm < fx) ? (fx, ix) : (fm, im)
    mapfoldl( ((k, v),) -> (v, k), cmp_tpl, pairs(data))
end

function error_matrix(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
    # indices = [i for i in 1:size(D, 2) if i != k]
    indices = deleteat!(collect(1:size(D, 2)), k)
    return Y - D[:, indices] * X[indices, :]
end

function error_matrix2(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
    return Y - (D * X - D[:, k:k] * X[k:k, :])
end
function error_matrix3(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
    D = copy(D); X = copy(X);
    D[:, k] .= 0; X[k, :] .= 0
    return Y - D * X
end
# function error_matrix4(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
#     mask = CUDA.CuVector(1:size(D,2) .!= k)
#     mask_lhs = reshape(mask, 1, size(D, 2))
#     mask_rhs = reshape(mask, size(X, 1), 1)
#     return Y - (D.*mask_lhs) * (mask_rhs.*X)
# end
