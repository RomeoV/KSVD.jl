import DataStructures: DefaultDict
import SparseArrays: sparsevec
import Random: AbstractRNG, default_rng
import Distributions: Binomial, quantile
import Base._typed_hcat
import StatsBase: sample
import Hungarian: hungarian

"Helper for `@threads for (i, idx) in enumerate(indices)` use case."
const cenumerate = collect ∘ enumerate

"Helper for `@threads for (lhs, rhs) in zip(foo, bar)` use case."
const czip = collect ∘ zip


function SparseArrays.sparsevec(d::DefaultDict{Int,T}, m::Int) where {T}
    SparseArrays.sparsevec(collect(keys(d)), collect(values(d)), m)
end

init_dictionary(n::Int, K::Int) = init_dictionary(Float64, n, K)
init_dictionary(::Type{T}, n::Int, K::Int) where {T} = init_dictionary(default_rng(), T, n, K)
function init_dictionary(rng::AbstractRNG, T::Type, n::Int, K::Int)
    # D must be a full-rank matrix
    D = rand(rng, T, n, K) .- 0.5
    while rank(D, rtol=sqrt(min(n, K) * eps())) != min(n, K)
        D = rand(rng, T, n, K) .- 0.5
    end
    D = convert(Matrix{T}, D)

    normalize!.(eachcol(D))
    return D
end

"""
    init_sparse_assignment(m::Int, n::Int, k::Int)
    init_sparse_assignment(::Type{T}, m::Int, n::Int, k::Int) where {T}
    init_sparse_assignment(fn::Function, m::Int, n::Int, k::Int)

Initialize a random matrix `X` in `m x n` with `k` nonzeros per column.
By default, each nonzero element is sampled from a uniform distribution with eltype `Float64`.
The eltype `T` or the initialization function with interface like `k -> rand(Float32, k).+1` can optionally be provided.
"""
init_sparse_assignment_mat(m::Int, n::Int, k::Int) = init_sparse_assignment_mat(Float64, m, n, k)
init_sparse_assignment_mat(::Type{T}, m::Int, n::Int, k::Int) where {T} = init_sparse_assignment_mat(k -> init_sparse_assignment_fn(T, k), m, n, k)
function init_sparse_assignment_mat(fn::Function, m::Int, n::Int, k::Int)
    X = reduce(hcat,
        (SparseVector(m, sort(sample(1:m, k; replace=false)), fn(k))
         for _ in 1:n)) |> SparseMatrixCSC
    return X
end
init_sparse_assignment_fn(T, k) = rand(T, k) .+ 1

function permute_D_X!(D, X, Dref::AbstractMatrix)
    distances = 1 .- abs.(D' * Dref)
    assignment, cost = hungarian(distances)
    D .= D[:, sortperm(assignment)]
    X .= X[sortperm(assignment), :]

    λs = sign.(dot.(eachcol(D), eachcol(Dref)))
    eachcol(D) .*= λs
    X .*= reshape(λs, :, 1)

    (; assignment, cost)  # s.t. D_rhs[:, assignment] ≈ D_lhs, and X_rhs[assignment, :] ≈ X_lhs
end

function permute_D_X!(D, X, Xref::AbstractSparseMatrix)
    distances = 1 .- abs.(X * Xref')
    assignment, cost = hungarian(distances)
    D .= D[:, sortperm(assignment)]
    X .= X[sortperm(assignment), :]

    λs = sign.([dot(X[i, :], Xref[i, :]) for i in axes(X, 1)])
    eachcol(D) .*= λs
    X .*= reshape(λs, :, 1)

    (; assignment, cost)  # s.t. D_rhs[:, assignment] ≈ D_lhs, and X_rhs[assignment, :] ≈ X_lhs
end



"""
    maybeview(mat::AbstractMatrix, ::Colon, idx::UnitRange)
    maybeview(mat::AbstractMatrix, ::Colon, idx)

Helper function to construct a matrix view if we have a continuous slice, and copy otherwise.
"""
maybeview(mat::AbstractMatrix, ::Colon, idx::UnitRange) = view(mat, :, idx)
maybeview(mat::AbstractMatrix, ::Colon, idx) = getindex(mat, :, idx)

""" Redefine findmax for vector of floats to not do nan-checks.

By default,`findmax` uses `isless`, which does a nan-check before computing `<(lhs, rhs)`.
We roll basically the same logic as in `Julia/Base/reduce.jl:findmax` but we directly use `<`, which gives us about a 1.5x speedup.
"""
function findmax_fast(data::Vector{T}) where {T}
    cmp_tpl((fm, im), (fx, ix)) = (fm < fx) ? (fx, ix) : (fm, im)
    mapfoldl(((k, v),) -> (v, k), cmp_tpl, pairs(data))
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
    D = copy(D)
    X = copy(X)
    D[:, k] .= 0
    X[k, :] .= 0
    return Y - D * X
end
# function error_matrix4(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
#     mask = CUDA.CuVector(1:size(D,2) .!= k)
#     mask_lhs = reshape(mask, 1, size(D, 2))
#     mask_rhs = reshape(mask, size(X, 1), 1)
#     return Y - (D.*mask_lhs) * (mask_rhs.*X)
# end

# we need our buffers to be at least as wide as we have nonzero value in any row.
# we use an upper bound for the percentage of nonzero values and model the distribution
# of nonzero values per row as a binomial distribution
# Here's a script to convince yourself
#
# gen() = sum(sample([0,1], ProbabilityWeights([0.99, 0.01]), 100_000))
# histogram([gen() for _ in 1:1000])
" Compute buffer size that is large enough with extremely high likelyhood."
function compute_reasonable_buffer_size(N, pct_nz; failure_chance=0)
    D = Binomial(N, pct_nz)
    quantile(D, 1 - failure_chance)
end

reorient!(vec::AbstractVector) = vec .*= sign(first(vec))
reorient(vec::AbstractVector) = vec .* sign(first(vec))

function Base._typed_hcat(::Type{T}, A::Base.AbstractVecOrTuple{SparseVector{T,Idx_t}}) where {T,Idx_t}
    K = length(first(A))
    N = length(A)
    X = let
        I = Idx_t[]
        J = Idx_t[]
        V = T[]
        for (i, v) in enumerate(A)
            append!(I, SparseArrays.nonzeroinds(v))
            append!(J, fill(Idx_t(i), SparseArrays.nnz(v)))
            append!(V, SparseArrays.nonzeros(v))
        end
        sparse(I, J, V, K, N)
    end
end

function descend_timer(::Nothing)
    return ()
end
function descend_timer(timer)
    return (timer.name, descend_timer(timer.prev_timer)...)
end
