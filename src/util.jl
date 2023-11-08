import DataStructures: DefaultDict
import SparseArrays: sparsevec
import Random: AbstractRNG, default_rng
import Distributions: Binomial, quantile

function SparseArrays.sparsevec(d::DefaultDict{Int, T}, m::Int) where T
    SparseArrays.sparsevec(collect(keys(d)), collect(values(d)), m)
end

init_dictionary(n::Int, K::Int) = init_dictionary(default_rng(), n, K)
init_dictionary(::Type{T}, n::Int, K::Int) where {T} = init_dictionary(default_rng(), T, n, K)
function init_dictionary(rng::AbstractRNG, T::Type, n::Int, K::Int)
    # D must be a full-rank matrix
    D = rand(rng, T, n, K) .- 0.5
    while rank(D, rtol=sqrt(min(n,K)*eps())) != min(n, K)
        D = rand(rng, T, n, K) .- 0.5
    end
    D = convert(Matrix{T}, D)

    normalize!.(eachcol(D))
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

# we need our buffers to be at least as wide as we have nonzero value in any row.
# we use an upper bound for the percentage of nonzero values and model the distribution
# of nonzero values per row as a binomial distribution
# Here's a script to convince yourself
#
# gen() = sum(sample([0,1], ProbabilityWeights([0.99, 0.01]), 100_000))
# histogram([gen() for _ in 1:1000])
" Compute buffer size that is large enough with extremely high likelyhood."
function compute_reasonable_buffer_size(N, pct_nz; failure_chance = eps())
    D = Binomial(N, pct_nz)
    quantile(D, 1-failure_chance)
end

reorient!(vec::AbstractVector) = vec .*= sign(first(vec))
reorient(vec::AbstractVector) = vec .* sign(first(vec))
