module KSVD

# This is an implementation of the K-SVD algorithm.
# The original paper:
# K-SVD: An Algorithm for Designing Overcomplete Dictionaries
# for Sparse Representation
# http://www.cs.technion.ac.il/~freddy/papers/120.pdf

# Variable names are based on the original paper.
# If you try to read the code, I recommend you to see Figure 2 first.
#

export ksvd, matching_pursuit

using ProgressMeter
using Base.Threads, Random, SparseArrays, LinearAlgebra
using TSVD
using Tullio


include("matching_pursuit.jl")

const default_sparsity_allowance = 0.9
const default_max_iter = 200
# const default_max_iter_mp = 200

Random.seed!(1234)  # for stability of tests


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
function error_matrix4(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, k::Int)
    mask = CUDA.CuVector(1:size(D,2) .!= k)
    mask_lhs = reshape(mask, 1, size(D, 2))
    mask_rhs = reshape(mask, size(X, 1), 1)
    return Y - (D.*mask_lhs) * (mask_rhs.*X)
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


abstract type KSVDMethod end
struct BasicKSVD <: KSVDMethod end
struct OptimizedKSVD <: KSVDMethod end
@kwdef struct ParallelKSVD <: KSVDMethod 
    prealloc_buffers::Bool=true
end

function ksvd(method::BasicKSVD, Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix)
    N = size(Y, 2)
    for k in 1:size(X, 1)
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        wₖ = findall(!iszero, xₖ)

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        Eₖ = error_matrix(Y, D, X, k)
        Ωₖ = sparse(wₖ, 1:length(wₖ), ones(length(wₖ)), N, length(wₖ))
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        U, S, V = svd(Eₖ * Ωₖ)
        D[:, k] = U[:, 1]
        X[k, wₖ] = V[:, 1] * S[1]
    end
    return D, X
end

function ksvd(method::ParallelKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; err_buffers=nothing, err_gamma_buffers=nothing) where T
    N = size(Y, 2)
    # Eₖ = Y .- D * X
    D_t = Matrix(transpose(D));
    Eₖ = transpose(.-X'*D_t .+ Y')
    X_cpy = copy(X)
    D_cpy = copy(D)

    # preallocate error buffers
    Eₖ_buffers = (!isnothing(err_buffers) ? copyto!.(err_buffers, [Eₖ]) :
                  (method.prealloc_buffers ? [copy(Eₖ) for _ in 1:Threads.nthreads()] : 
                   nothing))
    E_Ω_buffers = (!isnothing(err_buffers) ? err_gamma_buffers :
                  (method.prealloc_buffers ? [similar(Eₖ) for _ in 1:Threads.nthreads()] : 
                   nothing))

    # Note: we need :static to use threadid, see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/
    Threads.@threads :static for k in axes(X,1)
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        ωₖ = findall(!iszero, xₖ)
        Ωₖ = sparse(ωₖ, 1:length(ωₖ), ones(T, length(ωₖ)), N, length(ωₖ))

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        Eₖ_local = (method.prealloc_buffers ? Eₖ_buffers[Threads.threadid()] : copy(Eₖ))
        # Eₖ .+= D[:, k:k] * X[k:k, :]
        for (j, X_val) in zip(findnz(X[k, :])...)
            @inbounds @views Eₖ_local[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end

        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        # E_Ω = Eₖ_local * Ωₖ
        E_Ω = let
            buf = E_Ω_buffers[Threads.threadid()]
            @inbounds @view buf[:, 1:length(ωₖ)]
        end
        for (i, j, Ω_val) in zip(findnz(Ωₖ)...)
            @inbounds @views E_Ω[:, j] .=  Ω_val .* Eₖ_local[:, i]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end
        U, S, V = if size(E_Ω, 2) <= 5
            svd(E_Ω)
        else
            tsvd(E_Ω)                 # second hotspot
        end
        @inbounds @views D_cpy[:, k] .= U[:, 1]
        @inbounds @views X_cpy[k, ωₖ] .= S[1] .* V[:, 1]
        # reverse the local error matrix to be reused without copy
        # Eₖ .-= D[:, k:k] * X[k:k, :]
        for (j, X_val) in zip(findnz(X[k, :])...)
            @inbounds @views Eₖ_local[:, j] .-=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end
    end
    return D_cpy, X_cpy
end

function ksvd(method::OptimizedKSVD, Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix)
    N = size(Y, 2)
    Eₖ = Y - D * X
    @showprogress for k in 1:size(X, 1)
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        wₖ = findall(!iszero, xₖ)

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        # Eₖ = error_matrix(Y, D, X, k)
        # @tullio Eₖ[i, j] += D[i,$k] * X[$k,j]   # first hotspot
        Eₖ += D[:, k:k] * X[k:k, :]
        Ωₖ = sparse(wₖ, 1:length(wₖ), ones(length(wₖ)), N, length(wₖ))
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        # U, S, V = tsvd(Eₖ * Ωₖ, initvec=randn!(similar(Eₖ, size(Eₖ,1))))
        E_Ω = Eₖ * Ωₖ
        U, S, V = if size(E_Ω, 2) < 5
            svd(E_Ω)
        else
            tsvd(Eₖ * Ωₖ)                 # second hotspot
        end
        D[:, k] = U[:, 1]
        X[k, wₖ] = V[:, 1] * S[1]
        Eₖ -= D[:, k:k] * X[k:k, :]
        # @tullio Eₖ[i, j] += -D[i,$k] * X[$k,j]  # third hotspot
    end
    return D, X
end
ksvd(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix) = ksvd(OptimizedKSVD(), Y,D,X)
# ksvd(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix) = ksvd_parallel(Y,D,X)
# ksvd(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix) = ksvd_parallel(Y,D,X)


"""
    ksvd(Y::AbstractMatrix, n_atoms::Int;
         sparsity_allowance::Float64 = $default_sparsity_allowance,
         max_iter::Int = $default_max_iter,
         max_iter_mp::Int = $default_max_iter_mp)

Run K-SVD that designs an efficient dictionary D for sparse representations,
and returns X such that DX = Y or DX ≈ Y.

```
# Arguments
* `sparsity_allowance`: Stop iteration if the number of zeros in X / the number
    of elements in X > sparsity_allowance.
* `max_iter`: Limit of iterations.
* `max_iter_mp`: Limit of iterations in Matching Pursuit that `ksvd` calls at
    every iteration.
```
"""
function ksvd(Y::AbstractMatrix{T}, n_atoms::Int;
              sparsity_allowance = default_sparsity_allowance,
              ksvd_method = OptimizedKSVD(),
              sparse_coding_method = MatchingPursuit(),
              max_iter::Int = default_max_iter,
              verbose=false
              ) where T
              # max_iter_mp::Int = default_max_iter_mp)

    K = n_atoms
    n, N = size(Y)

    if !(0 <= sparsity_allowance <= 1)
        throw(ArgumentError("`sparsity_allowance` must be in range [0,1]"))
    end

    X = spzeros(T, K, N)  # just for making X global in this function
    max_n_zeros = ceil(Int, sparsity_allowance * length(X))

    # D is a dictionary matrix that contains atoms for columns.
    D = init_dictionary(n, K)  # size(D) == (n, K)

    p = Progress(max_iter)

    for i in 1:max_iter
        verbose && @info "Starting sparse coding"
        X_sparse = sparse_coding(sparse_coding_method, Y, D)
        verbose && @info "Starting svd"
        D, X = ksvd(ksvd_method, Y, D, X_sparse)

        # return if the number of zero entries are <= max_n_zeros
        if sum(iszero, X) > max_n_zeros
            return D, X
        end
        next!(p)
    end
    return D, X
end

end # module
