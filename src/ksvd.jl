import Random: shuffle
import Transducers: tcollect
using SparseMatricesCSR, ThreadedSparseCSR
using Polyester
using IterativeSolvers
import LinearAlgebra: normalize!
ksvd(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix) = ksvd(OptimizedKSVD(), Y,D,X)

abstract type KSVDMethod end
struct LegacyKSVD <: KSVDMethod end
@kwdef struct OptimizedKSVD <: KSVDMethod
    shuffle_indices::Bool = false
end
OptimizedKSVD(T::Type, emb_dim::Int, n_dict_vecs::Int, n_samples::Int; pct_nz=0.1) =
    OptimizedKSVD()
@kwdef struct ParallelKSVD{T} <: KSVDMethod where T
    E_buf::Matrix{T}
    E_Ω_bufs::Vector{Matrix{T}}
    D_cpy_buf::Matrix{T}
    shuffle_indices::Bool = false
end
ParallelKSVD(T::Type, emb_dim::Int, n_dict_vecs::Int, n_samples::Int; pct_nz=0.1) =
    ParallelKSVD(E_buf=Matrix{T}(undef, emb_dim, n_samples),
                 E_Ω_bufs=[Matrix{T}(undef, emb_dim, compute_reasonable_buffer_size(n_samples, pct_nz)) for _ in 1:Threads.nthreads()],
                 D_cpy_buf=Matrix{T}(undef, emb_dim, n_dict_vecs))

@kwdef struct BatchedParallelKSVD{T} <: KSVDMethod where T
    E_buf::Matrix{T}
    E_Ω_bufs::Vector{Matrix{T}}
    D_cpy_buf::Matrix{T}
    shuffle_indices::Bool = false
    batch_size_per_thread::Int = 1
end
BatchedParallelKSVD(T::Type, emb_dim::Int, n_dict_vecs::Int, n_samples::Int; pct_nz=0.1, kwargs...) =
    BatchedParallelKSVD(E_buf=Matrix{T}(undef, emb_dim, n_samples),
                        E_Ω_bufs=[Matrix{T}(undef, emb_dim, compute_reasonable_buffer_size(n_samples, pct_nz)) for _ in 1:Threads.nthreads()],
                        D_cpy_buf=Matrix{T}(undef, emb_dim, n_dict_vecs);
                        kwargs...)

function ksvd(method::LegacyKSVD, Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix)
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

" Just yields one big batch in the 'regular' case. "
function make_index_batches(method::ParallelKSVD, axis)
    basis_indices = (method.shuffle_indices ? shuffle(axis) : axis)
    return [basis_indices]
end
" Yields batches with one element per thread. "
function make_index_batches(method::BatchedParallelKSVD, axis)
    basis_indices = (method.shuffle_indices ? shuffle(axis) : axis)
    return Iterators.partition(basis_indices, method.batch_size_per_thread*Threads.nthreads())
end

@inbounds function ksvd(method::Union{ParallelKSVD, BatchedParallelKSVD}, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; svd_method=svd!) where T
    N = size(Y, 2)
    # First, we try to compute
    # E .= Y - D*X
    # Instead of just computing the `D*X` we use ThreadedSparseCSR for a portable and
    # fast multithreaded sparse matmu implementation.
    # Alternatively consider MKLSparse, but that's not portable and more proprietary, and also not faster.
    E = method.E_buf
    E .= Y;
    Xcsr_t = sparsecsr(findnz(X)[[2,1,3]]..., reverse(size(X))...);
    for (i, D_row) in enumerate(eachrow(D))
        # Signature bmul!(y, A, x, alpha, beta) produces
        # y = alpha*A*x + beta*y (y = A*x)
        @views ThreadedSparseCSR.bmul!(E[i, :], Xcsr_t, D_row, -1, 1)
    end

    X_cpy = copy(X)
    D_cpy = method.D_cpy_buf
    @assert all(≈(1.), norm.(eachcol(D)))
    E_Ω_buffers = method.E_Ω_bufs

    # We iterate over each basis vector, either in one big batch or many small batches,
    # depending on the method.
    index_batches = make_index_batches(method, axes(X, 1))
    for index_batch in index_batches
        # Note: we need :static to use threadid, see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/
        Threads.@threads :static for k in index_batch
            xₖ = X[k, :]
            all(iszero, xₖ) && continue
            ωₖ = findall(!iszero, xₖ)
            E_Ω = (!isnothing(E_Ω_buffers) ? let
                        buf = E_Ω_buffers[Threads.threadid()]  # See :static note above.
                        @view buf[:, 1:length(ωₖ)]
                end : similar(E, size(E, 1), length(ωₖ)))

            @views E_Ω[:, 1:length(ωₖ)] .= E[:, ωₖ]
            for (j_dest, j) in enumerate(ωₖ)
                # We basically do this, but the basic version is insanely slow...
                # E_Ω[:, j_dest] .+= D * X[:, j]
                # Instead we do the method inspired by See https://stackoverflow.com/a/52615073/5616591
                rs = nzrange(X, j)
                @views E_Ω[:, j_dest] .+=  D[:, rowvals(X)[rs]] * nonzeros(X)[rs]
            end
            for (j, X_val) in zip(findnz(X[k, ωₖ])...) # we could also remove this allocation, but it's not much.
                @views E_Ω[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
            end

            # truncated svd has some problems for column matrices. so then we just do svd.
            U, S, V = (size(E_Ω, 2) <= 3 ? svd!(E_Ω) : tsvd(E_Ω, 1; tolconv=1000*eps(eltype(E_Ω))))
            # Notice we fix the sign of U[1,1] to be positive to make the svd unique and avoid oszillations.
            @views D_cpy[:, k] .= U[:, 1] * sign(U[1,1])
            @views X_cpy[k, ωₖ] .= S[1] .* V[:, 1] * sign(U[1,1])
        end
        @views D[:, index_batch] .= D_cpy[:, index_batch]
        @views X[:, index_batch] .= X_cpy[:, index_batch]
    end
    return D, X
end

@inbounds function ksvd(method::OptimizedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T}
    N = size(Y, 2)
    Eₖ = Y - D * X
    E_Ω_buffer = similar(Eₖ)
    basis_indices = (method.shuffle_indices ? shuffle(axes(X, 1)) : axes(X, 1))
    for k in basis_indices
        xₖ = @view X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        ωₖ = findall(!iszero, xₖ)

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        # Eₖ = error_matrix(Y, D, X, k)
        # @tullio Eₖ[i, j] += D[i,$k] * X[$k,j]   # first hotspot
        # Eₖ .+= D[:, k:k] * X[k:k, :]
        for (j, X_val) in zip(findnz(X[k, :])...)
            @views Eₖ[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end

        Ωₖ = sparse(ωₖ, 1:length(ωₖ), ones(length(ωₖ)), N, length(ωₖ))
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        # U, S, V = tsvd(Eₖ * Ωₖ, initvec=randn!(similar(Eₖ, size(Eₖ,1))))
        # E_Ω = Eₖ * Ωₖ
        E_Ω = @view E_Ω_buffer[:, 1:length(ωₖ)]
        for (i, j, Ω_val) in zip(findnz(Ωₖ)...)
            # notice that Ωₖ only has one value per column
            # so we can just assign here (instead of summing)
            @inbounds @views E_Ω[:, j] .=  Ω_val .* Eₖ[:, i]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end
        U, S, V = (size(E_Ω, 2) < 5 ? svd!(E_Ω) : tsvd(E_Ω, 1; tolconv=10*eps(eltype(E_Ω))) )
        @views D[:, k] = U[:, 1]
        @views X[k, ωₖ] = V[:, 1] * S[1]

        # Eₖ .-= D[:, k:k] * X[k:k, :]
        for (j, X_val) in zip(findnz(X[k, :])...)
            @inbounds @views Eₖ[:, j] .-=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end
        # @tullio Eₖ[i, j] += -D[i,$k] * X[$k,j]  # third hotspot
    end
    return D, X
end
