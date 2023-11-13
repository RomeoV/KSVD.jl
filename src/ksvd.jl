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
end
BatchedParallelKSVD(T::Type, emb_dim::Int, n_dict_vecs::Int, n_samples::Int; pct_nz=0.01) =
    BatchedParallelKSVD(E_buf=Matrix{T}(undef, emb_dim, n_samples),
                 E_Ω_bufs=[Matrix{T}(undef, emb_dim, compute_reasonable_buffer_size(n_samples, pct_nz)) for _ in 1:Threads.nthreads()],
                 D_cpy_buf=Matrix{T}(undef, emb_dim, n_dict_vecs))

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

function ksvd(method::BatchedParallelKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; svd_method=svd!) where T
    N = size(Y, 2)
    # E .= Y - D*X
    # Instead of just computing the `D*X` we use ThreadedSparseCSR for a portable and
    # fast multithreaded sparse matmu implementation.
    # Alternatively consider MKLSparse, but that's not portable and more proprietary.
    # D_t = Matrix(transpose(D));
    # X_t = sparse(findnz(X)[[2, 1, 3]]..., size(X)[[2,1]]...)
    E = method.E_buf
    E .= Y;
    Xcsr_t = sparsecsr(findnz(X)[[2,1,3]]..., reverse(size(X))...);
    @inbounds for (i, D_row) in enumerate(eachrow(D))
        # Signature bmul!(y, A, x, alpha, beta) produces
        # y = alpha*A*x + beta*y (y = A*x)
        @views ThreadedSparseCSR.tmul!(E[i, :], Xcsr_t, D_row, -1, 1)
    end
    # @fastmath @batch for i in axes(D_t, 2)
    #     @views E[i, :] .-= X' * D_t[:, i]
    # end

    X_cpy = copy(X)
    D_cpy = method.D_cpy_buf
    @assert all(≈(1.), norm.(eachcol(D)))
    E_Ω_buffers = method.E_Ω_bufs

    # Note: we need :static to use threadid, see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/
    basis_indices = (method.shuffle_indices ? shuffle(axes(X, 1)) : axes(X, 1))
    index_batches = Iterators.partition(basis_indices, Threads.nthreads())
    for index_batch in index_batches
        Threads.@threads :static for k in index_batch
            xₖ = X[k, :]
            # ignore if the k-th row is zeros
            all(iszero, xₖ) && continue

            # wₖ is the column indices where the k-th row of xₖ is non-zero,
            # which is equivalent to [i for i in N if xₖ[i] != 0]
            ωₖ = findall(!iszero, xₖ)
            # Ωₖ = sparse(ωₖ, 1:length(ωₖ), ones(T, length(ωₖ)), N, length(ωₖ))

            # Eₖ * Ωₖ implies a selection of error columns that
            # correspond to examples that use the atom D[:, k]
            # Eₖ_local = (!isnothing(Eₖ_buffers) ? Eₖ_buffers[Threads.threadid()] : copy(Eₖ))
            # # Eₖ .+= D[:, k:k] * X[k:k, :]
            # for (j, X_val) in zip(findnz(X[k, :])...)
            #     @inbounds @views Eₖ_local[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
            # end

            # Note that S is a vector that contains diagonal elements of
            # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
            # Non-zero entries of X are set to
            # the first column of V multiplied by Δ(1, 1)
            # E_Ω = Eₖ_local * Ωₖ
            E_Ω = (!isnothing(E_Ω_buffers) ? let
                        buf = E_Ω_buffers[Threads.threadid()]
                        @inbounds @view buf[:, 1:length(ωₖ)]
                end : similar(E, size(E, 1), length(ωₖ)))

            @inbounds @views E_Ω[:, 1:length(ωₖ)] .= E[:, ωₖ]
            @inbounds @views for (j_dest, j) in enumerate(ωₖ)
                # this is insanely slow...
                # E_Ω[:, j_dest] .+= D * X[:, j]
                # See https://stackoverflow.com/a/52615073/5616591
                # This seems to be faster than `findnz` because it doesn't allocate at all.
                # These next two are pretty competitive.
                rs = nzrange(X, j)
                @fastmath E_Ω[:, j_dest] .+=  D[:, rowvals(X)[rs]] * nonzeros(X)[rs]
                # @fastmath for (k, X_val) in zip(rowvals(X)[rs], nonzeros(X)[rs])
                #     E_Ω[:, j_dest] .+=  X_val .* D[:, k]
                # end
                # for (X_val, k) in findnz(X[:, j])
                #     E_Ω[:, j_dest] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
                # end
            end
            for (j, X_val) in zip(findnz(X[k, ωₖ])...)
                @inbounds @views E_Ω[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
            end

            # Notice we fix the sign of U[1,1] to be positive to make the svd unique and avoid oszillations.
            if size(E_Ω, 2) <= 3
                U, S, V = svd!(E_Ω)
                @inbounds @views D_cpy[:, k] .= U[:, 1] * sign(U[1,1])
                @inbounds @views X_cpy[k, ωₖ] .= S[1] .* V[:, 1] * sign(U[1,1])
            else
                U, S, V = tsvd(E_Ω, 1; tolconv=10*eps())
                @inbounds @views D_cpy[:, k] .= U[:, 1] * sign(U[1,1])
                @inbounds @views X_cpy[k, ωₖ] .= S[1] .* V[:, 1] * sign(U[1,1])
            end
        end
        @views D[:, index_batch] .= D_cpy[:, index_batch]
        @views X[:, index_batch] .= X_cpy[:, index_batch]
    end
    return D, X
end

function ksvd(method::ParallelKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; svd_method=svd!) where T
    N = size(Y, 2)
    # E .= Y - D*X
    # Instead of just computing the `D*X` we use ThreadedSparseCSR for a portable and
    # fast multithreaded sparse matmu implementation.
    # Alternatively consider MKLSparse, but that's not portable and more proprietary.
    # D_t = Matrix(transpose(D));
    # X_t = sparse(findnz(X)[[2, 1, 3]]..., size(X)[[2,1]]...)
    E = method.E_buf
    E .= Y;
    Xcsr_t = sparsecsr(findnz(X)[[2,1,3]]..., reverse(size(X))...);
    @inbounds for (i, D_row) in enumerate(eachrow(D))
        # Signature bmul!(y, A, x, alpha, beta) produces
        # y = alpha*A*x + beta*y (y = A*x)
        @views ThreadedSparseCSR.tmul!(E[i, :], Xcsr_t, D_row, -1, 1)
    end
    # @fastmath @batch for i in axes(D_t, 2)
    #     @views E[i, :] .-= X' * D_t[:, i]
    # end

    X_cpy = copy(X)
    D_cpy = method.D_cpy_buf
    @assert all(≈(1.), norm.(eachcol(D)))
    E_Ω_buffers = method.E_Ω_bufs

    # Note: we need :static to use threadid, see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/
    basis_indices = (method.shuffle_indices ? shuffle(axes(X, 1)) : axes(X, 1))
    Threads.@threads :static for k in basis_indices
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        ωₖ = findall(!iszero, xₖ)
        # Ωₖ = sparse(ωₖ, 1:length(ωₖ), ones(T, length(ωₖ)), N, length(ωₖ))

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        # Eₖ_local = (!isnothing(Eₖ_buffers) ? Eₖ_buffers[Threads.threadid()] : copy(Eₖ))
        # # Eₖ .+= D[:, k:k] * X[k:k, :]
        # for (j, X_val) in zip(findnz(X[k, :])...)
        #     @inbounds @views Eₖ_local[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        # end

        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        # E_Ω = Eₖ_local * Ωₖ
        E_Ω = (!isnothing(E_Ω_buffers) ? let
                    buf = E_Ω_buffers[Threads.threadid()]
                    @inbounds @view buf[:, 1:length(ωₖ)]
               end : similar(E, size(E, 1), length(ωₖ)))

        @inbounds @views E_Ω[:, 1:length(ωₖ)] .= E[:, ωₖ]
        @inbounds @views for (j_dest, j) in enumerate(ωₖ)
            # this is insanely slow...
            # E_Ω[:, j_dest] .+= D * X[:, j]
            # See https://stackoverflow.com/a/52615073/5616591
            # This seems to be faster than `findnz` because it doesn't allocate at all.
            # These next two are pretty competitive.
            rs = nzrange(X, j)
            @fastmath E_Ω[:, j_dest] .+=  D[:, rowvals(X)[rs]] * nonzeros(X)[rs]
            # @fastmath for (k, X_val) in zip(rowvals(X)[rs], nonzeros(X)[rs])
            #     E_Ω[:, j_dest] .+=  X_val .* D[:, k]
            # end
            # for (X_val, k) in findnz(X[:, j])
            #     E_Ω[:, j_dest] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
            # end
        end
        for (j, X_val) in zip(findnz(X[k, ωₖ])...)
            @inbounds @views E_Ω[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end

        # Notice we fix the sign of U[1,1] to be positive to make the svd unique and avoid oszillations.
        if size(E_Ω, 2) <= 3
            U, S, V = svd!(E_Ω)
            @inbounds @views D_cpy[:, k] .= U[:, 1] * sign(U[1,1])
            @inbounds @views X_cpy[k, ωₖ] .= S[1] .* V[:, 1] * sign(U[1,1])
        else
            U, S, V = tsvd(E_Ω, 1; tolconv=10*eps())
            @inbounds @views D_cpy[:, k] .= U[:, 1] * sign(U[1,1])
            @inbounds @views X_cpy[k, ωₖ] .= S[1] .* V[:, 1] * sign(U[1,1])
        end
    end
    D, X = D_cpy, X_cpy
    return D, X
end

function ksvd(method::OptimizedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T}
    N = size(Y, 2)
    Eₖ = Y - D * X
    E_Ω_buffer = similar(Eₖ)
    basis_indices = (method.shuffle_indices ? shuffle(axes(X, 1)) : axes(X, 1))
    for k in basis_indices
        xₖ = X[k, :]
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
            @inbounds @views Eₖ[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
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
        U, S, V = if size(E_Ω, 2) < 5
            svd!(E_Ω)
        else
            # tsvd(E_Ω)                 # second hotspot
            # svdl(E_Ω, nsv=1, vecs=:both, maxiter=1000, tol=1e2*eps(T))[1]
            svd!(E_Ω)
        end
        D[:, k] = U[:, 1]
        X[k, ωₖ] = V[:, 1] * S[1]

        # Eₖ .-= D[:, k:k] * X[k:k, :]
        for (j, X_val) in zip(findnz(X[k, :])...)
            @inbounds @views Eₖ[:, j] .-=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end
        # @tullio Eₖ[i, j] += -D[i,$k] * X[$k,j]  # third hotspot
    end
    return D, X
end
