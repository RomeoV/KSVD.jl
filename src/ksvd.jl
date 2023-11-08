import Random: shuffle
import Transducers: tcollect
using SparseMatricesCSR, ThreadedSparseCSR
using Polyester
ksvd(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix) = ksvd(OptimizedKSVD(), Y,D,X)

abstract type KSVDMethod end
struct LegacyKSVD <: KSVDMethod end
@kwdef struct OptimizedKSVD <: KSVDMethod
    shuffle_indices::Bool = false
end
@kwdef struct ParallelKSVD <: KSVDMethod
    shuffle_indices::Bool = false
    prealloc_buffers::Bool=true
end

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

function ksvd2(method::ParallelKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; err_buffers=nothing, err_gamma_buffers=nothing) where T
    N = size(Y, 2)
    # Eₖ = Y .- D * X
    D_t = Matrix(transpose(D));
    Eₖ = transpose(.-X'*D_t .+ Y')
    X_cpy = copy(X)
    D_cpy = copy(D)

    # preallocate error buffers
    Eₖ_buffers = (!isnothing(err_buffers) ? tcollect(copyto!(buf, Eₖ) for buf in  err_buffers) :
                  (method.prealloc_buffers ? [copy(Eₖ) for _ in 1:Threads.nthreads()] :
                   nothing))
    E_Ω_buffers = (!isnothing(err_buffers) ? err_gamma_buffers :
                  (method.prealloc_buffers ? [similar(Eₖ) for _ in 1:Threads.nthreads()] :
                   nothing))

    # Note: we need :static to use threadid, see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/
    basis_indices = (method.shuffle_indices ? shuffle(axes(X, 1)) : axes(X, 1))
    Threads.@threads :static for k in basis_indices
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        ωₖ = findall(!iszero, xₖ)
        Ωₖ = sparse(ωₖ, 1:length(ωₖ), ones(T, length(ωₖ)), N, length(ωₖ))

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        Eₖ_local = (!isnothing(Eₖ_buffers) ? Eₖ_buffers[Threads.threadid()] : copy(Eₖ))
        # Eₖ .+= D[:, k:k] * X[k:k, :]
        for (j, X_val) in zip(findnz(X[k, :])...)
            @inbounds @views Eₖ_local[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end

        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        # E_Ω = Eₖ_local * Ωₖ
        E_Ω = (!isnothing(E_Ω_buffers) ? let
                    buf = E_Ω_buffers[Threads.threadid()]
                    @inbounds @view buf[:, 1:length(ωₖ)]
               end : similar(Eₖ, size(Eₖ, 1), length(ωₖ)))
        for (i, j, Ω_val) in zip(findnz(Ωₖ)...)
            @inbounds @views E_Ω[:, j] .=  Ω_val .* Eₖ_local[:, i]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end
        U, S, V = if size(E_Ω, 2) <= 5
            svd!(E_Ω)
        else
            # tsvd(E_Ω)                 # second hotspot
            svd!(E_Ω)
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

function ksvd(method::ParallelKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; err_buffers::Union{Nothing, Vector{Matrix{T}}}=nothing, err_gamma_buffers::Union{Nothing, Vector{Matrix{T}}}=nothing) where T
    N = size(Y, 2)
    E = err_buffers[1]
    # E .= Y - D*X
    # Instead of just computing the `D*X` we use ThreadedSparseCSR for a portable and
    # fast multithreaded sparse matmu implementation.
    # Alternatively consider MKLSparse, but that's not portable and more proprietary.
    D_t = Matrix(transpose(D));
    Xcsrt = sparsecsr(findnz(X)[[2,1,3]]...);
    # X_t = sparse(findnz(X)[[2, 1, 3]]..., size(X)[[2,1]]...)
    E .= Y;
    # @inbounds for (i, D_row) in enumerate(eachrow(D))
    #     # Signature bmul!(y, A, x, alpha, beta) produces
    #     # y = alpha*A*x + beta*y (y = A*x)
    #     @views ThreadedSparseCSR.bmul!(E[i, :], Xcsrt, D_row, -1, 1)
    # end
    @fastmath @batch for i in axes(D_t, 2)
        @views E[i, :] .-= X' * D_t[:, i]
    end

    X_cpy = copy(X)
    D_cpy = copy(D)

    # preallocate error buffers
    E_Ω_buffers = (!isnothing(err_buffers) ? err_gamma_buffers :
                  (method.prealloc_buffers ? [similar(E) for _ in 1:Threads.nthreads()] :
                   nothing))

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
            # E_Ω[:, j_dest] .+= D * X[:, j]
            rs = nzrange(X, j)
            for (k, X_val) in zip(rowvals(X)[rs], nonzeros(X)[rs])
                E_Ω[:, j_dest] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
            end
            # for (X_val, k) in findnz(X[:, j])
            #     E_Ω[:, j_dest] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
            # end
        end
        # for i in axes(X, 1)
        #     for (j, X_val) in (@inbounds @views zip(findnz(X[i, ωₖ])...))
        #         @inbounds @views E_Ω[:, j] .+=  X_val .* D[:, i]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        #     end
        # end
        for (j, X_val) in zip(findnz(X[k, ωₖ])...)
            @inbounds @views E_Ω[:, j] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        end
        U, S, V = if size(E_Ω, 2) <= 5
            svd!(E_Ω)
        else
            # tsvd(E_Ω)                 # second hotspot
            svd!(E_Ω)
        end
        @inbounds @views D_cpy[:, k] .= U[:, 1]
        @inbounds @views X_cpy[k, ωₖ] .= S[1] .* V[:, 1]
        # reverse the local error matrix to be reused without copy
        # Eₖ .-= D[:, k:k] * X[k:k, :]
        # for (j, X_val) in zip(findnz(X[k, :])...)
        #     @inbounds @views Eₖ_local[:, j] .-=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
        # end
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
