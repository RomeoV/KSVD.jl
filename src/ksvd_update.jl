import Random: shuffle
import SparseArrays: nzvalview

ksvd_update(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix) = ksvd_update(OptimizedKSVD(), Y, D, X)

abstract type KSVDMethod end

struct LegacyKSVD <: KSVDMethod end

@kwdef struct OptimizedKSVD <: KSVDMethod
    shuffle_indices::Bool = false
end

@kwdef mutable struct ParallelKSVD{precompute_error, T} <: KSVDMethod
    E_buf::Matrix{T} = T[;;]
    E_Ω_bufs::Vector{Matrix{T}} = Matrix{T}[]
    D_cpy_buf::Matrix{T} = T[;;]
    shuffle_indices::Bool = false
end

@kwdef mutable struct BatchedParallelKSVD{precompute_error, T} <: KSVDMethod
    E_buf::Matrix{T} = T[;;]
    E_Ω_bufs::Vector{Matrix{T}} = Matrix{T}[]
    D_cpy_buf::Matrix{T} = T[;;]
    shuffle_indices::Bool = false
    batch_size_per_thread::Int = 1
end

maybe_init_buffers!(method::KSVDMethod, emb_dim, n_dict_vecs, n_samples; pct_nz=1.) = nothing
function maybe_init_buffers!(method::Union{ParallelKSVD{I, T}, BatchedParallelKSVD{I, T}}, emb_dim, n_dict_vecs, n_samples; pct_nz=1.) where {I, T<:Real}
    method.E_buf=Matrix{T}(undef, emb_dim, n_samples)
    method.E_Ω_bufs=[Matrix{T}(undef, emb_dim, compute_reasonable_buffer_size(n_samples, pct_nz)) for _ in 1:Threads.nthreads()]
    method.D_cpy_buf=Matrix{T}(undef, emb_dim, n_dict_vecs)
end
is_initialized(method::KSVDMethod)::Bool = true
is_initialized(method::Union{ParallelKSVD, BatchedParallelKSVD})::Bool =
    length(method.E_buf) > 0 && length(method.E_Ω_bufs) > 0 && length(method.D_cpy_buf) > 0

function ksvd_update(method::LegacyKSVD, Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix)
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
        D[:, k]  =  sign(U[1,1])       .* @view(U[:, 1])
        X[k, wₖ] = (sign(U[1,1])*S[1]) .* @view(V[:, 1])
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

sparsecsr(M_t::Adjoint{SparseMatrixCSC}) = sparsecsr(findnz(parent(M_t))[[2,1,3]]..., size(Mt)...)
@inbounds function ksvd_update(method::Union{ParallelKSVD{false}, BatchedParallelKSVD{false}}, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; svd_method=svd!) where T
    @assert is_initialized(method) "Before using $method please call `maybe_initialize_buffers!(...)`"

    N = size(Y, 2)
    X_cpy = copy(X)
    D_cpy = method.D_cpy_buf
    @assert all(≈(1.), norm.(eachcol(D)))
    E_Ω_buffers = method.E_Ω_bufs

    # We iterate over each basis vector, either in one big batch or many small batches,
    # depending on the method. I.e. for ParallelKSVD, the first index_batch is just the entire dataset,
    # and for BatchedParallelKSVD it's split into more sub-batches.
    index_batches = make_index_batches(method, axes(X, 1))
    for index_batch in index_batches
        # Note: we need :static to use threadid, see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/
        Threads.@threads :static for k in index_batch
            xₖ = X[k, :]
            all(iszero, xₖ) && continue
            ωₖ = findall(!iszero, xₖ)
            E_Ω = let buf = E_Ω_buffers[Threads.threadid()]  # See :static note above.
                @view buf[:, 1:length(ωₖ)]
            end

            ##<BEGIN OPTIMIZED BLOCK>
            ## Original:
            # E_Ω .= Y[:, ωₖ] - D * X[:, ωₖ] + D[:, k] * X[k, ωₖ]'
            ## Faster:
            E_Ω .= @view Y[:, ωₖ]
            # Note: Make sure not to use `@view` on `X`, see https://github.com/JuliaSparse/SparseArrays.jl/issues/475
            fastdensesparsemul!(E_Ω, D, X[:, ωₖ], -1, 1)
            fastdensesparsemul_outer!(E_Ω, @view(D[:, k]), X[k, ωₖ], 1, 1)
            ## <END OPTIMIZED BLOCK>

            # truncated svd has some problems for column matrices. so then we just do svd.
            U, S, V = (size(E_Ω, 2) <= 3 ? svd!(E_Ω) : tsvd(E_Ω, 1; tolconv=100*eps(eltype(E_Ω))))
            # Notice we fix the sign of U[1,1] to be positive to make the svd unique and avoid oszillations.
            D_cpy[:, k]  .=  sign(U[1,1])       .* @view(U[:, 1])
            X_cpy[k, ωₖ] .= (sign(U[1,1])*S[1]) .* @view(V[:, 1])
        end
        D[:, index_batch] .= @view D_cpy[:, index_batch]
        # X[index_batch, :] .= X_cpy[index_batch, :]
        # we can exploit that the new nonzero indices don't change!
        # Note: This doesn't seem to help in the sparse copy above.
        row_indices = SparseArrays.rowvals(X) .∈ [index_batch]
        # Note: The next two lines are technically unnecessary, but sometimes the tests fail without them...
        # Maybe weirdly connected to a race condition?
        row_indices_cpy = SparseArrays.rowvals(X_cpy) .∈ [index_batch]
        @assert row_indices == row_indices_cpy
        nzvalview(X)[row_indices] .= nzvalview(X_cpy)[row_indices]

    end
    return D, X
end

@inbounds function ksvd_update(method::Union{ParallelKSVD{true}, BatchedParallelKSVD{true}}, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; svd_method=svd!) where T
    @assert is_initialized(method) "Before using $method please call `maybe_initialize_buffers!(...)`"

    N = size(Y, 2)
    E = method.E_buf
    # E = Y - D*X
    E .= Y
    fastdensesparsemul_threaded!(E, D, X, -1, 1)

    X_cpy = copy(X)
    D_cpy = method.D_cpy_buf
    @assert all(≈(1.), norm.(eachcol(D)))
    E_Ω_buffers = method.E_Ω_bufs

    # We iterate over each basis vector, either in one big batch or many small batches,
    # depending on the method. I.e. for ParallelKSVD, the first index_batch is just the entire dataset,
    # and for BatchedParallelKSVD it's split into more sub-batches.
    index_batches = make_index_batches(method, axes(X, 1))
    for index_batch in index_batches
        # Note: we need :static to use threadid, see https://julialang.org/blog/2023/07/PSA-dont-use-threadid/
        Threads.@threads :static for k in index_batch
            xₖ = X[k, :]
            all(iszero, xₖ) && continue
            ωₖ = findall(!iszero, xₖ)
            E_Ω = let buf = E_Ω_buffers[Threads.threadid()]  # See :static note above.
                @view buf[:, 1:length(ωₖ)]
            end

            E_Ω .= E[:, ωₖ]
            # E_Ω .+= @view(D[:, k:k]) * X[k:k, ωₖ]
            fastdensesparsemul_outer!(E_Ω, @view(D[:, k]), X[k, ωₖ], 1, 1)

            # truncated svd has some problems for column matrices. so then we just do svd.
            U, S, V = (size(E_Ω, 2) <= 3 ? svd!(E_Ω) : tsvd(E_Ω, 1; tolconv=100*eps(eltype(E_Ω))))
            # Notice we fix the sign of U[1,1] to be positive to make the svd unique and avoid oszillations.
            D_cpy[:, k]  .=  sign(U[1,1])       .* @view(U[:, 1])
            X_cpy[k, ωₖ] .= (sign(U[1,1])*S[1]) .* @view(V[:, 1])
        end

        # E .+= @view(D[:, index_batch]) * X[index_batch, :] - @view(D_cpy[:, index_batch]) * X_cpy[index_batch, :]
        fastdensesparsemul_threaded!(E, @view(D[:, index_batch]), X[index_batch, :], 1, 1)
        fastdensesparsemul_threaded!(E, @view(D_cpy[:, index_batch]), X_cpy[index_batch, :], -1, 1)
        D[:, index_batch] .= D_cpy[:, index_batch]
        # # <BEGIN OPTIMIZED BLOCK>.
        # # Original:
        # X[index_batch, :] .= X_cpy[index_batch, :]
        # # Optimized:
        # we can exploit that the new nonzero indices don't change!
        # Note: This doesn't seem to help in the sparse copy above.
        row_indices = SparseArrays.rowvals(X) .∈ [index_batch]
        # Note: The next two lines are technically unnecessary, but sometimes the tests fail without them...
        # Maybe weirdly connected to a race condition?
        row_indices_cpy = SparseArrays.rowvals(X_cpy) .∈ [index_batch]
        @assert row_indices == row_indices_cpy
        nzvalview(X)[row_indices] .= nzvalview(X_cpy)[row_indices]
        # # <END OPTIMIZED BLOCK>
    end
    return D, X
end

@inbounds function ksvd_update(method::OptimizedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}) where {T}
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
        Eₖ .+= D[:, k:k] * X[k:k, :]

        Ωₖ = sparse(ωₖ, 1:length(ωₖ), ones(length(ωₖ)), N, length(ωₖ))
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        # U, S, V = tsvd(Eₖ * Ωₖ, initvec=randn!(similar(Eₖ, size(Eₖ,1))))
        # E_Ω = Eₖ * Ωₖ
        E_Ω = @view E_Ω_buffer[:, 1:length(ωₖ)]
        E_Ω .= Eₖ[:, ωₖ]

        U, S, V = (size(E_Ω, 2) < 5 ? svd!(E_Ω) : tsvd(E_Ω, 1; tolconv=10*eps(eltype(E_Ω))) )
        D[:, k]  .=  sign(U[1,1])       .* @view(U[:, 1])
        X[k, ωₖ] .= (sign(U[1,1])*S[1]) .* @view(V[:, 1])

        Eₖ .-= D[:, k:k] * X[k:k, :]
    end
    return D, X
end
