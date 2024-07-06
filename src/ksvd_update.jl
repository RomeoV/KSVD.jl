import Random: shuffle
import SparseArrays: nzvalview, nonzeroinds
import OhMyThreads
import OhMyThreads: tforeach
# import OhMyThreads: SerialScheduler
import TimerOutputs: TimerOutput, @timeit
import ChunkSplitters: chunks
import Base.Threads: nthreads, threadpool

# set a default
ksvd_update(Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix, timer=TimerOutput()) = ksvd_update(OptimizedKSVD(), Y, D, X, timer)

"""
    ksvd_update(method::ParallelKSVD{false}, Y, D, X)
    ksvd_update(method::BatchedParallelKSVD{false}, Y, D, X)

Parallel KSVD method without preallocation.
In this implementation, the computation `E = Y - D*X` is not precomputed, but instead fused into the
later computation ` E_Ω .= Y[:, ωₖ] - D * X[:, ωₖ] + D[:, k] * X[k, ωₖ]`.
This can often be faster for large data sizes, and is also easier to parallelize.

    ksvd_update(method::ParallelKSVD{true}, Y, D, X)
    ksvd_update(method::BatchedParallelKSVD{true}, Y, D, X)

Parallel KSVD method with preallocation.
In this implementation, the computation `E = Y - D*X` is precomputed and a preallocated buffer is used.
Note that this uses a major part of the computation time, and any speedup is worth a lot.
For this reason, we use a specialized parallel implementation provided in the `ThreadedDenseSparseMul.jl` package.
The result of this precomputation can later be used in the computation ` E_Ω .= Y[:, ωₖ] - D * X[:, ωₖ] + D[:, k] * X[k, ωₖ]`.
by only having to compute the last part, and "resetting it" after using the result.
"""
function ksvd_update(method::ThreadedKSVDMethod, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T};
                     force_reinitialize_buffers::Bool=false, timer=TimerOutput(), merge_all_timers=true) where T
    @timeit timer "KSVD update" begin

    if force_reinitialize_buffers || !is_initialized(method) "Before using $method please call `maybe_initialize_buffers!(...)`"
        maybe_init_buffers!(method, size(D, 1), size(D, 2), size(Y, 2), timer)
    end

    @timeit_debug timer "copy X" begin
        X_cpy = copy(X)
    end
    D_cpy = method.D_cpy_buf
    @assert all(≈(1.), norm.(eachcol(D)))

    # this is a no-op if the template `precompute_error` is false.
    E = maybe_prepare_error_buffer!(method, Y, D, X; timer)

    timer_ch = Channel{TimerOutput}(nchunks(method))
    foreach(to->put!(timer_ch, to), [TimerOutput() for _ in 1:nchunks(method)])

    E_Ω_buf_ch = Channel{Matrix{T}}(length(method.E_Ω_bufs))
    foreach(buf->put!(E_Ω_buf_ch, buf), method.E_Ω_bufs)

    # We iterate over each basis vector, either in one big batch or many small batches, depending on the method.
    # I.e. for ParallelKSVD, the first index_batch is just the entire dataset, and for BatchedParallelKSVD it's split
    # into more sub-batches.
    index_batches = make_index_batches(method, axes(X, 1))
    scheduler = get_scheduler_t(method)()  # we do our own chunking
    @inbounds for index_batch in index_batches  # <- This is just one big batch for `ParallelKSVD`
        tforeach(index_batch; scheduler) do k
            # we use channels to manage the batch-local variables
            E_Ω_buf = take!(E_Ω_buf_ch); timer_ = take!(timer_ch)
            ksvd_update_k!(method, E_Ω_buf, D_cpy, X_cpy, E, Y, D, X, k, timer_)
            put!(E_Ω_buf_ch, E_Ω_buf); put!(timer_ch, timer_)
        end

        maybe_update_errors!(method, E, D_cpy, X_cpy, D, X, index_batch; timer)

        @timeit_debug timer "copy D results" begin
            D[:, index_batch] .= @view D_cpy[:, index_batch]
        end
        ksvd_update_X!(X, X_cpy, index_batch, timer)
    end

    close(timer_ch)
    if KSVD.timeit_debug_enabled()
        merge!(timer, (merge_all_timers ? collect(timer_ch) : [first(timer_ch)])...; tree_point=["KSVD update"])
    end

    end # @timeit
    return D, X
end

# This is the main composition method!
function ksvd_update_k!(method::ThreadedKSVDMethod, E_Ω_buf::AbstractMatrix{T}, D_cpy, X_cpy, E, Y, D, X, k,
                        timer=TimerOutput()) where {T}
    @timeit_debug timer "ksvd_update_k!" begin

    @timeit_debug timer "find nonzeros" begin
        xₖ = X[k, :]
        if all(iszero, xₖ)
            D_cpy[:, k] .= D[:, k];
            return
        end
        # ωₖ = findall(!iszero, xₖ)
        ωₖ = nonzeroinds(xₖ)
    end

    E_Ω = compute_E_Ω!(method, E_Ω_buf, E, Y, D, X, xₖ, ωₖ, k, timer)

    @timeit_debug timer "compute and copy tsvd" begin
        # truncated svd has some problems for column matrices. so then we just do svd.
        # U, S, V = (size(E_Ω, 2) <= 3 ? svd!(E_Ω) : tsvd(E_Ω, 1; tolconv=sqrt(eps(eltype(E_Ω)))))
        U, S, V = (size(E_Ω, 2) <= 3 ? svd!(E_Ω) : tsvd(E_Ω, 1; tolconv=10*(eps(T))))
        # Notice we fix the sign of U[1,1] to be positive to make the svd unique and avoid oszillations.
        D_cpy[:, k]  .=  sign(U[1,1])       .* @view(U[:, 1])
        X_cpy[k, ωₖ] .= (sign(U[1,1])*S[1]) .* @view(V[:, 1])
    end

    end  # @timeit
end

function ksvd_update_X!(X, X_cpy, index_batch, timer=TimerOutput())
    @timeit_debug timer "copy X results" begin
        # # <BEGIN OPTIMIZED BLOCK>
        # # Original:
        # X[index_batch, :] .= X_cpy[index_batch, :]
        # # Optimized:
        # we can exploit that the new nonzero indices don't change!
        # Note: This doesn't seem to help in the sparse copy above.
        row_indices = SparseArrays.rowvals(X) .∈ [index_batch]
        nzvalview(X)[row_indices] .= nzvalview(X_cpy)[row_indices]
        # # <END OPTIMIZED BLOCK>
    end
end

function compute_E_Ω!(::ThreadedKSVDMethodPrecomp{true}, E_Ω_buf, E, Y, D, X, xₖ, ωₖ, k, timer=TimerOutput())
    @timeit_debug timer "compute E_Ω" begin

    E_Ω = @view E_Ω_buf[:, 1:length(ωₖ)]

    @timeit_debug timer "copy" begin
        E_Ω .= E[:, ωₖ]
    end
    @timeit_debug timer "mul" begin
        # E_Ω .+= @view(D[:, k:k]) * X[k:k, ωₖ]
        fastdensesparsemul_outer!(E_Ω, @view(D[:, k]), xₖ[ωₖ], true, true)
    end

    end  # @timeit
end

function compute_E_Ω!(::ThreadedKSVDMethodPrecomp{false}, E_Ω_buf, E, Y, D, X, xₖ, ωₖ, k, timer=TimerOutput())
    @timeit_debug timer "compute E_Ω" begin

    E_Ω = @view E_Ω_buf[:, 1:length(ωₖ)]

    ##<BEGIN OPTIMIZED BLOCK>
    ## Original:
    # E_Ω .= Y[:, ωₖ] - D * X[:, ωₖ] + D[:, k] * X[k, ωₖ]'
    ## Optimized:
    @timeit_debug timer "copy data" begin
        E_Ω .= @view Y[:, ωₖ]
        # E_Ω .= Y[:, ωₖ]
    end
    @timeit_debug timer "compute matrix vector product" begin
        # # Note: Make sure not to use `@view` on `X`, see https://github.com/JuliaSparse/SparseArrays.jl/issues/475
        # fastdensesparsemul!(E_Ω, D, X[:, ωₖ], -1, 1)
        E_Ω .-= D*X[:, ωₖ]
    end
    @timeit_debug timer "compute outer product" begin
        # E_Ω .+= D[:, k] * xₖ[ωₖ]'
        fastdensesparsemul_outer!(E_Ω, @view(D[:, k]), xₖ[ωₖ], true, true)
    end
    ## <END OPTIMIZED BLOCK>

    end  # @timeit
end

function maybe_prepare_error_buffer!(method::ThreadedKSVDMethodPrecomp{false}, Y, D, X; timer=TimerOutput())
    method.E_buf
end
function maybe_prepare_error_buffer!(method::ThreadedKSVDMethodPrecomp{true}, Y, D, X; timer=TimerOutput())
    E = method.E_buf
    # E = Y - D*X
    @timeit_debug timer "Copy error buffer" begin
        E .= Y
    end
    @timeit_debug timer "Compute error buffer" begin
        E .-= D*X
        # fastdensesparsemul_threaded!(E, D, X, -1, 1)
    end
end


function maybe_update_errors!(::ThreadedKSVDMethodPrecomp{false}, E, D_cpy, X_cpy, D, X, index_batch; timer=TimerOutput()) end
function maybe_update_errors!(::ThreadedKSVDMethodPrecomp{true}, E, D_cpy, X_cpy, D, X, index_batch; timer=TimerOutput())
    @timeit_debug timer "Update errors" begin
        # We undo the operations in the lines above to leave the error buffer "unmodified".
        # # <BEGIN OPTIMIZED BLOCK>.
        # # Original:
        # E .+= @view(D[:, index_batch]) * X[index_batch, :] - @view(D_cpy[:, index_batch]) * X_cpy[index_batch, :]
        # # Optimized:
        fastdensesparsemul_threaded!(E, @view(D[:, index_batch]), X[index_batch, :], 1, 1)
        fastdensesparsemul_threaded!(E, @view(D_cpy[:, index_batch]), X_cpy[index_batch, :], -1, 1)
        ##<END OPTIMIZED BLOCK>
    end
end
