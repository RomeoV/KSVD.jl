import Random: shuffle
import SparseArrays: nzvalview, nonzeroinds, nonzeros
import OhMyThreads
import OhMyThreads: tforeach, @allow_boxed_captures, @localize
# import OhMyThreads: SerialScheduler
import TimerOutputs: TimerOutput, @timeit
import OhMyThreads.ChunkSplitters: chunks
import Base.Threads: nthreads, threadpool
import TSVD: tsvd

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

# TODO: This should probably be called `ksvd_update!` and note that it modifies D and X!
"""
function ksvd_update(method::ThreadedKSVDMethod, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T};
    force_reinitialize_buffers::Bool=false, timer=TimerOutput(), merge_all_timers=false) where {T}
    @timeit_debug timer "KSVD update" begin

        if force_reinitialize_buffers || !is_initialized(method)
            maybe_init_buffers!(method, size(D, 1), size(D, 2), size(Y, 2), timer)
        end

        @timeit_debug timer "copy X" begin
            X_cpy = copy(X)
        end
        D_cpy = method.D_cpy_buf
        @assert all(≈(1.0), norm.(eachcol(D))) "$(extrema(norm.(eachcol(D))))"

        # prepare variables for update ksvd_update_X!
        R = X.rowval
        Rsorted = sort(R)
        Rsortperm = sortperm(R)

        # this is a no-op if the template `precompute_error` is false.
        E = maybe_prepare_error_buffer!(method, Y, D, X; timer)

        timer_ch = Channel{TimerOutput}(length(method.E_Ω_bufs))
        foreach(to -> put!(timer_ch, to), [TimerOutput() for _ in 1:length(method.E_Ω_bufs)])

        E_Ω_buf_ch = Channel{Matrix{T}}(length(method.E_Ω_bufs))
        foreach(buf -> put!(E_Ω_buf_ch, buf), method.E_Ω_bufs)

        # We iterate over each basis vector, either in one big batch or many small batches, depending on the method.
        # I.e. for ParallelKSVD, the first index_batch is just the entire dataset, and for BatchedParallelKSVD it's split
        # into more sub-batches.
        index_batches = make_index_batches(method, axes(X, 1))
        scheduler = get_scheduler_t(method)()  # we do our own chunking

        @timeit_debug timer "Inner loop" begin
            @inbounds for index_batch in index_batches  # <- This is just one big batch for `ParallelKSVD`
                # avoid boxing: https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/boxing/boxing/#Non-race-conditon-boxed-variables
                @localize E_Ω_buf_ch timer_ch E D_cpy X_cpy tforeach(index_batch; scheduler) do k
                    # we use channels to manage the batch-local variables
                    E_Ω_buf = take!(E_Ω_buf_ch)
                    timer_ = take!(timer_ch)
                    try
                        ksvd_update_k!(method, E_Ω_buf, D_cpy, X_cpy, E, Y, D, X, k, timer_)
                    catch e
                        handle_ksvd_update_error(e, D_cpy, k)
                    end
                    put!(E_Ω_buf_ch, E_Ω_buf)
                    put!(timer_ch, timer_)
                end

                maybe_update_errors!(method, E, D_cpy, X_cpy, D, X, index_batch; timer)

                @timeit_debug timer "copy D results" begin
                    D[:, index_batch] .= @view D_cpy[:, index_batch]
                end
                ksvd_update_X!(X, X_cpy, index_batch, R, Rsorted, Rsortperm, timer)
            end
        end  # @timeit

        close(timer_ch)
        close(E_Ω_buf_ch)
        if KSVD.timeit_debug_enabled()
            TimerOutputs.merge!(timer, (merge_all_timers ? collect(timer_ch) : [first(timer_ch)])...;
                tree_point=["KSVD update", "Inner loop"])
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
                # If this dictionary vector had no matches at all, reinitialize it with a new random vector.
                # We write to `D_cpy`, which will later get copied into `D`.
                randn!(@view(D_cpy[:, k]))
                normalize!(@view(D_cpy[:, k]), 2)
                # D_cpy[:, k] .= D[:, k];
                return
            end
            # ωₖ = findall(!iszero, xₖ)
            ωₖ = nonzeroinds(xₖ)
        end

        E_Ω = compute_E_Ω!(method, E_Ω_buf, E, Y, D, X, xₖ, ωₖ, k, timer)

        @timeit_debug timer "compute and copy svd" begin
            # truncated svd has some problems for column matrices. so then we just do svd.
            # Use the solver from the method
            if size(E_Ω, 2) <= 3
                U, S, V = svd!(E_Ω)
            else
                U, S, V = compute_truncated_svd(method.svd_solver, E_Ω, 1)
            end
            # Notice we fix the sign of U[1,1] to be positive to make the svd unique and avoid oszillations.
            # We also re-normalize here. Even though the result should be normalized, we can have some numerical inaccuracies.
            # Finally, sometimes =U[1, 1]= is zero! Then sign(U[1,1]) would be zero... not good.
            # Instead we use =signbit= check to make sure we never zero out `U`.
            sgn = (signbit(U[1, 1]) ? -1 : 1)
            D_cpy[:, k] .= sgn .* @view(U[:, 1])
            X_cpy[k, ωₖ] .= (sgn * S[1]) .* @view(V[:, 1])
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

function ksvd_update_X!(X, X_cpy, index_batch, R, Rsorted, Rsortperm, timer=TimerOutput())
    @timeit_debug timer "copy X results" begin
        # # <BEGIN OPTIMIZED BLOCK>
        # # Original:
        # row_indices = SparseArrays.rowvals(X) .∈ [index_batch]
        # # Optimized:
        # We use a crucial insight here, which is that the nonzero indices don't change throughout the ksvd iterations.
        # It turns out that the rowvals ∈ index_batch actually takes a huge amount of time.
        # But we can operate on sorted indices, massively cutting down the time here.
        permindices = searchsortedfirst(Rsorted, first(index_batch)):searchsortedlast(Rsorted, last(index_batch))
        row_indices = Rsortperm[permindices]
        nzvalview(X)[row_indices] .= nzvalview(X_cpy)[row_indices]
        # # <END OPTIMIZED BLOCK>
    end
end

function compute_E_Ω!(::ThreadedKSVDMethodPrecomp{true}, E_Ω_buf, E, Y, D, X, xₖ, ωₖ, k, timer=TimerOutput())
    @timeit_debug timer "compute E_Ω" begin

        if size(E_Ω_buf, 2) <= length(ωₖ)
            @warn """
            The preallocated error buffer is too small: $(size(E_Ω_buf, 2)) vs $(length(ωₖ)). Not all errors will be computed. This is probably because
            `maybe_init_buffer!` has been called with a ratio_nonzero that's too small. Try setting it to `1`.
            """
            ωₖ = ωₖ[1:size(E_Ω_buf, 2)]
        end
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

        if size(E_Ω_buf, 2) <= length(ωₖ)
            @warn """
            The preallocated error buffer is too small. Not all errors will be computed. This is probably because
            `maybe_init_buffer!` has been called with a ratio_nonzero that's too small. Try setting it to `1`.
            """
            ωₖ = ωₖ[1:size(E_Ω_buf, 2)]
        end
        E_Ω = @view E_Ω_buf[:, 1:length(ωₖ)]

        ##<BEGIN OPTIMIZED BLOCK>
        ## Original:
        # E_Ω .= Y[:, ωₖ] - D * X[:, ωₖ] + D[:, k] * X[k, ωₖ]'
        ## Optimized:
        @timeit_debug timer "copy data" begin
            E_Ω .= @view Y[:, ωₖ]
            # E_Ω .= Y[:, ωₖ]
        end
        @timeit_debug timer "compute dense sparse matrix product" begin
            # # Note: Make sure not to use `@view` on `X`, see https://github.com/JuliaSparse/SparseArrays.jl/issues/475
            # fastdensesparsemul!(E_Ω, D, X[:, ωₖ], -1, 1)
            # this is actually slightly faster than our version
            E_Ω .-= D * X[:, ωₖ]
            # Benchmark results: multiply dominates, the indexing `X[:, ωₖ]` is almost free
        end
        @timeit_debug timer "compute dense sparse outer product" begin
            E_Ω .+= @view(D[:, k]) * nonzeros(xₖ)'
            # E_Ω .+= D[:, k] * xₖ[ωₖ]'
            # fastdensesparsemul_outer!(E_Ω, @view(D[:, k]), xₖ[ωₖ], true, true)
            # Benchmark results: multiply dominates, the indexing `xₖ[ωₖ]` is almost free
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
        # E .-= D*X
        fastdensesparsemul_threaded!(E, D, X, -1, 1)
    end
end


# no-op when we don't precompute the errors.
function maybe_update_errors!(::ThreadedKSVDMethodPrecomp{false}, E, D_cpy, X_cpy, D, X, index_batch; timer=TimerOutput()) end
function maybe_update_errors!(::ThreadedKSVDMethodPrecomp{true}, E, D_cpy, X_cpy, D, X, index_batch; timer=TimerOutput())
    @timeit_debug timer "Update errors" begin
        # We undo the operations in the lines above to leave the error buffer "unmodified".
        # # <BEGIN OPTIMIZED BLOCK>.
        # # Original:
        # E .+= @view(D[:, index_batch]) * X[index_batch, :] - @view(D_cpy[:, index_batch]) * X_cpy[index_batch, :]
        # # Optimized:
        @timeit_debug timer "extract Xs" begin
            x_batch = X[index_batch, :]
            x_cpy_batch = X_cpy[index_batch, :]
        end
        @timeit_debug timer "dense-sparse mul" begin
            fastdensesparsemul_threaded!(E, @view(D[:, index_batch]), x_batch, 1, 1)
            fastdensesparsemul_threaded!(E, @view(D_cpy[:, index_batch]), x_cpy_batch, -1, 1)
        end
        ##<END OPTIMIZED BLOCK>
    end
end


function handle_ksvd_update_error(::LinearAlgebra.LAPACKException, D_cpy, k)
    @warn "Handling LAPACKException by adding a bit of noise to one dictionary."
    D_cpy[:, k] .+= sqrt(eps(eltype(D_cpy))) * randn(size(D_cpy, 1))
    normalize!(@view(D_cpy[:, k]), 2)
end
handle_ksvd_update_error(e::Exception, _D_cpy, _k) = throw(e)
