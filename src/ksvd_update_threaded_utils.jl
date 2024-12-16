maybe_init_buffers!(method::KSVDMethod, emb_dim, n_dict_vecs, n_samples, timer=TimerOutput(); ratio_nonzero=1.) = nothing
function maybe_init_buffers!(method::Union{ParallelKSVD{precompute_error, T}, BatchedParallelKSVD{precompute_error, T}},
                             emb_dim, n_dict_vecs, n_samples, timer=TimerOutput(); ratio_nonzero=1.) where {precompute_error, T<:Real}
    @timeit_debug timer "init buffers" begin
        precompute_error && (method.E_buf=Matrix{T}(undef, emb_dim, n_samples);)
        method.E_Ω_bufs=[Matrix{T}(undef, emb_dim, compute_reasonable_buffer_size(n_samples, ratio_nonzero)) for _ in 1:KSVD.get_num_threads()]
        # method.E_Ω_bufs=[Matrix{T}(undef, emb_dim, n_samples) for _ in 1:KSVD.get_num_threads()]
        method.D_cpy_buf=Matrix{T}(undef, emb_dim, n_dict_vecs)
    end
end
function is_initialized(method::Union{ParallelKSVD{precompute_error}, BatchedParallelKSVD{precompute_error}})::Bool where {precompute_error}
    (length(method.E_Ω_bufs) > 0) && (length(method.D_cpy_buf) > 0) && (!precompute_error || length(method.D_cpy_buf) > 0)
end


"Just yields indices as one big batch in the 'regular' case. May shuffle, depending on ksvd method parameter."
function make_index_batches(method::ParallelKSVD, indices)
    indices = (method.shuffle_indices ? shuffle(indices) : indices)
    return [indices]
end
"Yields indices in batches. May shuffle before batching, depending on ksvd method parameter."
function make_index_batches(method::BatchedParallelKSVD, indices)
    basis_indices = (method.shuffle_indices ? shuffle(indices) : indices)
    # return Iterators.partition(basis_indices, method.batch_size_per_thread*Threads.nthreads())
    return chunks(basis_indices, size=method.batch_size_per_thread*KSVD.get_num_threads())
end

get_scheduler_t(::ParallelKSVD{precompute_error, T, Scheduler}) where {precompute_error, T, Scheduler<:OhMyThreads.Scheduler} =
    Scheduler
get_scheduler_t(::BatchedParallelKSVD{precompute_error, T, Scheduler}) where {precompute_error, T, Scheduler<:OhMyThreads.Scheduler} =
    Scheduler
