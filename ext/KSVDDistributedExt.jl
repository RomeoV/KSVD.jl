module KSVDDistributedExt
using Distributed
using KSVD
using LinearAlgebra: normalize!
using SparseArrays
using Distances: pairwise, Euclidean
using DistributedArrays: dzeros
using Clustering: kmedoids
using StatsBase: mean
import KSVD: DistributedKSVD
function KSVD.make_index_batches(method::DistributedKSVD, indices)
    basis_indices = (method.shuffle_indices ? shuffle(indices) : indices)
    return Iterators.partition(basis_indices, length(basis_indices)÷nworkers())
end
function KSVD.maybe_init_buffers!(method::DistributedKSVD, emb_dim, n_dict_vecs, n_samples; pct_nz=1.)
    for (p, submethod) in method.submethod_per_worker
        maybe_init_buffers_distributed!(submethod, p, emb_dim, n_dict_vecs, n_samples; pct_nz)
    end
end
maybe_init_buffers_distributed!(method::KSVD.KSVDMethod, p::Int, emb_dim, n_dict_vecs, n_samples; pct_nz=1.) = nothing
function maybe_init_buffers_distributed!(method::Union{ParallelKSVD{false, T}, BatchedParallelKSVD{false, T}}, p::Int, emb_dim, n_dict_vecs, n_samples; pct_nz=1.) where {T<:Real}
    method.E_Ω_bufs=[dzeros(T, (emb_dim, KSVD.compute_reasonable_buffer_size(n_samples, pct_nz)), [p])
                     for _ in 1:remotecall_fetch(Threads.nthreads, p)]
    method.D_cpy_buf=dzeros(T, (emb_dim, n_dict_vecs), [p])
end
function maybe_init_buffers_distributed!(
        method::Union{ParallelKSVD{true, T}, BatchedParallelKSVD{true, T}}, p::Int, emb_dim, n_dict_vecs, n_samples; pct_nz=1.
    ) where {T<:Real}
    method.E_buf=dzeros(T, (emb_dim, n_samples), [p])
    method.E_Ω_bufs=[dzeros(T, (emb_dim, KSVD.compute_reasonable_buffer_size(n_samples, pct_nz)), [p])
                     for _ in 1:remotecall_fetch(Threads.nthreads, p)]
    method.D_cpy_buf=dzeros(T, (emb_dim, n_dict_vecs), [p])
end


function KSVD.ksvd_update(method::DistributedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T};
                          reinitialize_buffers=false) where T
    # the revised plan:
    # Let's assume we already do a good job for moderately sized datasets.
    # Now, we may have a very large number of datapoints.
    # We can processes a subset of datapoints on each worker, but have to merge the resulting dictionary vectors together again later.
    # An important question is therefore if dictionary vector changes tend to be small (in which case averaging might be fair game),
    # or "jumpy". However, since it's supposed to eventually converge, I assume they will eventually be small, and averaging is fine...
    #
    # Let's split up the data, process each partition on the workers, and then average the results.

    index_batches = KSVD.make_index_batches(method, axes(Y, 2)) |> collect
    X_batches = [X[:, indices] for indices in index_batches]
    Y_batches = [Y[:, indices] for indices in index_batches]
    X_Y_batches = collect(zip(X_batches, Y_batches))

    if reinitialize_buffers
        max_pct_nz = maximum(X_batches) do X_
            maximum(1 .- mean.(iszero, eachrow(X_)))
        end
        KSVD.maybe_init_buffers!(method, size(D, 1), size(D, 2), length(index_batches[1]);
                                 pct_nz=min(1., 2*max_pct_nz))
    end

    # Here we do all the actual work. Note that we pass D and method aswell for scoping reasons.
    Ds_Xs = pmap(X_Y_batches, Iterators.repeated(D), Iterators.repeated(method)) do (X_, Y_), D, method
        # we have some problems with memory leaks, so we reinit the buffers here...
        let method_ = method.submethod_per_worker[myid()]
            ksvd_update(method_, Y_, copy(D), X_)
        end
    end
    Ds = getindex.(Ds_Xs, 1)
    Xs = getindex.(Ds_Xs, 2)

    # "summarize" Xs and Ds
    X = reduce(hcat, Xs)
    D = if method.reduction_method == :mean
        mean(Ds)
    elseif method.reduction_method == :clustering
        Ds_flat = reduce(hcat, Ds)
        cluster_res = kmedoids(pairwise(Euclidean(), Ds_flat), size(X, 1))
        Ds_flat[:, cluster_res.medoids]
    end
    return D, X
end
end
