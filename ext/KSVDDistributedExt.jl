module KSVDDistributedExt
using Distributed
using KSVD
using LinearAlgebra: normalize!
using SparseArrays
using Distances: pairwise, Euclidean
using Clustering: kmedoids
using StatsBase: mean
import KSVD: DistributedKSVD
function KSVD.make_index_batches(method::DistributedKSVD, indices)
    basis_indices = (method.shuffle_indices ? shuffle(indices) : indices)
    return Iterators.partition(basis_indices, length(basis_indices)÷nworkers())
end
function KSVD.maybe_init_buffers!(method::DistributedKSVD, emb_dim, n_dict_vecs, n_samples; pct_nz=1.)
    # for submethod in values(method.submethod_per_worker)
    #     KSVD.maybe_init_buffers!(submethod, emb_dim, n_dict_vecs, n_samples÷nworkers())
    # end
    nothing
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

    Ds_Xs = pmap(X_Y_batches, Iterators.repeated(D)) do (X_, Y_), D
        # we have some problems with memory leaks, so we reinit the buffers here...
        let method_ = typeof(method.submethod_per_worker[myid()])()
            KSVD.maybe_init_buffers!(method_, size(D, 1), size(D, 2), size(Y_, 2);
                                     pct_nz=min(1., 2*maximum(1 .- mean.(iszero, eachrow(X_)))))
            ksvd_update(method_, Y_, copy(D), X_)
        end
    end
    Ds = getindex.(Ds_Xs, 1)
    Xs = getindex.(Ds_Xs, 2)

    # "summarize" Ds
    X = reduce(hcat, Xs)
    D = if method.reduction_method == :mean
        sum(Ds)/length(Ds)
    elseif method.reduction_method == :clustering
        Ds_flat = reduce(hcat, Ds)
        cluster_res = kmedoids(pairwise(Euclidean(), Ds_flat), size(X, 1))
        Ds_flat[:, cluster_res.medoids]
    end
    return D, X
end
end
