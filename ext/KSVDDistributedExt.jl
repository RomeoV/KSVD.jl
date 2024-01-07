module KSVDDistributedExt
using Distributed
using KSVD
using LinearAlgebra: normalize!
import KSVD: DistributedKSVD, make_index_batches, ksvd
function KSVD.make_index_batches(method::DistributedKSVD, indices)
    basis_indices = (method.shuffle_indices ? shuffle(indices) : indices)
    return Iterators.partition(basis_indices, length(basis_indices)÷nworkers())
end

function KSVD.ksvd(method::DistributedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}) where T
    # the revised plan:
    # Let's assume we already do a good job for moderately sized datasets.
    # Now, we may have a very large number of datapoints.
    # We can processes a subset of datapoints on each worker, but have to merge the resulting dictionary vectors together again later.
    # An important question is therefore if dictionary vector changes tend to be small (in which case averaging might be fair game),
    # or "jumpy". However, since it's supposed to eventually converge, I assume they will eventually be small, and averaging is fine...
    #
    # Let's split up the data, process each partition on the workers, and then average the results.
    reducer_fn(tpls...) = begin
        Ds, Xs = [getindex.(tpls, i) for i ∈ 1:2]
        D = sum(Ds)/length(Ds); normalize!.(eachcol(D))
        X = reduce(hcat, Xs)  # use `reduce` version to make type stable.
        (D, X)
    end
    index_batches = make_index_batches(method, axes(Y, 2)) |> collect
    D, X = @sync @distributed reducer_fn for indices in index_batches
        let D = D, X = X
            ksvd(method.submethod_per_worker[myid()], copy(Y[:, indices]), copy(D), copy(X[:, indices]))
        end
    end
    return D, X
end

# function ksvd2(method::DistributedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}) where T
    # # the plan:
    # # we already have a sort of mini-batch structure, where we split the entire data into batches
    # # that have the size of the threads, and we "sync" after each batch to improve convergence speed
    # # (although it would be fine either way).
    # # Now, probably we do the same, but process multiple batches at the same time via `Distributed.jl`.
    # #
    # # More specifically, recall that we parallelize over the basis vectors, i.e. the second dim of `D` or the first dim of `X`.
    # # Let `index_batches` be something like `chunk(axes(D, 2), nthreads())`.
    # # Then we want
    # # 1. Take `pworkers` many batches.
    # # 2. Process all batches.
    # # 3. Sync both D and X.
    # # 4. Repeat until done.
    # #
    # # The syncing is most likely the slow part, but we can start by doing it via `SharedArrays` or via `DistributedArrays`, not clear yet.
    # # There is some tradeoff on how often to sync, which is dependent on the convergence speed and the details of the hardware, etc.
    # # We will start by just syncing all the time (after each batch).
    # #
    # # Notice that to process a batch with indices n:m, we need (in the precomputed case)
    # # IN : D[:, n:m], X[n:m, :]
    # # OUT: D[:, n:m], X[n:m, :]
    # # but in the sync step, E needs to be synced aswell,
    # # or in the not precomputed case
    # # IN : D[:,  : ], X[n:m, :]
    # # OUT: D[:, n:m], X[n:m, :].
    # # Since D continually changes, it's perhaps not a great idea to have to sync it all time time...
    # #
    # # Let's stick for now with the second case, i.e. not precomputed.
    # # I think after each batch, we need to sync all the modified D, and X rows/columns.
#
    # E_Ω_buffers = method.E_Ω_bufs[myid()] |> fetch
#
#
    # for k_batches = Iterators.partition(index_batches, nprocs())
        # Distributed.pmap(process_batch, k_batches)
#
#
    # end
    # return D, X
# end
end
