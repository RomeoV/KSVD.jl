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
            ksvd(method.submethod_per_worker[myid()],
                 copy(Y[:, indices]), copy(D), copy(X[:, indices]))
        end
    end
    return D, X
end
end
