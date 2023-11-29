module KSVD

# This is an implementation of the K-SVD algorithm.
# The original paper:
# K-SVD: An Algorithm for Designing Overcomplete Dictionaries
# for Sparse Representation
# http://www.cs.technion.ac.il/~freddy/papers/120.pdf

# Variable names are based on the original paper.
# If you try to read the code, I recommend you to see Figure 2 first.
#

export dictionary_learning, matching_pursuit, ksvd_update
export LegacyKSVD, OptimizedKSVD, ParallelKSVD, BatchedParallelKSVD
export LegacyMatchingPursuit, ParallelMatchingPursuit

using ProgressMeter
using Base.Threads, Random, SparseArrays, LinearAlgebra
using TSVD
import Transducers: tcollect
import LinearAlgebra: normalize!
using TimerOutputs

# importing StatsBase is also fine.
mean(vec::AbstractVector) = sum(vec)/length(vec)

# using ThreadedDenseSparseMul


include("util.jl")
include("matching_pursuit.jl")
include("ksvd_update.jl")

"""
    dictionary_learning(
         Y::AbstractMatrix, n_atoms::Int;
         sparsity_allowance::Float64 = 0.1,
         max_iter::Int = 10)

Run K-SVD that designs an efficient dictionary D for sparse representations,
and returns X such that DX = Y or DX ≈ Y.

```
# Arguments
* `sparsity_allowance`: Stop iteration if the number of zeros in X / the number
    of elements in X > sparsity_allowance.
* `max_iter`: Limit of iterations.
* `max_iter_mp`: Limit of iterations in Matching Pursuit that `ksvd` calls at
    every iteration.
```
"""
function dictionary_learning(Y::AbstractMatrix{T}, n_atoms::Int;
                             sparsity_allowance = 0.1,
                             ksvd_method = OptimizedKSVD(),
                             sparse_coding_method = MatchingPursuit(),
                             max_iter::Int = 10,
                             trace_convergence = false,
                             show_progress=true,
                             verbose=false
                             ) where T
    to = TimerOutput()
    K = n_atoms
    n, N = size(Y)

    if !(0 <= sparsity_allowance <= 1)
        throw(ArgumentError("`sparsity_allowance` must be in range [0,1]"))
    end

    X = spzeros(T, K, N)  # just for making X global in this function
    min_n_zeros = ceil(Int, sparsity_allowance * length(X))

    # D is a dictionary matrix that contains atoms for columns.
    @timeit to "Init dict" D = init_dictionary(T, n, K)  # size(D) == (n, K)
    @assert all(≈(1.0), norm.(eachcol(D)))

    p = Progress(max_iter)
    maybe_init_buffers!(ksvd_method, n, K, N; pct_nz=min(100*sparse_coding_method.max_nnz/K, 1))

    for i in 1:max_iter
        verbose && @info "Starting sparse coding"
        @timeit to "Sparse coding" X = sparse_coding(sparse_coding_method, Y, D)
        verbose && @info "Starting svd"
        @timeit to "KSVD" D, X = ksvd_update(ksvd_method, Y, D, X)
        trace_convergence && @spawn (@info "loss=$(norm(Y - D*X)), nnz_col=$(mean(sum.(!iszero, eachcol(X))))")

        # return if the number of zero entries are <= max_n_zeros
        if sum(iszero, X) > min_n_zeros
            show(to)
            return D, X
        end
        show_progress && next!(p)
    end
    show(to)
    return D, X
end

end # module
