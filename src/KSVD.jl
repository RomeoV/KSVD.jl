module KSVD

# This is an implementation of the K-SVD algorithm.
# The original paper:
# K-SVD: An Algorithm for Designing Overcomplete Dictionaries
# for Sparse Representation
# http://www.cs.technion.ac.il/~freddy/papers/120.pdf

# Variable names are based on the original paper.
# If you try to read the code, I recommend you to see Figure 2 first.
#

export ksvd, matching_pursuit

using ProgressMeter
using Base.Threads, Random, SparseArrays, LinearAlgebra
using TSVD
using Tullio
using TimerOutputs
# using MKLSparse


include("util.jl")
include("matching_pursuit.jl")
include("ksvd.jl")

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
    max_n_zeros = ceil(Int, sparsity_allowance * length(X))

    # D is a dictionary matrix that contains atoms for columns.
    @timeit to "Init dict" D = init_dictionary(T, n, K)  # size(D) == (n, K)

    p = Progress(max_iter)
    D_last = (trace_convergence ? similar(D) : nothing)

    Eₖ_buffers = [similar(Y) for _ in 1:1]
    E_Ω_buffers = [similar(Y) for _ in 1:Threads.nthreads()]

    for i in 1:max_iter
        verbose && @info "Starting sparse coding"
        @timeit to "Sparse coding" X_sparse = sparse_coding(sparse_coding_method, Y, D)
        trace_convergence && D_last .= copy(D)
        verbose && @info "Starting svd"
        @timeit to "KSVD" D, X = ksvd(ksvd_method, Y, D, X_sparse, err_buffers=Eₖ_buffers, err_gamma_buffers=E_Ω_buffers)
        trace_convergence && @info norm(D - D_last)

        # return if the number of zero entries are <= max_n_zeros
        if sum(iszero, X) > max_n_zeros
            show(to)
            return D, X
        end
        show_progress && next!(p)
    end
    show(to)
    return D, X
end

end # module
