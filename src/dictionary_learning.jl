"""
    dictionary_learning(
         Y::AbstractMatrix, n_atoms::Int;
         sparsity_allowance::Float64 = $default_sparsity_allowance,
         max_iter::Int = $default_max_iter,
         max_iter_mp::Int = $default_max_iter_mp)

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
                             sparsity_allowance = default_sparsity_allowance,
                             ksvd_method = OptimizedKSVD(),
                             sparse_coding_method = MatchingPursuit(),
                             max_iter::Int = default_max_iter,
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

    for i in 1:max_iter
        verbose && @info "Starting sparse coding"
        @timeit to "Sparse coding" X_sparse = sparse_coding(sparse_coding_method, Y, D)
        verbose && @info "Starting svd"
        @timeit to "KSVD" D, X = ksvd(ksvd_method, Y, D, X_sparse)

        # return if the number of zero entries are <= max_n_zeros
        if sum(iszero, X) > max_n_zeros
            show(to)
            return D, X
        end
        next!(p)
    end
    show(to)
    return D, X
end
