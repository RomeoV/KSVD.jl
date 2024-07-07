"""
    ksvd_update(method::LegacyKSVD, Y, D, X)

This is the original serial implementation written by Ishita Takeshi, mostly used as a reference for testing, and
for didactic reasons since the algorithm is the most clear here.

However, it is recommended to use `BatchedParallelKSVD`.
"""
function ksvd_update(::LegacyKSVD, Y::AbstractMatrix, D::AbstractMatrix, X::AbstractMatrix; timer=TimerOutput())
    @timeit_debug timer "KSVD update" begin

    N = size(Y, 2)
    # for k in 1:size(X, 1)
    for k in axes(X, 1)
        @timeit_debug timer "find nonzeros" begin
            xₖ = X[k, :]
            # ignore if the k-th row is zeros
            all(iszero, xₖ) && continue

            # wₖ is the column indices where the k-th row of xₖ is non-zero,
            # which is equivalent to [i for i in N if xₖ[i] != 0]
            wₖ = findall(!iszero, xₖ)
        end

        @timeit_debug timer "computer error matrix" begin
            # Eₖ * Ωₖ implies a selection of error columns that
            # correspond to examples that use the atom D[:, k]
            Eₖ = error_matrix(Y, D, X, k)
        end
        @timeit_debug timer "setup Ωₖ" begin
            Ωₖ = sparse(wₖ, 1:length(wₖ), ones(length(wₖ)), N, length(wₖ))
        end
        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        @timeit_debug timer "compute and copy svd" begin
            U, S, V = svd(Eₖ * Ωₖ)
            D[:, k]  =  sign(U[1,1])       .* @view(U[:, 1])
            X[k, wₖ] = (sign(U[1,1])*S[1]) .* @view(V[:, 1])
        end
    end

    end  # @timeit
    return D, X
end
