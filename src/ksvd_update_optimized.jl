import SparseArrays: nonzeroinds

@inbounds function ksvd_update(method::OptimizedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; timer=TimerOutput()) where {T}
    @timeit_debug timer "KSVD update" begin
        # Eₖ = Y - D * X
        E_buffer = similar(Y) .* false  # set to 0. Note that (.*0) doesn't work with NaN.
        basis_indices = (method.shuffle_indices ? shuffle(axes(X, 1)) : axes(X, 1))
        for k in basis_indices
            @timeit_debug timer "find nonzeros" begin
                xₖ = X[k, :]
                # ignore if the k-th row is zeros
                all(iszero, xₖ) && continue
            end

            # wₖ is the column indices where the k-th row of xₖ is non-zero,
            # which is equivalent to [i for i in N if xₖ[i] != 0]
            ωₖ = nonzeroinds(xₖ)
            @timeit_debug timer "allocate buffer" begin
                E_Ω = @view E_buffer[:, 1:length(ωₖ)]
            end

            @timeit_debug timer "do math" begin
                @timeit_debug timer "copy Y" begin
                    E_Ω .= @view Y[:, ωₖ]
                end
                @timeit_debug timer "E -= D X" begin
                    # E_Ω .-= D * X[:, ωₖ]
                    fastdensesparsemul_threaded!(E_Ω, D, X[:, ωₖ], -1, 1)
                end
                @timeit_debug timer "E += D_k X_k" begin
                    # E_Ω .+= @view(D[:, k]) * nonzeros(xₖ)'
                    fastdensesparsemul_outer_threaded!(E_Ω, @view(D[:, k]), xₖ[ωₖ], true, true)
                end
            end

            @timeit_debug timer "svd step" begin
                U, S, V = (size(E_Ω, 2) < 3 ? svd!(E_Ω) : compute_truncated_svd(method.svd_solver, E_Ω, 1))
                # U, S, V = (size(E_Ω, 2) <= 3 ? svd!(E_Ω) : krylov_svd(E_Ω, 1; tol=1e-10))
            end
            @timeit_debug timer "update D and X" begin
                D[:, k] .= sign(U[1, 1]) .* @view(U[:, 1])
                X[k, ωₖ] .= (sign(U[1, 1]) * S[1]) .* @view(V[:, 1])
            end
        end

    end  # @timeit_debug
    return D, X
end
