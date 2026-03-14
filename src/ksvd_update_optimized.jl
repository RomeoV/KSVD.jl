import SparseArrays: nonzeroinds

@inbounds function ksvd_update(method::OptimizedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T};
    timer=TimerOutput(), nonneg::Bool=false) where {T}
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
                    fastdensesparsemul_threaded!(E_Ω, D, X[:, ωₖ], -1, 1)
                end
                @timeit_debug timer "E += D_k X_k" begin
                    fastdensesparsemul_outer_threaded!(E_Ω, @view(D[:, k]), xₖ[ωₖ], true, true)
                end
            end

            E_Ω_copy = nonneg ? copy(E_Ω) : E_Ω

            @timeit_debug timer "svd step" begin
                U, S, V = (size(E_Ω, 2) < 3 ? svd!(E_Ω) : compute_truncated_svd(method.svd_solver, E_Ω, 1))
            end
            @timeit_debug timer "update D and X" begin
                sgn = sign(U[1, 1])
                d_new = sgn .* @view(U[:, 1])
                x_new = (sgn * S[1]) .* @view(V[:, 1])

                if nonneg
                    x_new = max.(x_new, zero(T))
                    if !all(iszero, x_new)
                        d_buf = similar(d_new)
                        for _ in 1:3
                            mul!(d_buf, E_Ω_copy, x_new)
                            nrm = norm(d_buf)
                            nrm > 0 || break
                            d_buf ./= nrm
                            mul!(x_new, E_Ω_copy', d_buf)
                            x_new .= max.(x_new, zero(T))
                        end
                        d_new = d_buf
                    end
                end

                D[:, k] .= d_new
                X[k, ωₖ] .= x_new
            end
        end

    end  # @timeit_debug
    return D, X
end
