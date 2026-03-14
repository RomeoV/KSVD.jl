"""
    ksvd_update(method::LegacyKSVD, Y, D, X)

This is the original serial implementation written by Ishita Takeshi, mostly used as a reference for testing, and
for didactic reasons since the algorithm is the most clear here.

However, it is recommended to use `BatchedParallelKSVD`.
"""
function ksvd_update(::LegacyKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T};
    timer=TimerOutput(), nonneg::Bool=false) where {T}
    @timeit_debug timer "KSVD update" begin

    N = size(Y, 2)
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
            Eₖ = error_matrix(Y, D, X, k)
        end
        @timeit_debug timer "setup Ωₖ" begin
            Ωₖ = sparse(wₖ, 1:length(wₖ), ones(length(wₖ)), N, length(wₖ))
        end
        @timeit_debug timer "compute and copy svd" begin
            E_Ωₖ = Eₖ * Ωₖ
            U, S, V = svd(E_Ωₖ)
            sgn = sign(U[1, 1])
            d_new = sgn .* @view(U[:, 1])
            x_new = (sgn * S[1]) .* @view(V[:, 1])

            if nonneg
                x_new = max.(x_new, zero(T))
                if !all(iszero, x_new)
                    d_buf = similar(d_new)
                    for _ in 1:3
                        mul!(d_buf, E_Ωₖ, x_new)
                        nrm = norm(d_buf)
                        nrm > 0 || break
                        d_buf ./= nrm
                        mul!(x_new, E_Ωₖ', d_buf)
                        x_new .= max.(x_new, zero(T))
                    end
                    d_new = d_buf
                end
            end

            D[:, k] = d_new
            X[k, wₖ] = x_new
        end
    end

    end  # @timeit
    return D, X
end
