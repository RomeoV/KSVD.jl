import SparseArrays: nonzeroinds

@inbounds function ksvd_update(method::OptimizedKSVD, Y::AbstractMatrix{T}, D::AbstractMatrix{T}, X::AbstractMatrix{T}; timer=TimerOutput()) where {T}
    @timeit_debug timer "KSVD update" begin

    Eₖ = Y - D * X
    E_Ω_buffer = similar(Eₖ) .* false  # set to 0. Note that (.*0) doesn't work with NaN.
    basis_indices = (method.shuffle_indices ? shuffle(axes(X, 1)) : axes(X, 1))
    for k in basis_indices
        xₖ = X[k, :]
        # ignore if the k-th row is zeros
        all(iszero, xₖ) && continue

        # wₖ is the column indices where the k-th row of xₖ is non-zero,
        # which is equivalent to [i for i in N if xₖ[i] != 0]
        ωₖ = nonzeroinds(xₖ)

        # Eₖ * Ωₖ implies a selection of error columns that
        # correspond to examples that use the atom D[:, k]
        Eₖ .+= D[:, k] * X[k, :]'

        # Ωₖ = sparse(ωₖ, 1:length(ωₖ), ones(length(ωₖ)), N, length(ωₖ))

        # Note that S is a vector that contains diagonal elements of
        # a matrix Δ such that Eₖ * Ωₖ == U * Δ * V.
        # Non-zero entries of X are set to
        # the first column of V multiplied by Δ(1, 1)
        # U, S, V = tsvd(Eₖ * Ωₖ, initvec=randn!(similar(Eₖ, size(Eₖ,1))))
        # E_Ω = Eₖ * Ωₖ
        E_Ω = @view E_Ω_buffer[:, 1:length(ωₖ)]
        E_Ω .= Eₖ[:, ωₖ]

        U, S, V = (size(E_Ω, 2) < 3 ? svd!(E_Ω) : tsvd(E_Ω, 1; tolconv=10*eps(eltype(E_Ω))) )
        D[:, k]  .=  sign(U[1,1])       .* @view(U[:, 1])
        X[k, ωₖ] .= (sign(U[1,1])*S[1]) .* @view(V[:, 1])

        Eₖ .-= D[:, k:k] * X[k:k, :]
    end

    end  # @timeit_debug
    return D, X
end
