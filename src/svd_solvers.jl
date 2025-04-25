using KrylovKit
import TSVD: tsvd
import Arpack: svds
# import ArnoldiMethod: arnoldi_svd
using ArnoldiMethod

import LinearAlgebra
using LinearAlgebra: mul!, eltype, size

abstract type AbstractTruncatedSVD end

@kwdef struct TSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tol::Float64 = 100 * (eps(real(T)))
    maxiter::Int = 50
end

@kwdef struct ArpackSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tol::Float64 = 100 * (eps(real(T)))
    maxiter::Int = 50
end

@kwdef struct KrylovSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tol::Float64 = 100 * (eps(real(T)))
    maxiter::Int = 50
end

@kwdef struct ArnoldiSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tol::Float64 = 100 * (eps(real(T)))
    maxiter::Int = 50
end

# Solver implementations
function compute_truncated_svd(solver::TSVDSolver, A::AbstractMatrix{T}, k::Int) where {T}
    tsvd(A, k; tolconv=solver.tol, maxiter=solver.maxiter)
end

function compute_truncated_svd(solver::ArpackSVDSolver, A::AbstractMatrix{T}, k::Int) where {T}
    # we copy here because we've had issues with @view and Arpack before...
    (U, S, V), _ = svds(copy(A); nsv=k, maxiter=solver.maxiter, tol=solver.tol)
    (U, S, V)
end

function compute_truncated_svd(solver::KrylovSVDSolver, A::AbstractMatrix{T}, k::Int) where {T}
    krylov_svd(A, k; tol=solver.tol)
end

function krylov_svd(A, nev=1; tol=1e-8)
    S, U, V, info = svdsolve(A, nev; tol)
    stack(U; dims=2)[:, 1:nev], S[1:nev], stack(V, dims=2)[:, 1:nev]
end


"To compute svd(A) = eigen(A*A'), but without doing the multiplication."
struct LazySym{A_t,V_t,T}
    A::A_t
    z::V_t
    LazySym(A) = new{typeof(A),Vector{eltype(A)},eltype(A)}(
        A, zeros(eltype(A), size(A, 2))
    )
end
Base.size(S::LazySym) = (size(S.A, 1), size(S.A, 1))
Base.size(S::LazySym, i) = size(S)[i]
Base.eltype(S::LazySym) = eltype(S.A)
"Computes A * A', lazily."
LinearAlgebra.mul!(y, S::LazySym, x) =
    let
        mul!(S.z, S.A', x)
        mul!(y, S.A, S.z)
    end

function compute_truncated_svd(solver::ArnoldiSVDSolver, A::AbstractMatrix{T}, k::Int) where {T}
    m, n = size(A)
    (U, S, V) = if m > n  # Tall matrix, decompose A^T * A
        (; Q, R, eigenvalues), _ = partialschur(Symmetric(A' * A); nev=k, tol=solver.tol)
        V = Q
        Sigma = sqrt.(real.(eigenvalues))
        U = A * V * Diagonal(1 ./ Sigma)
        (U, Sigma, V)
    else  # Wide matrix, decompose A * A^T
        (; Q, R, eigenvalues), _ = partialschur(Symmetric(A * A'); nev=k, tol=solver.tol)
        U = Q
        Sigma = sqrt.(real.(eigenvalues))
        V = A' * U * Diagonal(1 ./ Sigma)
        (U, Sigma, V)
    end
    return (U, S, V)
end
