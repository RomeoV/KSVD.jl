using KrylovKit
import TSVD: tsvd
import Arpack: svds
# import ArnoldiMethod: arnoldi_svd

abstract type AbstractTruncatedSVD end

@kwdef struct TSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tolconv::Float64 = sqrt(eps(real(T)))
    maxiter::Int = 50
end

@kwdef struct ArpackSolver{T<:Number} <: AbstractTruncatedSVD
    tolconv::Float64 = sqrt(eps(real(T)))
    maxiter::Int = 50
end

@kwdef struct KrylovSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tol::Float64 = sqrt(eps(real(T)))
    maxiter::Int = 50
end

@kwdef struct ArnoldiSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tol::Float64 = sqrt(eps(real(T)))
    maxiter::Int = 50
end

# Solver implementations
function compute_truncated_svd(solver::TSVDSolver, A::AbstractMatrix{T}, k::Int) where {T}
    tsvd(A, k; tolconv=solver.tolconv, maxiter=solver.maxiter)
end

function compute_truncated_svd(solver::ArpackSolver, A::AbstractMatrix{T}, k::Int) where {T}
    # we copy here because we've had issues with @view and Arpack before...
    (U, S, V), _ = svds(copy(A); nsv=k, maxiter=solver.maxiter, tol=solver.tolconv)
    (U, S, V)
end

function compute_truncated_svd(solver::KrylovSVDSolver, A::AbstractMatrix{T}, k::Int) where {T}
    krylov_svd(A, k; tol=solver.tol)
end

function compute_truncated_svd(solver::ArnoldiSVDSolver, A::AbstractMatrix{T}, k::Int) where {T}
    arnoldi_svd(A, k; tol=solver.tol)
end

function krylov_svd(A, nev=1; tol=1e-8)
    S, U, V, info = svdsolve(A, nev; tol)
    stack(U; dims=2), S, stack(V, dims=2)
end
