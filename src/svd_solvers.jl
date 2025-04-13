using KrylovKit
import TSVD: tsvd
# import ArnoldiMethod: arnoldi_svd

abstract type AbstractTruncatedSVD end

@kwdef struct TSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tolconv::Float64 = sqrt(eps(real(T)))
end

@kwdef struct KrylovSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tol::Float64 = sqrt(eps(real(T)))
end

@kwdef struct ArnoldiSVDSolver{T<:Number} <: AbstractTruncatedSVD
    tol::Float64 = sqrt(eps(real(T)))
end

# Solver implementations
function compute_truncated_svd(solver::TSVDSolver, A::AbstractMatrix{T}, k::Int) where {T}
    tsvd(A, k; tolconv=solver.tolconv)
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
