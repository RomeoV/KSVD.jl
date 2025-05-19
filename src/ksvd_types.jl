abstract type KSVDMethod end

struct LegacyKSVD <: KSVDMethod end

@kwdef struct OptimizedKSVD{SVDSol<:AbstractTruncatedSVD} <: KSVDMethod
    shuffle_indices::Bool = false
    svd_solver::SVDSol = ArnoldiSVDSolver{Float32}()
    OptimizedKSVD(shuffle_indices, svd_solver) = new{ArnoldiSVDSolver{Float32}}(shuffle_indices, svd_solver)
end

@kwdef mutable struct ParallelKSVD{precompute_error,T,Scheduler<:OhMyThreads.Scheduler,SVDSol<:AbstractTruncatedSVD} <: KSVDMethod
    E_buf::Matrix{T} = T[;;]
    E_Ω_bufs::Vector{Matrix{T}} = Matrix{T}[]
    D_cpy_buf::Matrix{T} = T[;;]
    shuffle_indices::Bool = false
    svd_solver::SVDSol = SVDSol()
end
# Backward compatibility constructor
ParallelKSVD{precompute_error,T}(; kwargs...) where {precompute_error,T} =
    ParallelKSVD{precompute_error,T,OhMyThreads.Schedulers.DynamicScheduler,TSVDSolver{T}}(; kwargs...)

function Base.show(io::IO, m::ParallelKSVD)
    out = IOBuffer()
    print(out, typeof(m))
    print(out, "(")
    print(out, typeof(m.E_buf), " size $(size(m.E_buf)), ")
    print(out, typeof(m.E_Ω_bufs), " length $(length(m.E_Ω_bufs)), ")
    print(out, typeof(m.D_cpy_buf), " size $(size(m.D_cpy_buf)), ")
    print(out, m.shuffle_indices, ", ")
    print(out, m.svd_solver, ")")
    print(io, String(take!(out)))
end

@kwdef mutable struct BatchedParallelKSVD{precompute_error,T,Scheduler<:OhMyThreads.Scheduler,SVDSol<:AbstractTruncatedSVD} <: KSVDMethod
    E_buf::Matrix{T} = T[;;]
    E_Ω_bufs::Vector{Matrix{T}} = Matrix{T}[]
    D_cpy_buf::Matrix{T} = T[;;]
    shuffle_indices::Bool = false
    batch_size_per_thread::Int = 1
    svd_solver::SVDSol = SVDSol()
end
# Backward compatibility constructor
BatchedParallelKSVD{precompute_error,T}(; kwargs...) where {precompute_error,T} =
    BatchedParallelKSVD{precompute_error,T,OhMyThreads.Schedulers.DynamicScheduler,ArnoldiSVDSolver{T}}(; kwargs...)

function Base.show(io::IO, m::BatchedParallelKSVD)
    out = IOBuffer()
    print(out, typeof(m))
    print(out, "(")
    print(out, typeof(m.E_buf), " size $(size(m.E_buf)), ")
    print(out, typeof(m.E_Ω_bufs), " length $(length(m.E_Ω_bufs)), ")
    print(out, typeof(m.D_cpy_buf), " size $(size(m.D_cpy_buf)), ")
    print(out, m.shuffle_indices, ", ")
    print(out, m.batch_size_per_thread, ", ")
    print(out, m.svd_solver, ")")
    print(io, String(take!(out)))
end

@kwdef mutable struct DistributedKSVD{T,SVDSol<:AbstractTruncatedSVD} <: KSVDMethod
    submethod_per_worker::Dict{Int,<:KSVDMethod} = Dict()
    shuffle_indices::Bool = false
    batch_size_per_thread::Int = 1
    reduction_method::Symbol = :clustering  # or :mean
    svd_solver::SVDSol = SVDSol()
end
# Backward compatibility constructor
DistributedKSVD{T}(; kwargs...) where {T} = DistributedKSVD{T,TSVDSolver{T}}(; kwargs...)

const ThreadedKSVDMethod = Union{BatchedParallelKSVD,ParallelKSVD}
const ThreadedKSVDMethodPrecomp{B} = Union{BatchedParallelKSVD{B},ParallelKSVD{B}}

const ThreadedKSVDMethod = Union{BatchedParallelKSVD,ParallelKSVD}
const ThreadedKSVDMethodPrecomp{B} = Union{BatchedParallelKSVD{B},ParallelKSVD{B}}
