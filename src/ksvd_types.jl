abstract type KSVDMethod end

struct LegacyKSVD <: KSVDMethod end

@kwdef struct OptimizedKSVD <: KSVDMethod
    shuffle_indices::Bool = false
end

@kwdef mutable struct ParallelKSVD{precompute_error, T, Scheduler<:OhMyThreads.Scheduler} <: KSVDMethod
    E_buf::AbstractMatrix{T} = T[;;]
    E_Ω_bufs::Vector{AbstractMatrix{T}} = Matrix{T}[]
    D_cpy_buf::AbstractMatrix{T} = T[;;]
    shuffle_indices::Bool = false
end
ParallelKSVD{precompute_error, T}(; kwargs...) where {precompute_error, T} = ParallelKSVD{precompute_error, T, OhMyThreads.Schedulers.DynamicScheduler}(; kwargs...)

@kwdef mutable struct BatchedParallelKSVD{precompute_error, T, Scheduler<:OhMyThreads.Scheduler} <: KSVDMethod
    E_buf::AbstractMatrix{T} = T[;;]
    E_Ω_bufs::Vector{AbstractMatrix{T}} = Matrix{T}[]
    D_cpy_buf::AbstractMatrix{T} = T[;;]
    shuffle_indices::Bool = false
    batch_size_per_thread::Int = 1
end
BatchedParallelKSVD{precompute_error, T}(; kwargs...) where {precompute_error, T} = BatchedParallelKSVD{precompute_error, T, OhMyThreads.Schedulers.DynamicScheduler}(; kwargs...)

@kwdef mutable struct DistributedKSVD{T} <: KSVDMethod
    submethod_per_worker::Dict{Int, <:KSVDMethod} = Dict()
    shuffle_indices::Bool = false
    batch_size_per_thread::Int = 1
    reduction_method::Symbol = :clustering  # or :mean
end

const ThreadedKSVDMethod = Union{BatchedParallelKSVD, ParallelKSVD}
const ThreadedKSVDMethodPrecomp{B} = Union{BatchedParallelKSVD{B}, ParallelKSVD{B}}
