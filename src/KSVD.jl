module KSVD

# This is an implementation of the K-SVD algorithm.
# The original paper:
# K-SVD: An Algorithm for Designing Overcomplete Dictionaries
# for Sparse Representation
# http://www.cs.technion.ac.il/~freddy/papers/120.pdf

# Variable names are based on the original paper.
# If you try to read the code, I recommend you to see Figure 2 first.
#

export ksvd, ksvd_update, sparse_coding
export LegacyKSVD, OptimizedKSVD, ParallelKSVD, BatchedParallelKSVD
export LegacyMatchingPursuit, ParallelMatchingPursuit, OrthogonalMatchingPursuit
export NormalLoop, MatryoshkaLoop

using Base.Threads, Random, SparseArrays, LinearAlgebra
import LinearAlgebra: normalize!
import ThreadedDenseSparseMul: fastdensesparsemul!, fastdensesparsemul_threaded!, fastdensesparsemul_outer!, fastdensesparsemul_outer_threaded!
import OhMyThreads
import TimerOutputs
import TimerOutputs: @timeit_debug
import StatsBase: mean
import ProgressMeter: Progress
using PrecompileTools: @compile_workload

include("set_num_threads.jl")
include("util.jl")
include("matching_pursuit.jl")
include("svd_solvers.jl")
include("ksvd_types.jl")
include("krylov_svd.jl")
include("ksvd_update.jl")
include("ksvd_update_legacy.jl")
include("ksvd_update_optimized.jl")
include("ksvd_update_threaded_utils.jl")
include("ksvd_loop.jl")
function __init__()
    set_num_threads(Threads.nthreads())
end
ext = Base.get_extension(@__MODULE__, :KSVDCudaExt)
if !isnothing(ext)
    const CUDAAcceleratedMatchingPursuit = ext.CUDAAcceleratedMatchingPursuit
    export CUDAAcceleratedMatchingPursuit
end



"""
    ksvd(Y::AbstractMatrix{T}, n_atoms::Int;
         ksvd_update_method = BatchedParallelKSVD{false, T}(; shuffle_indices=true, batch_size_per_thread=1),
         sparse_coding_method = ParallelMatchingPursuit(; max_nnz=max(n_atoms÷10, 1), rtol=5e-2),
         verbose=false,
         maxiters::Int=100,
         maxtime::Union{Nothing, <:Real}=nothing,
         abstol::Number=real(oneunit(T)) * (eps(real(one(T))))^(4 // 5),
         reltol::Number=real(oneunit(T)) * (eps(real(one(T))))^(4 // 5),
         nnz_per_col_target::Int=0,
         show_trace::Bool=false) where T

Run K-SVD algorithm to design an efficient dictionary D for sparse representations.
Returns dictionary `D` and sparse assignment matrix `X` such that Y ≈ DX.
Y is expected to be `(num_features x num_samples)`.
To set the number of nonzeros, set the `sparse_coding_method` (see Notes below).

Can also return losses and detailed timing results (see Notes below), and take a callback
function, e.g. to compute other losses or store intermediate solutions.

# Arguments
- `Y::AbstractMatrix{T}`: Input data matrix of size (num_features x num_samples)
- `n_atoms::Int`: Number of atoms (columns) in the dictionary

# Keyword Arguments
- `ksvd_update_method`: Method used for updating the dictionary (default: BatchedParallelKSVD)
- `sparse_coding_method`: Method used for sparse coding (default: ParallelMatchingPursuit)
- `verbose::Bool=false`: If true, print verbose output
- `maxiters::Int=100`: Maximum number of iterations
- `maxtime::Union{Nothing, <:Real}=nothing`: Maximum time limit (in seconds)
- `abstol::Number`: Absolute tolerance for convergence
- `reltol::Number`: Relative tolerance for convergence
- `nnz_per_col_target::Int=0`: Target number of non-zero entries per column
- `show_trace::Bool=false`: If true, show trace of the optimization

# Returns
A named tuple containing:
- `D`: The learned dictionary
- `X`: The sparse representation matrix
- `norm_results`: Array of norm values for each iteration
- `nnz_per_col_results`: Array of non-zero entries per column for each iteration
- `termination_condition`: The condition that caused termination
- `timer`: Timing information for various parts of the algorithm

# Notes
- Dictionary vectors are normalized such that ~all(norm.(eachcol(D), 2) .≈ 1)~.
- To enable timing outputs, run `TimerOutputs.enable_debug_timings(KSVD)`.
- To set the number of nonzeros, specify e.g. `sparse_coding_method=ParallelMatchingPursuit(; max_nnz=..., rtol=5e-2)`.
"""
function ksvd(Y::AbstractMatrix{T}, n_atoms::Int, max_nnz=max(3, n_atoms ÷ 100);
    ksvd_update_method=BatchedParallelKSVD{false,T}(; shuffle_indices=true, batch_size_per_thread=1),
    sparse_coding_method=ParallelMatchingPursuit(; max_nnz, rtol=5e-2),
    ksvd_loop_type::KSVDLoopType=NormalLoop(),
    D_init::Union{Nothing,<:AbstractMatrix{T}}=nothing,
    X_init::Union{Nothing,<:AbstractSparseMatrix}=nothing,
    minibatch_size=nothing,
    # termination conditions
    maxiters::Int=100,
    maxtime::Union{Nothing,<:Real}=nothing,
    abstol::Union{Nothing,<:Real}=real(oneunit(T)) * (eps(real(one(T))))^(4 // 5),
    reltol::Union{Nothing,<:Real}=real(oneunit(T)) * (eps(real(one(T))))^(4 // 5),
    nnz_per_col_target::Number=0.0,
    # tracing options
    show_trace::Bool=false,
    callback_fn::Union{Nothing,Function}=nothing,
    verbose=false,
    timer::TimerOutput=TimerOutput()
) where {T}
    @assert all(isfinite, Y) "All elements in Y must be finite. Probably there are some NaN or Inf."
    emb_dim, n_samples = size(Y)

    # D is a dictionary matrix that contains atoms for columns.
    @timeit_debug timer "Init dict" begin
        D = (isnothing(D_init) ? init_dictionary(T, emb_dim, n_atoms) : copy(D_init))  # size(D) == (n, K)
        @assert all(≈(1.0), norm.(eachcol(D)))
    end
    DtD = D' * D
    DtY = D' * Y
    X = (isnothing(X_init) ? sparse_coding(sparse_coding_method, Y, D; timer, DtD, DtY) : copy(X_init))

    # progressbar = Progress(maxiter)
    maybe_init_buffers!(ksvd_update_method, emb_dim, n_atoms, (isnothing(minibatch_size) ? n_samples : minibatch_size))

    norm_results, nnz_per_col_results = Float64[], Float64[]
    # if store_trace || show_trace
    trace_taskref = Ref{Task}()
    CH_T = Tuple{Int,Matrix{T},SparseMatrixCSC{T,Int64}}
    loggingtasks = OhMyThreads.StableTasks.StableTask{Nothing}[]
    trace_channel = Channel{CH_T}(maxiters; spawn=true, taskref=trace_taskref) do ch
        # tforeach(ch; scheduler=:greedy) do (iter, D, X)
        for (iter, D, X) in ch
            t = OhMyThreads.@spawn begin
                norm_val = (norm.(eachcol(Y - D * X)) ./ (norm.(eachcol(Y)) .+ eps(T))) |> mean
                nnz_per_col_val = nnz(X) / size(X, 2)
                show_trace && @info (iter, norm_val, nnz_per_col_val)
                (push!(norm_results, norm_val); push!(nnz_per_col_results, nnz_per_col_val))
                !isnothing(callback_fn) && callback_fn((; iter, Y, D, X, norm_val, nnz_per_col_val))
                return nothing
            end
            push!(loggingtasks, t)
        end
    end

    termination_condition = :nothing
    tic = time()
    for iter in 1:maxiters
        # note that D gets updated in place
        yidx = (isnothing(minibatch_size) ? axes(Y, 2) : sort(shuffle(axes(Y, 2))[1:minibatch_size]))
        X = ksvd_loop!(ksvd_loop_type, ksvd_update_method, sparse_coding_method,
            Y, D, X; timer, yidx, verbose)

        # put a task to compute the trace / termination conditions.
        push!(trace_channel, (iter, copy(D), copy(X)))

        # Check termination conditions.
        # Notice that this is typically not using the most recent results yet. So we might only later realize that we
        # should terminate.
        if iter == maxiters
            termination_condition = :maxiter
            break
        elseif !isnothing(maxtime) && (time() - tic) > maxtime
            termination_condition = :maxtime
            break
        elseif (!isnothing(abstol) && !isnothing(reltol)) && length(norm_results) > 1 && isapprox(norm_results[end], norm_results[end-1]; atol=abstol, rtol=reltol)
            termination_condition = :converged
            break
        elseif !isempty(nnz_per_col_results) && last(nnz_per_col_results) <= nnz_per_col_target
            termination_condition = :nnz_per_col_target
            break
        end
    end
    close(trace_channel)
    TimerOutputs.complement!(timer)
    wait(trace_taskref[])  # make sure trace has finished
    foreach(wait, loggingtasks)
    return (; D, X, norm_results, nnz_per_col_results, termination_condition, timer)
end

const dictionary_learning = ksvd  # for compatibility

@compile_workload begin
    ksvd(rand(Float32, 10, 20), 15; maxiters=2)
    ksvd(rand(Float64, 10, 20), 15; maxiters=2)
end

end # module
