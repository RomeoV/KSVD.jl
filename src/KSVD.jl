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
export LegacyMatchingPursuit, ParallelMatchingPursuit

using Base.Threads, Random, SparseArrays, LinearAlgebra
import TSVD: tsvd
import LinearAlgebra: normalize!
import ThreadedDenseSparseMul: fastdensesparsemul!, fastdensesparsemul_threaded!, fastdensesparsemul_outer!, fastdensesparsemul_outer_threaded!
import OhMyThreads
import TimerOutputs
import TimerOutputs: @timeit_debug
import StatsBase: mean
import ProgressMeter: Progress


include("set_num_threads.jl")
include("util.jl")
include("matching_pursuit.jl")
include("ksvd_types.jl")
include("ksvd_update.jl")
include("ksvd_update_legacy.jl")
include("ksvd_update_optimized.jl")
include("ksvd_update_threaded_utils.jl")
function __init__()
    set_num_threads(Threads.nthreads())
end


"""
    ksvd(Y::AbstractMatrix, n_atoms::Int, max_nnz=max(n_atoms÷10, 1);
         sparsity_allowance::Float64 = 0.1,
         maxiter::Int = 10)

Run K-SVD that designs an efficient dictionary D for sparse representations,
and returns X such that DX = Y or DX ≈ Y.

Y is expected to be `(num_features x num_samples)`.
"""
function ksvd(Y::AbstractMatrix{T}, n_atoms::Int, max_nnz=n_atoms÷10;
              ksvd_update_method = BatchedParallelKSVD{false, T}(; shuffle_indices=true, batch_size_per_thread=1),
              sparse_coding_method = ParallelMatchingPursuit(; max_nnz, rtol=5e-2),
              verbose=false,
              # termination conditions
              maxiters::Int=100, #: The maximum number of iterations to perform. Defaults to 100.
              maxtime::Union{Nothing, <:Real}=nothing,# : The maximum time for solving the nonlinear system of equations. Defaults to nothing which means no time limit. Note that setting a time limit does have a small overhead.
              abstol::Number=real(oneunit(T)) * (eps(real(one(T))))^(4 // 5), #: The absolute tolerance. Defaults to real(oneunit(T)) * (eps(real(one(T))))^(1 // 2).
              reltol::Number=real(oneunit(T)) * (eps(real(one(T))))^(4 // 5), #: The relative tolerance. Defaults to real(oneunit(T)) * (eps(real(one(T))))^(1 // 2).
              nnz_per_col_target::Int=0,
              # tracing options
              show_trace::Bool=false,
              # store_trace::Bool,
              ) where T
    timer = TimerOutput()
    emb_dim, n_samples = size(Y)

    # D is a dictionary matrix that contains atoms for columns.
    @timeit_debug timer "Init dict" begin
        D = init_dictionary(T, emb_dim, n_atoms)  # size(D) == (n, K)
        @assert all(≈(1.0), norm.(eachcol(D)))
    end
    X = sparse(zeros(T, 0, 0))  # to assign to later

    # progressbar = Progress(maxiter)
    maybe_init_buffers!(ksvd_update_method, emb_dim, n_atoms, n_samples)

    norm_results, nnz_per_col_results = Float64[], Float64[];
    # if store_trace || show_trace
    trace_channel = Channel{Tuple{Matrix{T}, SparseMatrixCSC{T, Int64}}}(; spawn=true) do ch
        for (D, X) in ch
            norm_val = norm(Y - D*X)
            nnz_per_col_val = nnz(X) / size(X, 2)
            show_trace && @info norm_val, nnz_per_col_val
            (push!(norm_results, norm_val); push!(nnz_per_col_results, nnz_per_col_val))
        end
    end

    termination_condition = :nothing
    tic = time()
    for iter in 1:maxiters
        verbose && @info "Starting sparse coding"
        X = sparse_coding(sparse_coding_method, Y, D; timer)
        verbose && @info "Starting svd"
        D, X = ksvd_update(ksvd_update_method, Y, D, X; timer)

        # put a task to compute the trace / termination conditions.
        push!(trace_channel, (copy(D), copy(X)))

        # Check termination conditions.
        # Notice that this is typically not using the most recent results yet. So we might only later realize that we
        # should terminate.
        if iter == maxiters
            termination_condition = :maxiter; break
        elseif !isnothing(maxtime) && (time() - tic) > maxtime
            termination_condition = :maxtime; break
        elseif length(norm_results) > 1 && isapprox(norm_results[end], norm_results[end-1]; atol=abstol, rtol=reltol)
            termination_condition = :converged; break
        elseif !isempty(nnz_per_col_results) && last(nnz_per_col_results) <= nnz_per_col_target
            termination_condition = :nnz_per_col_target
        end
    end
    TimerOutputs.complement!(timer)
    return (; D, X, norm_results, nnz_per_col_results, termination_condition, timer)
end

const dictionary_learning = ksvd  # for compatibility

end # module
