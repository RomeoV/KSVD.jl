module KSVDCudaExt
using CUDA
import KSVD
import KSVD: sparse_coding, czip
using SparseArrays: sparse, nonzeroinds, nonzeros
# include("batched_device_op.jl")

""" CUDA Accelerated version pipelining the dictionary * data computation
on the gpu, and using the result on the cpu. Always performs batching, i.e. uses
limited memory even for large number of samples (however not for large embedding dimension).

For description of parameters see `MatchingPursuit`.
Notice that this method never precomputes the full product matrix `D'*Y`. Instead
batches of data with size number of samples `batch_size` are loaded, moved to the GPU, multiplied there,
and the result is moved back to the cpu. This happens asynchronously, so that the memory movement
CPU->GPU and GPU->CPU, aswell as the computation on the CPU, are pipelined using Julia's `Channel`s.
"""
@kwdef struct CUDAAcceleratedMatchingPursuit <: KSVD.GPUAcceleratedMatchingPursuit
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4 * max_nnz
    rtol = default_rtol
    precompute_products = true
    CUDAAcceleratedMatchingPursuit(args...) = (KSVD.validate_mp_args(args...); new(args...))
end

function KSVD.sparse_coding(method::CUDAAcceleratedMatchingPursuit,
    data::AbstractMatrix{T}, dictionary::AbstractMatrix{T};
    DtD=let Ddev = cu(dictionary)
        Matrix(Ddev' * Ddev)
    end,
    Ydev=cu(data),
    DtY=let Ddev = cu(dictionary)
        Matrix(Ddev' * Ydev)
    end) where {T}

    noncudamethod = KSVD.ParallelMatchingPursuit(;
        max_nnz=method.max_nnz,
        max_iter=method.max_iter,
        rtol=method.rtol,
        precompute_products=method.precompute_products)
    KSVD.sparse_coding(noncudamethod, data, dictionary; DtD, DtY)
end

#function KSVD.sparse_coding2(method::CUDAAcceleratedMatchingPursuit,
#    data::AbstractMatrix{T}, dictionary::AbstractMatrix{T}) where {T}
#    K = size(dictionary, 2)
#    N = size(data, 2)
#
#    DtD = dictionary' * dictionary
#    Dt_dev = CuMatrix(dictionary')
#
#    op(Y_batch::CuMatrix) = Dt_dev * Y_batch
#    if isnothing(method.workspace)
#        method.workspace = make_batched_device_op_workspace(op, data)
#    end
#    # workspace = BatchedDeviceOpWorkspace(4096, 2*512, 2*512)
#    # ch, task = batched_device_op(op, workspace, Y)
#    ch, task = batched_device_op(op, method.workspace, data)
#
#    X_batches = map(ch) do (idx, DtY)
#        idx => sparse_coding(ParallelMatchingPursuit(), @view(data[:, idx]), dictionary; DtY, DtD)
#    end
#    # X_batches = collect(ch)
#    # @info size(X_batches), fetch(task)
#    hcat(last.(sort(collect(X_batches); by=first))...)
#end

end # KSVDCudaExt
