module KSVDCudaExt
using CUDA
import KSVD
import KSVD: sparse_coding, czip, compute_truncated_svd
import LinearAlgebra: Symmetric, Diagonal
import KSVD.ArnoldiMethod: partialschur
import KSVD.OhMyThreads
import KSVD.TimerOutputs: TimerOutput, @timeit_debug
using SparseArrays: sparse, nonzeroinds, nonzeros

export CUDAAcceleratedMatchingPursuit, CUDAAcceleratedArnoldiSVDSolver
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

function KSVD.sparse_coding(method::KSVD.CUDAAcceleratedMatchingPursuit,
    data::AbstractMatrix{T}, dictionary::AbstractMatrix{T};
    timer=TimerOutput(),
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
    KSVD.sparse_coding(noncudamethod, data, dictionary; DtD, DtY, timer)
end

# We try to implement pipelining here, but it's not going very well.
# It works, but it's not super fast.
#function KSVD.sparse_coding(method::CUDAAcceleratedMatchingPursuit,
#    data::AbstractMatrix{T}, dictionary::AbstractMatrix{T};
#    Ddev=cu(dictionary),
#    DtD=Symmetric(Matrix(Ddev' * Ddev))
#) where {T}
#    Y = data
#    D = dictionary
#
#    # DtD = Symmetric(Matrix(Ddev' * Ddev))
#    batchsize = 8 * 1024
#    idx_batches = Iterators.partition(axes(Y, 2), batchsize) |> collect
#    DtYb_ch = Channel{Tuple{UnitRange{Int},Matrix{T}}}(8; spawn=true) do ch
#        OhMyThreads.tforeach(idx_batches) do idx_batch
#            # Ddev′ = cu(D)
#            Ybdev = cu(@view Y[:, idx_batch])
#            DtYb = Matrix(Ddev' * Ybdev)
#            put!(ch, (idx_batch, DtYb))
#        end
#        close(ch)
#    end
#
#    noncudamethod = KSVD.ParallelMatchingPursuit(;
#        max_nnz=method.max_nnz,
#        max_iter=method.max_iter,
#        rtol=method.rtol,
#        precompute_products=method.precompute_products)
#
#    Xbatches = map(DtYb_ch) do (idx_batch, DtYb)
#        (idx_batch, KSVD.sparse_coding(noncudamethod, @view(Y[:, idx_batch]), D; DtD, DtY=DtYb))
#    end
#    sort!(Xbatches; by=first)
#    X = reduce(hcat, last.(Xbatches))
#    GC.gc()
#    return X
#end

#function KSVD.sparse_coding2(method::CUDAAcceleratedMatchingPursuit,
#    data::AbstractMatrix{T}, dictionary::AbstractMatrix{T}) where {T}
#    K = size(dictionary, 2)
#    N = size(data, 2)
#
#    DtD = let Ddev = cu(dictionary)
#      Symmetric(Matrix(Ddev'*Ddev))
#    end
#    Ddev = cu(dictionary)
#
#    op(Y_batch::CuMatrix) = Ddev' * Y_batch
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

function KSVD.compute_truncated_svd(solver::KSVD.CUDAAcceleratedArnoldiSVDSolver{T}, A::AbstractMatrix{T}, k::Int) where {T<:Number}
    m, n = size(A)
    (U, S, V) = if m > n  # Tall matrix, decompose A^T * A
        AtA = let Adev = cu(Matrix(A))
            AtA = Symmetric(Matrix(Adev' * Adev))
            CUDA.unsafe_free!(Adev)
            AtA
        end
        (; Q, R, eigenvalues), _ = partialschur(AtA; nev=k, tol=solver.tol)
        V = Q
        Sigma = sqrt.(real.(eigenvalues))
        U = A * V * Diagonal(1 ./ Sigma)
        (U, Sigma, V)
    else  # Wide matrix, decompose A * A^T
        AAt = let Adev = cu(Matrix(A))
            AAt = Symmetric(Matrix(Adev * Adev'))
            CUDA.unsafe_free!(Adev)
            AAt
        end
        (; Q, R, eigenvalues), _ = partialschur(AAt; nev=k, tol=solver.tol)
        U = Q
        Sigma = sqrt.(real.(eigenvalues))
        V = A' * U * Diagonal(1 ./ Sigma)
        (U, Sigma, V)
    end
    return (U, S, V)
end
end # KSVDCudaExt
