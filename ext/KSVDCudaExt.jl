module KSVDCudaExt
using CUDA
import KSVD
import KSVD: sparse_coding, czip, compute_truncated_svd
import LinearAlgebra: Symmetric, Diagonal
import KSVD.ArnoldiMethod: partialschur
import KSVD.OhMyThreads
import KSVD.OhMyThreads: chunks
import KSVD.TimerOutputs: TimerOutput, @timeit_debug
using SparseArrays: sparse, nonzeroinds, nonzeros

export CUDAAcceleratedMatchingPursuit, CUDAAcceleratedArnoldiSVDSolver

"""
    KSVD.sparse_coding(method::KSVD.CUDAAcceleratedMatchingPursuit,
    data::AbstractMatrix{T}, dictionary::AbstractMatrix{T};
    timer=TimerOutput(),
    DtD=<...>) where {T}

CUDA Accelerated version pipelining the dictionary * data computation
on the gpu, and using the result on the cpu. Always performs batching, i.e. uses
limited memory even for large number of samples (however not for large embedding dimension).

For description of parameters see `MatchingPursuit`.
Notice that this method never precomputes the full product matrix `D'*Y`. Instead
batches of data with size number of samples `batchsize` are loaded, moved to the GPU, multiplied there,
and the result is moved back to the cpu. This happens asynchronously, so that the memory movement
CPU->GPU and GPU->CPU, aswell as the computation on the CPU, are pipelined.
"""
function KSVD.sparse_coding(method::KSVD.CUDAAcceleratedMatchingPursuit,
    data::AbstractMatrix{T}, dictionary::AbstractMatrix{T};
    timer=TimerOutput(),
    DtD=let Ddev = cu(dictionary)
        Matrix(Ddev' * Ddev)
    end) where {T}
    batchsize = method.batchsize
    idx_batches = chunks(axes(data, 2); size=batchsize)

    # batchsize = 2^10

    # We precompute DtY on the GPU. To maximize throughput, we try to overlap memory transfer and
    # gpu computation by running three streams in parallel: host to device, device computation,
    # and device to host transfer. We use some pinned buffers to help transfer speed, see e.g. https://cuda.juliagpu.org/stable/usage/multitasking/#Task-based-programming.
    ch_HtoD = Channel{CuMatrix{T}}(2; spawn=true) do ch
        Yslice_host_pinned = CUDA.pin(similar(data, (size(data, 1), batchsize)))
        foreach(idx_batches) do idx
            copyto!(Yslice_host_pinned, @view(data[:, idx]))
            put!(ch, cu(Yslice_host_pinned))
        end
        close(ch)
    end

    ch_DtoD = Channel{CuMatrix{T}}(2; spawn=true) do ch
        Ddev = cu(dictionary)
        foreach(ch_HtoD) do Yslice
            DtYslice = Ddev' * Yslice
            put!(ch, DtYslice)
        end
        close(ch)
    end

    ch_DtoH = Channel{Matrix{T}}(2; spawn=true) do ch
        DtYslice_host_pinned = CUDA.pin(similar(dictionary, (size(dictionary, 2), batchsize)))
        foreach(ch_DtoD) do DtYslice
            copyto!(DtYslice_host_pinned, DtYslice)
            put!(ch, copy(DtYslice_host_pinned))
        end
        close(ch)
    end

    noncudamethod = KSVD.ParallelMatchingPursuit(;
        max_nnz=method.max_nnz,
        max_iter=method.max_iter,
        rtol=method.rtol,
        precompute_products=method.precompute_products,
        refit_coeffs=method.refit_coeffs)

    mapreduce(hcat, zip(idx_batches, ch_DtoH)) do (idx, DtYslice)
        KSVD.sparse_coding(noncudamethod, @view(data[:, idx]), dictionary; DtD, DtY=DtYslice, timer)
    end
end

function KSVD.compute_truncated_svd(solver::KSVD.CUDAAcceleratedArnoldiSVDSolver{T}, A::AbstractMatrix{T}, k::Int) where {T<:Number}
    m, n = size(A)
    (U, S, V) = if m > n  # Tall matrix, decompose A^T * A
        # we assume that here we would not get any benefit from transferring to the GPU.
        AtA = Symmetric(Matrix(Adev' * Adev))
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
