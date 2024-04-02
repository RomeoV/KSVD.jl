module KSVDCudaExt
using CUDA
using KSVD
import KSVD: sparse_coding, CUDAAcceleratedMatchingPursuit, czip
using SparseArrays: sparse, nonzeroinds, nonzeros

function KSVD.sparse_coding(method::CUDAAcceleratedMatchingPursuit, data::AbstractMatrix{T}, dictionary::AbstractMatrix{T}) where T
    K = size(dictionary, 2)
    N = size(data, 2)

    DtD = dictionary'*dictionary
    Dt_gpu = CuMatrix(dictionary')

    # data_iter = chunks(eachcol(data), 10)
    data_iter = Iterators.partition(axes(data, 2), method.batch_size)
    # Move first batch of data to gpu (asynchronously), compute matrix matrix product there,
    # then move back to cpu. This should be pipelined, and the next part of the computation (sparse matmul)
    # can get started as soon as the first batch is done computing.
    ch_cpu_to_gpu = Channel{CuMatrix{T}}() do ch
        foreach(data_iter) do idx
            CUDA.@sync data_batch = CuMatrix(data[:, idx])
            put!(ch, data_batch)
        end
    end
    ch_gpu_to_cpu = Channel{Matrix{T}}() do ch
        foreach(ch_cpu_to_gpu) do data_batch
            CUDA.@sync products_batch = Matrix(Dt_gpu * data_batch)
            put!(ch, products_batch)
        end
    end


    I_buffers = [Int[] for _ in 1:size(data, 2)]; V_buffers = [T[] for _ in 1:size(data, 2)];

    @debug "Getting ready for processing"
    for (j_batch, products_batch) in zip(data_iter, ch_gpu_to_cpu)
        @debug "Processing $j_batch"
        Threads.@threads for (j_, j) in czip(j_batch, axes(products_batch, 2))
            datacol = @view data[:, j]; productcol = @view products_batch[:, j]
            data_vec = KSVD.matching_pursuit_(
                            method,
                            datacol,
                            dictionary,
                            DtD;
                            products_init=productcol
                        )
            append!(I_buffers[j_], nonzeroinds(data_vec))
            append!(V_buffers[j_], nonzeros(data_vec))
        end
    end

    I = vcat(I_buffers...); V = vcat(V_buffers...)
    J = vcat([fill(j, size(I_buf)) for (j, I_buf) in enumerate(I_buffers)]...)
    X = sparse(I,J,V, K, N)
    return X
end

end # KSVDCudaExt
