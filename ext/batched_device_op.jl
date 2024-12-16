struct BatchedDeviceOpWorkspace{T}
    batchsize::Int
    n_bufs::Int
    M_dst::Vector{Matrix{T}}  # list of pinned host memory buffers
    M_dev::Vector{CuMatrix{T}}  # list of device memory buffers
    # op::O_t
    BatchedDeviceOpWorkspace(T, batchsize, sz_in::Int, sz_out::Int, n_bufs=5) = new{T}(
        batchsize,
        n_bufs,
        [CUDA.pin(Matrix{T}(undef, sz_out, batchsize)) for _ in 1:n_bufs],
        [CuMatrix{T}(undef, sz_in, batchsize) for _ in 1:n_bufs])
end
BatchedDeviceOpWorkspace(batchsize, sz_in::Int, sz_out::Int) = BatchedDeviceOpWorkspace(Float32, batchsize, sz_in, sz_out)

function make_batched_device_op_workspace(op, M::Matrix)
    ten_MB = 10 * 1024^2
    # heuristic for batchsize
    # sizeof(T) * size(M, 1) * batchsize = ten_MB
    batchsize = 2^round(Int, log2(ten_MB / (sizeof(eltype(M)) * size(M, 1))))

    batch_dev = cu(M[:, 1:min(100, size(M, 2))])
    sz_out = size(op(batch_dev), 1)
    BatchedDeviceOpWorkspace(eltype(M), batchsize, size(M, 1), sz_out)
end

batched_device_op(op, M::Matrix) = batched_device_op(op, make_batched_device_op_workspace(op, M), M)
function batched_device_op(op, workspace::BatchedDeviceOpWorkspace, M::Matrix{T}) where {T}
    CUDA.is_pinned(pointer(M)) || CUDA.pin(M)
    (; batchsize, n_bufs, M_dst, M_dev) = workspace
    idx_batches = Iterators.partition(axes(M, 2), batchsize)
    idx_dev = CartesianIndices((axes(M, 1), 1:batchsize))

    dst_ch = Channel{Matrix{eltype(M)}}(n_bufs)
    put!.([dst_ch], M_dst)

    dev_ch = Channel{CuMatrix{eltype(M)}}(n_bufs)
    put!.([dev_ch], M_dev)

    ready_ch = Channel{Pair{UnitRange{Int}, Matrix{eltype(M)}}}(n_bufs)

    op_task = Threads.@spawn begin
        Threads.@threads for idx_ in idx_batches |> collect
            # M_dev = CuMatrix{eltype(M)}(undef, size(M, 1), batchsize)
            idx = CartesianIndices((axes(M, 1), idx_))
            M_dst_buf = take!(dst_ch)  # this blocks until a pinned host buffer is available
            M_dev_buf = take!(dev_ch)

            copyto!(M_dev_buf, idx_dev, M, idx)
            res = op(M_dev_buf)
            copyto!(M_dst_buf, res)

            # We need to keep `idx_` as we may run into a race condition...
            put!(ready_ch, idx_=>M_dst_buf)
            put!(dev_ch, M_dev_buf)
        end
        close(ready_ch); close(dev_ch)
    end

    results_ch = Channel{Pair{UnitRange{Int}, Matrix{T}}}(n_bufs; spawn=true) do results_ch
        foreach(ready_ch) do (idx, M_buf)
            put!(results_ch, idx=>copy(M_buf))
            put!(dst_ch, M_buf)
        end
        close(dst_ch)
    end
    results_ch, op_task
end

"""
@test begin

T = Float32
Y = rand(T, 2*512, 4096*32)
D = rand(T, 2*512, 2*512)
D_dev = CuMatrix(D)
op(rhs::CuMatrix) = D_dev*rhs
op(rhs::Matrix) = D*rhs
workspace = BatchedDeviceOpWorkspace(4096, 2*512, 2*512)
ch, task = batched_device_op(op, workspace, Y)
res = hcat(last.(sort(collect(ch); by=first))...);
@test res ≈ D*Y
@test res ≈ op(Y)

end
"""
