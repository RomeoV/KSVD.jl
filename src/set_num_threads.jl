num_threads = Ref(1)

"""
    ThreadedSparseCSR.set_num_threads(n::Int)

Sets the number of threads used in sparse csr matrix - vector multiplication.
"""
function set_num_threads(n::Int)

    0 < n <=  Threads.nthreads(Threads.threadpool()) || throw(DomainError("The numbers of threads must be > 0 and <= $(Threads.nthreads(Threads.threadpool()))."))

    KSVD.num_threads[] = n

end

"""
    ThreadedSparseCSR.get_num_threads()

Gets the number of threads used in sparse csr matrix - vector multiplication.
"""
function get_num_threads()

    return KSVD.num_threads[]

end
