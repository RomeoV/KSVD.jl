using KSVD, SparseArrays

function generate_benchmark_data(N, E, K, avg_nnz_per_sample)
    basis = KSVD.init_dictionary(E, K)
    X = sprand(K, N, avg_nnz_per_sample/K)
    data = basis * X + rand(size(basis, 1))
    (; data, basis, X)
end
