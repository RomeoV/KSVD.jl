using SparseArrays, BenchmarkTools

suite = BenchmarkGroup();
L = 1_000; N = 50_000;
suite[:noskip] = @benchmarkable begin
        Ω = sparse(ω, 1:nnz, ones(nnz), N, nnz)
        EΩ .= E*Ω
    end setup=begin
        E = rand(L, N);
        nnz = N÷50
        ω = sample(1:N, nnz, replace=false)
        EΩ = zeros(L, nnz)
    end
suite[:noskip_pre] = @benchmarkable EΩ .= E*Ω setup=begin
        E = rand(L, N);
        nnz = N÷50
        ω = sample(1:N, nnz, replace=false)
        Ω = sparse(ω, 1:nnz, ones(nnz), N, nnz)
        EΩ = zeros(L, nnz)
    end
suite[:skip] = @benchmarkable EΩ .= E[:, ω] setup=begin
        E = rand(L, N);
        nnz = N÷50
        ω = sample(1:N, nnz, replace=false)
        EΩ = zeros(L, nnz)
    end
tune!(suite)
res = run(suite)
