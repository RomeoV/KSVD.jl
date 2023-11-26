using BenchmarkTools, SparseArrays
using Plots, StatsPlots, BenchmarkPlots

suite = BenchmarkGroup();
L = 1_000; K = 2_000; N = 50_000;

suite[:baseline0] = @benchmarkable begin
        @views EΩ .+= D * X[:, ωₖ]
    end setup=begin
        D = rand($L, $K); X = sprand($K, $N, 0.05);
        xₖ = X[1, :]; ωₖ = findall(!iszero, xₖ)
        EΩ = zeros(L, length(ωₖ))
    end samples=10
suite[:baseline] = @benchmarkable begin
        @inbounds for (j_dest, j) in enumerate(ωₖ)
            EΩ[:, j_dest] .+= D * X[:, j]
        end
    end setup=begin
        D = rand($L, $K); X = sprand($K, $N, 0.05);
        xₖ = X[1, :]; ωₖ = findall(!iszero, xₖ)
        EΩ = zeros(L, length(ωₖ))
    end
suite[:broadcast] = @benchmarkable begin
        @inbounds @views for (j_dest, j) in enumerate(ωₖ)
            rs = nzrange(X, j)
            EΩ[:, j_dest] .+=  D[:, rowvals(X)[rs]] * nonzeros(X)[rs]
        end
    end setup=begin
        D = rand($L, $K); X = sprand($K, $N, 0.05);
        xₖ = X[1, :]; ωₖ = findall(!iszero, xₖ)
        EΩ = zeros(L, length(ωₖ))
    end
suite[:loop_noalloc] = @benchmarkable begin
        @inbounds @views for (j_dest, j) in enumerate(ωₖ)
            rs = nzrange(X, j)
            for (k, X_val) in zip(rowvals(X)[rs], nonzeros(X)[rs])
                EΩ[:, j_dest] .+=  X_val .* D[:, k]
            end
        end
    end setup=begin
        D = rand($L, $K); X = sprand($K, $N, 0.05);
        xₖ = X[1, :]; ωₖ = findall(!iszero, xₖ)
        EΩ = zeros(L, length(ωₖ))
    end samples=10
suite[:loop_alloc] = @benchmarkable begin
        @inbounds @views for (j_dest, j) in enumerate(ωₖ)
            rs = nzrange(X, j)
            for (k, X_val) in zip(findnz(X[:, j])...)
                EΩ[:, j_dest] .+=  X_val .* D[:, k]  # this compiles to something similar to axpy!, i.e. no allocations. Notice we need the dot also for the scalar mul.
            end
        end
    end setup=begin
        D = rand($L, $K); X = sprand($K, $N, 0.05);
        xₖ = X[1, :]; ωₖ = findall(!iszero, xₖ)
        EΩ = zeros(L, length(ωₖ))
    end samples=10

tune!(suite); @info "Finished tuning."
res2 = run(suite)
