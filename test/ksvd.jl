import LinearAlgebra: norm
import SparseArrays: sprand
import Random: TaskLocalRNG, seed!
import StatsBase: std
import OhMyThreads

function allapproxequal(xs; kwargs...)
    isempty(xs) && return true
    x1 = first(xs)
    all(x -> isapprox(x, x1; kwargs...), xs)
end


@testset "Compare to Legacy Implementation." begin
    if Threads.nthreads == 1
        @warn "Parallel implementation will only be tested properly if test is launched with multiple threads!"
    end

    @testset for T in [Float64, Float32]
        rng = TaskLocalRNG()
        seed!(rng, 1)

        E, N = 100, 1000
        data = rand(rng, T, E, N)
        D = KSVD.init_dictionary(rng, T, E, 2 * E)
        nnz_per_col = 10
        X = sprand(rng, T, 2 * E, N, nnz_per_col * N / (2 * E * N))

        D_baseline, X_baseline = ksvd_update(KSVD.LegacyKSVD(), data, copy(D), copy(X))
        @test all(≈(1.0), norm.(eachcol(D_baseline)))
        @test eltype(D_baseline) == eltype(X_baseline) == T

        # The following tests will only pass if we're running single-threaded...
        KSVD.set_num_threads(1)
        # The solution is only unique up to the sign of each factor / the direction of each basis vector.
        # Therefore, we multiply each basis vector and factor with the sign of it's first element.
        # That forces the first element to be positive, and makes the solution unique and let's us run the comparison.
        @testset for method in [
            KSVD.OptimizedKSVD(shuffle_indices=false),
            # KSVD.ParallelKSVD{true, T}(shuffle_indices=false),  These are not equivalent, as they do a sort of "full-batch" update.
            # KSVD.ParallelKSVD{false, T}(shuffle_indices=false),

            KSVD.BatchedParallelKSVD{true,T}(shuffle_indices=false, batch_size_per_thread=1),
            KSVD.BatchedParallelKSVD{false,T}(shuffle_indices=false, batch_size_per_thread=1),
        ]
            KSVD.maybe_init_buffers!(method, E, 2 * E, N)
            D_res, X_res = ksvd_update(method, data, copy(D), copy(X))

            @test D_res ≈ D_baseline rtol = sqrt(eps(T))
            @test X_res ≈ X_baseline rtol = sqrt(eps(T))

            @test all(≈(1.0), norm.(eachcol(D_res)))
            @test eltype(D_res) == eltype(X_res) == T
        end

        # KSVD.set_num_threads(Threads.nthreads())
        # We don't expect to get the same results after one iteration for out-of-order operations.
        # Instead we test if they do approximately as well as the baseline.
        @testset for method in [
            KSVD.BatchedParallelKSVD{true,T}(shuffle_indices=true, batch_size_per_thread=1),
            KSVD.BatchedParallelKSVD{false,T}(shuffle_indices=true, batch_size_per_thread=1),
        ]
            KSVD.maybe_init_buffers!(method, E, 2 * E, N)
            D_res, X_res = ksvd_update(method, data, copy(D), copy(X))

            @test D_res ≈ D_baseline rtol = sqrt(eps(T)) skip = true
            @test X_res ≈ X_baseline rtol = sqrt(eps(T)) skip = true
            @test std(data - D_res * X_res) ≈ std(data - D_baseline * X_baseline) rtol = 1e-2

            @test all(≈(1.0), norm.(eachcol(D_res)))
            @test eltype(D_res) == eltype(X_res) == T
        end

        # Finally we test the parallel version, which has much worse convergence.
        @testset for method in [
            KSVD.ParallelKSVD{false,T}(shuffle_indices=false),
            KSVD.ParallelKSVD{true,T}(shuffle_indices=false),
        ]
            KSVD.maybe_init_buffers!(method, E, 2 * E, N)
            D_res, X_res = ksvd_update(method, data, copy(D), copy(X))

            @test D_res ≈ D_baseline rtol = sqrt(eps(T)) skip = true
            @test X_res ≈ X_baseline rtol = sqrt(eps(T)) skip = true

            @test all(≈(1.0), norm.(eachcol(D_res)))
            @test eltype(D_res) == eltype(X_res) == T
        end
    end

    @testset "Compare SVD implementations" begin
        using KSVD, Random, StatsBase, SparseArrays, LinearAlgebra
        m, n = 2 * 64, 2 * 256
        nsamples = 10_000
        nnzpercol = 5
        T = Float32

        D = rand(Float32, m, n)
        X = stack(
            (SparseVector(n, sort(sample(1:n, nnzpercol; replace=false)), rand(T, nnzpercol))
             for _ in 1:nsamples);
            dims=2)
        Y = D * X + T(0.05) * randn(T, size(D * X))

        idx = findall(!iszero, X[1, :])
        Y_ = Y[:, idx]

        (U_tsvd, S_tsvd, V_tsvd) = KSVD.compute_truncated_svd(KSVD.TSVDSolver{Float32}(), Y[:, idx], 1)
        λ = sign.(U_tsvd[1, :])
        U_tsvd *= λ
        V_tsvd *= λ'

        (U_kryl, S_kryl, V_kryl) = KSVD.compute_truncated_svd(KSVD.KrylovSVDSolver{Float32}(), Y[:, idx], 1)
        λ = sign.(U_kryl[1, :])
        U_kryl *= λ
        V_kryl *= λ'

        (U_arno, S_arno, V_arno) = KSVD.compute_truncated_svd(KSVD.ArnoldiSVDSolver{Float32}(), Y[:, idx], 1)
        λ = sign.(U_arno[1, :])
        U_arno *= λ
        V_arno *= λ'

        (U_arpa, S_arpa, V_arpa) = KSVD.compute_truncated_svd(KSVD.ArpackSVDSolver{Float32}(), Y[:, idx], 1)
        λ = sign.(U_arpa[1, :])
        U_arpa *= λ
        V_arpa *= λ'
        @test allapproxequal([U_tsvd, U_kryl, U_arno, U_arpa])
        @test allapproxequal([S_tsvd, S_kryl, S_arno, S_arpa])
        @test allapproxequal([V_tsvd, V_kryl, V_arno, V_arpa])


        ksvd_update_method_tsvd = BatchedParallelKSVD{false,T,OhMyThreads.DynamicScheduler,KSVD.TSVDSolver{T}}(; shuffle_indices=false, batch_size_per_thread=1)
        ksvd_update_method_kryl = BatchedParallelKSVD{false,T,OhMyThreads.DynamicScheduler,KSVD.KrylovSVDSolver{T}}(; shuffle_indices=false, batch_size_per_thread=1)
        ksvd_update_method_arno = BatchedParallelKSVD{false,T,OhMyThreads.DynamicScheduler,KSVD.ArnoldiSVDSolver{T}}(; shuffle_indices=false, batch_size_per_thread=1)
        # Arpack crashes when called from multiple threads at once!
        # ksvd_update_method_arpa = BatchedParallelKSVD{false,T,OhMyThreads.DynamicScheduler,KSVD.ArpackSVDSolver{T}}(; shuffle_indices=false, batch_size_per_thread=1)
        D_init = KSVD.init_dictionary(Float32, m, n)
        Random.seed!(42)
        (D_tsvd, X_tsvd) = ksvd(Y, n, nnzpercol; ksvd_update_method=ksvd_update_method_tsvd, maxiters=10, D_init=copy(D_init))
        Random.seed!(42)
        (D_kryl, X_kryl) = ksvd(Y, n, nnzpercol; ksvd_update_method=ksvd_update_method_kryl, maxiters=10, D_init=copy(D_init))
        Random.seed!(42)
        (D_arno, X_arno) = ksvd(Y, n, nnzpercol; ksvd_update_method=ksvd_update_method_arno, maxiters=10, D_init=copy(D_init))
        # (D_arpa, X_arpa) = ksvd(Y, n, nnzpercol; ksvd_update_method=ksvd_update_method_arpa, maxiters=10, D_init)
        @test mean(norm.(eachcol(Y - D_tsvd * X_tsvd)) ./ norm.(eachcol(Y))) ≈ mean(norm.(eachcol(Y - D_kryl * X_kryl)) ./ norm.(eachcol(Y))) atol = 0.005
        @test mean(norm.(eachcol(Y - D_tsvd * X_tsvd)) ./ norm.(eachcol(Y))) ≈ mean(norm.(eachcol(Y - D_arno * X_arno)) ./ norm.(eachcol(Y))) atol = 0.005
        # @test mean(norm.(eachcol(Y - D_tsvd * X_tsvd)) ./ norm.(eachcol(Y))) ≈ mean(norm.(eachcol(Y - D_arpa * X_arpa)) ./ norm.(eachcol(Y))) atol = 0.005
    end
end

## # basic one
## Y = [
##      0  3;
##      1  0;
##      1 -3;
##     -2  3;
##      0  0;
## ]
##
## D, X = ksvd(Y, 8, max_iter_mp = 600, sparsity_allowance = 1.0)
## @test norm(Y-D*X) < 1e-4
##
## # sparsity_allowance must be > 0
## @test_throws ArgumentError ksvd(Y, 20, sparsity_allowance = -0.1)
## @test_throws ArgumentError ksvd(Y, 20, sparsity_allowance = 1.1)
##
## # should work normally
## ksvd(Y, 20, max_iter = 1)
## ksvd(Y, 20, sparsity_allowance = 0.0)
## ksvd(Y, 20, sparsity_allowance = 1.0)
##
## # But should work well when size(Y, 1) == n_atoms
## Y = [
##     -1 1 2;
##      1 0 1
## ]
## D, X = ksvd(Y, 2, max_iter_mp = 4000)
## @test norm(Y-D*X) < 0.001  # relax the constraint since the dictionary is small
##
## # Return only if X is sparse enough
## Y = [
## 	0  2  3 -1  1;
## 	1 -3  1  3  0
## ]
##
## # More than 20% of elements in X must be zeros
## sparsity_allowance = 0.2
## D, X = ksvd(Y, 5, max_iter = Int(1e10), sparsity_allowance = sparsity_allowance)
## @test sum(iszero, X) / length(X) > sparsity_allowance
##
##
## # compare error_maxtrix and error_matrix2
## Y, D, X = rand(30, 30), rand(30, 20), rand(20, 30);
## KSVD.error_matrix(Y, D, X, 10) ≈ KSVD.error_matrix2(Y, D, X, 10)
##
## # compare with tullio approach
## Eₖ = Y - D * X
## for k in 1:size(X,1)
##     # let Xₖ = (@view X[k, :]), Dₖ = (@view D[:, k])
##     @tullio Eₖ[i, j] += D[i, $k] * X[$k, j]
##     # end
##     @test Eₖ ≈ KSVD.error_matrix2(Y, D, X, k)
##     D[:, k] = rand(size(D, 1))
##     X[k, :] = rand(size(X, 2))
##     # let Xₖ = (@view X[k, :]), Dₖ = (@view D[:, k])
##     @tullio Eₖ[i, j] += -D[i, $k] * X[$k, j]
##     # end
## end
