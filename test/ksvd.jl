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

        d, n = 100, 1000
        m = 2 * d
        data = rand(rng, T, d, n)
        D = KSVD.init_dictionary(rng, T, d, m)
        nnz_per_col = 10
        X = KSVD.init_sparse_assignment_mat(T, m, n, nnz_per_col)

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
            KSVD.maybe_init_buffers!(method, d, m, n)
            D_res, X_res = ksvd_update(method, data, copy(D), copy(X))

            @test D_res ≈ D_baseline rtol = sqrt(method.svd_solver.tol)
            @test X_res ≈ X_baseline rtol = sqrt(method.svd_solver.tol)

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
            KSVD.maybe_init_buffers!(method, d, m, n)
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
            KSVD.maybe_init_buffers!(method, d, m, n)
            D_res, X_res = ksvd_update(method, data, copy(D), copy(X))

            @test D_res ≈ D_baseline rtol = sqrt(eps(T)) skip = true
            @test X_res ≈ X_baseline rtol = sqrt(eps(T)) skip = true

            @test all(≈(1.0), norm.(eachcol(D_res)))
            @test eltype(D_res) == eltype(X_res) == T
        end
    end

    @testset "Compare SVD implementations" begin
        using KSVD, Random, StatsBase, SparseArrays, LinearAlgebra
        d, m = 2 * 64, 2 * 256
        n = nsamples = 10_000
        nnzpercol = 5
        T = Float32

        D = KSVD.init_dictionary(Float32, d, m)
        X = KSVD.init_sparse_assignment_mat(Float32, m, nsamples, nnzpercol)
        Y = D * X
        ycolmean = mean(norm.(eachcol(Y)))
        # we want the noise to be about some percentage ϵ of the signal.
        # we can recall that `norm(σ*randn(d)) ≈ σ*sqrt(d)`.
        # shockingly, we can recover the true x adjacency indices for noise as large as the signal!
        ϵ = 0.1
        σ = ϵ * ycolmean / sqrt(d)
        Y .+= σ * randn!(similar(Y))

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
        D_init = KSVD.init_dictionary(Float32, d, m)
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
