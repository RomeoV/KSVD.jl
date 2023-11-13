import LinearAlgebra: norm
import SparseArrays: sprand
import Random: TaskLocalRNG, seed!

@testset "Compare to Legacy Implementation." begin
    if Threads.nthreads == 1
        @warn "Parallel implementation will only be tested properly if test is launched with multiple threads!"
    end

    @testset for T in [Float64, Float32]
        rng = TaskLocalRNG();
        seed!(rng, 1)

        E, N = 100, 1000
        data = rand(rng, T, E, N)
        D = KSVD.init_dictionary(rng, T, E, 2*E)
        X = sprand(rng, T, 2*E, N, 0.1)

        D_baseline, X_baseline = KSVD.ksvd(KSVD.LegacyKSVD(), data, copy(D), copy(X))
        signs1_baseline = sign.(eachcol(D_baseline) .|> first)
        @test all(≈(1.), norm.(eachcol(D_baseline)))
        @test eltype(D_baseline) == eltype(X_baseline) == T

        # The solution is only unique up to the sign of each factor / the direction of each basis vector.
        # Therefore, we multiply each basis vector and factor with the sign of it's first element.
        # That forces the first element to be positive, and makes the solution unique and let's us run the comparison.
        @testset for method in [KSVD.OptimizedKSVD(shuffle_indices=false), ]
            D_res, X_res = KSVD.ksvd(method, data, copy(D), copy(X))
            signs1_res = sign.(eachcol(D_res) .|> first)
            @test all(≈(1.), norm.(eachcol(D_res)))
            @test D_res.*signs1_res' ≈ D_baseline.*signs1_baseline' rtol=10*sqrt(eps(T))
            @test X_res.*signs1_res ≈ X_baseline.*signs1_baseline rtol=10*sqrt(eps(T))
            @test eltype(D_res) == eltype(X_res) == T
        end
        # We don't expect to get the same results after one iteration for out-of-order operations.
        # We will test later for convergence though.
        @testset for method in [KSVD.OptimizedKSVD(shuffle_indices=true),
                                KSVD.ParallelKSVD(Float64, E, 2*E, N),
                                KSVD.ParallelKSVD(Float64, E, 2*E, N),
                                KSVD.BatchedParallelKSVD(Float64, E, 2*E, N; batch_size_per_thread=1),
                                KSVD.BatchedParallelKSVD(Float64, E, 2*E, N; batch_size_per_thread=4),
                                ]
            D_res, X_res = KSVD.ksvd(method, data, copy(D), copy(X))
            signs1_res = sign.(eachcol(D_res) .|> first)
            @test all(≈(1.), norm.(eachcol(D_res)))
            @test D_res.*signs1_res' ≈ D_baseline.*signs1_baseline' rtol=10*sqrt(eps(T)) skip=true;
            @test X_res.*signs1_res ≈ X_baseline.*signs1_baseline rtol=10*sqrt(eps(T)) skip=true;
            @test eltype(D_res) == eltype(X_res) == T
        end
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
