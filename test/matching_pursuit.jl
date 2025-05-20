import Random: seed!, TaskLocalRNG, rand!
import KSVD.Accessors: @set
import SparseArrays
import SparseArrays: sparse, findnz, nnz, nonzeroinds
import StatsBase: sample, mean, middle
import LinearAlgebra: norm
import Random: randn!
test_cuda_ext = try
    using CUDA
    true
catch
    false
end

@testset "Check 'correctness'" begin
    # We can recall from our paper that small `k` compared to a large `d` (k = O(sqrt(d))) we can
    # have recoverability. Further, in the noise free case, we can hope for perfect recovery.
    # In the noisy case we may not have perfect value recovery, but still can get the adjacencies right.
    sparse_coding_methods = [
        KSVD.MatchingPursuit(; refit_coeffs=true),
        ParallelMatchingPursuit(; refit_coeffs=true),
        OrthogonalMatchingPursuit(;),
    ]
    test_cuda_ext && append!(sparse_coding_methods, CUDAAccelMatchingPursuit())
    rng = TaskLocalRNG()
    seed!(rng, 1)
    d = 1000  # sample dimension
    n = 100  # num samples
    m = 2 * d  # dictionary dimension >= sample_dimension
    @testset for T in [Float32, Float64]
        D = KSVD.init_dictionary(rng, T, d, m)
        DtD = D' * D
        @testset "Noise free case" begin
            @testset for k in [3, 10]  # Note that this doesn't pass for larger nnz...
                @testset for sparse_coding_method in sparse_coding_methods
                    Xtrue = KSVD.init_sparse_assignment_mat(T, m, n, k)
                    Y = D * Xtrue
                    sparse_coding_method = @set sparse_coding_method.max_nnz = k
                    Xest = sparse_coding(sparse_coding_method, Y, D; DtD)
                    @test Xtrue ≈ Xest
                end
            end
        end
        @testset "Noisy case" begin
            @testset for k in [3, 10]  # Note that this doesn't pass for larger nnz...
                @testset for ϵ in [0.1, 0.3, 0.5, 0.8]
                    @testset for sparse_coding_method in sparse_coding_methods
                        Xtrue = KSVD.init_sparse_assignment_mat(T, m, n, k)
                        Y = D * Xtrue
                        ycolmean = mean(norm.(eachcol(Y)))
                        # we want the noise to be about some percentage ϵ of the signal.
                        # we can recall that `norm(σ*randn(d)) ≈ σ*sqrt(d)`.
                        # shockingly, we can recover the true x adjacency indices for noise as large as the
                        # signal!
                        σ = ϵ * ycolmean / sqrt(d)
                        Y .+= σ * randn!(similar(Y))
                        sparse_coding_method = @set sparse_coding_method.max_nnz = k
                        Xest = sparse_coding(sparse_coding_method, Y, D; DtD)
                        @test findall(!iszero, Xest) == findall(!iszero, Xtrue)
                    end
                end
            end
        end
    end
end

@testset "permute_D_X! test" begin
    rng = TaskLocalRNG()
    seed!(rng, 1)
    d = 1000  # sample dimension
    n = 10_000  # num samples
    m = 1200  # dictionary dimension >= sample_dimension
    D = KSVD.init_dictionary(rng, T, d, m)

    @testset "D as reference" begin
        X = KSVD.init_sparse_assignment_mat(T, m, n, k)
        for k in 3:10  # Note that this doesn't pass for larger nnz...
            idxshuf = randperm(size(D, 2))
            Dperm = D[:, idxshuf]
            Xperm = X[idxshuf, :]
            λs = rand((-1, 1), size(Dperm, 2))
            eachcol(Dperm) .*= λs
            Xperm .*= reshape(λs, :, 1)

            permute_D_X!(Dperm, Xperm, D)
            @test Dperm ≈ D
            @test Xperm ≈ X
        end
    end

    @testset "X as reference" begin
        X = KSVD.init_sparse_assignment_mat(T, m, n, k)
        for k in 3:10  # Note that this doesn't pass for larger nnz...
            idxshuf = randperm(size(D, 2))
            Dperm = D[:, idxshuf]
            Xperm = X[idxshuf, :]
            λs = rand((-1, 1), size(Dperm, 2))
            eachcol(Dperm) .*= λs
            Xperm .*= reshape(λs, :, 1)

            permute_D_X!(Dperm, Xperm, X)
            @test Dperm ≈ D
            @test Xperm ≈ X
        end
    end
end
