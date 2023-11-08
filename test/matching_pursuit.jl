import Random: seed!, TaskLocalRNG, rand!
import SparseArrays
import SparseArrays: sparse
import StatsBase: sample

@testset "Check 'correctness'" begin
    # this test is actually very wonky and I'm not sure about the theory.
    # For a very small number of true non-zero values, it seems we can make an algorithm that usually picks
    # the correct basis and then can compute the coefficients exactly.
    # However, for nnz >10 or so, it already performs much worse.
    # It's definitely important to have a large dimensionality, so that the chance of two basis-vectors being "exchangable" is very low.
    @testset for T in [Float32, Float64]
        @testset for nnz in [3, 10]  # Note that this doesn't pass for larger nnz...
            rng = TaskLocalRNG();
            seed!(rng, 1)
            D = 1000  # sample dimension
            N = 100  # num samples
            # nnz = 10  # num nonzeros
            K = 1200  # dictionary dimension >= sample_dimension
            basis = KSVD.init_dictionary(rng, T, D, K)
            Is = vcat([sample(rng, 1:K, nnz, replace=false) for _ in 1:N]...);
            Js = vcat([fill(j, nnz) for j in 1:N]...)
            Vs = rand!(rng, similar(Js, T)) .+ 1
            X_true = sparse(Is, Js, Vs, K, N)
            Y = basis * X_true

            for _ in 1:10
                # pick the sparse assignments
                X_recovered = KSVD.sparse_coding(KSVD.LegacyMatchingPursuit(max_nnz=nnz, max_iter=typemax(Int), tolerance=0), Y, basis)

                # update the coefficients after the basis is picked
                for (i, col) in enumerate(eachcol(X_recovered))
                    inds = SparseArrays.nonzeroinds(col)
                    local_basis = basis[:, inds]
                    coeffs = local_basis \ Y[:, i]
                    X_recovered[inds, i] .= coeffs
                end

                # For large D, the basis vectors should be sufficiently different that this method above mostly works.
                @test SparseArrays.nnz(abs.(X_recovered - X_true) .> 100*sqrt(eps(T))) == 0
            end
        end
    end
end

@testset "Compare to Legacy Implementation." begin
    @testset for T in [Float64, Float32]
        rng = TaskLocalRNG();
        seed!(rng, 1)

        D, N = 100, 1000
        data = rand(rng, T, D, N)
        B = KSVD.init_dictionary(rng, T, D, 2*D)

        res_baseline = KSVD.sparse_coding(KSVD.LegacyMatchingPursuit(), data, B)
        @test eltype(res_baseline) == T

        if Threads.nthreads == 1
            @warn "Parallel implementation will only be tested properly if test is launched with multiple threads!"
        end
        @testset for method in [KSVD.MatchingPursuit(),
                                KSVD.ParallelMatchingPursuit(),
                                KSVD.FasterParallelMatchingPursuit(),
                                KSVD.CUDAAcceleratedMatchingPursuit(batch_size=300) # batch_size<N and batch_size÷N != 0 for test
                                ]

            res = KSVD.sparse_coding(method, data, B)
            @test res ≈ res_baseline rtol=sqrt(eps(T))
            @test eltype(res) == T
        end
        @testset for method in [KSVD.FullBatchMatchingPursuit,]
            res = KSVD.sparse_coding(method(), data, B)
            @test res ≈ res_baseline broken=true
            @test eltype(res) == T
        end
    end
end

# D = [
# 	-5  -9  -9   7   0;
# 	 7   0   4   0   9;
# 	 8   0   5   4  -6
# ]


# # when data is representad as a vector
# Y = [
#     -1;
#      0;
#      5
# ]

# X = matching_pursuit(Y, D)
# @test norm(Y-D*X) < 1e-6


# # when data is representad as a matrix
# Y = [
#     -1  2;
#      0  3;
#      5 -5
# ]

# X = matching_pursuit(Y, D)
# @test norm(Y-D*X) < 1e-6


# # when max_iter < 0
# @test_throws ArgumentError matching_pursuit(Y, D; max_iter = 0)
# @test_throws ArgumentError matching_pursuit(Y, D; max_iter = -1)


# # when tolerance < 0
# @test_throws ArgumentError matching_pursuit(Y, D, tolerance = 0.)
# @test_throws ArgumentError matching_pursuit(Y, D; tolerance = -1.0)


# # when dimensions of data and atoms do not match

# Y = [
#      2;
#     -5
# ]

# @test_throws ArgumentError matching_pursuit(Y, D)
