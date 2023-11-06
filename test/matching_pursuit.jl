import Random: seed!, TaskLocalRNG


@testset "Compare Matching Pursuit implementations" begin
    rng = TaskLocalRNG();
    seed!(rng, 1)

    T = Float64
    D, N = 100, 1000
    data = rand(rng, D, N)
    B = KSVD.init_dictionary(T, D, 2*D)

    res_baseline = KSVD.sparse_coding(KSVD.LegacyMatchingPursuit(), data, B)

    if Threads.nthreads == 1
        @warn "Parallel implementation will only be tested properly if test is launched with multiple threads!"
    end
    @testset for method in [KSVD.MatchingPursuit(),
                            KSVD.ParallelMatchingPursuit(),
                            KSVD.FasterParallelMatchingPursuit(),
                            KSVD.CUDAAcceleratedMatchingPursuit(batch_size=300) # batch_size<N and batch_size÷N != 0 for test
                            ]

        res = KSVD.sparse_coding(method, data, B)
        @test res ≈ res_baseline
    end
    @testset for method in [KSVD.FullBatchMatchingPursuit,]
        res = KSVD.sparse_coding(method(), data, B)
        @test res ≈ res_baseline broken=true
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
