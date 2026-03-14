using KSVD
using LinearAlgebra
# using ReTest
using Test
using SparseArrays
import SparseArrays.nzvalview
import OhMyThreads
using Random

@testset "Test ksvd algo" begin
    include("ksvd.jl")
end
@testset "Test matching pursuit algo" begin
    include("matching_pursuit.jl")
end

@testset "Run the whole thing" begin
    data = rand(100, 500)
    ksvd(data, 200)
    @test true
end

@testset "Non-negative coefficients" begin
    Random.seed!(42)
    T = Float32

    @testset "Sparse coding produces non-negative coefficients" begin
        d, m, n = 64, 128, 200
        D = KSVD.init_dictionary(T, d, m)
        # Ground truth with positive coefficients (init_sparse_assignment_mat produces values in [1, 2))
        Xtrue = KSVD.init_sparse_assignment_mat(T, m, n, 3)
        Y = D * Xtrue
        DtD = D' * D

        @testset for method in [
            ParallelMatchingPursuit(; max_nnz=3, refit_coeffs=false),
            ParallelMatchingPursuit(; max_nnz=3, refit_coeffs=true),
            KSVD.MatchingPursuit(; max_nnz=3, refit_coeffs=false),
            OrthogonalMatchingPursuit(; max_nnz=3),
            LegacyMatchingPursuit(; max_nnz=3),
        ]
            Xest = sparse_coding(method, Y, D; DtD, nonneg=true)
            @test all(>=(0), nonzeros(Xest))
        end
    end

    @testset "Full ksvd with nonneg=true" begin
        data = rand(T, 32, 256)
        result = ksvd(data, 64; maxiters=5, nonneg=true)
        @test all(>=(0), nonzeros(result.X))
    end

    @testset "Nonneg feature grid" begin
        data = rand(T, 32, 256)
        @testset for ksvd_update_method in [LegacyKSVD(), BatchedParallelKSVD{false,T}()],
            sparse_coding_method in [
                ParallelMatchingPursuit(; max_nnz=3),
                OrthogonalMatchingPursuit(; max_nnz=3),
            ]

            result = ksvd(data, 64;
                sparse_coding_method, ksvd_update_method,
                maxiters=3, nonneg=true)
            @test all(>=(0), nonzeros(result.X))
        end
    end
end

@testset "Test feature grid" begin
    T = Float32
    ksvd_update_methods = [
        LegacyKSVD(),
        BatchedParallelKSVD{true,T}(),
        BatchedParallelKSVD{false,T}(),
        BatchedParallelKSVD{false,Float32,OhMyThreads.Schedulers.DynamicScheduler,KSVD.TSVDSolver{T}}(),
    ]
    minibatch_sizes = [
        nothing, 100
    ]
    ksvd_loop_types = [
        NormalLoop(),
        MatryoshkaLoop(; log2min=1)
    ]
    sparse_coding_methods = [
        LegacyMatchingPursuit(; max_nnz=3),
        ParallelMatchingPursuit(; max_nnz=3),
        ParallelMatchingPursuit(; max_nnz=3, refit_coeffs=true),
        OrthogonalMatchingPursuit(; max_nnz=3),
    ]

    data = rand(Float32, 32, 256)
    @testset for ksvd_update_method in ksvd_update_methods, minibatch_size in minibatch_sizes,
        ksvd_loop_type in ksvd_loop_types, sparse_coding_method in sparse_coding_methods

        ksvd(data, 64;
            sparse_coding_method, ksvd_update_method,
            ksvd_loop_type, minibatch_size,
            maxiters=5)
        @test true
    end
end

@testset "Test utils" begin
    @testset "Test fast sparse hcat" begin
        X = sprand(Float32, 128, 2048, 0.01)
        vecs = [X[:, i] for i in axes(X, 2)]
        X_rec = reduce(hcat, vecs)
        @test X == X_rec
    end

    @testset "Test copy X" begin
        X = sprand(Float32, 128, 2048, 0.01)
        X_cpy = copy(X)
        rand!(SparseArrays.nzvalview(X_cpy))
        index_set = 12:30

        KSVD.ksvd_update_X!(X, X_cpy, index_set)
        @test X[index_set, :] == X_cpy[index_set, :]


        R = X.rowval
        Rsorted = sort(R)
        Rsortperm = sortperm(R)

        rand!(SparseArrays.nzvalview(X_cpy))
        KSVD.ksvd_update_X!(X, X_cpy, index_set, Rsorted, Rsortperm)
        @test X[index_set, :] == X_cpy[index_set, :]

        # index set may be non-continuous and shuffled
        index_set = [15, 10, 21]
        rand!(SparseArrays.nzvalview(X_cpy))
        KSVD.ksvd_update_X!(X, X_cpy, index_set, Rsorted, Rsortperm)
        @test X[index_set, :] == X_cpy[index_set, :]
        @test all(nzvalview(X[setdiff(axes(X, 1), index_set), :]) .!= nzvalview(X_cpy[setdiff(axes(X, 1), index_set), :]))
    end
end
