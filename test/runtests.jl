using KSVD
using LinearAlgebra
# using ReTest
using Test
using SparseArrays
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
    end
end
