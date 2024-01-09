using KSVD
using LinearAlgebra
# using ReTest
using Test

@testset "Test ksvd algo" begin
    include("ksvd.jl")
end
@testset "Test matching pursuit algo" begin
    include("matching_pursuit.jl")
end

@testset "Run the whole thing" begin
    data = rand(100, 500)
    dictionary_learning(data, 200)
end
