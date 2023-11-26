using KSVD
using LinearAlgebra
# using ReTest
using Test
try using CUDA
catch end

@testset "Test ksvd algo" begin
    include("ksvd.jl")
end
@testset "Test matching pursuit algo" begin
    include("matching_pursuit.jl")
end
