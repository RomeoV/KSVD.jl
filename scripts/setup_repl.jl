# useful to create some dummy vectors to play around with.
Pkg.activate(".")
using KSVD
import Random: seed!, TaskLocalRNG, rand!
import SparseArrays
import SparseArrays: sparse
import StatsBase: sample
using ProfileView, BenchmarkTools
T = Float64
N = 100_000
nnz=10

rng = TaskLocalRNG();
seed!(rng, 1)
D = 1000  # sample dimension
# nnz = 10  # num nonzeros
K = 2000  # dictionary dimension >= sample_dimension
basis = KSVD.init_dictionary(rng, T, D, K)
Is = vcat([sample(rng, 1:K, nnz, replace=false) for _ in 1:N]...);
Js = vcat([fill(j, nnz) for j in 1:N]...)
Vs = rand!(rng, similar(Js, T)) .+ 1
X_true = sparse(Is, Js, Vs, K, N)
Y = basis * X_true
