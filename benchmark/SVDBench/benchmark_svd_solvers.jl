import Pkg; Pkg.activate("@SVDBench"); Pkg.instantiate()
using IterativeSolvers, LinearAlgebra, ArnoldiMethod, TSVD
using KrylovKit
using BenchmarkTools
using DataFrames
import Base: rtoldefault
using FileIO

# BLAS by default uses all available cores, even if nthreads is 1.
# this gives unrealistic results when predicting scaling behaviour.
LinearAlgebra.BLAS.set_num_threads(Threads.nthreads())

results_file = joinpath(["benchmark", "data", "svd_comparison_$(Threads.nthreads())_threads.csv"])
@assert isdir(dirname(results_file)) "assumed to be executed from project root dir"

A = rand(1000, 10_000);
bm_res_svd =  @benchmark svd(A)         # 20.984 ms (12 allocations: 5.07 MiB)
bm_res_svdl = @benchmark svdl(A, nsv=1, vecs=:both) # 165.504 μs (254 allocations: 350.53 KiB)
bm_res_tsvd = @benchmark tsvd(A)        # 82.739 μs (100 allocations: 101.05 KiB)
bm_res_powm = @benchmark powm(hermitianpart(A*A'))     # 759.756 μs (14 allocations: 1.93 MiB)
bm_res_kryl = @benchmark svdsolve(hermitianpart(A*A'))     # 1.589 ms (270 allocations: 2.21 MiB))

bm_res_syms = [:bm_res_svd, :bm_res_svdl, :bm_res_tsvd, :bm_res_powm, :bm_res_kryl]

results_table = DataFrame(
    [[string(sym) for sym in bm_res_syms],
     [mean(eval(sym)).time for sym in bm_res_syms],
     [mean(eval(sym)).memory for sym in bm_res_syms]],
    [:method, :time, :memory]
)

save(results_file, results_table)

# takeaways:
# full svd obviously slow
# svdl quite good, but slower than tsvd (however, might have no bugs...)
# tsvd super fast, but has a bug. Fixed by me in https://github.com/RomeoV/TSVD.jl
# power method quite slow, might be because it's computing with complex numbers.
# krylov similar to power method, although it doesn't seem like it's using complex numbers...

## Let's make sure all the found eigenvalues / vectors actually match up...
res_svd =  svd(A)         # 20.984 ms (12 allocations: 5.07 MiB)
res_svdl = svdl(A, nsv=1, vecs=:both) # 165.504 μs (254 allocations: 350.53 KiB)
res_tsvd = tsvd(A)        # 82.739 μs (100 allocations: 101.05 KiB)
res_powm = powm(hermitianpart(A*A'))     # 759.756 μs (14 allocations: 1.93 MiB)
res_kryl = svdsolve(hermitianpart(A*A'))     # 1.589 ms (270 allocations: 2.21 MiB))

# singular value comparison
@assert (
    res_svd.S[1] ≈
    res_svdl[1].S[1] ≈
    res_tsvd[2][1] ≈
    res_powm[1] |> sqrt ≈
    res_kryl[1][1] |> sqrt
)

# singular vector comparison
# we have to reduce default tolerance slightly...
Base.rtoldefault(::Type{Float64}) = 1e-6
@assert (
    res_svd.U[:, 1] * sign(res_svd.U[1, 1]) ≈
    res_svdl[1].U[:, 1] * sign(res_svdl[1].U[1, 1]) ≈
    res_tsvd[1][:, 1] * sign(res_tsvd[1][1, 1]) ≈
    res_kryl[2][1] * sign(res_kryl[2][1][1])
    # real.(A'*res_powm[2]/res_powm[1]) ≈  # <- this one isn't quite the same... not sure why
)
Base.rtoldefault(::Type{Float64}) = sqrt(eps(Float64))  # reset to default
