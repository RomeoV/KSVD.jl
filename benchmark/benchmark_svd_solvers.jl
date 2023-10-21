import Pkg; Pkg.activate("@SVDBench")
using IterativeSolvers, LinearAlgebra, ArnoldiMethod, TSVD
using KrylovKit
using BenchmarkTools


A = rand(500, 300);
res_svd =  @btime svd(A)         # 20.984 ms (12 allocations: 5.07 MiB)
res_svdl = @btime svdl(A, nsv=1, vecs=:both) # 165.504 μs (254 allocations: 350.53 KiB)
res_tsvd = @btime tsvd(A)        # 82.739 μs (100 allocations: 101.05 KiB)
res_powm = @btime powm(A*A')     # 759.756 μs (14 allocations: 1.93 MiB)
res_kryl = @btime svdsolve(A*A')     # 1.589 ms (270 allocations: 2.21 MiB)

"singular value comparison"
@assert (
    res_svd.S[1] ≈
    res_svdl[1][1] ≈
    res_tsvd[2][1] ≈
    real(res_powm[1] |> sqrt)
)

"singular vector comparison"
res_svd.U[:, 1]
res_svdl[1].U[:, 1]
res_tsvd[1][:, 1]
real.(A'*res_powm[2]/res_powm[1])  # <- this one isn't quite the same...

B = rand(5000, 3000);
res2_svdl = @btime svdl(B, nsv=1, vecs=:both) # 165.504 μs (254 allocations: 350.53 KiB)
res2_tsvd = @btime tsvd(B)        # 82.739 μs (100 allocations: 101.05 KiB)
