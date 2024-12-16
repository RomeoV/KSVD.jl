# import Pkg; Pkg.activate("@SVDBench"); Pkg.instantiate()
import LinearAlgebra: svd, normalize!
import ArnoldiMethod: partialschur, partialeigen
import TSVD: tsvd
import IterativeSolvers: svdl
import KrylovKit: svdsolve
import Arpack: svds
import Chairmarks: @be, median
import DataFrames: DataFrame
using CSVFiles, FileIO

import LinearAlgebra
using LinearAlgebra: mul!, eltype, size

"To compute svd(A) = eigen(A*A'), but without doing the multiplication."
struct LazySym{A_t, V_t, T}
    A::A_t
    z::V_t
    LazySym(A) = new{typeof(A), Vector{eltype(A)}, eltype(A)}(
        A, zeros(eltype(A), size(A,2))
    )
end
Base.size(S::LazySym) = (size(S.A, 1), size(S.A, 1))
Base.size(S::LazySym, i) = size(S)[i]
Base.eltype(S::LazySym) = eltype(S.A)
"Computes A * A', lazily."
LinearAlgebra.mul!(y, S::LazySym, x) = let
    mul!(S.z, S.A', x)
    mul!(y, S.A, S.z)
end

# BLAS by default uses all available cores, even if nthreads is 1.
# this gives unrealistic results when predicting scaling behaviour.
LinearAlgebra.BLAS.set_num_threads(1)


tol = 1e-10
check_approx(v1_v2) = check_approx(v1_v2[1], v1_v2[2])
check_approx(v1, v2; tol=tol) = let
    isapprox(v1, v2; atol=1e-4) || isapprox(-v1, v2; atol=1e-4)
end

skip_check = true
results = DataFrame([String[], DataType[], Float64[], Int64[]],
                    [:method, :eltype, :median_runtime, :m])

for T in [Float64, Float32], m in Int.(exp10.(1:5))
    @info T, m
    make_A_v(; tol=tol) = let
        A = randn(T, 1_000, m); normalize!.(eachcol(A))
        v = tsvd(Float64.(A), 1; tolconv=tol*1e-2)[1]
        (A, v)
    end
    #
    if m < 5e4
        bm_res_svdl = (@be make_A_v (A, v)::Tuple->(
                A, svdl(A; nsv=1, vecs=:left, tol=tol)[1].U, v
            ) $skip_check || check_approx(_[2:3]) || error() seconds=1) |> median
        push!(results, (; method="IterativeSolvers.svdl", eltype=T, median_runtime=bm_res_svdl.time, m))
    end
    bm_res_arpack = (@be make_A_v (A, v)::Tuple->(
            A, svds(A; nsv=1, ritzvec=true, tol=tol)[1].U, v
        ) $skip_check || check_approx(_[2:3]) || error() seconds=1) |> median
    push!(results, (; method="Arpack.svds", eltype=T, median_runtime=bm_res_arpack.time, m))
    bm_res_tsvd = (@be make_A_v (A, v)::Tuple->(
            tsvd(A, 1; tolconv=tol)[1], v
        ) $skip_check || check_approx(_) || error() seconds=1) |> median
    push!(results, (; method="TSVD.tsvd", eltype=T, median_runtime=bm_res_tsvd.time, m))
    bm_res_kryl = (@be make_A_v (A, v)::Tuple->(
            svdsolve(A, 1; tol)[2][1], v
        ) $skip_check || check_approx(_) || error() seconds=1 ) |> median
    push!(results, (; method="KrylovKit.svdsolve", eltype=T, median_runtime=bm_res_kryl.time, m))
    bm_res_arnoldi = (@be make_A_v (A, v)::Tuple->(
            # partialeigen(partialschur(A*A'; nev=1, tol)[1])[2], v
            partialschur(A*A'; nev=1, tol)[1].Q, v
        ) $skip_check || check_approx(_) || error() seconds=1 ) |> median
    push!(results, (; method="ArnoldiMethod.partialeigen", eltype=T, median_runtime=bm_res_arnoldi.time, m))
    bm_res_lazy_arnoldi = (@be make_A_v (A, v)::Tuple->(
            partialschur(LazySym(A); nev=1, tol)[1].Q, v
        ) $skip_check || check_approx(_) || error() seconds=1 ) |> median
    push!(results, (; method="ArnoldiMethod.partialeigen (lazy)", eltype=T, median_runtime=bm_res_lazy_arnoldi.time, m))
end

save("bmres.csv", results)
