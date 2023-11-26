""" The goal is to compute
E = Y - DX[idx, :]
as fast as possible, where X is sparse and the rest dense.

The main problem is that dense times sparse isn't really widely implemented, so we try a variety of
verion transposing (DX) etc.

Long story short:
For very sparse matrices, the ThreadedSparseCSR way has a huge speedup (2-3x) over basically all the others.
For more dense matrices, plain matmul is fastest!
"""

using Polyester
using SparseArrays
using SparseMatricesCSR, ThreadedSparseCSR
using MKLSparse
import LinearAlgebra: Adjoint
import SparseArrays: mul!

# this seems to be fastest, which is nice :)
function no_transpose_polyester_mul!(C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC, α::Number, β::Number)
    @inbounds @batch for j in axes(B, 2)
        C[:, j] .*= β
        C[:, j] .+= A * (α.*B[:, j])
    end
    return C
end
function no_transpose_polyester_mul!(C::AbstractMatrix, X, D, idx)
    no_transpose_polyester_mul!(E, D[:, idx], X[idx, :], -1, 1)
end

# rs = nzrange(X, j)
# @views E_Ω[:, j_dest] .+=  D[:, rowvals(X)[rs]] * nonzeros(X)[rs]
function another_mul2!(C::AbstractMatrix, A::AbstractMatrix, B::SparseMatrixCSC, α::Number, β::Number)
    @batch for j in axes(B, 2)
        # @views E_Ω[:, j_dest] .+=  D[:, rowvals(X)[rs]] * nonzeros(X)[rs]
        rs = nzrange(B, j)
        C[:, j] .*= β
        C[:, j] .+= A[:, rowvals(B)[rs]] * (α.*nonzeros(B)[rs])
    end
    return C
end

# 32ms, 85MB
function manual_polyester!(E, X::AbstractSparseArray, D, idx)
    D_t = Matrix(transpose(@view D[:, idx]))
    @fastmath @batch for i in axes(D_t, 2)
        E[i, :] .-= X[idx, :]' * @view D_t[:, i]
    end
    return E
end

# 32ms, 85MB
function manual_polyester_no_t!(E, X::AbstractSparseArray, D, idx)
    # D_t = Matrix(transpose(D))
    @batch for i in axes(D, 1)
        E[i, :] .-= X[idx, :]' * @view D[i, idx]
    end
    return E
end



# 27ms, 7MB
mysparsecsr(M_t::Adjoint{<:Real, <:SparseMatrixCSC}) = sparsecsr(findnz(parent(M_t))[[2,1,3]]..., size(M_t)...)
function threadedsparsemul!(E, X::AbstractSparseArray, D, idx)
    Xcsrt = mysparsecsr(X[idx, :]')
    @inbounds for (i, D_row) in enumerate(eachrow(@view D[:, idx]))
        # Signature bmul!(y, A, x, alpha, beta) produces
        # y = alpha*A*x + beta*y (y = A*x)
        ThreadedSparseCSR.bmul!((@view E[i, :]), Xcsrt, D_row, -1, 1)
    end
    return E
end
function threadedsparsemul_no_polyester!(E, X::AbstractSparseArray, D, idx)
    # Xcsrt = sparsecsr(findnz(X[idx, :])[[2,1,3]]...);
    Xcsrt = mysparsecsr(X[idx, :]')
    @inbounds for (i, D_row) in enumerate(eachrow(@view D[:, idx]))
        # Signature bmul!(y, A, x, alpha, beta) produces
        # y = alpha*A*x + beta*y (y = A*x)
        @views ThreadedSparseCSR.tmul!(E[i, :], Xcsrt, D_row, -1, 1)
    end
    return E
end

# 52ms, 89MB
function intel_mkl_mul!(E, X::AbstractSparseArray, D)
    D_t = Matrix(transpose(D));
    E .-= transpose(.-X'*D_t)
end
function intel_mkl_mul!(E, X::AbstractSparseArray, D, idx)
    D_t = Matrix(transpose(D[:, idx]));
    E .-= transpose(.-X[idx, :]'*D_t)
end

function intel_mkl_mul_no_t!(E, X::AbstractSparseArray, D, idx)
    D_t = Matrix(transpose(D[:, idx]));
    E .-= (@view D[:,idx])*X[idx,:]
end

# 49ms, 76MB
function plain_matmul!(E, X::AbstractSparseArray, D, idx)
    E .-= (@view D[:, idx])*X[idx, :]
end

spmul_functions = [manual_polyester!,
                   manual_polyester_no_t!,
                   threadedsparsemul!,
                   # threadedsparsemul_no_polyester!,  # this is insanely slow...
                   intel_mkl_mul!,
                   intel_mkl_mul_no_t!,
                   no_transpose_polyester_mul!,
                   plain_matmul!,
                   ]

function run_sparse_matmul_benchmarks_idx(; N=10_000, d=1_000, K=2_000, pct_nnz=0.01)
    E = rand(d, N)
    X = sprand(K, N, pct_nnz)
    D = rand(d, K)
    idx = [1:10...]

    bmres = Dict(
        string(f) => @benchmark $f($E, $X, $D, $idx)
        for f in spmul_functions
    )

end

function compare_sparse_matmul_implementations(; N=10_000, d=1_000, K=2_000, pct_nnz=0.01)
    E = rand(d, N)
    X = sprand(K, N, pct_nnz)
    D = rand(d, K)
    idx=[1:20...]

    eval_res = Dict(
        string(f) => f(copy(E), X, D, idx)
        for f in spmul_functions
    )
    baseline = eval_res["plain_matmul!"]
    for (k, v) in eval_res
        @info k=>isapprox(baseline, v, rtol=100*sqrt(eps(eltype(E))))
    end
end

"""Results:
julia> run_sparse_matmul_benchmarks()  # i.e. E=10_000, K=1200
Dict{String, BenchmarkTools.Trial} with 5 entries:
  "threadedsparsemul!"     => 35.357 ms
  "plain_matmul!"          => 57.196 ms
  "intel_mkl_mul!"         => 64.072 ms
  "manual_polyester_no_t!" => 47.209 ms
  "manual_polyester!"      => 50.076 ms

julia> run_sparse_matmul_benchmarks(; N=20_000)
Dict{String, BenchmarkTools.Trial} with 5 entries:
  "threadedsparsemul!"     => 91.524 ms
  "plain_matmul!"          => 135.337 ms
  "intel_mkl_mul!"         => 145.085 ms
  "manual_polyester_no_t!" => 109.332 ms
  "manual_polyester!"      => 112.892 ms

julia> run_sparse_matmul_benchmarks(; K=2_000)
Dict{String, BenchmarkTools.Trial} with 5 entries:
  "threadedsparsemul!"     => 37.783 ms
  "plain_matmul!"          => 61.879 ms
  "intel_mkl_mul!"         => 68.938 ms
  "manual_polyester_no_t!" => 54.968 ms
  "manual_polyester!"      => 53.199 ms
julia> run_sparse_matmul_benchmarks(; pct_nnz=0.1)
Dict{String, BenchmarkTools.Trial} with 5 entries:
  "threadedsparsemul!"     => 442.652 ms
  "plain_matmul!"          => 395.263 ms
  "intel_mkl_mul!"         => 519.875 ms
  "manual_polyester_no_t!" => 506.361 ms
  "manual_polyester!"      => 512.548 ms

julia> run_sparse_matmul_benchmarks(; pct_nnz=0.001)
Dict{String, BenchmarkTools.Trial} with 5 entries:
  "threadedsparsemul!"     => 11.967 ms
  "plain_matmul!"          => 23.167 ms
  "intel_mkl_mul!"         => 32.219 ms
  "manual_polyester_no_t!" => 31.151 ms
  "manual_polyester!"      => 35.449 ms
"""
