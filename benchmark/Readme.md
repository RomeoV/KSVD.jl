# Benchmark results


## SVD benchmarking
One of the original bottlenecks of the KSVD algorithm is the SVD step.
Since we only need the maximum singular value / vector, there's a lot of optimization potential over computing the full svd.

We consider the following SVD implementations:
- full `svd` (Base.LinearAlgebra)
- `svdl` (ArnoliMethods.jl)
- `tsvd` (TSVD.jl -> Truncated svd)
- `powm` (IterativeMethods.jl -> Power method for modified eigenvalue problem)
- `svdsolve` (KrylovKit.jl)

We note that all solvers, except for `svd`, allow for just computing the maximum singular vector.
We compare numerical results of all solvers, which are consistent up to `sqrt(eps(Float64))` for the singular values, and `100*sqrt(eps(Float64))` for the L2-norm of the difference of the maximum singular vector.

### Results
Results were collected with a 11th Gen Intel(R) Core(TM) i7-11800H @ 2.30GHz, using the `@benchmark` macro from BenchmarkTools.jl, which performs multiple runs + warmup run, and the mean is reported.
No special effort was taken do reduce "background noise, so results should be taken with a grain of salt. However, we expect the general trends to hold.

For the single-thread results, we note that we also explicitly fix the BLAS threads to 1 (not true by default).

*Single core*
| method      | time    | memory |
| ----------- | ------- | ------ |
| bm_res_svd  | 2.34e9  | 200MB  |
| bm_res_svdl | 4.55e7  | 5MB    |
| bm_res_tsvd | 2.07e7  | 904kB  |
| bm_res_powm | 2.00e8  | 16MB   |
| bm_res_kryl | 1.96e8  | 16MB   |

*16 core*
| method       | time   | memory |
| -----------  | ------ | ------ |
| bm_res_svd"  | 2.25e9 | 200MB  |
| bm_res_svdl" | 9.39e7 | 5MB    |
| bm_res_tsvd" | 4.54e7 | 904kB  |
| bm_res_powm" | 2.21e8 | 16MB   |
| bm_res_kryl" | 2.93e8 | 16MB   |

Interestingly, the single core performance is slightly better!
Either way, we can see that `tsvd` is the clear winner, with `svdl` being the runner up.
