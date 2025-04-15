# KSVD.jl

[K-SVD](https://en.wikipedia.org/wiki/K-SVD) is an algorithm for creating overcomplete dictionaries for sparse representations.  

This package implements:

* K-SVD as described in the original paper: [K-SVD: An Algorithm for Designing Overcomplete Dictionaries for Sparse Representation](http://www.cs.technion.ac.il/~freddy/papers/120.pdf)
* [Matching Pursuit](https://en.wikipedia.org/wiki/Matching_pursuit) for representing signals using a given dictionary.

In particular, substantial effort has been put in speeding up this implementation.
This includes:
- Custom parallel dense-sparse matrix multiplication via [`ThreadedDenseSparseMul.jl`](https://github.com/RomeoV/ThreadedDenseSparseMul.jl)
- Custom pipelined GPU offloading for matching pursuit (`CUDAAcceleratedMatchingPursuit`)
- Threaded matching pursuit (`ParallelMatchingPursuit`) (mostly "embarassingly parallel")
- Threaded ksvd implementation (`ParallelKSVD`) (*not* "embarassingly parallel")
- Threaded and batched ksvd implementation (`BatchedParallelKSVD`) (*not* "embarassingly parallel")
- Extensive efforts removing allocations by preallocating buffers
- Extensive benchmark-driven optimizations utilizing [`ProfileView.jl`](https://github.com/timholy/ProfileView.jl)
- Many other modification experiments.

# Usage

Assume that each column of Y represents a feature vector, and each column of D a dictionary vector, with n dictionaries (`size(D, 2) == n`).
K-SVD derives D and X such that DX ≈ Y from only Y.
Finally, let `nnzpercol` be the maximum number of nonzero dictionary elements per sample.
Then we can just run

```julia
(; D, X) = ksvd(Y, n, nnzpercol)
```

Runnable example:
``` julia
using KSVD, Random, StatsBase, SparseArrays, LinearAlgebra
m, n = 4*64, 4*256
nsamples = 50_000
nnzpercol=5
T = Float32

D = rand(Float32, m, n);
X = stack(
  (SparseVector(n, sample(1:n, nnzpercol; replace=false), rand(T, nnzpercol))
   for _ in 1:nsamples);
  dims=2)
Y = D*X + T(0.05)*randn(T, size(D*X))

(; D, X) = ksvd(Y, n, nnzpercol)
norm.(eachcol(Y - D*X)) |> mean  # approx 0.4, i.e. recovers 60% of the signal
```

You can get some more information about what's happening through `show_trace=true`, and by turning on the built-in timer.
``` julia
using TimerOutputs
TimerOutputs.enable_debug_timings(KSVD)
(; D, X) = ksvd(Y, n, nnzpercol; show_trace=true)
```

# we can control the matching pursuit stage and ksvd stage through method structs
ksvd_update_method = BatchedParallelKSVD{false, Float64}(shuffle_indices=true, batch_size_per_thread=1)
sparse_coding_method = ParallelMatchingPursuit(max_nnz=25, rtol=5e-2)
result = ksvd(Y, 256;
              ksvd_update_method,
              sparse_coding_method,
              maxiters=100,
              abstol=1e-6,
              reltol=1e-6,
              show_trace=true)

# Access additional information
println("Termination condition: ", result.termination_condition)
println("Norm results: ", result.norm_results)
println("NNZ per column results: ", result.nnz_per_col_results)
println("Timing information: ", result.timer)
```

Of course we can also just run one step of matching pursuit/sparse coding, or one step of the ksvd update:

```julia
basis = KSVD.init_dictionary(size(Y, 1), 2*size(Y,2))
X = sparse_coding(OrthogonalMatchingPursuit(max_nnz=25), Y, basis)

(; D, X) = ksvd_update(ksvd_update_method, Y, basis, X)
```

[Matching Pursuit](https://en.wikipedia.org/wiki/Matching_pursuit) derives X from D and Y such that DX = Y in constraint that X be as sparse as possible.


# Performance improvements

Here is an overview of the performance improvements in the `ksvd_update` provided in this package, broken down by computation type.
The data is computed using different commits on the `experiments` branch.
More details will be added later.

![benchmark results](/ksvd_benchmarks/figs/benchmark_results.png)
