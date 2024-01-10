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

# Installation
Package registration is WIP (https://github.com/JuliaRegistries/General/pull/98593).
Until then, you can add this package like so:

```julia
] add https://github.com/RomeoV/KSVD.jl
```

# Usage

Assume that each column of Y represents a feature vector (or an input signal from some system).  
D is a dictionary. Each column of D represents an atom.  
K-SVD derives D and X such that DX ≈ Y from only Y.  

```julia
D, X = dictionary_learning(Y, 256)

# we can control the matching pursuit stage and ksvd stage through method structs
ksvd_method = BatchedParallelKSVD()
mp_Method = ParallelMatchingPursuit()
D, X = dictionary_learning(Y, 256;
                           ksvd_method=ksvd_method,
                           sparse_coding_method=mp_method)
```

Of course we can also just run one step of matching pursuit/sparse coding, or one step of the ksvd update:

``` julia
basis = KSVD.init_dictionary(size(Y, 1), 2*size(Y,2))
X::SparseMatrix = sparse_coding(mp_method, Y, basis

D::Matrix, X::SparseMatrix = ksvd_update(ksvd_method, Y, basis, X)
```

[Matching Pursuit](https://en.wikipedia.org/wiki/Matching_pursuit) derives X from D and Y such that DX = Y in constraint that X be as sparse as possible.
