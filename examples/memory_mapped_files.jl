"""
Here we show how we can compute matching pursuit for files that are too large to fit into memory.
Fortunately matching pursuit is "embarassingly parallel" (https://en.wikipedia.org/wiki/Embarrassingly_parallel)
along the columns of the data.

Therefore we can use a memory mapped file for the data and then just call the matching pursuit algorithm as usual!
However, we have to make sure not to precompute the `dictionary' * data` matrix.
"""

using KSVD, Mmap, Random

# fname = tempname()
fname = joinpath("examples", "mmap.bin")
@warn """
Running this script will create a large (30GB or so) file in your temp directory, each time!
We try to remove the file at the end of the program, but if the program crashes the file might remain.
Feel free to delete it manually in that case! The filename is '$(fname)'.
"""
# Create very large datafile with random values.
s = open(fname, "w+")
A = mmap(s, Matrix{Float64}, (1_000, 4_000_000));
for col in eachcol(A)
    rand!(col)
end
Mmap.sync!(A)
A = nothing; close(s)

s2 = open(fname, "r")
A2 = mmap(s2, Matrix{Float64}, (1000, 4_000_000); shared=false);

"""
Notice that for the non-batching versions we turn off precomputing the products,
as the memory requirement for that is O(K*N), and we have N>>1.

Possible method choices:
- `KSVD.MatchingPursuit(precompute_products=false)`
- `KSVD.ParallelMatchingPursuit(precompute_products=false)`
- `KSVD.CUDAAcceleratedMatchingPursuit()`
"""
method = KSVD.ParallelMatchingPursuit(precompute_products=false)
KSVD.sparse_coding(method, A2, KSVD.init_dictionary(1000, 1000))
A2 = nothing; close(s2)

@warn "Removing $(fname)"
Base.Filesystem.rm(fname)

# and that's it!
