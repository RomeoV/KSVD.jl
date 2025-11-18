# DB-KSVD: Scalable Alternating Optimization for Disentangling High-Dimensional Embedding Spaces
> Supplementary code

## Installation Instructions
This repository is a "multilingual" repository and requires an installation of both python with the `uv` package manger ([see here](https://docs.astral.sh/uv/getting-started/installation/ "uv installation instructions")) and a Julia ([see here](https://julialang.org/install/ "julia installation instructions")).

To set up all dependencies first instantiate the python repository with

``` sh
uv sync
```

Then copy the Julia environment specification into the generated `.venv` directory with

``` sh
cp -r julia_env .venv
```

To launch Julia simply run

``` sh
# to get an interactive REPL
julia --project=. --threads=auto  # multiple threads to go fast
# run `import Pkg; Pkg.instantiate()` the first time
# or to launch a script
julia --project=. --threads=auto scripts/<...>.jl
```
This works because the Julia files `Project.toml` and `Manifest.toml` are symlinked to `.venv/julia_env`.  
However, `KSVD` will only be installed when `import juliacall` in run from the first time from python (see below).

*Note: For the first startup Julia will need to precompile a bunch of things still which may take a moment.*

To launch python, simply run

``` sh
# to get an interactive REPL
uv run ipython
# or to launch a script
uv run ipython scripts/<...>.py
```

and `uv` will automatically use the correct environment.


## Using KSVD.jl functionality from python
The main functionality of `KSVD.jl` is written in Julia.
To use the functionality from python, we therefore have to jump through some small hoops.
However, integration should be mostly easy.

#### Basic example: calling `ksvd` from python
``` python
import numpy, torch
import juliacall; jl = juliacall.Main
jl.seval("""
    using KSVD
""")

Y = torch.rand(128, 5_000, dtype=torch.float32)
# numpy arrays are understood by Julia
res = jl.ksvd(Y.numpy(), 256, 3)  # m=256, k=3
print(res.D, res.X)
# or
Dtorch = torch.from_numpy(numpy.array(res.D))
Xtorch = torch.sparse_csc_tensor(res.X.colptr, res.X.rowval, res.X.nzval, size=(res.X.m, res.X.n), dtype=torch.float32)
print(Dtorch, Xtorch)
```

We can also easily just do sparse coding (Matching Pursuit):
``` python
D = jl.KSVD.init_dictionary(jl.Float32, 128, 256)
X = jl.sparse_coding(Y.numpy(), D, 3)
```

#### Using multiple threads
When launching python and `juliacall` as above, Julia will be launched on a single thread only.
To use all available threads, first set these two environment variables:

``` python
import os
# see https://juliapy.github.io/PythonCall.jl/stable/juliacall/#py-multi-threading-signal-handling
# will disable Ctrl-Cing python code, as the signal is now caught by Julia
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"]="yes"
# use all your CPUs!
os.environ["PYTHON_JULIACALL_THREADS"]="auto"

import juliacall
# continue as before ...
```

#### Using GPU acceleration
To use GPU acceleration we have two options. First, we can simply use python `torch` for some operations.
For example:

``` python
D = jl.KSVD.init_dictionary(jl.Float32, 128, 256)
Ddev = torch.from_numpy(numpy.array(D)).to('cuda')
DtD = (Ddev.T @ Ddev).cpu().numpy()
X = jl.sparse_coding(Y.numpy(), D, 3, DtD=DtD)
```

However, we can also use KSVD's CUDA acceleration. For this we first have to add and load the CUDA package:

``` python
jl.seval("""
import Pkg; Pkg.add("CUDA")
using KSVD, KSVD.OhMyThreads, CUDA
""")
```
Then, `CUDAAcceleratedMatchingPursuit` and `CUDAAcceleratedArnoldiSVDSolver` are available.

``` python
D = jl.KSVD.init_dictionary(jl.Float32, 128, 256)
sparse_coding_method = jl.KSVD.CUDAAcceleratedMatchingPursuit(max_nnz=3)
X = jl.sparse_coding(sparse_coding_method, Y.numpy(), D)  # k is now stored in the sparse coding method

ksvd_update_method = jl.BatchedParallelKSVD[False, jl.Float32, jl.OhMyThreads.DynamicScheduler, jl.KSVD.CUDAAcceleratedArnoldiSVDSolver[jl.Float32]]()
res = jl.ksvd(Y.numpy(), 256, sparse_coding_method=sparse_coding_method, ksvd_update_method=ksvd_update_method)
```

For more `ksvd` options please refer to the `KSVD.jl` documentation.
