# The implementation is referencing the wikipedia page
# https://en.wikipedia.org/wiki/Matching_pursuit#The_algorithm
using LinearAlgebra
using DataStructures
using Transducers
import SparseArrays: nonzeroinds

const default_max_nnz = 10
const default_rtol = 1e-6

abstract type SparseCodingMethod end
""" 'Baseline' single threaded but optimized implementation.


`max_nnz` controls the maximum number of non-zero values (i.e. of basis vectors) that
are summed up to reconstruct a data sample.

`rtol` controls when the search for more/improved basis vectors may be stopped,
i.e. when `norm(y - Dx) < tol` (that is the L2 norm).

`precompute_products` controls whether the computation `D'*Y` is computed once in the beginning.
This is generally faster than computing the products for each sample individually, but
may use too much memory, e.g. if the data is too large to fit into memory (see `examples/memory_mapped_files.jl`).
"""
@kwdef struct MatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4*max_nnz
    rtol = default_rtol
    precompute_products=true
    MatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
@kwdef struct OrthogonalMatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4*max_nnz
    rtol = default_rtol
    precompute_products=true
    OrthogonalMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end


""" Multithreaded version of `MatchingPursuit`.
Essentially falls back to single-threaded version automatically if julia is launched with
only one thread.

For description of parameters see `MatchingPursuit`.
"""
@kwdef struct ParallelMatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4*max_nnz
    rtol = default_rtol
    precompute_products=true
    ParallelMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end

"""
    SparseCodingMethoddGPU

To be instantiated in extensions, e.g. `CUDAAccelMatchingPursuit`.
"""
abstract type GPUAcceleratedMatchingPursuit <: SparseCodingMethod end;

""" CUDA Accelerated version pipelining the dictionary * data computation
on the gpu, and using the result on the cpu. Always performs batching, i.e. uses
limited memory even for large number of samples (however not for large embedding dimension).

For description of parameters see `MatchingPursuit`.
Notice that this method never precomputes the full product matrix `D'*Y`. Instead
batches of data with size number of samples `batch_size` are loaded, moved to the GPU, multiplied there,
and the result is moved back to the cpu. This happens asynchronously, so that the memory movement
CPU->GPU and GPU->CPU, aswell as the computation on the CPU, are pipelined using Julia's `Channel`s.
"""
@kwdef struct CUDAAcceleratedMatchingPursuit <: GPUAcceleratedMatchingPursuit
    max_nnz::Int = KSVD.default_max_nnz
    max_iter::Int = 4*max_nnz
    rtol = KSVD.default_rtol
    batch_size::Int = 1_000
    CUDAAcceleratedMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end

""" Original implementation by https://github.com/IshitaTakeshi/KSVD.jl.
Useful for comparison and didactic purposes, but much much slower. """
@kwdef struct LegacyMatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4*max_nnz
    rtol = default_rtol
    LegacyMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
function validate_mp_args(max_nnz, max_iter, rtol, other_args...)
    max_nnz >= 1 || throw(ArgumentError("`max_nnz` must be > 0"))
    max_iter >= 1 || throw(ArgumentError("`max_iter` must be > 0"))
    0. <= rtol <= 1. || throw(ArgumentError("`rtol` must be in [0,1]"))
end

sparse_coding(data::AbstractMatrix, dictionary::AbstractMatrix) = sparse_coding(ParallelMatchingPursuit(), data, dictionary)

get_method_collect_fn(::MatchingPursuit) = collect
get_method_collect_fn(::ParallelMatchingPursuit) = tcollect

"""
    sparse_coding(method::Union{MatchingPursuit, ParallelMatchingPursuit},
                  data::AbstractMatrix, dictionary::AbstractMatrix)

Find ``X`` such that ``DX = Y`` or ``DX ≈ Y`` where Y is `data` and D is `dictionary`.
"""
function sparse_coding(method::Union{MatchingPursuit, ParallelMatchingPursuit}, data::AbstractMatrix{T}, dictionary::AbstractMatrix{T}) where T
    K = size(dictionary, 2)
    N = size(data, 2)

    collect_fn = get_method_collect_fn(method)

    DtD = dictionary'*dictionary
    # if the data is very large we might not want to precompute this.
    products = (method.precompute_products ? (dictionary' * data) : fill(nothing, 1, size(data, 2)))

    X_::Vector{SparseVector{T, Int}} = collect_fn(
        matching_pursuit_(
                method,
                datacol,
                dictionary,
                DtD;
                products_init=(method.precompute_products ? productcol : nothing)
            )
        for (datacol, productcol) in zip(eachcol(data), eachcol(products))
    )
    # The "naive" version of `cat`ing the columns in X_ run into type inference problems for some reason.
    # I tried `hcat(X_...)` and `X = spzeros(Float64, K, N); for (i, col) in enumerate(X_); X[:, i]=col; end` they were both very slow.
    # This version seems to overcome the type inference issues and makes the code much faster.
    # Note that there's a multithreaded version of this too here: https://github.com/RomeoV/KSVD.jl/blob/b3e089c925a9123692f766b2106934eebb13edcc/src/matching_pursuit.jl#L168-L183
    X = let
        I = Int[]; J = Int[]; V = T[]
        for (i, v) in enumerate(X_)
            append!(I, SparseArrays.nonzeroinds(v))
            append!(J, fill(i, SparseArrays.nnz(v)))
            append!(V, SparseArrays.nonzeros(v))
        end
        sparse(I, J, V, K, N)
    end
    return X
end

@inbounds function matching_pursuit_(
        method::Union{OrthogonalMatchingPursuit},
        data::AbstractVector{T}, dictionary::AbstractMatrix{T}, DtD::AbstractMatrix{T};
        products_init::Union{Nothing, AbstractVector{T}}=nothing) :: SparseVector{T, Int} where T
    (; max_nnz, max_iter, rtol) = method

    n_atoms = size(dictionary, 2)
    residual = copy(data)
    xdict = DefaultDict{Int, T}(zero(T))
    norm_data = norm(data)

    products = (isnothing(products_init) ? (dictionary' * residual) : products_init)
    products_abs = abs.(products)  # prealloc

    for i in 1:max_iter
        if norm(residual)/norm_data < rtol
            return sparsevec(xdict, n_atoms)
        end
        if length(xdict) > max_nnz
            pop!(xdict, findmin(abs, xdict)[2])
            return sparsevec(xdict, n_atoms)
        end

        # find an atom with maximum inner product
        products_abs .= abs.(products)
        _, maxindex = findmax_fast(products_abs)

        beta = 1/(1 - v_k'*b_k)
        v_k = DtD[inds, maxindex]
        b_k = A_inv*v_k
        A_inv = [A_inv -beta*b_k;
                 -beta*b_k' 1]

        atom = @view dictionary[:, maxindex]
        γ_k = atom - dictionary[:, inds] * b_k
        α_k = inv(sum(abs2, γ_k)) * products[maxindex]
        factors .-= α_k*b_k
        products .-= α_k .* @view DtD[:, maxindex]

        push!(inds, maxindex)
        push!(factors, α_k)

        reconstruction .+= α_k * atom
        residual .-= α_k * atom
    end
    xdict = Dict(zip(inds, factors))
    return sparsevec(xdict, n_atoms)
end


"""
Optimization rationale:
For any given residual, we compute `products = dictionary' * residual` and find the maximizer over the dictionary elements of this.
We then assign `residual -= (d_i' * residual) * d_i`


```
products = dictionary' * residual
_, maxindex = findmax(abs.(products))
maxval = products[maxindex]
atom = dictionary[:, maxindex]

a = maxval / sum(abs2, atom)  # equivalent to maxval / norm(atom)^2
residual -= atom * a
```

Then the new products become
```
a = max(abs.(products[t]))
products[t+1] = dictionary' * (residual[t] - dictionary[idx] * a)
              = products[t] - (dictionary' * dictionary[idx] * a)
```
"""

@inbounds function matching_pursuit_(
        method::Union{MatchingPursuit, ParallelMatchingPursuit, GPUAcceleratedMatchingPursuit},
        data::AbstractVector{T}, dictionary::AbstractMatrix{T}, DtD::AbstractMatrix{T};
        products_init::Union{Nothing, AbstractVector{T}}=nothing) :: SparseVector{T, Int} where T
    (; max_nnz, max_iter, rtol) = method

    n_atoms = size(dictionary, 2)
    residual = copy(data)
    xdict = DefaultDict{Int, T}(zero(T))
    norm_data = norm(data)

    products = (isnothing(products_init) ? (dictionary' * residual) : products_init)
    products_abs = abs.(products)  # prealloc

    for i in 1:max_iter
        if norm(residual)/norm_data < rtol
            return sparsevec(xdict, n_atoms)
        end
        if length(xdict) > max_nnz
            pop!(xdict, findmin(abs, xdict)[2])
            return sparsevec(xdict, n_atoms)
        end

        # find an atom with maximum inner product
        products_abs .= abs.(products)
        _, maxindex = findmax_fast(products_abs)

        a = products[maxindex]
        atom = @view dictionary[:, maxindex]
        @assert norm(atom) ≈ 1. norm(atom)

        residual .-= a .* atom
        products .-= a .* @view DtD[:, maxindex]

        xdict[maxindex] += a
    end
    inds, vals = collect(keys(xdict)), collect(values(xdict))
    if false # !isnothing(method.regularize_results_factor)
        # abuse products_abs buffer
        buf = let products_abs = reshape(products_abs, size(dictionary, 1), :)
            @view products_abs[:, 1:length(inds)]
        end
        buf .= dictionary[:, inds]
        vals = (buf'*buf + 0.1*I)*buf'*data  # ridge regression
    end
    return sparsevec(inds, vals)
end

" This is the original implementation by https://github.com/IshitaTakeshi, useful for
numerical comparison and didactic purposes. "
function sparse_coding(method::LegacyMatchingPursuit,
                       data::AbstractMatrix{T},
                       dictionary::AbstractMatrix{T}) where T
    K = size(dictionary, 2)
    N = size(data, 2)

    X = spzeros(T, K, N)

    for i in 1:N
        X[:, i] = matching_pursuit_(
            method,
            vec(data[:, i]),
            dictionary
        )
    end
    return X
end
function matching_pursuit_(method::LegacyMatchingPursuit,
                           data::AbstractVector{T},
                           dictionary::AbstractMatrix{T}) where T
    (; max_nnz, max_iter, rtol) = method
    n_atoms = size(dictionary, 2)

    residual = copy(data)
    norm_data = norm(data)

    xdict = DefaultDict{Int, T}(zero(T))
    for i in 1:max_iter
        if norm(residual)/norm_data < rtol
            return sparsevec(xdict, n_atoms)
        end
        if length(xdict) > max_nnz
            pop!(xdict, findmin(abs, xdict)[2])
            return sparsevec(xdict, n_atoms)
        end

        # find an atom with maximum inner product
        products = dictionary' * residual
        _, maxindex = findmax(abs.(products))
        maxval = products[maxindex]
        atom = dictionary[:, maxindex]

        # c is the length of the projection of data onto atom
        a = maxval / sum(abs2, atom)  # equivalent to maxval / norm(atom)^2
        residual -= atom * a

        xdict[maxindex] += a
    end
    return sparsevec(xdict, n_atoms)
end
