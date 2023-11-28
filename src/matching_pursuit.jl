# The implementation is referencing the wikipedia page
# https://en.wikipedia.org/wiki/Matching_pursuit#The_algorithm
using LinearAlgebra
using DataStructures
using Transducers
import SparseArrays: nonzeroinds
import Base: @kwdef

const default_max_nnz = 10
const default_tolerance = 1e-6

abstract type SparseCodingMethod end
""" 'Baseline' single threaded but optimized implementation.


`max_nnz` controls the maximum number of non-zero values (i.e. of basis vectors) that
are summed up to reconstruct a data sample.

`tolerance` controls when the search for more/improved basis vectors may be stopped,
i.e. when `norm(y - Dx) < tol` (that is the L2 norm).

`precompute_products` controls whether the computation `D'*Y` is computed once in the beginning.
This is generally faster than computing the products for each sample individually, but
may use too much memory, e.g. if the data is too large to fit into memory (see `examples/memory_mapped_files.jl`).
"""
@kwdef struct MatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4*max_nnz
    tolerance = default_tolerance
    precompute_products=true
    MatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
""" Multithreaded version of `MatchingPursuit`.
Essentially falls back to single-threaded version automatically if julia is launched with
only one thread.

For description of parameters see `MatchingPursuit`.
"""
@kwdef struct ParallelMatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4*max_nnz
    tolerance = default_tolerance
    precompute_products=true
    ParallelMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end

"""
    SparseCodingMethoddGPU

To be instantiated in extensions, e.g. `CUDAAccelMatchingPursuit`.
"""
abstract type GPUAcceleratedMatchingPursuit <: SparseCodingMethod end;

""" Original implementation by https://github.com/IshitaTakeshi/KSVD.jl.
Useful for comparison and didactic purposes, but much much slower. """
@kwdef struct LegacyMatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4*max_nnz
    tolerance = default_tolerance
    LegacyMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
function validate_mp_args(max_nnz, max_iter, tolerance, other_args...)
    max_nnz >= 1 || throw(ArgumentError("`max_nnz` must be > 0"))
    max_iter >= 1 || throw(ArgumentError("`max_iter` must be > 0"))
    tolerance >= 0. || throw(ArgumentError("`tolerance` must be >= 0"))
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
        method::Union{MatchingPursuit, ParallelMatchingPursuit, GPUAcceleratedMatchingPursuit},
        data::AbstractVector{T}, dictionary::AbstractMatrix{T}, DtD::AbstractMatrix{T};
        products_init::Union{Nothing, AbstractVector{T}}=nothing) :: SparseVector{T, Int} where T
    (; max_nnz, max_iter, tolerance) = method

    n_atoms = size(dictionary, 2)
    residual = copy(data)
    xdict = DefaultDict{Int, T}(0.)

    products = (isnothing(products_init) ? (dictionary' * residual) : products_init)
    products_abs = abs.(products)  # prealloc

    for i in 1:max_iter
        if norm(residual) < tolerance
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
    return sparsevec(xdict, n_atoms)
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
    (; max_nnz, max_iter, tolerance) = method
    n_atoms = size(dictionary, 2)

    residual = copy(data)

    xdict = DefaultDict{Int, T}(0.)
    for i in 1:max_iter
        if norm(residual) < tolerance
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
