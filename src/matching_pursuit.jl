using DataStructures

# The implementation is referencing the wikipedia page
# https://en.wikipedia.org/wiki/Matching_pursuit#The_algorithm
using CUDA
using LoopVectorization
using LinearAlgebra
using Transducers
using Match

const default_max_iter_mp = 20
const default_tolerance = 1e-6


function SparseArrays.sparsevec(d::DefaultDict{Int, Float64}, m::Int)
    SparseArrays.sparsevec(collect(keys(d)), collect(values(d)), m)
end

abstract type SparseCodingMethod end
@kwdef struct MatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
end
@kwdef struct ParallelMatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
end

function matching_pursuit_(data::AbstractVector, dictionary::AbstractMatrix,
                           max_iter::Int, tolerance::Float64) :: SparseVector{Float64, Int}
    n_atoms = size(dictionary, 2)

    residual = copy(data)

    xdict = DefaultDict{Int, Float64}(0.)
    products = similar(residual, size(dictionary, 2))

    for i in 1:max_iter
        if norm(residual) < tolerance
            return sparsevec(xdict, n_atoms)
        end

        # find an atom with maximum inner product
        products .= dictionary' * residual
        # maxval, maxidx = findmax(products)
        # minval, minidx = findmin(products)
        _, maxindex = findmax(abs.(products))
        maxval = products[maxindex]
        atom = dictionary[:, maxindex]

        # # val, idx = (abs(maxval) > abs(minval) ? (maxval, maxidx) : (minval, minidx))
        # # maxval = products[maxindex]
        # atom = if data isa CuArray
        #     idx = idx[1]  # is cartesian index with second element always 1
        #     mask = CUDA.CuMatrix(I(n_atoms)[idx:idx, :])
        #     sum(dictionary .* mask, dims=2)
        # else
        #     dictionary[:, idx]
        # end

        # c is the length of the projection of data onto atom
        a = maxval / sum(abs2, atom)  # equivalent to maxval / norm(atom)^2
        residual -= atom * a

        xdict[maxindex] += a
    end
    return sparsevec(xdict, n_atoms)
end


"""
    matching_pursuit(data::Vector, dictionary::AbstractMatrix;
                     max_iter::Int = $default_max_iter_mp,
                     tolerance::Float64 = $default_tolerance)

Find ``x`` such that ``Dx = y`` or ``Dx ≈ y`` where y is `data` and D is `dictionary`.
```
# Arguments
* `max_iter`: Hard limit of iterations
* `tolerance`: Exit when the norm of the residual < tolerance
```
"""
function matching_pursuit(data::AbstractVector, dictionary::AbstractMatrix;
                          max_iter::Int = default_max_iter_mp,
                          tolerance = default_tolerance) :: SparseVector{Float64, Int}

    if tolerance <= 0
        throw(ArgumentError("`tolerance` must be > 0"))
    end

    if max_iter <= 0
        throw(ArgumentError("`max_iter` must be > 0"))
    end

    if size(data, 1) != size(dictionary, 1)
        throw(ArgumentError(
            "Dimensions must match: `size(data, 1)` and `size(dictionary, 1)`."
        ))
    end

    matching_pursuit_(data, dictionary, max_iter, tolerance)
end


"""
    matching_pursuit(data::AbstractMatrix, dictionary::AbstractMatrix;
                     max_iter::Int = $default_max_iter_mp,
                     tolerance::Float64 = $default_tolerance)

Find ``X`` such that ``DX = Y`` or ``DX ≈ Y`` where Y is `data` and D is `dictionary`.
```
# Arguments
* `max_iter`: Hard limit of iterations
* `tolerance`: Exit when the norm of the residual < tolerance
```
"""
function sparse_coding(method::Union{MatchingPursuit, ParallelMatchingPursuit}, data::AbstractMatrix, dictionary::AbstractMatrix)
                          # max_iter::Int = default_max_iter_mp,
                          # tolerance::Float64 = default_tolerance)
    K = size(dictionary, 2)
    N = size(data, 2)

    # data = CUDA.CuArray(data)
    # dictionary = CUDA.CuArray(dictionary)

    X_::Vector{SparseVector{Float64, Int}} = tcollect(
    # X_::Vector{SparseVector{Float64, Int}} = ThreadsX.collect(
    collect_fn = @match method begin
        ::MatchingPursuit => collect
        ::ParallelMatchingPursuit => tcollect
        _ => collect
    end
        matching_pursuit_(
                datacol,
                dictionary,
                method.max_iter,
                method.tolerance
            )
        for datacol in eachcol(data)
    )
    X = spzeros(K, N)
    for (i, col) in enumerate(X_)
        X[:, i] = col  # this is still bad somehow...
    end
    return X
end

sparse_coding(data::AbstractMatrix, dictionary::AbstractMatrix) = sparse_coding(MatchingPursuit(), data, dictionary)
