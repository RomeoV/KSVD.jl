# The implementation is referencing the wikipedia page
# https://en.wikipedia.org/wiki/Matching_pursuit#The_algorithm
using LinearAlgebra
using DataStructures
using Transducers
using Match
import SparseArrays: nonzeroinds
using Destruct
import Profile

const default_max_iter_mp = 20
const default_tolerance = 1e-6


function SparseArrays.sparsevec(d::DefaultDict{Int, Float64}, m::Int)
    SparseArrays.sparsevec(collect(keys(d)), collect(values(d)), m)
end

abstract type SparseCodingMethod end
function validate_mp_args(max_iter, tolerance)
    max_iter >= 1 || throw(ArgumentError("`max_iter` must be > 0"))
    tolerance > 0. || throw(ArgumentError("`tolerance` must be > 0"))
end
@kwdef struct LegacyMatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
    LegacyMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
@kwdef struct MatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
    MatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
@kwdef struct ParallelMatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
    ParallelMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
@kwdef struct FasterParallelMatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
    FasterParallelMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
@kwdef struct FullBatchMatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
    FullBatchMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end

""" Redefine findmax for vector of floats to not do nan-checks.

By default,`findmax` uses `isless`, which does a nan-check before computing `<(lhs, rhs)`.
We roll basically the same logic as in `Julia/Base/reduce.jl:findmax` but we directly use `<`, which gives us about a 1.5x speedup.
"""
function findmax_fast(data::Vector{T}) where T
    cmp_tpl((fm, im), (fx, ix)) = (fm < fx) ? (fx, ix) : (fm, im)
    mapfoldl( ((k, v),) -> (v, k), cmp_tpl, pairs(data))
end

@inbounds function matching_pursuit_(
        method::Union{MatchingPursuit, ParallelMatchingPursuit, FasterParallelMatchingPursuit, CUDAAcceleratedMatchingPursuit},
        data::AbstractVector, dictionary::AbstractMatrix, DtD::AbstractMatrix;
        products_init::Union{Nothing, AbstractVector}=nothing) :: SparseVector{Float64, Int}
    (; tolerance, max_iter) = method

    n_atoms = size(dictionary, 2)
    residual = copy(data)
    xdict = DefaultDict{Int, Float64}(0.)

    products = (isnothing(products_init) ? dictionary' * residual : products_init)
    products_abs = abs.(products)  # prealloc

    for i in 1:max_iter
        if norm(residual) < tolerance
            return sparsevec(xdict, n_atoms)
        end

        # find an atom with maximum inner product
        products_abs .= abs.(products)
        _, maxindex = findmax_fast(products_abs)

        maxval = products[maxindex]
        @inbounds atom = @view dictionary[:, maxindex]

        # c is the length of the projection of data onto atom
        a = maxval / sum(abs2, atom)  # equivalent to maxval / norm(atom)^2
        residual .-= a .* atom
        products .-= a .* @view DtD[:, maxindex]

        xdict[maxindex] += a
    end
    return sparsevec(xdict, n_atoms)
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
    K = size(dictionary, 2)
    N = size(data, 2)

    collect_fn = @match method begin
        ::MatchingPursuit => collect
        ::ParallelMatchingPursuit => tcollect
    end

    DtD = dictionary'*dictionary
    products = dictionary' * data

    X_::Vector{SparseVector{Float64, Int}} = collect_fn(
        matching_pursuit_(
                method,
                datacol,
                dictionary,
                DtD;
                products_init=productcol
            )
        for (datacol, productcol) in zip(eachcol(data), eachcol(products))
    )
    # The "naive" version of `cat`ing the columns in X_ run into type inference problems for some reason.
    # I first tried `hcat(X_...)`, but it was somewhat slow.
    # Then I tried `X = spzeros(Float64, K, N); for (i, col) in enumerate(X_); X[:, i]=col; end` but that
    # was also bad somehow.
    # This version seems to overcome the type inference issues and makes the code much faster.
    X = let
        I = Int[]; J = Int[]; V = Float64[]
        for (i, v) in enumerate(X_)
            append!(I, v.nzind)
            append!(J, fill(i, nnz(v)))
            append!(V, v.nzval)
        end
        sparse(I,J,V, K, N)
    end
    return X
end

"Similar to `ParallelMatchingPursuit`, but we deal with the collection of indices slightly differently.
Doesn't seem to make a great difference..."
function sparse_coding(method::FasterParallelMatchingPursuit, data::AbstractMatrix{T}, dictionary::AbstractMatrix{T}) where T
    K = size(dictionary, 2)
    N = size(data, 2)

    DtD = dictionary'*dictionary
    products = dictionary' * data

    # reset the profiler here to only profile the threaded section
    # otherwise we get a pretty large area of "poptask" even though all cores (but not all threads(!)) are busy with the dense matmul
    # Profile.clear()

    I_buffers = [Int[] for _ in 1:size(data, 2)]; V_buffers = [T[] for _ in 1:size(data, 2)];
    Threads.@threads for j in axes(data, 2)
        datacol = @view data[:, j]; productcol = @view products[:, j]
        data_vec = matching_pursuit_(
                        method,
                        datacol,
                        dictionary,
                        DtD;
                        products_init=productcol
                    )
        append!(I_buffers[j], nonzeroinds(data_vec))
        append!(V_buffers[j], nonzeros(data_vec))
    end

    I = vcat(I_buffers...); V = vcat(V_buffers...)
    J = vcat([fill(j, size(I_buf)) for (j, I_buf) in enumerate(I_buffers)]...)
    X = sparse(I,J,V, K, N)
    return X
end

sparse_coding(data::AbstractMatrix, dictionary::AbstractMatrix) = sparse_coding(ParallelMatchingPursuit(), data, dictionary)

"This turns out to be slow /and/ algorithmically wrong. Don't use..."
function sparse_coding(method::FullBatchMatchingPursuit, data::AbstractMatrix, dictionary::AbstractMatrix)
    K = size(dictionary, 2)
    N = size(data, 2)
    max_iter = method.max_iter

    DtD = dictionary'*dictionary
    products = dictionary' * data
    products_abs = abs.(products)

    I_buffers = [Int[] for _ in 1:size(data, 2)]; V_buffers = [Float64[] for _ in 1:size(data, 2)];

    # max_product_inds = sortperm.(eachcol(products_abs), rev=true, lt=(<))
    Threads.@threads for data_idx in axes(data, 2)
        hp = BinaryHeap(Base.By(last, DataStructures.FasterReverse()),
                        collect(pairs(@view products_abs[:, data_idx])))
        inds, ps = [Tuple(pop!(hp)) for _ in 1:max_iter] |> destruct

        coeffs = ps ./ sum(abs2, dictionary[:, inds], dims=1)[1, :]
        residuals = data[:, data_idx] .- cumsum(coeffs' .* dictionary[:, inds], dims=2)
        residual_norms = norm.(eachcol(residuals)).^2
        last_idx::Int = let
            idx = findfirst(<(method.tolerance), residual_norms)
            (isnothing(idx) ? max_iter : idx)
        end

        data_vec = sparsevec(inds[1:last_idx], coeffs[1:last_idx])
        append!(I_buffers[data_idx], nonzeroinds(data_vec))
        append!(V_buffers[data_idx], nonzeros(data_vec))
    end
    I = vcat(I_buffers...); V = vcat(V_buffers...)
    J = vcat([fill(j, size(I_buf)) for (j, I_buf) in enumerate(I_buffers)]...)

    X = sparse(I,J,V, K, N)
    return X
end

" This is the original implementation by https://github.com/IshitaTakeshi, useful for
numerical comparison and didactic purposes. "
function sparse_coding(method::LegacyMatchingPursuit,
                          data::AbstractMatrix,
                          dictionary::AbstractMatrix)
    K = size(dictionary, 2)
    N = size(data, 2)

    X = spzeros(K, N)

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
                           data::AbstractVector,
                           dictionary::AbstractMatrix)
    (; max_iter, tolerance) = method
    n_atoms = size(dictionary, 2)

    residual = copy(data)

    xdict = DefaultDict{Int, Float64}(0.)
    for i in 1:max_iter
        if norm(residual) < tolerance
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
