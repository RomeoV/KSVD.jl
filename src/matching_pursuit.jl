using DataStructures

# The implementation is referencing the wikipedia page
# https://en.wikipedia.org/wiki/Matching_pursuit#The_algorithm
using LinearAlgebra
using Transducers
using Match
import SparseArrays: nonzeroinds

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
@kwdef struct FasterParallelMatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
end
@kwdef struct FullBatchMatchingPursuit <: SparseCodingMethod
    max_iter::Int = default_max_iter_mp
    tolerance = default_tolerance
end

""" Redefine findmax for vector of floats to not do nan-checks.

By default,`findmax` uses `isless`, which does a nan-check before computing `<(lhs, rhs)`.
We roll basically the same logic as in `Julia/Base/reduce.jl:findmax` but we directly use `<`, which gives us about a 1.5x speedup.
"""
function findmax(data::Vector{Float64})
    cmp_tpl((fm, im), (fx, ix)) = (fm < fx) ? (fx, ix) : (fm, im)
    mapfoldl( ((k, v),) -> (v, k), cmp_tpl, pairs(data))
end

@inbounds function matching_pursuit_(
        data::AbstractVector, dictionary::AbstractMatrix, DtD::AbstractMatrix,
        max_iter::Int, tolerance::Float64;
        products_init::Union{Nothing, AbstractVector}=nothing) :: SparseVector{Float64, Int}
    n_atoms = size(dictionary, 2)

    residual = copy(data)

    xdict = DefaultDict{Int, Float64}(0.)
    products = (isnothing(products_init) ? dictionary' * residual : products_init)
    products_abs = abs.(products)  # prealloc
    # products_abs_heap = MutableBinaryHeap(Base.By(last, DataStructures.FasterReverse()),
    #                                       collect(enumerate(products_abs)))
    # products_abs_heap = BinaryHeap(Base.By(last, DataStructures.FasterReverse()),
    #                                collect(enumerate(products_abs)))

    # pre-sorting doesn't provide much speedup and is algorithmically questionable...
    # sorted_ind = sortperm(products_abs, order=Base.Order.Reverse)
    for i in 1:max_iter
        if norm(residual) < tolerance
            return sparsevec(xdict, n_atoms)
        end

        # find an atom with maximum inner product
        # @inbounds products .= dictionary' * residual
        # maxval, maxidx = findmax(products)
        # minval, minidx = findmin(products)
        # products_abs .= products  # basically) abs. without alloc
        # products_abs .*= (-1 .^ signbit.(products))

        products_abs .= abs.(products)

        _, maxindex = findmax(products_abs)
        # maxindex, _ = pop!(products_abs_heap)
        # maxindex = sorted_ind[i]

        maxval = products[maxindex]
        @inbounds atom = @view dictionary[:, maxindex]

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
        residual .-= a .* atom
        @inbounds products .-= a .* @view DtD[:, maxindex]

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

    collect_fn = @match method begin
        ::MatchingPursuit => collect
        ::ParallelMatchingPursuit => tcollect
        _ => collect
    end

    DtD = dictionary'*dictionary
    products = dictionary' * data

    X_::Vector{SparseVector{Float64, Int}} = collect_fn(
        matching_pursuit_(
                datacol,
                dictionary,
                DtD,
                method.max_iter,
                method.tolerance,
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

function sparse_coding(method::FasterParallelMatchingPursuit, data::AbstractMatrix, dictionary::AbstractMatrix)
    K = size(dictionary, 2)
    N = size(data, 2)

    DtD = dictionary'*dictionary
    products = dictionary' * data

    # I_buffers = [Int[] for _ in 1:Threads.nthreads()]
    # J_buffers = [Int[] for _ in 1:Threads.nthreads()]
    # V_buffers = [Float64[] for _ in 1:Threads.nthreads()]
    I = Int[]; J = Int[]; V = Float64[];

    # Threads.@threads :static for j  in axes(data, 2)
    #     datacol = @view data[:, j]; productcol = @view products[:, j]
    #     data_vec = matching_pursuit_(
    #                     datacol,
    #                     dictionary,
    #                     DtD,
    #                     method.max_iter,
    #                     method.tolerance,
    #                     products_init=productcol
    #                 )
    #     I_buffer = I_buffers[Threads.threadid()]; J_buffer = J_buffers[Threads.threadid()]; V_buffer = V_buffers[Threads.threadid()];
    #     append!(I_buffer, nonzeroinds(data_vec)),
    #     append!(J_buffer, fill(j, nnz(data_vec))),
    #     append!(V_buffer, nonzeros(data_vec))
    # end

    I_buffers = [Int[] for _ in 1:size(data, 2)]; V_buffers = [Float64[] for _ in 1:size(data, 2)];
    # @floop for (j, (datacol, productcol)) in enumerate(zip(eachcol(data), eachcol(products)))
    # ThreadPools.@bthreads for j in axes(data, 2)
    Threads.@threads for j in axes(data, 2)
        # (j, (datacol, productcol)) in enumerate(zip(eachcol(data), eachcol(products)))
        datacol = @view data[:, j]; productcol = @view products[:, j]
        data_vec = matching_pursuit_(
                        datacol,
                        dictionary,
                        DtD,
                        method.max_iter,
                        method.tolerance,
                        products_init=productcol
                    )
        # I_buffer = I_buffers[Threads.threadid()]; J_buffer = J_buffers[Threads.threadid()]; V_buffer = V_buffers[Threads.threadid()];
        append!(I_buffers[j], nonzeroinds(data_vec))
        append!(V_buffers[j], nonzeros(data_vec))
        # @reduce(
        #     I = append!!(EmptyVector{Int}(), nonzeroinds(data_vec)),
        #     J = append!!(EmptyVector{Int}(), fill(j, nnz(data_vec))),
        #     V = append!!(EmptyVector{Float64}(), nonzeros(data_vec))
        # )
    end

    # The "naive" version of `cat`ing the columns in X_ run into type inference problems for some reason.
    # I first tried `hcat(X_...)`, but it was somewhat slow.
    # Then I tried `X = spzeros(Float64, K, N); for (i, col) in enumerate(X_); X[:, i]=col; end` but that
    # was also bad somehow.
    # This version seems to overcome the type inference issues and makes the code much faster.
    # I = vcat(I_buffers...)
    # J = vcat(J_buffers...)
    # V = vcat(V_buffers...)
    I = vcat(I_buffers...); V = vcat(V_buffers...)
    J = vcat([fill(j, size(I_buf)) for (j, I_buf) in enumerate(I_buffers)]...)
    X = sparse(I,J,V, K, N)
    return X
end

sparse_coding(data::AbstractMatrix, dictionary::AbstractMatrix) = sparse_coding(ParallelMatchingPursuit(), data, dictionary)

function sparse_coding(method::FullBatchMatchingPursuit, data::AbstractMatrix, dictionary::AbstractMatrix)
    K = size(dictionary, 2)
    N = size(data, 2)
    max_iter = method.max_iter

    DtD = dictionary'*dictionary
    products = dictionary' * data
    products_abs = abs.(products)

    # I = Int[]; J = Int[]; V = Float64[];
    I_buffers = [Int[] for _ in 1:size(data, 2)]; V_buffers = [Float64[] for _ in 1:size(data, 2)];

    max_product_inds = sortperm.(eachcol(products_abs), rev=true)
    @floop for data_idx in axes(data, 2)
    # for data_idx in axes(data, 2)
        inds = max_product_inds[data_idx][1:max_iter]

        coeffs = products[inds, data_idx] ./ sum(abs2, dictionary[:, inds], dims=1)[:]
        residuals = data[:, data_idx] .- cumsum(coeffs' .* dictionary[:, inds], dims=2)
        residual_norms = norm.(eachcol(residuals)).^2
        last_idx::Int = let
            idx = findfirst(<(method.tolerance), residual_norms)
            (isnothing(idx) ? max_iter : idx)
        end

        data_vec = sparsevec(inds[1:last_idx], coeffs[1:last_idx])
        append!(I_buffers[data_idx], nonzeroinds(data_vec))
        append!(V_buffers[data_idx], nonzeros(data_vec))
        # append!(I, nonzeroinds(data_vec))
        # append!(J, fill(data_idx, nnz(data_vec)))
        # append!(V, nonzeros(data_vec))
        # @reduce(
        #     I = append!!(EmptyVector{Int}(), nonzeroinds(data_vec)),
        #     J = append!!(EmptyVector{Int}(), fill(data_idx, nnz(data_vec))),
        #     V = append!!(EmptyVector{Float64}(), nonzeros(data_vec))
        # )
    end
    I = vcat(I_buffers...); V = vcat(V_buffers...)
    J = vcat([fill(j, size(I_buf)) for (j, I_buf) in enumerate(I_buffers)]...)

    X = sparse(I,J,V, K, N)
    return X
end

function matching_pursuit_original_(data::AbstractVector, dictionary::AbstractMatrix,
                           max_iter::Int, tolerance::Float64)
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
