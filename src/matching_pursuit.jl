# The implementation is referencing the wikipedia page
# https://en.wikipedia.org/wiki/Matching_pursuit#The_algorithm
using LinearAlgebra
using DataStructures
using OhMyThreads: tmap
import SparseArrays: nonzeroinds

const default_max_nnz = 10
const default_rtol = 5e-2

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
    max_iter::Int = 4 * max_nnz
    rtol = default_rtol
    precompute_products = true
    refit_coeffs = false
    MatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end

@kwdef struct OrthogonalMatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4 * max_nnz
    rtol = default_rtol
    precompute_products = true
    OrthogonalMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end


""" Multithreaded version of `MatchingPursuit`.
Essentially falls back to single-threaded version automatically if julia is launched with
only one thread.

For description of parameters see `MatchingPursuit`.
"""
@kwdef struct ParallelMatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4 * max_nnz
    rtol = default_rtol
    precompute_products = true
    refit_coeffs = true
    ParallelMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end

"""
    SparseCodingMethodGPU

To be instantiated in extensions, e.g. `CUDAAccelMatchingPursuit`.
"""
abstract type GPUAcceleratedMatchingPursuit <: SparseCodingMethod end;
@kwdef struct CUDAAcceleratedMatchingPursuit <: KSVD.GPUAcceleratedMatchingPursuit
    max_nnz::Int = KSVD.default_max_nnz
    max_iter::Int = 4 * max_nnz
    rtol = KSVD.default_rtol
    precompute_products = true
    refit_coeffs = true
    batchsize = 2^12
    CUDAAcceleratedMatchingPursuit(args...) = (KSVD.validate_mp_args(args...); new(args...))
end

""" Original implementation by https://github.com/IshitaTakeshi/KSVD.jl.
Useful for comparison and didactic purposes, but much much slower. """
@kwdef struct LegacyMatchingPursuit <: SparseCodingMethod
    max_nnz::Int = default_max_nnz
    max_iter::Int = 4 * max_nnz
    rtol = default_rtol
    LegacyMatchingPursuit(args...) = (validate_mp_args(args...); new(args...))
end
function validate_mp_args(max_nnz, max_iter, rtol, other_args...)
    max_nnz >= 1 || throw(ArgumentError("`max_nnz` must be > 0"))
    max_iter >= 1 || throw(ArgumentError("`max_iter` must be > 0"))
    0.0 <= rtol <= 1.0 || throw(ArgumentError("`rtol` must be in [0,1]"))
end

function sparse_coding(data::AbstractMatrix, dictionary::AbstractMatrix, max_nnz=max(size(dictionary, 2) ÷ 10, 1);
    timer=TimerOutput(), DtD=nothing, sparse_coding_kwargs...)

    sparse_coding_method = ParallelMatchingPursuit(; max_nnz, sparse_coding_kwargs...)
    sparse_coding(sparse_coding_method, data, dictionary; DtD, timer)
end

get_method_map_fn(::MatchingPursuit) = map
get_method_map_fn(::ParallelMatchingPursuit) = tmap
get_method_map_fn(::OrthogonalMatchingPursuit) = tmap

"""
    sparse_coding(method::Union{MatchingPursuit, ParallelMatchingPursuit},
                  data::AbstractMatrix, dictionary::AbstractMatrix)

Find ``X`` such that ``DX = Y`` or ``DX ≈ Y`` where Y is `data` and D is `dictionary`.
"""
function sparse_coding(method::Union{MatchingPursuit,ParallelMatchingPursuit,OrthogonalMatchingPursuit},
    data::AbstractMatrix{T}, dictionary::AbstractMatrix{T}; timer=TimerOutput(), DtD=nothing, DtY=nothing) where {T}
    @timeit_debug timer "Sparse coding" begin

        map_fn = get_method_map_fn(method)

        @timeit_debug timer "Precompute DtD" begin
            DtD = (isnothing(DtD) ? dictionary' * dictionary : DtD)
        end
        # if the data is very large we might not want to precompute this.
        @timeit_debug timer "precompute products" begin
            products = (isnothing(DtY) ? (method.precompute_products ? (dictionary' * data) : fill(nothing, 1, size(data, 2)))
                        : copy(DtY))  # we need to copy here because we're about to operate in place on these
        end

        @timeit_debug timer "matching pursuit" begin
            X_ = let DtD = DtD  # avoid boxing: https://juliafolds2.github.io/OhMyThreads.jl/stable/literate/boxing/boxing/#Non-race-conditon-boxed-variables
                map_fn(czip(eachcol(data), eachcol(products))) do (datacol, productcol)
                    matching_pursuit_(
                        method,
                        datacol,
                        dictionary,
                        DtD;
                        products_init=(method.precompute_products ? productcol : nothing)
                    )
                end
            end
        end

        @timeit_debug timer "cat sparse arrays" begin
            # The "naive" version of `cat`ing the columns in X_ run into type inference problems for some reason.
            # The Julia base version of `reduce(hcat, X_)` is super slow.
            # This uses an overloaded definition defined in this package (in `utils.jl`).
            X = reduce(hcat, X_)
        end

    end  # @timeit
    return X
end

"""
OMP implementation. See Pati 1993
Orthogonal Matching Pursuit: Recursive Function Approximat ion with Applications to Wavelet Decomposition
(https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=342465)
"""
function matching_pursuit_(
    method::Union{OrthogonalMatchingPursuit},
    data::AbstractVector{T}, dictionary::AbstractMatrix{T}, DtD::AbstractMatrix{T};
    products_init::Union{Nothing,AbstractVector{T}}=nothing,
    debug=false) where {T}
    (; max_nnz, max_iter, rtol) = method

    n_atoms = size(dictionary, 2)
    residual = copy(data)
    # xdict = DefaultDict{Int, T}(zero(T))
    norm_data = norm(data)

    products_init = (isnothing(products_init) ? (dictionary' * residual) : products_init)
    products = copy(products_init)
    products_abs = abs.(products)  # prealloc
    # A_inv = zeros(T, 0, 0)
    A = zeros(T, 0, 0)
    inds = Int[]
    factors = T[]
    # atoms = Vector{T}[]
    b_k = zeros(T, 0)
    v_k = zeros(T, 0)
    dicts_chosen_buf = zeros(T, size(dictionary, 1), max_nnz)
    DtD_chosen_buf = zeros(T, size(dictionary, 2), max_nnz)
    # reconstruction = zeros(T, size(data))

    # will mask out "used indices" when finding next basis vector
    # mask = ones(Bool, size(products))
    mask = BitArray(undef, size(products))
    mask .= 1;

    for i in 1:max_nnz
        if norm(residual) / norm_data < rtol
            return SparseArrays.sparsevec(inds, factors, n_atoms)
        end

        # find an atom with maximum inner product
        products_abs .= abs.(products)
        _, maxindex = findmax_fast(products_abs .* mask)  # make sure to not pick used index
        dicts_chosen_buf[:, i] .= @view dictionary[:, maxindex]
        DtD_chosen_buf[:, i] .= @view DtD[:, maxindex]
        dicts_chosen = @view dicts_chosen_buf[:, 1:i]
        DtD_chosen = @view DtD_chosen_buf[:, 1:i]

        #==
        # The block matrix inversion updates =A_inv=, which stores =(D_I' * D_I)^-1=, where =D_I= is the sub-dictionary of atoms selected so far (=dictionary[:, inds]=). When a new atom (let's call it =d_new=, corresponding to =dictionary[:, maxindex]=) is added to =D_I=, the matrix =D_I' * D_I= is augmented.
        # This comes from the Sherman-Morrison-Woodbury equation, which inverts $(A + UV^\top)^{-1}$ blockwise.
        # Here, $A = Diagonal(G_{\rm old}, s)$ and $U=[u_1 u_2]$ and $W=[w_1 w_2]$ with $u_1 = [v; 0]$ and $w_1 = e_{k+1}$ and $u_2 = e_{k+1}$ and $w_2 = [v;0]$.
        ==#
        # beta = 1 / (1 - v_k' * b_k)
        v_k = @view DtD[inds, maxindex]
        b_k = A \ v_k
        # A_inv = [(A_inv+beta*(b_k*b_k')) -beta*b_k;
        #     -beta*b_k' beta]
        A = @view DtD_chosen[inds, :]
        # A_inv = inv(A)

        atom = @view dictionary[:, maxindex]
        γ_k = atom .- @view(dicts_chosen[:, 1:end-1]) * b_k
        α_k = products[maxindex] / sum(abs2, γ_k)
        factors .-= α_k * b_k

        push!(inds, maxindex)
        push!(factors, α_k)
        # push!(atoms, normalize(γ_k))
        mask[maxindex] = false
        products .= products_init .- DtD_chosen * factors

        # reconstruction .+= α_k * atom
        residual .-= α_k .* γ_k
    end
    return SparseArrays.sparsevec(inds, factors, n_atoms)
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

function refitcoefficients!(x::SparseVector, y, D)
    local_basis = @view D[:, nonzeroinds(x)]
    x.nzval .= (local_basis \ y)
    return x
end

@inbounds function matching_pursuit_(
    method::Union{MatchingPursuit,ParallelMatchingPursuit,GPUAcceleratedMatchingPursuit},
    data::AbstractVector{T}, dictionary::AbstractMatrix{T}, DtD::AbstractMatrix{T};
    products_init::Union{Nothing,AbstractVector{T}}=nothing) where {T}
    (; max_nnz, max_iter, rtol) = method

    n_atoms = size(dictionary, 2)
    residual = copy(data)
    xdict = DefaultDict{Int,T}(zero(T))
    norm_data = norm(data)

    products = (isnothing(products_init) ? (dictionary' * residual) : products_init)
    products_abs = abs.(products)  # prealloc

    for i in 1:max_iter
        # @assert(norm(residual)) && @show norm(residual), residual
        # @assert(isfinite(norm_data))
        if norm(residual) == 0 || norm(residual) / norm_data < rtol
            x = sparsevec(xdict, n_atoms)
            method.refit_coeffs && refitcoefficients!(x, data, dictionary)
            return x
        end
        if length(xdict) > max_nnz
            pop!(xdict, findmin(abs, xdict)[2])
            x = sparsevec(xdict, n_atoms)
            method.refit_coeffs && refitcoefficients!(x, data, dictionary)
            return x
        end

        # find an atom with maximum inner product
        products_abs .= abs.(products)
        _, maxindex = findmax_fast(products_abs)

        a = products[maxindex]
        atom = @view dictionary[:, maxindex]
        # @assert norm(atom) ≈ 1. norm(atom)

        residual .-= a .* atom
        products .-= a .* @view DtD[:, maxindex]

        xdict[maxindex] += a
    end
    x = sparsevec(xdict, n_atoms)
    method.refit_coeffs && refitcoefficients!(x, data, dictionary)
    return x
end

""" This is the original implementation by https://github.com/IshitaTakeshi, useful for
numerical comparison and didactic purposes.
`DtD` and `DtY` are just available for API completeness.
"""
function sparse_coding(method::LegacyMatchingPursuit, data::AbstractMatrix{T}, dictionary::AbstractMatrix{T};
    DtD=nothing, DtY=nothing, timer=TimerOutput()) where {T}
    @timeit_debug timer "Sparse coding" begin

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

    end  # @timeit
    return X
end

function matching_pursuit_(method::LegacyMatchingPursuit,
    data::AbstractVector{T},
    dictionary::AbstractMatrix{T}) where {T}
    (; max_nnz, max_iter, rtol) = method
    n_atoms = size(dictionary, 2)

    residual = copy(data)
    norm_data = norm(data)

    xdict = DefaultDict{Int,T}(zero(T))
    for i in 1:max_iter
        if norm(residual) / norm_data < rtol
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
