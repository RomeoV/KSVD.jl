using Accessors
abstract type KSVDLoopType end

struct NormalLoop <: KSVDLoopType end

@kwdef struct MatryoshkaLoop <: KSVDLoopType
    log2min::Int = 6
end


"""
    ksvd_loop!(
    ksvd_loop_type::KSVDLoopType,
    ksvd_update_method,
    sparse_coding_method,
    Y,
    D,
    X;
    timer=TimerOutput(),
    verbose=false
)

`D` will be overwritten in this loop.
`Y` may be a (potentially shuffled) subset of `Y`.
"""
function ksvd_loop!(
    ksvd_loop_type::NormalLoop,
    ksvd_update_method,
    sparse_coding_method,
    Y,
    D,
    X;
    timer=TimerOutput(),
    yidx=axes(Y, 2),
    verbose=false
)
    verbose && @info "Starting svd"
    ksvd_update(ksvd_update_method, maybeview(Y, :, yidx), D, X[:, yidx]; timer)
    verbose && @info "Starting sparse coding"
    X = sparse_coding(sparse_coding_method, Y, D; timer)
end

"""
    ksvd_loop!(
    ksvd_loop_method::MatryoshkaLoop,
    ksvd_update_method,
    sparse_coding_method,
    Y,
    D,
    X;
    timer=TimerOutput(),
    yidx=axes(Y, 2),
    verbose=false
)

This is the "heart" of KSVD, together with the Matryoshka modification.
We assume that we already have computed `X`.
Then, the dictionaries are "grouped" into disjoint groups of exponentially growing size.
E.g., by default the first and second group have 64 elements, the third 128, the fourth 256 and so forth.
Each group gets an equal share of the `nnz` budget, although the value is rounded up (`ceil`),
so in practice the true nnz may be slightly larger than `max_nnz`.

Then for each group we first update (in-place) D, and the compute a new `X` (out-of-place).
Then, we compute the error matrix just using the current slices of `D` and `X`, and continue to the next group, but focussing only on the error matrix.

`yidx` may be provided in a similar way to the `NormalLoop` implementation.
"""
function ksvd_loop!(
    ksvd_loop_method::MatryoshkaLoop,
    ksvd_update_method,
    sparse_coding_method,
    Y,
    D,
    X;
    timer=TimerOutput(),
    yidx=axes(Y, 2),
    verbose=false
)
    E = copy(Y)
    X_slices = typeof(X)[]

    Msets = constructM(size(D, 2); log2min=ksvd_loop_method.log2min)
    nnzbudget = sparse_coding_method.max_nnz
    localnnzbudget = ceil(Int, nnzbudget / length(Msets))
    sparse_coding_method′ = @set sparse_coding_method.max_nnz = localnnzbudget

    for midx in Msets
        D′ = @view D[:, midx]
        X′ = copy(X[midx, :])
        verbose && @info "Running ksvd update for $(midx)"
        ksvd_update(ksvd_update_method, maybeview(E, :, yidx), D′, X′[:, yidx]; timer)
        verbose && @info "Running sparse coding"
        X′ = sparse_coding(sparse_coding_method′, E, D′; timer)
        E .-= D′ * X′
        push!(X_slices, X′)
    end
    X = reduce(vcat, X_slices)
    return X
end

constructM(K; log2min=6) =
    let
        mmax = unique([2 .^ (log2min:floor(Int, log2(K))); K])
        map(zip([0; mmax[1:end-1]], mmax)) do (lhs, rhs)
            (lhs+1):rhs
        end
    end

function sparse_coding_matryoshka(Y::AbstractMatrix{T}, D, max_nnz;
    log2min=8, DtD=D' * D, DtY=D' * Y) where {T}
    sparse_coding_matryoshka(ParallelMatchingPursuit(; max_nnz), Y, D; log2min, DtD, DtY)
end

function sparse_coding_matryoshka(sparse_coding_method::SparseCodingMethod,
    Y::AbstractMatrix{T}, D; log2min=8, DtD=D' * D, DtY=D' * Y) where {T}
    E = copy(Y)
    X_slices = SparseMatrixCSC{T}[]

    Msets = constructM(size(D, 2); log2min)
    nnzbudget = sparse_coding_method.max_nnz
    localnnzbudget = round(Int, nnzbudget / length(Msets))
    sparse_coding_method′ = @set sparse_coding_method.max_nnz = localnnzbudget

    for midx in Msets
        D′ = @view D[:, midx]
        D′tD′ = @view DtD[midx, midx]
        D′tY = @view DtY[midx, :]
        X′ = sparse_coding(sparse_coding_method′, E, D′; DtD=D′tD′, DtY=D′tY)
        E .-= D′ * X′
        push!(X_slices, X′)
    end
    X = reduce(vcat, X_slices)
    return X
end
