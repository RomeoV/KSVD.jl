using Distributions

abstract type AbstractDictionaryTrackingMethod end
@kwdef struct EnergyBasedReplacement <: AbstractDictionaryTrackingMethod
    beta::Float64 = 2
    proposal_strategy = ()
    maxreplacements::Int = 5
end

struct NoReplacement <: AbstractDictionaryTrackingMethod end

abstract type AbstractProposalStrategy end
struct TSVDProposalStrategy <: AbstractProposalStrategy end
@kwdef struct KSVDProposalStrategy <: AbstractProposalStrategy
    ndicts::Int = 10
    nnzpercol::Int = 3
    kwargs = (;)
end
struct ErrorSamplingProposalStrategy <: AbstractProposalStrategy end

function fasterror!(E, Y, D, X; timer::TimerOutput=TimerOutput())
    @timeit_debug timer "compute fast error" begin
        E .= copy(Y)
        fastdensesparsemul_threaded!(E, D, X, -1, 1)
        return E
    end
end

"""
    replace_atoms!(strategy::AbstractAtomReplacementStrategy, tracker, ...)

Main dispatch function for atom replacement strategies.
"""
function replace_atoms!(
    strategy::EnergyBasedReplacement,
    tracker::AbstractDictionaryTrackingMethod,
    Y::AbstractMatrix{T},
    D::AbstractMatrix{T},
    X::AbstractSparseMatrix{T},
    sparse_coding_method::SparseCodingMethod;
    timer::TimerOutput=TimerOutput(),
    E::AbstractMatrix{T}=fasterror!(similar(Y), Y, D, X; timer),
    DtD=D' * D,
    DtY=D' * Y,
    verbose::Bool=false
) where {T}

    num_replaced = 0
    fn = energyfunction(tracker)

    for attempt in 1:strategy.maxreplacements
        # DtD = D' * D
        # DtY = D' * Y
        # E = Y - D * X

        energy_old, idx = findmin(values(tracker))
        d_new = proposecandidate(strategy.proposal_strategy, Y, D, X; timer, verbose, E)
        energy_new = evaluate_candidate_energy(d_new, Y, D, sparse_coding_method, fn; timer, DtD, DtY)

        @show (energy_new, energy_old)
        if energy_new > strategy.beta * energy_old
            (d_old, X_old) = D[:, idx], copy(X)
            replaceatom!(D, idx, d_new, Y; timer, DtD, DtY)
            @assert DtY ≈ D' * Y
            X = sparse_coding(sparse_coding_method, Y, D; timer, DtD, DtY)

            fasterror!(E, Y, D, X)
            # updateerror!(E, Y, D, d_old, d_new, X_old, X, idx)
            resetstats!(tracker, idx, energy_new)
            num_replaced += 1
            @assert DtY ≈ D' * Y
        else
            break # Stop if the best candidate isn't accepted
        end
    end # End replacement attempts loop

    return num_replaced
end

function evaluate_candidate_energy(d_new, Y, D, sparse_coding_method, fn=abs2; timer=TimerOutput(), DtD=D' * D, DtY=D' * Y)
    D′ = [D d_new]
    Dtdnew = D' * d_new
    D′tD′ = [DtD Dtdnew;
        Dtdnew' one(eltype(DtD))]
    D′tY = [DtY;
        d_new' * Y]
    X = sparse_coding(sparse_coding_method, Y, D′; DtD=D′tD′, DtY=D′tY)
    # X = sparse_coding(sparse_coding_method, Y, D′)
    return sum(fn, X[end, :])
end

function proposecandidate(strat::ErrorSamplingProposalStrategy, Y, D, X, np::Int=0;
    timer::TimerOutput=TimerOutput(),
    E=fasterror!(similar(Y), Y, D, X),
    verbose=false)
    @timeit_debug timer "propose candidate (error sampling)" begin
        m = size(E, 1)
        errs = norm.(eachcol(E)) ./ norm(eachcol(Y))
        idx = rand(Categorical(normalize(errs, 1)))
        return Y[:, idx]
    end
end

function proposecandidate(strat::KSVDProposalStrategy, Y, D, X, np::Int=min(4 * size(Y, 1), size(Y, 2));
    timer::TimerOutput=TimerOutput(),
    E=fasterror!(similar(Y), Y, D, X),
    verbose=false
)
    @timeit_debug timer "propose candidate (ksvd proposal)" begin
        m = size(E, 1)
        errs = norm.(eachcol(E)) ./ norm(eachcol(Y))
        p = sortperm(errs, rev=true)
        Ebuf = E[:, p[1:np]]
        (; D, X) = ksvd(Ebuf, strat.ndicts, strat.nnzpercol;
            ksvd_update_method=KSVD.OptimizedKSVD(; shuffle_indices=true),
            timer, maxiters=50, abstol=1e-3, show_trace=true, verbose,
            strat.kwargs...
        )
        _, idx = findmax(sum(abs2, X; dims=2))
        return D[:, idx]
    end
end

function proposecandidate(strat::TSVDProposalStrategy, E, Y, D, X, np::Int=min(size(E)...);
    timer::TimerOutput=TimerOutput(), verbose=false
)

    m = size(E, 1)
    errs = norm.(eachcol(E)) ./ norm(eachcol(Y))
    p = sortperm(errs, rev=true)
    Ebuf = E[:, p[1:np]]
    U, S, Vt = compute_truncated_svd(ArnoldiSVDSolver{eltype(Ebuf)}(), Ebuf, 1)
    return U[:, 1]
end

function replaceatom!(D, idx, d_new, Y; timer=TimerOutput(), DtD=nothing, DtY=nothing)
    @timeit_debug timer "Atom Replace: Perform Update" begin
        D[:, idx] .= d_new
        if !isnothing(DtD)
            Dtd = D' * d_new
            DtD[:, idx] .= Dtd
            DtD[idx, :] .= Dtd
            DtD[idx, idx] = 1
        end
        if !isnothing(DtY)
            DtY[idx, :] = d_new' * Y
        end
    end
end
function updateerror!(E, d_old, d_new, Xold, Xnew, idx)
    error("""
This actually doesn't work the way I thought.
Since the new sparse coding doesn't just update values related to the new dictionary, but may also update other elements, we have to check all new updates.
""")
    xₖ = Xold[idx, :]
    ωₖ = nonzeroinds(xₖ)
    E[:, ωₖ] .+= d_old * xₖ[ωₖ]'

    xₖ = Xnew[idx, :]
    ωₖ = nonzeroinds(xₖ)
    E[:, ωₖ] .-= d_new * xₖ[ωₖ]'
end

# Method for NoReplacement strategy
function replace_atoms!(
    strategy::NoReplacement,
    tracker, Y_batch, D, X_batch, sparse_coding_method, timer; verbose=false
)
    return 0 # Do nothing
end
