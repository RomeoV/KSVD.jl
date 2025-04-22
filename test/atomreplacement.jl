using StatsBase
using SparseArrays
using Random
using LinearAlgebra
using Test
using Distributions


m, n = 2 * 64, 2 * 256
nsamples = 10_000
nnzpercol = 5
T = Float32

Dgt = KSVD.init_dictionary(Float32, m, n)
Xgt = stack(
    (SparseVector(n, sort(sample(1:n, nnzpercol; replace=false)), rand(T, nnzpercol))
     for _ in 1:nsamples);
    dims=2)
Ygt = Dgt * Xgt
Ymeas = Ygt + T(0.05) * randn(T, size(Ygt))

Dhat = copy(Dgt)
idx = rand(axes(Dhat, 2))
normalize!(Distributions.randn!(@view Dhat[:, idx]))
d_old = Dhat[:, idx]
Xhat = KSVD.sparse_coding(Ymeas, Dhat, nnzpercol)
Ehat = Ymeas - Dhat * Xhat


# test whether this recovers the "missing" dictionary vector
d_new = KSVD.proposecandidate((), Ehat, Ymeas, Dhat, Xhat)
@test abs(dot(d_new, Dgt[:, idx])) > 0.9


DtD = Dhat' * Dhat;
DtY = Dhat' * Ymeas;

energy = KSVD.evaluate_candidate_energy(d_new, Ymeas, Dhat, KSVD.ParallelMatchingPursuit(; max_nnz=nnzpercol); DtD, DtY)
let
    D′ = [Dhat d_new]
    X = sparse_coding(Ymeas, D′, nnzpercol)
    @test energy ≈ sum(abs2, X[end, :])
    @test energy > sum(abs2, X[idx, :])
end

KSVD.replaceatom!(Dhat, idx, d_new, Ymeas; DtD, DtY)
@assert DtD ≈ Dhat' * Dhat
@assert DtY ≈ Dhat' * Ymeas



let
    Dgt = KSVD.init_dictionary(Float32, m, n)
    Xgt = stack(
        (SparseVector(n, sort(sample(1:n, nnzpercol; replace=false)), rand(T, nnzpercol))
         for _ in 1:nsamples);
        dims=2)
    Ygt = Dgt * Xgt
    Ymeas = Ygt + T(0.0) * randn(T, size(Ygt))

    Dhat = copy(Dgt)
    idx = rand(axes(Dhat, 2))
    normalize!(Distributions.randn!(@view Dhat[:, idx]))
    d_old = Dhat[:, idx]
    Xhat = KSVD.sparse_coding(Ymeas, Dhat, nnzpercol)
    Ehat = Ymeas - Dhat * Xhat

    tracker = KSVD.EWMAUsageTracking(n)
    for i in axes(X, 1)
        KSVD.resetstats!(tracker, i, sum(Xhat[i, :]))
    end


    # @info sum(Xhat), sum(Dhat)
    KSVD.replace_atoms!(
        KSVD.EnergyBasedReplacement(; beta=2),
        tracker,
        Ymeas,
        Dhat,
        Xhat,
        KSVD.ParallelMatchingPursuit(; max_nnz=nnzpercol),
    )
    # @info sum(Xhat), sum(Dhat)
    # Xhat = sparse_coding(Ymeas, Dhat, nnzpercol)
    # KSVD.replace_atoms!(
    #     KSVD.EnergyBasedReplacement(; beta=2, maxreplacements=1),
    #     tracker,
    #     Ymeas,
    #     Dhat,
    #     Xhat,
    #     KSVD.ParallelMatchingPursuit(; max_nnz=nnzpercol),
    # )
    # @info sum(Xhat), sum(Dhat)
end

m, n = 1024, 4 * 1024
nsamples = 32_000
nnzpercol = 40
T = Float32
for ndicts in 1:5
    Dgt = KSVD.init_dictionary(Float32, m, n)
    Xgt = stack(
        (SparseVector(n, sort(sample(1:n, nnzpercol; replace=false)), 1 .+ rand(T, nnzpercol))
         for _ in 1:nsamples);
        dims=2)
    Ygt = Dgt * Xgt
    Ymeas = Ygt + T(0.00) * randn(T, size(Ygt))

    Dhat = copy(Dgt)
    indices = sample(axes(Dhat, 2), ndicts; replace=false)
    for idx in indices
        normalize!(Distributions.randn!(@view Dhat[:, idx]))
    end
    ds_old = [Dhat[:, i] for i in indices]
    Xhat = KSVD.sparse_coding(Ymeas, Dhat, nnzpercol)
    Ehat = KSVD.fasterror!(similar(Ymeas), Ymeas, Dhat, Xhat)

    tracker = KSVD.EWMAUsageTracking(n)
    for i in axes(X, 1)
        KSVD.resetstats!(tracker, i, sum(Xhat[i, :]))
    end

    # @info sum(Xhat), sum(Dhat)
    nreplaced = KSVD.replace_atoms!(
        KSVD.EnergyBasedReplacement(; beta=4, maxreplacements=10),
        tracker,
        Ymeas,
        Dhat,
        Xhat,
        KSVD.ParallelMatchingPursuit(; max_nnz=nnzpercol);
        E=Ehat
    )
    @test ndicts == nreplaced
    @info nreplaced
end

m, n = 2*1024, 4 * 2*1024
nsamples = 64_000
nnzpercol = 40
T = Float32
for ndicts in 1:3
    Dgt = KSVD.init_dictionary(Float32, m, n)
    Xgt = stack(
        (SparseVector(n, sort(sample(1:n, nnzpercol; replace=false)), 1 .+ rand(T, nnzpercol))
         for _ in 1:nsamples);
        dims=2)
    Ygt = Dgt * Xgt
    Ymeas = Ygt + T(0.01) * randn(T, size(Ygt))

    Dhat = copy(Dgt)
    indices = sample(axes(Dhat, 2), ndicts; replace=false)
    for idx in indices
        normalize!(Distributions.randn!(@view Dhat[:, idx]))
    end
    ds_old = [Dhat[:, i] for i in indices]
    Xhat = KSVD.sparse_coding(Ymeas, Dhat, nnzpercol)
    Ehat = KSVD.fasterror!(similar(Ymeas), Ymeas, Dhat, Xhat)

    tracker = KSVD.EWMAUsageTracking(n)
    for i in axes(Xhat, 1)
        KSVD.resetstats!(tracker, i, sum(Xhat[i, :]))
    end

    # @info sum(Xhat), sum(Dhat)
    nreplaced = KSVD.replace_atoms!(
        KSVD.EnergyBasedReplacement(; beta=1.5, maxreplacements=10,
            proposal_strategy=KSVD.KSVDProposalStrategy()),
        tracker,
        Ymeas,
        Dhat,
        Xhat,
        KSVD.ParallelMatchingPursuit(; max_nnz=nnzpercol);
        E=Ehat
    )
    @test ndicts == nreplaced
    @info nreplaced
end

# updateerror! doesn't work currently.
# Xold = copy(Xhat)
# Xnew = sparse_coding(Ymeas, Dhat, nnzpercol)
# KSVD.updateerror!(Ehat, d_old, d_new, Xold, Xnew, idx)
# @assert Ehat ≈ Ymeas - Dhat * Xnew



# for eps in [0.01, 0.05, 0.1, 0.5, 1], repeat in 1:5
#     Ymeas = Ygt + T(eps) * randn(T, size(Ygt))
#     Dhat = copy(Dgt)
#     idx = rand(axes(Dhat, 2))
#     normalize!(Distributions.randn!(@view Dhat[:, idx]))
#     Xhat = KSVD.sparse_coding(Ymeas, Dhat, nnzpercol)
#     Ehat = Ymeas - Dhat * Xhat
#     d_new = KSVD.proposecandidate((), Ehat, Ymeas, Dhat, Xhat, size(Ehat, 2))
#     @info eps, dot(d_new, Dgt[:, idx])
#     # we find that for eps <= 0.1, we can recover the missing dict pretty well.
#     # this is interestingly pretty much the same for 128 samples and 10k samples.
#     # So the identifyability seems more related to the signal to noise ratio,
#     # rather than the number of samples.
# end
