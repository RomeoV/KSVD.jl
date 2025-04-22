using OnlineStats
import OnlineStats: fit!, value, Mean, ExponentialWeight # Specify types used

"""
    AbstractDictionaryTrackingMethod

Abstract supertype for methods that track the utility or statistics
of dictionary atoms during iterative learning (like KSVD).
"""
abstract type AbstractDictionaryTrackingMethod end

"""
    update!(method::AbstractDictionaryTrackingMethod, D, X, Y)

Update the internal statistics of the tracking `method` based on the
dictionary `D`, sparse coefficients `X`, and optionally data `Y`
from the latest iteration or batch.

For the provided implementations, only `X` is actively used.
"""
function update!(method::AbstractDictionaryTrackingMethod, X::AbstractSparseMatrix)
    # Default fallback if a concrete type doesn't implement it
    error("update! not implemented for type $(typeof(method))")
end

"""
    values(method::AbstractDictionaryTrackingMethod) -> Vector{Float64}

Return a vector of values, one for each dictionary atom, indicating their
current tracked utility or status. Lower values generally indicate candidates
for replacement.
"""
function values(method::AbstractDictionaryTrackingMethod)
    error("values not implemented for type $(typeof(method))")
end

"""
    reset_stats!(method::AbstractDictionaryTrackingMethod, indices)

Reset the statistics for the atoms specified by `indices`. This is typically
called after replacing these atoms in the dictionary.
"""
function resetstats!(method::AbstractDictionaryTrackingMethod, indices)
    error("resetstats! not implemented for type $(typeof(method))")
end

function resetstats!(method::AbstractDictionaryTrackingMethod, indices, val)
    error("resetstats! not implemented for type $(typeof(method))")
end


# --- Strategy 1: EWMA of Usage ---

"""
    EWMAUsageTracking(K::Int, alpha::Real=0.01; usage_threshold::Real=1e-8)

Tracks the Exponentially Weighted Moving Average (EWMA) of atom usage.
An atom is considered "used" in a batch if the sum of absolute values of its
coefficients exceeds `usage_threshold`.
"""
struct EWMAUsageTracking{fn} <: AbstractDictionaryTrackingMethod
    stats::Vector{Mean{Float64,ExponentialWeight}}
    EWMAUsageTracking(fn, K::Int, lookback::Int=10) = new{fn}(
        [Mean(weight=ExponentialWeight(lookback)) for _ in 1:K]
    )
    EWMAUsageTracking(K::Int, lookback::Int=10) = EWMAUsageTracking(abs2, K, lookback)
end

energyfunction(::EWMAUsageTracking{fn}) where {fn} = fn

function update!(method::EWMAUsageTracking{fn}, X::AbstractSparseMatrix) where {fn}
    length(method.stats) == size(X, 1) || error("Mismatch between X rows ($(size(X,1))) and tracker size ($(length(method.stats))).")
    energies = sum(fn, X; dims=2)
    fit!.(method.stats, energies)
end

function values(method::EWMAUsageTracking)
    return value.(method.stats)
end

resetstats!(method::EWMAUsageTracking, idx::Int) = resetstats!(method, [idx])
resetstats!(method::EWMAUsageTracking, idx::Int, val) = resetstats!(method, [idx], val)
function resetstats!(method::EWMAUsageTracking, indices::AbstractVector)
    for k in indices
        method.stats[k] = Mean(methods.stats[k].weight)
    end
end
function resetstats!(method::EWMAUsageTracking, indices::AbstractVector, val)
    for k in indices
        method.stats[k] = Mean(Float64(val), method.stats[k].weight, 0)
    end
end


# --- Example Usage (Conceptual) ---
# Assume K is dictionary size, N is batch size, P is signal dimension

# K = 100 # Number of atoms
# tracker = EWMAUsageTracking(K, 0.01)
# # tracker = MeanUsageTracking(K)
# # tracker = MeanEnergyTracking(K)

# for i in 1:num_batches
#     # --- Get Mini-Batch ---
#     # Y_batch = get_next_batch(...) # Matrix P x N
#
#     # --- Sparse Coding ---
#     # X_batch = perform_sparse_coding(D, Y_batch) # Matrix K x N
#
#     # --- Dictionary Update ---
#     # D_new = update_dictionary(D, X_batch, Y_batch) # Matrix P x K
#     # D = D_new
#
#     # --- Update Tracking Stats ---
#     # Use D_new or D depending on when you check
#     # update!(tracker, D, X_batch, Y_batch)
#
#     # --- Periodic Replacement Check ---
#     # if i % replacement_check_interval == 0
#     #     replacement_threshold = 0.05 # Example threshold
#     #     candidate_indices = get_candidate_indices(tracker, replacement_threshold)
#
#     #     if !isempty(candidate_indices)
#     #         println("Replacing atoms at indices: ", candidate_indices)
#     #         # Find high-error signals from Y_batch or recent batches
#     #         # high_error_signals = find_high_error_signals(D, Y_batch, X_batch, length(candidate_indices))
#
#     #         # Perform replacement in D
#     #         # for (idx, signal_to_replace_with) in zip(candidate_indices, high_error_signals)
#     #         #     D[:, idx] = normalize(signal_to_replace_with)
#     #         # end
#
#     #         # Reset stats for replaced atoms
#     #         # reset_stats!(tracker, candidate_indices)
#     #     end
#     # end
# end
