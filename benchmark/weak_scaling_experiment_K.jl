import Pkg; Pkg.activate("@SVDBench"); Pkg.instantiate()
using KSVD, BenchmarkTools, Serialization, CSV, DataFrames
import StatsBase: std
import LinearAlgebra
datafile = joinpath([pkgdir(KSVD), "benchmark", "data", "embeddings.jls"])
data = deserialize(datafile) |> Matrix
@info "nthreads: $(Threads.nthreads())"

K_max, M_max = size(data)
N_procs = Threads.nthreads()
LinearAlgebra.BLAS.set_num_threads(N_procs)
N_procs_max = 8
K_per_proc = K_max÷N_procs_max
K = K_per_proc * N_procs

if length(ARGS) > 0 && ARGS[1] == "DRY"
    bres = @benchmark rand(100, 100)\rand(100) samples=7
else
    bres = @benchmark dictionary_learning($data[1:K, 1:10_000], K, max_iter=1,
                           ksvd_method=KSVD.ParallelKSVD(),
                           sparse_coding_method=KSVD.ParallelMatchingPursuit()) samples=2
end

results_row = DataFrame([
    :nproc=>N_procs,
    :Kfactor=>1,
    :min=>minimum(bres).time,
    :max=>maximum(bres).time,
    :mean=>mean(bres).time,
    :median=>median(bres).time,
    :std=>std(bres.times)
])

resultsfile = joinpath([pkgdir(KSVD), "benchmark", "data", "weak_scaling_in_K_results.csv"])
results_df_so_far = if isfile(resultsfile)
    CSV.read(resultsfile, DataFrame)
else
    DataFrame(fill(Float64[], ncol(results_row)), names(results_row))
end
results_df = append!(results_df_so_far, results_row)
CSV.write(resultsfile, results_df)
