import Pkg; Pkg.activate("benchmark"); Pkg.instantiate()
using KSVD, BenchmarkTools
import StatsBase: std
using DataStructures
using JSON
include("generate_benchmark_data.jl")

E = 1000; K=2000; avg_nnz_per_sample=10
suite = BenchmarkGroup()
suite[:sparse_coding] = BenchmarkGroup([],Dict(
    string(typeof(method))=>BenchmarkGroup([], Dict(
        N => try; @benchmarkable KSVD.sparse_coding($method, data, dictionary) setup=begin
            data = generate_benchmark_data($N, $E, $K, $avg_nnz_per_sample)[:data]
            dictionary = KSVD.init_dictionary($E, $K)
        end; catch; nothing; end
        for N in round.(Int, 10 .^ (2:0.5:5.5))))
    for method in [KSVD.MatchingPursuit(), KSVD.ParallelMatchingPursuit(), KSVD.CUDAAcceleratedMatchingPursuit()]))
suite[:basis_improvement] = BenchmarkGroup([], Dict(
    string(method_t)=>BenchmarkGroup([], Dict(
        N => try; @benchmarkable KSVD.ksvd(method, data, dictionary, X) setup=begin
            tpl = generate_benchmark_data($N, $E, $K, $avg_nnz_per_sample)
            data = tpl.data
            dictionary = KSVD.init_dictionary($E, $K)
            X = KSVD.sparse_coding(KSVD.ParallelMatchingPursuit(), data, dictionary)
            method = $method_t(Float64, $E, $K, $N, ; pct_nz=0.1)
        end teardown=(GC.gc()); catch; nothing; end
        for N in round.(Int, 10 .^ (4.0:1.5:5.5))))
    for method_t in [
        KSVD.OptimizedKSVD,
        KSVD.ParallelKSVD,
        KSVD.BatchedParallelKSVD
    ]))
tune!(suite)
bmres = run(suite)
BenchmarkTools.save(joinpath(pkgdir(KSVD), "benchmark", "weak_scaling_experiment_N.json"), bmres)





# fig = Figure()
# ax = Axis(fig[1,1];
#           xscale=log10, yscale=log2,
#           xlabel="Number of samples",
#           ylabel="Time [s]")
# for (method, grp) in bmres["sparse_coding"]
#     data = SortedDict(k=>t/1e9 for (k,t) in time(mean(grp)))
#     err  = SortedDict(k=>t/1e9 for (k,t) in time( std(grp)))
#     scatterlines!(ax, Tuple.(collect(data)), label=replace(string(method), "KSVD."=>""), marker=:x)
#     errorbars!(ax, Tuple.(collect(data)), collect(values(err)))
# end
# fig[1,2] = Legend(fig, ax)
# fig
