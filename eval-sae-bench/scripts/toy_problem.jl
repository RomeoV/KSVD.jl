using KSVD, Random, SparseArrays, LinearAlgebra, StatsBase, HDF5

T = Float32
m = 256
n = 4 * m
nsamples = Int(2^14)
nnzpercol = 20
D = KSVD.init_dictionary(T, m, n)
Xrand = reduce(hcat,
    [SparseVector(n, sort(sample(1:n, nnzpercol)), 1e0 .+ rand(T, nnzpercol))
     for _ in 1:nsamples])
Y = D * Xrand;

ϵ = 0.05
Y .+= ϵ * randn!(similar(Y));

fractions_known = [0 // 16, 1 // 16, 2 // 16, 4 // 16, 8 // 16, 16 // 16];

norm_results = map(fractions_known) do fraction_known
    println("fraction_known = $fraction_known")
    X_init = false .* similar(Xrand)
    let idx = 1:Int(fraction_known * n)
        X_init[idx, :] .= sign.(Xrand[idx, :])
    end
    res = ksvd(Y, n, nnzpercol; X_init, maxiters=1000, show_trace=true, verbose=true)
    res.norm_results[end]
end

h5open("coherence_values_ours.h5", "w") do file
    write(file, "fractions_known", fractions_known)
    write(file, "norm_results", norm_results)
end
