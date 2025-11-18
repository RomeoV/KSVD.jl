using HDF5, NPZ, LinearAlgebra

h5open("coherence_values_ours.h5", "w") do file
    for nnz in [20, 40, 80], m in [4096, 16384]
        # Baseline
        D_baseline_path = "gemma_2b_2ishM-result-baseline-2621440-65536-1-$(m)-$(nnz).npy"
        D_baseline = npzread(D_baseline_path)
        values_baseline = map(maximum, eachrow(abs.(D_baseline' * D_baseline - I(size(D_baseline, 2)))))
        write(file, "baseline_m_$(m)_nnz_$(nnz)", values_baseline)

        # Matryoshka
        D_matryoshka_path = "gemma_2b_2ishM-result-matryoshka-2621440-65536-1-$(m)-$(nnz).npy"
        D_matryoshka = npzread(D_matryoshka_path)
        values_matryoshka = map(maximum, eachrow(abs.(D_matryoshka' * D_matryoshka - I(size(D_matryoshka, 2)))))
        write(file, "matryoshka_m_$(m)_nnz_$(nnz)", values_matryoshka)
    end
end

println("Data saved to output_data.h5")
