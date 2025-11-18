using HDF5
using Wandb
using Base.Iterators
using KSVD
using OhMyThreads
using NPZ
using LinearAlgebra
using Random
using ProgressBars
using StatsBase
using SparseArrays

DATAFILE = "gemma_2b_5M_layer12_largebuf.h5"
data = h5open(DATAFILE, "r")["layer_activations"]
nsamples = 2^20

# Sample validation set
val_size = 2^14
val_data = data[:, (end-(val_size-1)):end]

explainedsignal(Y, D, X; E=(Y - D * X)) = mean(norm.(eachcol(E)) ./ norm.(eachcol(Y)))
explainedvariance(Y, D, X; E=(Y - D * X)) = 1 - sum(var(E; dims=2)) / sum(var(Y; dims=2))

issomething(x) = !isnothing(x)
function make_filename(config; basename="/mnt/data", extra="", force_recompute=false)
    nsamples = config["nsamples"]
    batchsize = config["batchsize"]
    itersperbatch = config["itersperbatch"]
    ndicts = config["ndicts"]
    nnzpercol = config["nnzpercol"]

    filename = "$(basename)/gemma_2b_2ishM-result-$(extra)-$(nsamples)-$(batchsize)-$(itersperbatch)-$(ndicts)-$(nnzpercol).npy"
    if force_recompute && isfile(filename)
        repeat_counter = 1
        filename = "$(basename)/gemma_2b_2ishM-result-$(extra)-$(nsamples)-$(batchsize)-$(itersperbatch)-$(ndicts)-$(nnzpercol).$(repeat_counter).npy"
        while isfile(filename)
            repeat_counter += 1
            filename = "$(basename)/gemma_2b_2ishM-result-$(extra)-$(nsamples)-$(batchsize)-$(itersperbatch)-$(ndicts)-$(nnzpercol).$(repeat_counter).npy"
        end
    end
    return filename
end

for n in [2^12, 2^14], nnzpercol in [20, 40, 80], ksvd_loop_type in [KSVD.NormalLoop(), KSVD.MatryoshkaLoop(; log2min=8)]
    sparse_coding_method = KSVD.ParallelMatchingPursuit(; max_nnz=nnzpercol, refit_coeffs=false)
    itersperbatch = 1
    batchsize = 2^16
    minibatch_size = nothing
    force_recompute = true
    doinit = false
    initdictsize = nothing  # 32k
    nrepeats = 3

    # Configure WandB for this run
    config = Dict(
        "ksvd_loop_type" => ksvd_loop_type,
        "batchsize" => batchsize,
        "itersperbatch" => itersperbatch,
        "ndicts" => n,
        "nnzpercol" => nnzpercol,
        "minibatch_size" => minibatch_size,
        "doinit" => doinit,
        "initdictsize" => initdictsize,
        "nsamples" => nsamples,
        "datafile" => DATAFILE,
        "sparse_coding_method" => sparse_coding_method,
        "nrepeats" => nrepeats
    )
    filename_extra = if ksvd_loop_type isa KSVD.NormalLoop
        "baseline"
    elseif ksvd_loop_type isa KSVD.MatryoshkaLoop
        "matryoshka"
    else
        "other"
    end
    filename = make_filename(config; extra=filename_extra, force_recompute)
    if isfile(filename)
        @info "Config $(config) already computed. Skipping."
        continue
    else
        @info "Computing config $(config)"
    end

    # Initialize WandB
    lg = WandbLogger(; project="ksvdgemma", tags=[readchomp(`hostname`)], name=nothing, config)

    D_init = nothing
    doinit && @info "Computed initial dict"
    D = nothing

    function callback_fn((; iter, Y, D, X, norm_val, nnz_per_col_val))
        # Compute validation metrics
        variance_expl = explainedvariance(Y, D, X)
        val_X = sparse_coding(sparse_coding_method, val_data, D)
        val_norm_val = explainedsignal(val_data, D, val_X)
        val_variance_expl = explainedvariance(val_data, D, val_X)
        val_nnz_per_col_val = nnz(val_X) / size(val_X, 2)
        usagecounts = countmap(X.rowval)
        minusage = minimum(values(usagecounts))
        numunuseddicts = length(setdiff(axes(X, 1), unique(sort(X.rowval))))

        # Log both training and validation metrics
        Wandb.log(lg, Dict(
            "num_unuseddicts" => numunuseddicts,
            "dictminusage" => minusage,
            "norm_val" => norm_val,
            "variance_expl" => variance_expl,
            "nnz_per_col" => nnz_per_col_val,
            "val_norm_val" => val_norm_val,
            "val_variance_expl" => val_variance_expl,
            "val_nnz_per_col" => val_nnz_per_col_val
        ))
        return nothing
    end

    save_ch = Channel{Matrix{Float32}}(10; spawn=true) do ch
        foreach(ch) do D
            npzwrite(filename, D)
        end
    end

    for rep in 1:nrepeats
        for (i, batch_idx) in enumerate(Iterators.partition(axes(data, 2)[1:nsamples], batchsize))
            if length(batch_idx) != batchsize
                continue
            end

            Y = copy(data[:, batch_idx])

            D_init = isnothing(D) ? D_init : copy(D)
            @show i, batch_idx
            res = ksvd(Y, n;
                sparse_coding_method,
                ksvd_loop_type=ksvd_loop_type,
                verbose=true,
                show_trace=true,
                D_init,
                maxiters=itersperbatch, callback_fn,
                abstol=nothing, reltol=nothing)
            D = res.D

            push!(save_ch, copy(D))
        end
    end

    close(save_ch)
    close(lg)
end
