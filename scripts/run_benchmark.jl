using KSVD
import Random: seed!, TaskLocalRNG, rand!
import SparseArrays
import SparseArrays: sparse
import StatsBase: sample
using ProfileView, BenchmarkTools
using TimerOutputs, OhMyThreads
import TOML
TimerOutputs.enable_debug_timings(KSVD)

T = Float32
N = 100_000
D = 768  # sample dimension
K = 4*D  # dictionary dimension >= sample_dimension
nnz = 10

rng = TaskLocalRNG();
seed!(rng, 1)

basis = KSVD.init_dictionary(rng, T, D, K)
Is = vcat([sample(rng, 1:K, nnz, replace=false) for _ in 1:N]...);
Js = vcat([fill(j, nnz) for j in 1:N]...)
Vs = rand!(rng, similar(Js, T)) .+ 1
X_true = sparse(Is, Js, Vs, K, N)
Y = basis * X_true

ksvd_method = let
  ksvd_method = KSVD.BatchedParallelKSVD{false, T, DynamicScheduler}(; batch_size_per_thread=8)
  KSVD.maybe_init_buffers!(ksvd_method, size(Y, 1), size(basis, 2), size(Y, 2); pct_nz=0.03)
  ksvd_method
end;

@info "Starting warmup run."
ksvd_update(ksvd_method, Y[:, 1:1_000], copy(basis), X_true[:, 1:1_000])
@info "Finished warmup run."
timer = TimerOutput();
@info "Starting actual run."
ksvd_update(ksvd_method, Y, copy(basis), X_true;
            timer=timer, merge_all_timers=false)
TimerOutputs.complement!(timer)
@info "Finished actual run."

ksvd_benchmark_dir = joinpath(dirname(pathof(KSVD)), "..", "ksvd_benchmarks")
open(joinpath(ksvd_benchmark_dir, "bmark2.toml"), "w") do ofile
    rev = `git rev-parse --short HEAD` |> readchomp
    timer_dict = TimerOutputs.todict(timer)
    timer_dict["ksvd_pkg_rev"] = rev
    TOML.print(ofile, timer_dict)
end
