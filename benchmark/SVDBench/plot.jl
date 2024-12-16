import Pkg; Pkg.activate("MakieEnv", shared=true)
using AlgebraOfGraphics, CairoMakie
using DataFrames, FileIO

results = load("bmres.csv") |> DataFrame

plt = data(results)
plt *= mapping(:m=>"Num samples "*L"$m$", :median_runtime=>"Runtime [s]", color=:method)
plt *= (visual(Scatter)*mapping(; marker=:eltype) + mapping(linestyle=:eltype)*visual(Lines))

axis=(; xscale=log10, yscale=log10)
fig = draw(plt; axis)
save("bmres.png", fig)
save("bmres.svg", fig)
