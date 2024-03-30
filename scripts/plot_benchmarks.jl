import Pkg; Pkg.activate("KSVDViz", shared=true)
import TOML
using AlgebraOfGraphics, CairoMakie
using Makie.MathTeXEngine  # for texfont
import AlgebraOfGraphics: sorter, renamer
using DataFrames
using LinearAlgebra

function load_data(inpath)
# inpath = "/tmp/bmark1.toml"
    timer = TOML.parsefile(inpath)
    ksvd_timer = timer["inner_timers"]["ksvd_update"]["inner_timers"]

    matrix_vector_product = try
        ksvd_timer["compute E_Ω"]["inner_timers"]["compute matrix vector product"]["time_ns"]
    catch ; 0; end
    outer_product = try
        ksvd_timer["compute E_Ω"]["inner_timers"]["compute outer product"]["time_ns"]
    catch ; 0; end
    E_Ω_data_copy = try
        ksvd_timer["compute E_Ω"]["inner_timers"]["copy data"]["time_ns"]
    catch ; 0; end
    find_nonzeros = try
        ksvd_timer["find nonzeros"]["time_ns"]
    catch ; 0; end
    tsvd = try
        ksvd_timer["compute and copy tsvd"]["time_ns"]
    catch ; 0; end
    # copy_X = ksvd_timer["copy X"]["time_ns"]
    copy_results = try
        ksvd_timer["copy results"]["time_ns"]
    catch ; 0; end
    update_errors = try
        ksvd_timer["Update errors"]["time_ns"]
    catch ; 0; end
    copy_and_precompute_buffers = try
        ksvd_timer["Copy and precompute buffers"]["time_ns"]
    catch ; 0; end
    # these two match up very well!
    # other1 = timer["inner_timers"]["ksvd_update"]["time_ns"] - sum([
    #     matrix_vector_product, outer_product, E_Ω_data_copy, find_nonzeros, tsvd,
    #     copy_X, copy_results
    # ])
    other2 = ksvd_timer["~ksvd_update~"]["time_ns"]

    infile = basename(inpath)
    results = [
        (; inpath, infile, group="other", time_ns = other2),
        (; inpath, infile, group="copy results", time_ns = copy_results),
        (; inpath, infile, group="prepare buffers", time_ns = copy_and_precompute_buffers),
        (; inpath, infile, group="update errors", time_ns = update_errors),
        (; inpath, infile, group="compute svd/tsvd", time_ns = tsvd),
        (; inpath, infile, group="find nonzeros", time_ns = find_nonzeros),
        (; inpath, infile, group="E_Ω data copy", time_ns = E_Ω_data_copy),
        (; inpath, infile, group="compute outer product", time_ns = outer_product),
        (; inpath, infile, group="compute matrix vector product", time_ns = matrix_vector_product),
    ]
    df = DataFrame(results)
    df[!, :time_normalized] = normalize(df.time_ns, 1)
    df[!, :time_normalized_str] = ByRow(x->round(Int, x*100))(df.time_normalized)
    df
end

order = [
  "update errors",
  "compute matrix vector product",
  "compute outer product",
  "E_Ω data copy",
  "find nonzeros",
  "compute svd/tsvd",
  "prepare buffers",
  "copy results",
  "other",
]
# plt =(data(insertcols!(load_data("/tmp/bmark1.toml"), 1, :id=>1)) +
#       data(insertcols!(load_data("/tmp/bmark2.toml"), 1, :id=>1)))

root_path = "/home/romeo/Documents/github/KSVD.jl/ksvd_benchmarks/"
input_fnames = [
 "bmark9.toml",
 # "bmark7.toml",
 "bmark6.toml",
 "bmark5.toml",
 "bmark4.toml",
 "bmark3.toml",
 "bmark2.toml",
 # "bmark1.toml",
 "bmark0.toml",
 "bmark-1.toml",
 "bmark-2.toml",
]
xsorter = renamer("bmark9.toml"=>"original paper",
                  "bmark7.toml"=>"+restructure error comp",
                  "bmark6.toml"=>"+out-of-place error",
                  "bmark5.toml"=>"+svd->tsvd",
                  "bmark4.toml"=>"+parallel",
                  "bmark3.toml"=>"+f64->f32",
                  "bmark2.toml"=>"+fast sparse copy",
                  # "bmark1.toml"=>"+optimize outer product",
                  "bmark0.toml"=>"+mul!",
                  "bmark-1.toml"=>raw"+precompute X.T",
                  "bmark-2.toml"=>raw"+large scale",
                  )
dataframes = [load_data(root_path*fname) for fname in input_fnames]

mysorter1 = sorter(order)
mysorter2 = sorter(reverse(order))
mapping_ = mapping(# :infile=>xsorter, :time_normalized=>"Workload share\nby computation";
                   :infile=>xsorter, :time_normalized=>"";
                   stack=:group=>mysorter2=>"Computation",
                   color=:group=>mysorter1=>"Computation",
                   bar_labels=:time_normalized,
                   )

plt = sum(data, dataframes)
plt *= mapping_

#     mapping(:inpath, :time_normalized;
#                          stack=:group=>mysorter2, color=:group=>mysorter1) #, bar_labels=:time_normalized)
plt *= visual(BarPlot;
              flip_labels_at=0.,
              label_offset = 0,
              label_formatter = x->(x<0.055 ? "" : "$(round(Int, x*100))%"),
              # bar_label_formatter = x->"",
              label_align=(:center, :top),
              label_size=10,
              label_font=:bold,
              )#; bar_labels=fill(1, 6))

textheme = Theme(fonts=(; regular=texfont(:text),
                          bold=texfont(:bold),
                          italic=texfont(:italic),
                          bold_italic=texfont(:bolditalic)))

factor(df) = let infile = df.infile[1]
    infile == "bmark-2.toml" && return 1/100
    infile == "bmark9.toml" && return 1000
    return 1
end

fig = with_theme(textheme) do
    fig = Makie.Figure()
    ag = draw!(fig[2, 1], plt;
               axis=(;
                    xticklabelrotation=deg2rad(45/2),
                    xlabelvisible=false, height=300, width=300),
               palettes=(; color=cgrad(:seaborn_colorblind, 10, categorical=true),)
               )
    for (i, df) in enumerate(dataframes)
        text = let val = sum(df.time_ns)/1e9 * factor(df)
            function fmt(val)
                val > 1000 && return string(round(Int, val/1000))*"ks"
                val > 10 && return string(round(Int, val))*"s"
                return string(round(val, digits=1))*"s"
            end
            [fmt(val)]
        end
        text!(ag[1,1].axis, [(i, 1.)]; text, align=(:center, :bottom))
    end
    legend!(fig[2, 2], ag)
    sc = scatterlines(fig[1,1], axes(dataframes, 1), [sum(df.time_ns)/1e9*factor(df) for df in dataframes];
                      axis=(; height=150, ylabel="",# ylabel="Wall-clock time [sec]\nper 100'000 samples",
                            xticks=axes(dataframes,1), xticklabelsvisible=false, yscale=log10))
    hlines!(fig[1,1], [124, 82.5e3], linestyle=:dash, linewidth=1, color=:gray, overdraw=false)
    arrows!(fig[1,1], [2], [82.5e3], [0], [120-82.3e3])
    arrows!(fig[1,1], [9], [120], [0], [0.8-120])
    #
    linkxaxes!(ag[1,1].axis, sc.axis)
    Label(fig[1,1, Top()], text="Runtime improvements to\nprocess 100k samples\n\n", tellwidth=false, tellheight=true, font=:bold, fontsize=16, valign=:top)
    Label(fig[1,1, TopLeft()], text="Wall-clock time [sec]\nper 100'000 samples", tellwidth=true, tellheight=false, font=:italic, halign=:right, valign=:bottom)
    Label(fig[2,1, Left()], text="workload share by computation",
          tellwidth=true, tellheight=true, font=:italic,
          halign=:right, valign=:top, justification=:right,
          padding=(0f0, 30f0, 0f0, -5f0),
          word_wrap=true)
    Makie.resize_to_layout!(fig)
    tooltip_y_pos = sum(log10.([124, 0.8]))/2
    tooltip!(fig.scene, Makie.shift_project(sc.axis.scene, Point2f(9, tooltip_y_pos));
             text="Additional\n~250x speedup", placement=:right, color=:red, outline_linewidth=0.5f0,
             fontsize=12, font=:bold, backgroundcolor=(:lightgray, 0.5),
             overdraw=true, depth=0f0
             )
    tooltip_y_pos2 = sum(log10.([82.5e3, 124]))/2
    tooltip!(fig.scene, Makie.shift_project(sc.axis.scene, Point2f(2, tooltip_y_pos2));
             text="Replace naïve\nerror computation:\n~600x speedup", placement=:right, color=:red, outline_linewidth=0.5f0,
             fontsize=12, font=:bold, backgroundcolor=(:lightgray, 0.5),
             overdraw=true, depth=0f0
             )
    fig
end;
fig

save(joinpath(root_path, "figs", "benchmark_results.png"), fig)
save(joinpath(root_path, "figs", "benchmark_results.pdf"), fig)
save(joinpath(root_path, "figs", "benchmark_results.svg"), fig)
