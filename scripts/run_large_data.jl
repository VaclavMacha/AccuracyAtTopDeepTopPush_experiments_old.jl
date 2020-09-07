using DrWatson
@quickactivate "AccuracyAtTop_aaai"

using AccuracyAtTop, EvalMetrics, Plots, CSV, DataFrames

# plotlyjs()
pyplot() # for savefig

include(srcdir("datasets.jl"))
include(srcdir("models.jl"))
include(srcdir("utilities.jl"))
include(srcdir("evalutilities.jl"))

# ------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------
d = BSON.load(datadir("large_batches.bson"))

xlims = (1e-6, 1)
xscale = :log10

plt = plot(title = "Test");
for key in keys(d)
    y = d[key][:targets]
    s = d[key][:scores]
    plot_roccurve!(plt, s, y; label =  string(key), xscale = xscale)
end
for arg in [1e-4, 1e-3, 1e-2]
    vline!(plt, [arg]; label = "fpr=$(arg)", linestyle = :dash)
end
plot!(legend = :outertopright, size = (1000, 400), xscale = xscale, xlims = xlims);
display(plt)
savefig(plt, plotsdir("large_batches.png"))

# save as csv
fprs = logrange(1e-5, 1; length = 298)
fprs = sort(vcat(fprs, 0.01, 0.05))

df = DataFrame()
for key in keys(d)
    y = d[key][:targets]
    s = d[key][:scores]
    ts = threshold_at_fpr(y, s, fprs)
    fprates, tprates = roccurve(y, s, ts)

    df[Symbol(key, "_", "fpr")] = fprates
    df[Symbol(key, "_", "tpr")] = tprates
end
mkpath(datadir("results"))
CSV.write(datadir("results", "large_data.csv"), df)
