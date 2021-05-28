using DrWatson
@quickactivate "AccuracyAtTop_DeepTopPush"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------
collect_benchmark()

# ------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------
df = collect_metrics(; key = :test, epochs = 200)

table_1 = @linq df |>
    where(:batchsize .== 32)

CSV.write(datadir("results", "metrics_mean.csv"), table_1)


# ------------------------------------------------------------------------------------------
# Curves
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => CIFAR100,
    :posclass => 1,
)

Train_Settings = Dict(
    :batchsize => 32,
    :epochs => 200,
    :optimiser => Descent,
    :steplength => 0.0001,
    :seed => 1,
)

Model_Settings = Dict(
    :type => [BaseLine, DeepTopPush, PatMatNP, APPerf, TFCO],
    :arg => [missing, missing, 0.01, 0.01, 0.01],
    :surrogate => [missing, quadratic, quadratic, missing, missing],
    :reg => 1e-3,
    :buffer => [missing, true, missing, missing, missing],
)

key = :test
xlims = (1e-3, 1)
collect_curves(Dataset_Settings,Train_Settings,Model_Settings; key = key, xlims = xlims)
