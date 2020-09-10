using DrWatson
@quickactivate "AccuracyAtTop_aaai"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------
collect_benchmark()

# ------------------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------------------
df = collect_metrics(; key = :test)

table_1 = @linq df |>
    where(:batchsize .== 32) |>
    where(:dataset .!= "Molecules")

CSV.write(datadir("results", "metrics_mean.csv"), table_1)

table_2 = @linq df |>
    where(:batchsize .== 32) |>
    where(:dataset .== "Molecules")

CSV.write(datadir("results", "molecules.csv"), table_2)


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
    :type => [BaseLine, DeepTopPush, PatMatNP, PatMatNP, APPerf, APPerf, TFCO, TFCO],
    :arg => [missing, missing, 0.01, 0.05, 0.01, 0.05, 0.01, 0.05],
    :surrogate => [missing, quadratic, quadratic, quadratic, missing, missing, missing, missing],
    :reg => 1e-3,
    :buffer => [missing, true, missing, missing, missing, missing, missing, missing],
)

key = :test
xlims = (1e-3, 1)
collect_curves(Dataset_Settings,Train_Settings,Model_Settings; key = key, xlims = xlims)
