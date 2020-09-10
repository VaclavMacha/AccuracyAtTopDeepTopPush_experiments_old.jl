using DrWatson
@quickactivate "AccuracyAtTop_aaai"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => [FashionMNIST, CIFAR100, Molecules],
    :posclass => [0, 1, 1],
)

Train_Settings = Dict(
    :batchsize => 32,
    :epochs => 200,
    :optimiser => Descent,
    :steplength => 0.0001,
    :seed => 1,
)

Model_Settings = Dict(
    :type => [APPerf, APPerf],
    :arg => [0.01, 0.05],
    :surrogate => missing,
    :reg => 1e-3,
    :buffer => missing,
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings)
run_evaluation(Dataset_Settings)
