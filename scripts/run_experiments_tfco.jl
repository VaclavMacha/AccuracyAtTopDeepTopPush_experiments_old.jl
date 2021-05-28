using DrWatson
@quickactivate "AccuracyAtTop_aaai"

include(srcdir("utilities.jl"))
include(srcdir("tfco.jl"))

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
    :seed => collect(1:10),
)

Model_Settings = Dict(
    :type => [TFCO],
    :arg => [0.01],
    :surrogate => missing,
    :reg => 1e-3,
    :buffer => missing,
)

run_simulations_tfco(Dataset_Settings, Train_Settings, Model_Settings)
run_evaluation(Dataset_Settings)
