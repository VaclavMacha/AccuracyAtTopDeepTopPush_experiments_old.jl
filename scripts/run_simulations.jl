using DrWatson
@quickactivate "AccuracyAtTop_aaai"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => [FashionMNIST, CIFAR100, SVHN2Full, Molecules],
    :posclass => [0, 1, 1, 1],
)

Train_Settings = Dict(
    :batchsize => 32,
    :epochs => 200,
    :optimiser => Descent,
    :steplength => 0.0001,
    :seed => collect(1:10),
)

Model_Settings = Dict(
    :type => [BaseLine, DeepTopPush, DeepTopPush, PatMatNP, PatMatNP],
    :arg => [missing, missing, missing, 0.01, 0.05],
    :surrogate => [missing, quadratic, quadratic, quadratic, quadratic],
    :reg => 1e-3,
    :buffer => [missing, false, true, missing, missing],
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings)
run_evaluation(Dataset_Settings)
