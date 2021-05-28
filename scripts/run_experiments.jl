using DrWatson
@quickactivate "AccuracyAtTop_DeepTopPush"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Basic datasets
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
    :type => [BaseLine, DeepTopPush, PatMatNP],
    :arg => [missing, missing, 0.01],
    :surrogate => [missing, quadratic, quadratic],
    :reg => 1e-3,
    :buffer => [missing, true, missing],
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings; nepochs_save = 50)
run_evaluation(Dataset_Settings)

# ------------------------------------------------------------------------------------------
# APPerf
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
    :type => [APPerf],
    :arg => [0.01],
    :surrogate => missing,
    :reg => 1e-3,
    :buffer => missing,
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings)
run_evaluation(Dataset_Settings)

# ------------------------------------------------------------------------------------------
# Imagenet
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => [ImageNet],
    :posclass => [33:37], # turtles
)

Train_Settings = Dict(
    :batchsize => 32,
    :epochs => 200,
    :optimiser => Descent,
    :steplength => 0.0001,
    :seed => 1,
)

Model_Settings = Dict(
    :type => [DeepTopPush, PatMatNP, PatMatNP],
    :arg => [missing, 0.01, 0.05],
    :surrogate => [hinge, hinge, hinge],
    :reg => 1e-3,
    :buffer => [false, missing, missing],
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings; nepochs_save = 10)
run_evaluation(Dataset_Settings)

@info "All experiments finished"
