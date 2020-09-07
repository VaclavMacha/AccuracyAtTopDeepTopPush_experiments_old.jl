using DrWatson
@quickactivate "AccuracyAtTop_aaai"

using AccuracyAtTop, EvalMetrics, Plots, Flux, CUDA
using Flux: gpu

plotlyjs()

include(srcdir("datasets.jl"))
include(srcdir("models.jl"))
include(srcdir("utilities.jl"))
include(srcdir("evalutilities.jl"))

# ------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => [FashionMNIST, CIFAR100, SVHN2Full],
    :posclass => [0, 1, 1],
)

batchsize = [32, 1000]
seed = 1:10

Train_Settings = Dict(
    :batchsize => repeat(batchsize, outer=length(seed)),
    :epochs => 200,
    :optimiser => Descent,
    :steplength => 0.0001,
    :seed => repeat(seed, inner=length(batchsize)),
)

Model_Settings = Dict(
    :type => [BaseLine, DeepTopPush, PatMatNP, PatMatNP],
    :arg => [missing, missing, 0.01, 0.05],
    :surrogate => [missing, quadratic, quadratic, quadratic],
    :reg => 1e-3,
    :buffer => [missing, false, missing, missing],
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings)
run_evaluation(Dataset_Settings)
