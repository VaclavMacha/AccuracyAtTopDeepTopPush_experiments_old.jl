using DrWatson
@quickactivate "AccuracyAtTop_aaai"

using AccuracyAtTop, EvalMetrics, Plots, Flux, CUDA
using Flux: gpu

plotlyjs()

include(srcdir("datasets.jl"))
include(srcdir("models.jl"))
include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------
Dataset_Settings = Dict(
    :dataset => [FashionMNIST, CIFAR100],
    :posclass => [0, 1]
)

batchsize = [32]
seed = [1]

Train_Settings = Dict(
    :batchsize => repeat(batchsize, outer=length(seed)),
    :epochs => 200,
    :optimiser => Descent,
    :steplength => 0.0001,
    :seed => repeat(seed, inner=length(batchsize)),
)

Model_Settings = Dict(
    :type => APPerf,
    :arg => [0.01, 0.05],
    :surrogate => missing,
    :reg => 1e-3,
    :buffer => missing,
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings)
run_evaluation(Dataset_Settings)
