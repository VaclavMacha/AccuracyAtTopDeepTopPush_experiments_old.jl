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
id = [1]

Train_Settings = Dict(
    :batchsize => repeat(batchsize, outer=length(id)),
    :epochs => 200,
    :optimiser => Descent,
    :steplength => 0.0001,
    :runid => repeat(id, inner=length(batchsize)),
)

Model_Settings = Dict(
    :type => APPerf,
    :arg => [0.01, 0.05],
    :surrogate => missing,
    :reg => 1e-3,
    :buffer => false,
)

run_simulations(Dataset_Settings, Train_Settings, Model_Settings)
