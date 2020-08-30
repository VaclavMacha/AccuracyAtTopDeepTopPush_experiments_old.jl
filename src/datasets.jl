using Flux
using MLDataPattern
using Random

import MLDatasets

# -------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------
abstract type Dataset; end

function load(D::Type{<:Dataset}, T = Float32; kwargs...)
    return reshape_dataset.(load_raw(D, T); kwargs...)
end

function reshape_dataset((x,y)::Tuple; labelmap = identity)
    return (reshape_samples(x), reshape_labels(labelmap.(y)))
end

reshape_labels(y) = y
reshape_labels(y::AbstractVector) = Array(reshape(y, 1, :))
reshape_samples(x) = x

function reshape_samples(x::AbstractArray{T, 3}) where T
    return Array(reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)))
end

# -------------------------------------------------------------------------------
# MNIST-like datasets
# -------------------------------------------------------------------------------
abstract type AbstractMNIST <: Dataset end
abstract type MNIST <: AbstractMNIST end

function load_raw(::Type{MNIST}, T)
    train = MLDatasets.MNIST.traindata(T)
    test = MLDatasets.MNIST.testdata(T)

    return train, test
end

abstract type FashionMNIST <: AbstractMNIST end

function load_raw(::Type{FashionMNIST}, T)
    train = MLDatasets.FashionMNIST.traindata(T)
    test = MLDatasets.FashionMNIST.testdata(T)

    return train, test
end

function build_network(::Type{<:AbstractMNIST}; seed = 1234)
    Random.seed!(seed)

    return Chain(
        # First convolution
        Conv((5, 5), 1=>20, stride=(1,1), relu),
        MaxPool((2,2)),

        # Second convolution
        Conv((5, 5), 20=>50, stride=(1,1), relu),
        MaxPool((2,2)),

        flatten,
        Dense(800, 500),
        Dense(500, 1)
    )
end

# -------------------------------------------------------------------------------
# CIFAR dataset
# -------------------------------------------------------------------------------
abstract type AbstractCIFAR <: Dataset end
abstract type CIFAR10 <: AbstractCIFAR end

function load_raw(::Type{CIFAR10}, T)
    train = MLDatasets.CIFAR10.traindata(T)
    test = MLDatasets.CIFAR10.testdata(T)

    return train, test
end

abstract type CIFAR20 <: AbstractCIFAR end

function load_raw(::Type{CIFAR20}, T)
    train = MLDatasets.CIFAR100.traindata(T)[[1,2]]
    test = MLDatasets.CIFAR100.testdata(T)[[1,2]]

    return train, test
end

abstract type CIFAR100 <: AbstractCIFAR end

function load_raw(::Type{CIFAR100}, T)
    train = MLDatasets.CIFAR100.traindata(T)[[1,3]]
    test = MLDatasets.CIFAR100.testdata(T)[[1,3]]

    return train, test
end

function build_network(::Type{<:AbstractCIFAR}; seed = 1234)
    Random.seed!(seed)

    return Chain(
        # First convolution, operating upon a 32x32 image
        Conv((3, 3), 3=>64, pad=(1,1), relu),
        MaxPool((2,2)),

        # Second convolution, operating upon a 16x16 image
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),

        # Third convolution, operating upon a 4x4 image
        Conv((3, 3), 128=>128, pad=(1,1), relu),
        MaxPool((2,2)),

        flatten,
        Dense(2048, 1)
    )
end

# -------------------------------------------------------------------------------
# SVHN2 dataset
# -------------------------------------------------------------------------------
abstract type AbstractSVHN2 <: Dataset end
abstract type SVHN2 <: AbstractSVHN2 end

function load_raw(::Type{SVHN2}, T)
    train = MLDatasets.SVHN2.traindata(T)
    test = MLDatasets.SVHN2.testdata(T)

    return train, test
end

abstract type SVHN2Full <: AbstractSVHN2 end

function load_raw(::Type{SVHN2Full}, T)
    x1, y1 = MLDatasets.SVHN2.traindata(T)
    x2, y2 = MLDatasets.SVHN2.extradata(T)

    train = (cat(x1, x2; dims = ndims(x1)), cat(y1, y2; dims = ndims(y1)))
    test = MLDatasets.SVHN2.testdata(T)

    return train, test
end

function build_network(::Type{<:AbstractSVHN2}; seed = 1234)
    Random.seed!(seed)

    return Chain(
        # First convolution, operating upon a 32x32 image
        Conv((3, 3), 3=>64, pad=(1,1), relu),
        MaxPool((2,2)),

        # Second convolution, operating upon a 16x16 image
        Conv((3, 3), 64=>128, pad=(1,1), relu),
        MaxPool((2,2)),

        # Third convolution, operating upon a 4x4 image
        Conv((3, 3), 128=>128, pad=(1,1), relu),
        MaxPool((2,2)),

        flatten,
        Dense(2048, 1)
    )
end
