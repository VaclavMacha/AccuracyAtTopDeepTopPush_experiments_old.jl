using Flux
import MLDatasets
using MLDataPattern

abstract type Dataset; end

# -------------------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------------------
function load(D::Type{<:Dataset}, T = Float32; kwargs...)
    return reshape_dataset.(load_raw(D, T); kwargs...)
end

function reshape_dataset((x,y)::Tuple; onehot = false, labelmap = identity)
    ym = labelmap.(y)
    if onehot
        labels = Flux.onehotbatch(vec(ym), sort(unique(ym)))
    else
        labels = reshape_labels(ym)
    end
    return (reshape_samples(x), labels)
end

reshape_labels(y) = y
reshape_labels(y::AbstractVector) = Array(reshape(y, 1, :))
reshape_samples(x) = x

function reshape_samples(x::AbstractArray{T, 3}) where T
    return Array(reshape(x, size(x, 1), size(x, 2), 1, size(x, 3)))
end

# -------------------------------------------------------------------------------
# Image datasets
# -------------------------------------------------------------------------------
abstract type MNIST <: Dataset; end

function load_raw(::Type{MNIST}, T)
    train = MLDatasets.MNIST.traindata(T)
    test = MLDatasets.MNIST.testdata(T)

    return train, test
end

abstract type FashionMNIST <: Dataset; end

function load_raw(::Type{FashionMNIST}, T)
    train = MLDatasets.FashionMNIST.traindata(T)
    test = MLDatasets.FashionMNIST.testdata(T)

    return train, test
end

abstract type CIFAR10 <: Dataset; end

function load_raw(::Type{CIFAR10}, T)
    train = MLDatasets.CIFAR10.traindata(T)
    test = MLDatasets.CIFAR10.testdata(T)

    return train, test
end

abstract type CIFAR20 <: Dataset; end

function load_raw(::Type{CIFAR20}, T)
    train = MLDatasets.CIFAR100.traindata(T)[[1,2]]
    test = MLDatasets.CIFAR100.testdata(T)[[1,2]]

    return train, test
end

abstract type CIFAR100 <: Dataset; end

function load_raw(::Type{CIFAR100}, T)
    train = MLDatasets.CIFAR100.traindata(T)[[1,3]]
    test = MLDatasets.CIFAR100.testdata(T)[[1,3]]

    return train, test
end

abstract type SVHN2 <: Dataset; end

function load_raw(::Type{SVHN2}, T)
    train = MLDatasets.SVHN2.traindata(T)
    test = MLDatasets.SVHN2.testdata(T)

    return train, test
end

abstract type SVHN2Full <: Dataset; end

function load_raw(::Type{SVHN2Full}, T)
    x1, y1 = MLDatasets.SVHN2.traindata(T)
    x2, y2 = MLDatasets.SVHN2.extradata(T)

    train = (cat(x1, x2; dims = ndims(x1)), cat(y1, y2; dims = ndims(y1)))
    test = MLDatasets.SVHN2.testdata(T)

    return train, test
end
