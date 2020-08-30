using Flux: params, binarycrossentropy, sigmoid
using Zygote: @nograd
using AccuracyAtTop

# -------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------
abstract type Model end
abstract type BaseLine <: Model end

function weights(y)
    n_pos = sum(y .== 1)
    n_neg = length(y) - n_pos
    return y ./ n_pos .+ (1 .- y) ./ n_neg
end

@nograd weights

function build_model(type::Type{BaseLine}, arg, surrogate, reg; T = Float32)

    function loss(x, y, model, pars)
        s = model(x)
        w = weights(y)
        return sum(binarycrossentropy.(sigmoid.(s), y) .* w) + reg * sum(sqsum, pars)
    end
    return loss
end

# threshold models
abstract type DeepTopPush <: Model end
abstract type PatMat <: Model end
abstract type PatMatNP <: Model end

build_threshold(::DeepTopPush, arg) = Maximum(; samples = NegSamples)
build_threshold(::PatMat, τ::Real) = Quantile(τ; samples = AllSamples)
build_threshold(::PatMatNP, τ::Real) = FPRate(τ)

sqsum(x) = sum(abs2, x)

function build_model(type, arg, surrogate, reg; T = Float32)
    thres = build_threshold(type, arg)

    function loss(x, y, model, pars)
        s = model(x)
        t = threshold(thres, y, s)
        return fnr(y, s, t, surrogate) + reg * sum(sqsum, pars)
    end
    return loss
end
