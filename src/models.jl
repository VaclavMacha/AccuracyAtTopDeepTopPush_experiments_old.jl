using Flux: params, sigmoid
using Flux.Losses: binarycrossentropy
using Zygote: @nograd
using AccuracyAtTop

# -------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------
abstract type Model end

# Baseline model
abstract type BaseLine <: Model end

weights(y, s) = oftype(s, y ./ sum(y .== 1) .+ (1 .- y) ./ sum(y .== 0))

@nograd weights

function build_loss(::Type{BaseLine}, arg, surrogate, reg)

    function loss(x, y, model, pars)
        return loss(y, model(x), pars)
    end

    function loss(y, s, pars)
        w = weights(y, s)
        return binarycrossentropy(sigmoid.(s), y; agg = x -> sum(w .* x)) + eltype(s)(reg) * sum(sqsum, pars)
    end
    return loss
end

# Threshold models
abstract type DeepTopPush <: Model end
abstract type PatMat <: Model end
abstract type PatMatNP <: Model end

build_threshold(::Type{DeepTopPush}, arg) = Maximum(; samples = NegSamples)
build_threshold(::Type{PatMat}, τ::Real) = Quantile(τ; samples = AllSamples, rev = true)
build_threshold(::Type{PatMatNP}, τ::Real) = FPRate(τ)

sqsum(x) = sum(abs2, x)

function build_loss(type, arg, surrogate, reg)
    thres = build_threshold(type, arg)

    function loss(x, y, model, pars)
        return loss(y, model(x), pars)
    end

    function loss(y, s, pars)
        t = threshold(thres, y, s)
        return fnr(y, s, t, surrogate) + eltype(s)(reg) * sum(sqsum, pars)
    end
    return loss
end
