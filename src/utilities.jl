using ProgressMeter
using Random
using StatsBase

using Base.Iterators: partition
using BSON: @save

# -------------------------------------------------------------------------------
# Data processing
# -------------------------------------------------------------------------------
getdim(A::AbstractArray, d::Integer, i) = getindex(A, Base._setindex(i, d, axes(A)...)...)


function batch_provider(x, y, batchsize)
    neg = findall(vec(y) .== 0)
    pos = findall(vec(y) .== 1)

    n_neg = batchsize ÷ 2
    n_pos = batchsize - n_neg

    x_obs = ndims(x)
    y_obs = ndims(y)

    last_batch = sample(1:batchsize, batchsize; replace = false)

    function make_batch(; buffer::Bool = false)
        inds = vcat(
            sample(neg, n_neg; replace = false),
            sample(pos, n_pos; replace = false),
        )
        if buffer
            addind = AccuracyAtTop.BUFFER[].ind
            if 0 < addind <= batchsize
                inds[rand(1:batchsize)] = last_batch[addind]
            end
            last_batch .= inds
        end
        return (getdim(x, x_obs, inds), getdim(y, y_obs, inds))
    end
    return make_batch
end

function compute_scores(model, x; chunksize = 512)
    x_obs = ndims(x)
    n = size(x, x_obs)
    scores = zeros(eltype(x), n)

    for inds in partition(1:n, chunksize)
        scores[inds] .= model(getdim(x, x_obs, inds))[:]
    end
    return scores
end

# -------------------------------------------------------------------------------
# Callback function
# -------------------------------------------------------------------------------
Base.@kwdef mutable struct CallBack
    iters::Int
    epochlength::Int = iters
    bar::Progress = Progress(iters, 1, "Training:")
    showat::Int = 100
    showfunc::Function = () -> []
    saveat::Int = 1000
    savefunc::Function = () -> nothing
    counter::Int = 0
    usershows = showfunc()
end

function CallBack(iters, epochlength = iters; kwargs...)
    return CallBack(; iters = iters, epochlength = epochlength, kwargs...)
end

function (c::CallBack)()
    c.counter += 1

    if mod(c.counter, c.showat) == 0 || c.counter == 1
        c.usershows = c.showfunc()
    end
    if mod(c.counter, c.saveat) == 0
        c.savefunc()
    end
    next!(c.bar; showvalues = vcat(
        epochcounter(c),
        epochtimer(c),
        itercounter(c),
        itertimer(c),
        c.usershows
    ))
    return
end

function itercounter(c::CallBack)
    return ("Iteration", string(c.counter, "/", c.iters))
end

function itertimer(c::CallBack)
    tm = round((c.bar.tlast - c.bar.tfirst)/c.bar.counter; sigdigits = 2)
    return ("Average time per iteration", string(tm, "s"))
end

function epochcounter(c::CallBack)
    epoch = floor(Int, c.counter/c.epochlength)
    maxepoch = floor(Int, c.iters/c.epochlength)
    return maxepoch <= 1 ? [] : ("Epoch", string(epoch, "/", maxepoch))
end

function epochtimer(c::CallBack)
    tm = round(c.epochlength*(c.bar.tlast - c.bar.tfirst)/c.bar.counter; sigdigits = 2)
    return ("Average time per epoch", string(tm, "s"))
end
