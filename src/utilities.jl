using BSON
using ProgressMeter
using Random
using StatsBase
using ValueHistories

using Base.Iterators: partition
using Flux.Optimise: runall, update!, StopException, batchmemaybe
using Zygote: Params, gradient


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
# Custom train!
# -------------------------------------------------------------------------------
function custom_train!(loss, ps, data, opt; cb = () -> ())
  ps = Params(ps)
  cb = runall(cb)

  local loss_val

  for d in data
    try
      gs = gradient(ps) do
        loss_val = loss(batchmemaybe(d)...)
        return loss_val
      end
      update!(opt, ps, gs)
      cb(loss_val, batchmemaybe(d)...)
    catch ex
      if ex isa StopException
        break
      else
        rethrow(ex)
      end
    end
  end
end

# -------------------------------------------------------------------------------
# Callback function
# -------------------------------------------------------------------------------
Base.@kwdef mutable struct CallBack
    iters::Int
    epochlength::Int = iters
    title::String = "Training:"
    bar::Progress = Progress(iters, 5, title)
    showat::Int = 100
    showfunc::Function = (c) -> []
    saveat::Int = 1000
    savefunc::Function = (c) -> nothing
    counter::Int = 0
    usershows = []
    loss = History(Float32)
end

function CallBack(iters, epochlength = iters; kwargs...)
    return CallBack(; iters = iters, epochlength = epochlength, kwargs...)
end

function (c::CallBack)(loss_val, x, y)
    c.counter += 1
    push!(c.loss, c.counter, eltype(c.loss.values)(loss_val))

    if mod(c.counter, c.showat) == 0 || c.counter == 1
        c.usershows = c.showfunc(c)
    end
    if mod(c.counter, c.saveat) == 0
        c.savefunc(c, x, y)
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

# -------------------------------------------------------------------------------
# Saving simulations
# -------------------------------------------------------------------------------
allowedtypes(args...) = (Real, String, Symbol, DataType, Function, args...)

function save_simulation(
    c::CallBack,
    dataset_settings::Dict,
    train_settings_in::Dict,
    model_settings::Dict,
    model,
    x_train,
    y_train,
    x_test,
    y_test,
    x_mini,
    y_mini,
)

    train_settings = deepcopy(train_settings_in)
    train_settings[:epochs] = floor(Int, c.counter/c.epochlength)
    train_settings[:epochlength] = c.epochlength
    train_settings[:iters] = c.counter

    tm = (c.bar.tlast - c.bar.tfirst)/c.bar.counter

    simulation = Dict(
        :dataset_settings => deepcopy(dataset_settings),
        :train_settings => deepcopy(train_settings),
        :model_settings => deepcopy(model_settings),
        :time_per_iter => tm,
        :time_per_epoch => tm * c.epochlength,
        :model => cpu(model),
        :loss => c.loss.values,
        :minibatch => Dict(
            :targets => cpu(vec(y_mini)),
            :scores => cpu(compute_scores(model, x_mini)),
        ),
        :train => Dict(
            :targets => cpu(vec(y_train)),
            :scores => cpu(compute_scores(model, x_train)),
        ),
        :test => Dict(
            :targets => cpu(vec(y_test)),
            :scores => cpu(compute_scores(model, x_test)),
        ),
    )

    # save
    model_dict = deepcopy(model_settings)
    model_dict[:epochs] = simulation[:train_settings][:epochs]
    model_dict[:iters] = simulation[:train_settings][:iters]


    dataset_dir = savename(dataset_settings; allowedtypes = allowedtypes())
    delete!(train_settings, :epochs)
    delete!(train_settings, :iters)
    train_dir = savename(train_settings; allowedtypes = allowedtypes())
    simul_name = string(savename(model_dict; allowedtypes = allowedtypes()), ".bson")

    isdir(datadir(dataset_dir, train_dir)) || mkpath(datadir(dataset_dir, train_dir))
    bson(datadir(dataset_dir, train_dir, simul_name), simulation)
    return
end
