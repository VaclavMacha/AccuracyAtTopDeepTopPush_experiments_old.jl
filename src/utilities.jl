using DrWatson

include(srcdir("datasets.jl"))
include(srcdir("models.jl"))
include(srcdir("utilities_train.jl"))
include(srcdir("utilities_evaluation.jl"))

DrWatson.datadir(args...) = joinpath("/disk/macha/data_deeptoppush", args...)

# -------------------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------------------
getdim(A::AbstractArray, d::Integer, i) = getindex(A, Base._setindex(i, d, axes(A)...)...)

function compute_scores(model, x; chunksize = 10000, device = identity)
    x_obs = ndims(x)
    n = size(x, x_obs)
    scores = zeros(eltype(x), n)

    for inds in partition(1:n, chunksize)
        scores[inds] .= model(device(getdim(x, x_obs, inds)))[:]
    end
    return scores
end

function dict_list_simple(d::Dict)
    ls = map(values(d)) do val
        typeof(val) <: AbstractVector ? length(val) : 1
    end
    if length(unique(ls)) > 2
        @error "not supported"
    else
        return map(1:maximum(ls)) do k
            Dict(key => typeof(val) <: AbstractVector ? val[k] : val for (key, val) in d)
        end
    end
end

logrange(x1, x2; kwargs...) = exp10.(range(log10(x1), log10(x2); kwargs...))

# -------------------------------------------------------------------------------
# Names....
# -------------------------------------------------------------------------------
allowedtypes(args...) = (Real, String, Symbol, DataType, Function, args...)

function dataset_savename(dataset_settings; digits = 4)
    return savename(dataset_settings; allowedtypes = allowedtypes(), digits = digits)
end

function train_savename(train_settings_in; digits = 4)
    train_settings = deepcopy(train_settings_in)
    delete!(train_settings, :epochlength)
    delete!(train_settings, :iters)
    return savename(train_settings; allowedtypes = allowedtypes(), digits = digits)
end

function model_savename(model_settings_in; digits = 4)
    model_settings = deepcopy(model_settings_in)
    delete!(model_settings, :seed)
    return savename(model_settings; allowedtypes = allowedtypes(), digits = digits)
end

function modeldir(
    dataset_settings,
    train_settings,
    model_settings;
    digits = 4,
    agg = datadir,
)

    datdir = dataset_savename(dataset_settings; digits = digits)
    traindir = train_savename(train_settings; digits = digits)
    moddir = model_savename(model_settings; digits = digits)

    return agg("models", datdir, traindir, moddir)
end

simulation_name(epoch) = string("model_epoch=", epoch, ".bson")
extract_model_type(d::Dict) = d[:model_settings][:type]
model_name(d::Dict) = model_name(d[:model_settings][:type], d[:model_settings])

function model_name(T::Type{<:Model}, d::Dict)
    vals = [string(d[key]) for key in model_args(T)]
    return string(T, "(", join(vals, ", "), ")")
end

model_args(::Type{<:Model}) = [:arg]
model_args(::Type{BaseLine}) = []
model_args(::Type{DeepTopPush}) = [:arg, :buffer]
