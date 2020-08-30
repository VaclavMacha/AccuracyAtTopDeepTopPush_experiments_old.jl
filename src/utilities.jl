using Random
using StatsBase
using Base.Iterators

function batch_provider(x, y, batchsize)
    neg = findall(vec(y) .== 0)
    pos = findall(vec(y) .== 1)

    n_neg = batchsize ÷ 2
    n_pos = batchsize - n_neg

    x_obs = ndims(x)
    y_obs = ndims(y)

    last_batch = sample(1:batchsize, batchsize; replace = false)

    function make_batch(addsample::Int = 0)
        inds = vcat(
            sample(neg, n_neg; replace = false),
            sample(pos, n_pos; replace = false),
        )
        if 0 < addsample <= batchsize
            inds[rand(1:batchsize)] = last_batch[addsample]
        end
        last_batch .= inds

        return (selectdim(x, x_obs, inds), selectdim(y, y_obs, inds))
    end
    return make_batch
end

function compute_scores(model, x; T = Float32, chunksize = 512)
    x_obs = ndims(x)
    n = size(x, x_obs)
    scores = zeros(T, n)

    for inds in partition(1:n, chunksize)
        scores[inds] .= model(selectdim(x, x_obs, inds))
    end
    return scores
end
