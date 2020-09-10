using DrWatson
@quickactivate "AccuracyAtTop_aaai"

include(srcdir("utilities.jl"))

# ------------------------------------------------------------------------------------------
# Settings
# ------------------------------------------------------------------------------------------
dataset = CIFAR100
posclass = 1
batch_size = 32
epochs = 10

type = DeepTopPush
arg = missing
surrogate = quadratic
reg = 1e-3
seed = 1

optimiser = Descent
steplength =  0.000001

# loading data
@info "Dataset: $(dataset), positive class label: $(posclass)"
labelmap = (y) -> y == posclass
(x_train, y_train), ~ = load(dataset; labelmap = labelmap) |> gpu
make_batch = batch_provider(x_train, y_train, batch_size)

epochlength = length(y_train) ÷ batch_size
iters = epochs * epochlength

# building model
model = build_network(dataset; seed = seed) |> gpu
objective = build_loss(type, arg, surrogate, reg)
thres = build_threshold(type, arg)
pars = params(model)

loss(x, y) = objective(x, y, model, pars)

history = MVHistory()
x_train_neg = getdim(x_train, ndims(x_train), findall(vec(y_train) .== 0));
iter = 0

function callback_buffer(l, x, y)
    global iter
    iter += 1
    if iter > epochlength*(epochs-2)
        @info string(iter, "/", iters)
        s = cpu(model(x))
        y = cpu(y)
        t_mini_buffer = maximum(s[findall(vec(y) .== 0)])
        t_mini = maximum(s[findall(vec(y[1:(end-1)]) .== 0)])
        t_true = maximum(compute_scores(model, x_train_neg))

        push!(history, :loss, iter, l)
        push!(history, :t_minibatch, iter, t_mini)
        push!(history, :t_minibatch_aatp, iter-1, AccuracyAtTop.BUFFER[].t)
        push!(history, :t_minibatch_buffer, iter, t_mini_buffer)
        push!(history, :t_true, iter, t_true)
    end
end

# training
opt = optimiser(steplength)
batches = (make_batch(; buffer = true) for iter in 1:iters) |> gpu

custom_train!(loss, pars, batches, opt; cb = callback_buffer)

BSON.bson(datadir("thresholds.bson"), Dict(:history => history))

plt = plot(
    xlabel = "iteration",
    ylabel = "threshold",
    title = "$(dataset): batchsize = $(batch_size)",
    legend = :outertopright,
    size = (1000, 400),
)
for key in [:t_minibatch, :t_minibatch_buffer, :t_true]
    h = history[key]
    plot!(plt, h.iterations, h.values, label = string(key))
end
savefig(plt, plotsdir("thresholds.png"))
