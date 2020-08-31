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
    :dataset => [FashionMNIST, CIFAR100, SVHN2],
    :posclass => 0,
)

Train_Settings = Dict(
    :batchsize => 1000,
    :epochs => 200,
    :optimiser => Descent,
    :steplength => 0.0005,
    :runid => 1,
)

Model_Settings = Dict(
    :type => [DeepTopPush, DeepTopPush, PatMatNP, PatMatNP, PatMat, PatMat, BaseLine],
    :arg => [missing, missing, 0.01, 0.05, 0.01, 0.05, missing],
    :surrogate => [quadratic, quadratic, quadratic, quadratic, quadratic, quadratic, missing],
    :reg => 1e-3,
    :buffer => [true, false, false, false, false, false, false],
    :seed => 1234,
)

# ------------------------------------------------------------------------------------------
# Run experiments
# ------------------------------------------------------------------------------------------
for dataset_settings in dict_list_simple(Dataset_Settings)
    @unpack dataset, posclass = dataset_settings
    @info "Dataset: $(dataset), positive class label: $(posclass)"

    labelmap = (y) -> y == posclass
    (x_train, y_train), (x_test, y_test) = load(dataset; labelmap = labelmap) |> gpu

    for train_settings in dict_list_simple(Train_Settings)
        @unpack batchsize, epochs, runid = train_settings
        @info "Batchsize: $(batchsize), runid: $(runid)"

        epochlength = length(y_train) ÷ batchsize
        iters = epochs * epochlength
        make_batch = batch_provider(x_train, y_train, batchsize)


        for model_settings in dict_list_simple(Model_Settings)
            @unpack type, arg, surrogate, reg, buffer, seed = model_settings

                # create model
            model = build_network(dataset; seed = seed) |> gpu
            objective = build_loss(type, arg, surrogate, reg)
            pars = params(model)

            loss(x, y) = objective(x, y, model, pars)

            # create callback
            savefunc(c, x, y) = save_simulation(
                c,
                dataset_settings,
                train_settings,
                model_settings,
                model,
                x_train,
                y_train,
                x_test,
                y_test,
                x,
                y,
            )

            cb = CallBack(
                title = string(type),
                iters = iters,
                epochlength = epochlength;
                saveat = epochlength,
                savefunc = savefunc,
            )

            # training
            @unpack optimiser, steplength = train_settings

            batches = (make_batch(; buffer = buffer) for iter in 1:iters)
            opt = optimiser(steplength)
            custom_train!(loss, pars, batches, opt; cb = cb)
        end
    end
end
