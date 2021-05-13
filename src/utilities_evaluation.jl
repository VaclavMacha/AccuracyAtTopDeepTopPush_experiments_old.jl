using CSV
using DataFrames
using DataFramesMeta
using EvalMetrics
using Plots
using Statistics

pyplot()

using ProgressMeter: Progress, next!

# -------------------------------------------------------------------------------
# Eval functions
# -------------------------------------------------------------------------------
function tpr_at_fpr(targets, scores, rate)
    t = threshold_at_fpr(targets, scores, rate)
    return true_positive_rate(targets, scores, t)
end

function tpr_at_top(targets, scores)
    t = maximum(scores[targets .== 0])
    return true_positive_rate(targets, scores, t)
end

function tpr_at_top(targets, scores)
    t = maximum(scores[targets .== 0])
    return true_positive_rate(targets, scores, t)
end

function partial_auroc(targets, scores, fprmax)
    rates = logrange(1e-5, fprmax; length=1000)
    ts = unique(threshold_at_fpr(targets, scores, rates))

    fprs = false_positive_rate(targets, scores, ts)
    inds = unique(i -> fprs[i], 1:length(fprs))
    fprs = fprs[inds]
    tprs = true_positive_rate(targets, scores, ts[inds])

    auc_max = abs(maximum(fprs) - minimum(fprs))

    return 100*EvalMetrics.auc_trapezoidal(fprs, tprs)/auc_max
end

# -------------------------------------------------------------------------------
# Compute evaluation and add it to existing model files
# -------------------------------------------------------------------------------
function add_targets_scores(file::String, train::Tuple, test::Tuple; device = identity)
    if !endswith(file, ".bson")
        return false
    end

    d = BSON.load(file)
    overwrite1 = add_targets_scores!(d, train..., :train; device)
    overwrite2 = add_targets_scores!(d, test..., :test; device)
    overwrite3 = add_loss!(d, :minibatch)
    overwrite4 = add_metrics!(d, :train)
    overwrite5 = add_metrics!(d, :test)
    overwrite = any([overwrite1, overwrite2, overwrite3, overwrite4, overwrite5])

    overwrite && bson(file, d)
    return overwrite
end

function add_targets_scores!(d::Dict, x, y, key::Symbol; device = identity)
    overwrite = false
    if haskey(d, key)
        if !haskey(d[key], :targets)
            d[key][:targets] = cpu(vec(y))
            overwrite = true
        end
        if !haskey(d[key], :scores)
            model = d[:model] |> gpu
            d[key][:scores] = cpu(compute_scores(model, x; device))
            overwrite = true
        end
    else
        model = d[:model] |> gpu
        d[key] = Dict(
            :targets => cpu(vec(y)),
            :scores => cpu(compute_scores(model, x; device)),
        )
        overwrite = true
    end
    overwrite = add_loss!(d, key) || overwrite
    return overwrite
end

function add_loss!(d, key::Symbol)
    overwrite = false
    type = extract_model_type(d)
    if !haskey(d, key)
        return overwrite
    end
    if !haskey(d[key], :loss) && !(type <: APPerf || type <: TFCO)
        pars = params(d[:model])
        targets = d[key][:targets]
        scores = d[key][:scores]

        @unpack type, arg, surrogate, reg = d[:model_settings]
        loss = build_loss(type, arg, surrogate, reg)

        d[key] = convert(Dict{Symbol, Any}, d[key])
        d[key][:loss] = loss(targets, scores, pars)
        overwrite = true
    end
    return overwrite
end

function add_metrics!(d, key::Symbol)
    overwrite = false
    targets = d[key][:targets]
    scores = d[key][:scores]

    if !(typeof(d[key]) <: Dict{Symbol, Any})
        d[key] = convert(Dict{Symbol, Any}, d[key])
        overwrite = true
    end
    if !haskey(d[key], :tpr_at_top)
        d[key][:tpr_at_top] = tpr_at_top(targets, scores)
        overwrite = true
    end
    if !haskey(d[key], :tpr_at_fpr_1)
        d[key][:tpr_at_fpr_1] = tpr_at_fpr(targets, scores, 0.01)
        overwrite = true
    end
    if !haskey(d[key], :tpr_at_fpr_5)
        d[key][:tpr_at_fpr_5] = tpr_at_fpr(targets, scores, 0.05)
        overwrite = true
    end
    if !haskey(d[key], :auroc_1)
        d[key][:auroc_1] = partial_auroc(targets, scores, 0.01)
        overwrite = true
    end
    if !haskey(d[key], :auroc_5)
        d[key][:auroc_5] = partial_auroc(targets, scores, 0.05)
        overwrite = true
    end
    return overwrite
end

function run_evaluation(Dataset_Settings)
    for dataset_settings in dict_list_simple(Dataset_Settings)
        @unpack dataset, posclass = dataset_settings
        @info "Dataset: $(dataset), positive class label: $(posclass)"

        labelmap = (y) -> y == posclass
        train, test = load(dataset; labelmap = labelmap)

        dataset_dir = datadir("models", dataset_savename(dataset_settings))
        all_files = String[]
        for (root, dirs, files) in walkdir(dataset_dir)
            append!(all_files, joinpath.(root, files))
        end

        skipped = 0
        overwritten = 0
        p = Progress(length(all_files))
        for file in all_files
            overwrite = false
            try
                overwrite = add_targets_scores(file, train, test; device = gpu)
            catch
                @warn "Problem with: $file"
            end
            if overwrite
                overwritten += 1
            else
                skipped += 1
            end
            next!(p; showvalues = [(:Skipped, skipped), (:Overwritten, overwritten)])
        end
    end
    return
end

# ------------------------------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------------------------------
function create_plots(
    Dataset_Settings,
    Train_Settings,
    Model_Settings;
    save::Bool = false,
    prefix = "",
    xscale = :log10,
    xlims = (1e-4, 1),
    kwargs...
)

    plts = []
    for dataset_settings in dict_list_simple(Dataset_Settings)
        @unpack dataset = dataset_settings
        for train_settings in dict_list_simple(Train_Settings)
            @unpack batchsize, epochs = train_settings

            m_train = plot(title = "Train"; kwargs...)
            m_test = plot(title = "Train"; kwargs...)
            args = []
            for model_settings in dict_list_simple(Model_Settings)
                type = model_settings[:type]
                if type <: TFCO || type <: APPerf
                    train_settings_2 = deepcopy(train_settings)
                    train_settings_2[:seed] = 1
                else
                    train_settings_2 = deepcopy(train_settings)
                end

                dir = modeldir(dataset_settings, train_settings_2, model_settings)

                file = joinpath(dir, simulation_name(epochs))
                if !isfile(file)
                    @warn "File does not exist: $(file)"
                    continue
                end

                d = BSON.load(file)
                push!(args, d[:model_settings][:arg])
                plot_roccurve!(m_train, d, :train; xscale = xscale, xlims = xlims, kwargs...)
                plot_roccurve!(m_test, d, :test; xscale = xscale, xlims = xlims, kwargs...)
            end
            for arg in unique(skipmissing(args))
                vline!(m_train, [arg]; label = "fpr=$(arg)")
                vline!(m_test, [arg]; label = "fpr=$(arg)")
            end
            plot!(m_train, title = "Train")
            plot!(m_test, title = "Test")

            title = join([dataset_savename(dataset_settings), train_savename(train_settings)], "_")

            plt = plot(
                plot_title(title),
                m_train, m_test,
                layout = @layout([A{0.01h}; B; C]),
                size = (1000, 800),
                legend = :outertopright
            )
            push!(plts, plt)

            if save
                dir = plotsdir(dataset_savename(dataset_settings))
                mkpath(dir)
                if isempty(prefix)
                    file_name = string(train_savename(train_settings), ".png")
                else
                    file_name = string(prefix, "_",  train_savename(train_settings), ".png")
                end
                savefig(plt, joinpath(dir, file_name))
            else
                display(plt)
            end
        end
    end
    return plts
end

function plot_title(title)
    return plot(title = title, grid = false, showaxis = false, titlefont=font(9))
end

function plot_roccurve!(plt, d::Dict, key::Symbol; kwargs...)
    y = d[key][:targets]
    s = d[key][:scores]
    plot_roccurve!(plt, s, y; label = model_name(d), kwargs...)
end

function plot_roccurve!(plt, s, y; label = "", kwargs...)
    fprs = logrange(1e-5, 1; length = 298)
    fprs = sort(vcat(fprs, 0.01, 0.05))
    ts = threshold_at_fpr(y, s, fprs)

    auc_score = auc_trapezoidal(roccurve(y, s))
    auc_label = string.("auc: ", round.(100 * auc_score, digits = 2))

    if isempty(label)
        label = "auc: $(auc_label)"
    else
        label = string(label, " (auc: $(auc_label))")
    end
    plot!(plt, roccurve(y, s, ts); label = label, xlabel = "fpr", ylabel = "tpr",seriestype=:steppost, kwargs...)
end

function plot_activity(file)
    a_train, a_test = load_activity(Molecules, Float32)
    (~, y_train), (~, y_test) = load(Molecules; labelmap = (y) -> y == 1)

    d = BSON.load(file)
    plt = plot(
        plot_activity(y_train, a_train, d[:train][:scores]),
        plot_activity(y_test, a_test, d[:test][:scores]),
        legend = :outertopright,
        size = (1000, 400),
    )
    display(plt)
    return plt
end

function plot_activity(y, a, s; title = "")
    neg = findall(vec(y) .== 0)
    pos = findall(vec(y) .== 1)

    plt = scatter(s[neg], a[neg]; xlabel = "scores", ylabel = "activity", title = title, label = "negatives")
    scatter!(plt, s[pos], a[pos]; label = "positives", legend = :outertopright)
    return plt
end

# ------------------------------------------------------------------------------------------
# Collecting results...
# ------------------------------------------------------------------------------------------
function add_missing!(d)
    get!(d, :dataset, missing)
    get!(d, :posclass, missing)
    get!(d, :batchsize, missing)
    get!(d, :iters, missing)
    get!(d, :optimiser, missing)
    get!(d, :steplength, missing)
    get!(d, :type, missing)
    get!(d, :arg, missing)
    get!(d, :surrogate, missing)
    get!(d, :reg, missing)
    get!(d, :buffer, missing)
end

function collect_benchmark(path = datadir("benchmarks"); save = true)
    dfs = []
    for (root, dirs, files) in walkdir(path)
        for file in files
            fl = joinpath(root, file)
            dir_rel = relpath(fl, path)
            ~, bench = parse_savename(replace(dir_rel, "/" => "_"))
            bench = Dict(Symbol(key) => val for (key, val) in bench)

            d = BSON.load(fl)
            add_missing!(bench)
            bench[:time_per_epoch] = mean(d[:times])/d[:iters_in_run]*d[:epochlength]
            push!(dfs, DataFrame(bench))
        end
    end
    df = reduce(vcat, dfs)
    select!(df, Not(["arg", "iters", "optimiser", "posclass", "reg", "steplength", "surrogate"]))
    df = df[["dataset", "batchsize", "type", "buffer", "time_per_epoch"]]
    rename!(df, "type" => "model")
    if save
        mkpath(datadir("results"))
        CSV.write(datadir("results", "benchmarks.csv"), df)
    end
    return df
end

err(x) = std(x; corrected = false)


function collect_metrics(path = datadir("models"); key::Symbol = :test, epochs = 200, save = true)
    dfs = []
    for (root, dirs, files) in walkdir(path)
        for file in files
            contains(file, string(epochs)) || continue
            fl = joinpath(root, file)
            dir_rel = relpath(root, path)
            ~, dict = parse_savename(replace(dir_rel, "/" => "_"))
            dict = Dict(Symbol(key) => val for (key, val) in dict)
            add_missing!(dict)

            d = BSON.load(fl)
            dict[:tpr_at_top] = d[key][:tpr_at_top]
            dict[:tpr_at_fpr_1] = d[key][:tpr_at_fpr_1]
            dict[:tpr_at_fpr_5] = d[key][:tpr_at_fpr_5]
            dict[:auroc_1] = d[key][:auroc_1]
            dict[:auroc_5] = d[key][:auroc_5]
            push!(dfs, DataFrame(dict))
        end
    end
    df = reduce(vcat, dfs)
    select!(df, Not(["seed", "iters", "optimiser", "posclass", "reg", "steplength", "surrogate"]))
    rename!(df, "type" => "model")

    gdf = groupby(df, ["arg", "batchsize", "buffer", "dataset", "epochs", "model"])
    combs = [
        "auroc_1" => mean,
        "auroc_1" => err,
        "auroc_5" => mean,
        "auroc_5" => err,
        "tpr_at_fpr_1" => mean,
        "tpr_at_fpr_1" => err,
        "tpr_at_fpr_5" => mean,
        "tpr_at_fpr_5" => err,
        "tpr_at_top" => mean,
        "tpr_at_top" => err
    ]
    df2 = combine(gdf, combs)

    if save
        mkpath(datadir("results"))
        CSV.write(datadir("results", "metrics.csv"), df2)
    end
    return df2
end


function collect_curves(
    dataset_settings,
    train_settings,
    Model_Settings;
    key = :train,
    epochs = 200,
    xlims = (1e-4, 1),
    npoints = 300,
)

    rates = logrange(xlims...; length = npoints - 1)
    rates = sort(vcat(rates, 0.01, 0.05))
    df = DataFrame()

    for model_settings in dict_list_simple(Model_Settings)
        files = String[]
        for seed in 1:10
            train_settings[:seed] = seed
            dir = modeldir(dataset_settings, train_settings, model_settings)
            file = joinpath(dir, simulation_name(epochs))
            isfile(file) || continue
            push!(files, file)
        end

        fprs = zeros(length(rates))
        tprs = zeros(length(rates))

        for file in files
            d = BSON.load(file)
            y = d[key][:targets]
            s = d[key][:scores]
            ts = threshold_at_fpr(y, s, rates)
            fp, tp = roccurve(y, s, ts)

            fprs .+= fp
            tprs .+= tp
        end
        fprs ./= length(files)
        tprs ./= length(files)

        @unpack arg, type = model_settings
        mdl = ismissing(arg) ? string(type) : "$(type)($(arg))"
        df["$(mdl)_fprates"] = fprs
        df["$(mdl)_tprates"] = tprs
    end
    delete!(dataset_settings, :posclass)
    delete!(train_settings, :seed)
    delete!(train_settings, :seed)
    delete!(train_settings, :optimiser)
    delete!(train_settings, :epochs)
    delete!(train_settings, :steplength)

    sett = merge(dataset_settings, train_settings)
    filename = savename(sett; allowedtypes = allowedtypes(), digits = 4)
    CSV.write(datadir("results", string(filename, "_$(key).csv")), df)
    return
end
