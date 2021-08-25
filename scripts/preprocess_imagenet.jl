using DrWatson
@quickactivate "AccuracyAtTop_DeepTopPush"

using Tar, ProgressMeter, MAT, CSV, DataFrames, DelimitedFiles

# extraction
const IMAGENET_DIR = "/disk/macha/data_deeptoppush/datasets/ImageNet2012"
const IMAGENET_VERSION = "ILSVRC2012"

function extract_train(file)
    dir = joinpath(IMAGENET_DIR, "train")
    tmp1 = joinpath(IMAGENET_DIR, "tmp")
    mkpath(dir)
    Tar.extract(file, tmp1)

    @showprogress for file in readdir(tmp1; join = true)
        tmp2 = Tar.extract(file)
        datadir = joinpath(dir, basename(file)[1:end-4])
        mkpath(datadir)
        files = readdir(tmp2)
        mv.(joinpath.(tmp2, files), joinpath.(datadir, files))
    end
    rm(tmp1; recursive = true)
    return 
end

function extract_test(file)
    dir = joinpath(IMAGENET_DIR, "test")
    mkpath(dir)
    Tar.extract(file, dir)
    return 
end

extract_train(IMAGENET_DIR * "/$(IMAGENET_VERSION)_img_train.tar")
extract_test(IMAGENET_DIR * "/$(IMAGENET_VERSION)_img_val.tar")

# label extraction
function create_label_map(meta::String)
    m = matread(meta)["synsets"]
    return Dict(String.(m["WNID"]) .=> Int.(m["$(IMAGENET_VERSION)_ID"]))
end

function data_train(label_map)
    dir_train = joinpath(IMAGENET_DIR, "train")

    dfs = map(readdir(dir_train)) do dir
        return DataFrame(
            files = readdir(joinpath(dir_train, dir); join = true),
            labels = label_map[dir],
            category = dir,
        )
    end
    df = reduce(vcat, dfs)
    CSV.write(joinpath(IMAGENET_DIR, "data_train.csv"), df)
    return df
end

function data_test(file, label_map)
    dir_test = joinpath(IMAGENET_DIR, "test")
    rev_map = Dict(values(label_map) .=> keys(label_map))
    labels = readdlm(file, ' ', Int, '\n')[:]

    df = DataFrame(
        files = readdir(dir_test; join = true),
        labels = labels,
        category = [rev_map[label] for label in labels],
    )
    CSV.write(joinpath(IMAGENET_DIR, "data_test.csv"), df)
    return df
end

label_map = create_label_map(IMAGENET_DIR * "/meta.mat")

data_train(label_map)
data_test(IMAGENET_DIR * "/$(IMAGENET_VERSION)_validation_ground_truth.txt", label_map)
