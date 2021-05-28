using FileIO, JLD2, NPZ

datadir(args...) = joinpath("/disk/macha/data_deeptoppush/datasets/ImageNet", args...)

function npy_to_jld2(i, dataset::String; remove = false)
    dataset in ["val", "train"] || error("dataset must be one of: train, val")
    @info "Loading data $(dataset)_$(i)"
    @time begin
        y = npzread(datadir("$(dataset)_labels_$(i).npy"))
        x = permutedims(npzread(datadir("$(dataset)_samples_$(i).npy")), (2, 1))
        s = permutedims(npzread(datadir("$(dataset)_scores_$(i).npy")), (2, 1))
    end


    @info "Saving data $(dataset)_$(i)"
    @time FileIO.save(datadir("$(dataset)_$(i).jld2"), Dict("x" => x, "y" => y, "s" => s))

    if remove
        @info "Removing .npy files"
        rm(datadir("$(dataset)_labels_$(i).npy"))
        rm(datadir("$(dataset)_samples_$(i).npy"))
        rm(datadir("$(dataset)_scores_$(i).npy"))
    end
    return y, x, s
end

for i in 1:26
    @info i
    @time y, x, s = npy_to_jld2(i, "train"; remove = true);
end
