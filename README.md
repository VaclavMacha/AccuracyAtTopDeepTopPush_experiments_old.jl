# AccuracyAtTop_DeepTopPush

This repository is a complementary material to our paper *DeepTopPush: Simple and Scalable Method for Accuracy at the Top*. This paper was submitted to the [Thirty-fifth Conference on Neural Information Processing Systems NeurIPS 2021](https://nips.cc/).

# Running the codes

All required packages are listed in the `Project.toml` file. Before running any of provided scripts, go to the project directory and from the Pkg REPL run the following commands

```julia
(@v1.5) pkg> activate .
(AccuracyAtTop_DeepTopPush) pkg> instantiate
```

For more information see the [manual.](https://julialang.github.io/Pkg.jl/v1/environments/#Using-someone-else's-project-1)

The repository consists of two folders. The `src` folder contains the auxiliary functions needed for experiments, and the `scripts` folder contains scripts for all experiments. Only files which name starts with `run_*` contains the experiments. The rest of the files are used to visualize and export the results.

# Datasets

Not all of the used datasets can be downloaded automatically. ImageNet dataset can be downloaded from the official [page](https://www.image-net.org/) and preprocessed using provided scripts: `preprocess_imagenet.py` and then by `preprocess_imagenet.jl`. Absolute paths defined in the scripts must be changed.

Preprocessed Molecules dataset is provided in the `datasets` folder. 
