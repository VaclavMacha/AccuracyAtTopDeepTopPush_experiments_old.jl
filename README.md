# AccuracyAtTop_DeepTopPush

This repository is a complementary material to our paper *[DeepTopPush: Simple and Scalable Method for Accuracy at the Top]()*. This paper was submitted to the [AAAI Conference on Artificial Intelligence](https://aaai.org/Conferences/AAAI-21/).

# Running the codes

All required packages are listed in the `Project.toml` file. Before running any of provided scripts, go to the project directory and from the Pkg REPL run the following commands
```julia
(@v1.5) pkg> activate .
(AccuracyAtTop_DeepTopPush) pkg> instantiate
```
For more information see the [manual.](https://julialang.github.io/Pkg.jl/v1/environments/#Using-someone-else's-project-1)

The repository consists of two folders. The `src` folder contains the auxiliary functions needed for experiments, and the `scripts` folder contains scripts for all experiments. Only files which name starts with `run_*` contains the experiments. The rest of the files are used to visualize and export the results.
