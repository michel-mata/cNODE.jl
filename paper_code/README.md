#### Usage
Download: Julia 0.6.4 from the [GitHub repository](https://github.com/JuliaLang/julia/tree/v0.6.4) or the [webpage](https://julialang.org/downloads/oldreleases/)

For generating synthetic data using the GLV model, run:
```
julia test_syntetic_generation.jl
```
This will create the folder `test_data` and save the parameters, species collections and steady-states.
For running the hyperparameter search run:
```
julia test_hyperparameter_search.jl
```
This will create the folder `test_data/Hyperparameters` with the results of a leave-one-out cross validation.
To validate the performance of cNODE using this hyperparameters, run:
```
julia test_syntetic_validation.jl
```
This will create the folder `test_results` with the training and test loss for every dataset.
