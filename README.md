# cNODE.jl

[![Build status (Github Actions)](https://github.com/michel-mata/cNODE.jl/workflows/CI/badge.svg)](https://github.com/michel-mata/cNODE.jl/actions)
[![Documentation](https://github.com/michel-mata/cNODE.jl/actions/workflows/Documentation.yml/badge.svg)](https://github.com/michel-mata/cNODE.jl/actions/workflows/Documentation.yml)
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://michel-mata.github.io/cNODE.jl)
---

This repository contains a Julia package for using compositional Neural Ordinary Differential Equations (`cNODE`) associated with the paper: "Predicting microbiome compositions from species assemblages through deep learning" (doi: https://doi.org/10.1101/2021.06.17.448886).

---
## Documentation
Check out the [documentation](https://michel-mata.github.io/cNODE.jl/).


---
## Usage

This package is build for Julia at the long term support release.
Download: Julia 1.6.5 from the [webpage](https://julialang.org/downloads/#long_term_support_release).

To install `cNODE.jl`, on Julia REPL press `]` to acces Pkg utilities and add package:
```
(@v1.6) pkg> add https://github.com/michel-mata/cNODE.jl.git
```

---
## Tutorial

**Synthetic Data**.
For generating synthetic data and validating cNODE, check the tutorial:
```
tutorial_synthetic_data.jl
```

**Real Data**.
For using cNODE with real data, check the tutorial:
```
tutorial_real_data.jl
```

---
## Paper
The results from our [paper](https://doi.org/10.1101/2021.06.17.448886) were obtained in Julia 0.6.4, check the [repository](https://github.com/michel-mata/cNODE-paper) for this version.
