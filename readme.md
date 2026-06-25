# AutoTune: A Unified Benchmark for Highway Traffic Microsimulation Calibration

AutoTune is an open-source benchmark for highway traffic microsimulation calibration. It is designed to help transportation researchers and practitioners compare calibration methods on a shared, reproducible setting built around the I-24 MOTION testbed and the SUMO traffic microsimulator.

This repository accompanies the paper **"AutoTune: A Unified Benchmark for Highway Traffic Microsimulation Calibration"**. A copy is included in this repository as [`AutoTune_Paper.pdf`](AutoTune_Paper.pdf).

The project is intended to support transparent, extensible work in microsimulation calibration: common networks, common data processing tools, common objective functions, and open implementations of several calibration approaches.

## Project Goals

Traffic microsimulation is widely used for transportation planning, traffic operations, and intelligent vehicle research, but simulation results are only as useful as the calibration behind them. AutoTune provides a shared benchmark for evaluating calibration methods across:

- **Macroscopic traffic measurements**, such as detector-level speeds, counts, and flow-related metrics.
- **Microscopic trajectory information**, such as vehicle headways and trajectory-derived velocity fields.
- **Multiple parameter categories**, including origin-destination flows, car-following parameters, lane-changing parameters, traffic heterogeneity, and simulator settings.
- **Multiple calibration approaches**, ranging from SUMO defaults to gradient-free optimization and Bayesian car-following calibration.

The benchmark is built around a SUMO replica of the I-24 MOTION corridor near Nashville, Tennessee, and tooling for deriving benchmark-ready data from I-24 MOTION / INCEPTION trajectories. More details on I-24 can be found [here](https://i24motion.org/).

## Repository Layout

Most project code currently lives under [`benchmark/`](benchmark/). The top-level directory contains repository metadata, the paper PDF, and this README.

```text
microsim-cal/
+-- AutoTune_Paper.pdf
+-- LICENSE
+-- readme.md
+-- benchmark/
    +-- cal_methods/
    |   +-- GA_and_SPSA/
    |   +-- bilevel/
    |   +-- maidm/
    |   +-- sumo_baseline/
    +-- detector_preprocessing/
    +-- sim_files/
    +-- build_data/
    +-- reference_data/
    +-- notebooks/
    +-- error_funcs.py
    +-- utils.py
    +-- environment.yml
    +-- requirements-imports.txt
```

Key directories:

- [`benchmark/cal_methods/`](benchmark/cal_methods/) contains calibration method implementations.
- [`benchmark/detector_preprocessing/`](benchmark/detector_preprocessing/) contains tools for mapping detectors, inferring lane boundaries, and converting I-24 trajectory data into detector-style measurements.
- [`benchmark/sim_files/`](benchmark/sim_files/) contains SUMO network and configuration files.
- [`benchmark/build_data/`](benchmark/build_data/) contains detector metadata and network mapping support files.
- [`benchmark/reference_data/`](benchmark/reference_data/) contains reference arrays used by selected objectives and analyses.
- [`benchmark/notebooks/`](benchmark/notebooks/) contains exploratory notebooks and data-analysis workflows.
- [`benchmark/error_funcs.py`](benchmark/error_funcs.py) and [`benchmark/utils.py`](benchmark/utils.py) contain shared evaluation and utility code.

## Calibration Methods

AutoTune is designed to compare calibration methods with different modeling assumptions and parameter scopes. The paper discusses six methods:

- **SUMO Default**: SUMO default parameters with SUMO's flowrouter used for route and flow generation.
- **MA-IDM**: A Bayesian car-following calibration method using memory-augmented IDM-style modeling.
- **Genetic Algorithm (GA)**: A gradient-free optimization approach for broad parameter calibration.
- **SPSA**: Simultaneous perturbation stochastic approximation, another gradient-free approach well suited to simulation-based optimization.
- **Bilevel calibration**: A two-level calibration approach adapted for coarse and fine-grained traffic data.
- **Sim-in-the-Loop (SL)**: A simulation-in-the-loop framework discussed in the paper as part of the broader benchmark. See [here](https://github.com/yanb514/CorridorCalibration) for the open-source implementation of this method. 

This repository currently includes active code for the SUMO baseline, MA-IDM support workflows, GA/SPSA, and bilevel calibration. Some workflows are more research-prototype-like than package-like; users should expect to inspect configuration files and paths before running large experiments.

## Networks

The SUMO network files live in [`benchmark/sim_files/`](benchmark/sim_files/). The repository currently includes three network scales:

- **I-24 MOTION network**: [`sumo_test.net.xml`](benchmark/sim_files/sumo_test.net.xml) is the main I-24 MOTION corridor network used in the AutoTune paper. It represents the benchmark's largest and most realistic setting, with ramp connections, detector support files, and trajectory-derived evaluation workflows.
- **Medium network**: [`mediumnet.net.xml`](benchmark/sim_files/mediumnet.net.xml) is a mid-sized highway corridor. It is useful for faster calibration experiments that still exercise multi-segment network behavior. See below for source. 
- **Small network**: [`smallnet.net.xml`](benchmark/sim_files/smallnet.net.xml) is a compact synthetic bottleneck network with a short mainline and ramp connection. It is useful for lightweight examples, quick smoke tests, and optimizer debugging. See below for source. 

The small and medium networks are drawn from the open-source [CorridorCalibration](https://github.com/yanb514/CorridorCalibration) project, with permission of the original author. 

## Data

AutoTune is associated with the I-24 MOTION testbed and the INCEPTION vehicle trajectory dataset. The benchmark uses these data to derive:

- detector-style aggregate measurements,
- lane-level counts and speeds,
- microscopic trajectory samples,
- velocity-grid and headway-based objective functions,
- scenario-specific calibration/evaluation inputs.

Large raw or generated data files are intentionally not tracked in Git. Several local data and output directories are ignored, including:

- `benchmark/i24_data/`
- `benchmark/detector_measurements/`
- `benchmark/detector_measurements_scenarios/`
- `benchmark/cal_methods/*/outputs/`
- `benchmark/cal_methods/*/cached_trajectories/`
- `benchmark/sim_files/scen*/`
- `benchmark/sim_files/maptiles/`

If you are reproducing experiments, you will need to place the relevant local data in the expected directories or update the paths in the corresponding configuration files. I-24 MOTION (INCEPTION) data can be found [here](https://i24motion.org/).

## Installation

The Python environment is defined in [`benchmark/environment.yml`](benchmark/environment.yml), with a small pip supplement in [`benchmark/requirements-imports.txt`](benchmark/requirements-imports.txt).

From the repository root:

```bash
cd benchmark
conda env create -f environment.yml
conda activate i24_bench
```

If the environment already exists and you want to update it:

```bash
cd benchmark
conda env update -f environment.yml --prune
```

### SUMO

Eclipse SUMO is an external dependency for the simulation workflows. Do **not** install the Conda package named `sumo`; that package is unrelated to the Eclipse SUMO traffic simulator.

Install Eclipse SUMO separately using the official installer, a system package manager, Homebrew, or Docker. Before running SUMO-based workflows, make sure:

- `SUMO_HOME` points to your SUMO installation.
- SUMO command-line tools such as `sumo`, `sumo-gui`, `duarouter`, `od2trips`, and `flowrouter.py` are available.
- Python can import SUMO tools such as `sumolib` and `traci`, either through `SUMO_HOME/tools` or through appropriate Python packages.

## Common Workflows

### 1. Build Detector Inputs

Detector preprocessing tools live in [`benchmark/detector_preprocessing/`](benchmark/detector_preprocessing/).

- [`build_detectors.py`](benchmark/detector_preprocessing/build_detectors.py) maps detector positions onto the SUMO network and writes detector XML files.
- [`infer_lane_boundaries.py`](benchmark/detector_preprocessing/infer_lane_boundaries.py) infers lane boundaries from I-24 trajectory data.
- [`get_detections.py`](benchmark/detector_preprocessing/get_detections.py) converts trajectory data into aggregate detector measurements.
- [`run_batch_get_detections.sh`](benchmark/detector_preprocessing/run_batch_get_detections.sh) runs detection extraction over a batch of time windows.

These scripts assume the benchmark directory structure and local I-24 data files are available.

### 2. Run the SUMO Baseline

The SUMO baseline is in:

```text
benchmark/cal_methods/sumo_baseline/
```

This workflow uses SUMO's routing tools and default parameters to create a baseline simulation for comparison.

### 3. Run GA or SPSA Calibration

The GA/SPSA workflow is in:

```text
benchmark/cal_methods/GA_and_SPSA/
```

The primary configuration files are:

- [`input_param_smallnet.yaml`](benchmark/cal_methods/GA_and_SPSA/run_params/input_param_smallnet.yaml)
- [`input_param_mediumnet.yaml`](benchmark/cal_methods/GA_and_SPSA/run_params/input_param_mediumnet.yaml)
- [`input_param.yaml`](benchmark/cal_methods/GA_and_SPSA/run_params/input_param.yaml)

These YAML files define the simulation input paths, objective-function type, calibrated parameter blocks, optimizer settings, and evaluation files.

### 4. Work With MA-IDM Support Files

MA-IDM support code and notebooks live in:

```text
benchmark/cal_methods/maidm/
```

The notebooks under `run_maidm/` document data-processing and car-following pair extraction workflows. Some of these depend on external I-24V, NGSIM, or highD-style data layouts and may require additional setup beyond the main environment.

## Objective Functions and Evaluation

Evaluation utilities are centered in [`benchmark/error_funcs.py`](benchmark/error_funcs.py). Supported or partially supported evaluation styles include:

- macroscopic detector speed/count errors,
- speed intersection-over-union metrics,
- microscopic headway distribution comparisons,
- Wasserstein-style trajectory distribution metrics,
- FCD-derived velocity-grid objectives,
- synthetic smallnet and mediumnet benchmark objectives.

The configuration key `error_func_type` controls which objective family a workflow uses. Current config examples include values such as:

- `macro`
- `micro`
- `velocity_grid`
- `smallnet`
- `mediumnet`

## Reproducibility Notes

This is a research codebase. It aims to make calibration methods, benchmark data processing, and evaluation logic open and inspectable, but some workflows still reflect the realities of active academic development:

- Some notebooks contain exploratory or historical paths.
- Some large data artifacts must be supplied locally.
- SUMO version and local `SUMO_HOME` setup can affect results.
- Stochastic calibration methods should be run with attention to seeds, evaluation budgets, and repeated trials.
- Some generated support files are intentionally kept out of Git.
- **The code is under ongoing development, so expect more updates to come!** (And please excuse any rough edges or let me know if you encounter issues.)

Please reach out if you have any suggestions for features or questions on the above. 

When adapting the benchmark, we recommend recording:

- SUMO version,
- Python environment,
- calibration method and config file,
- scenario/date/time window,
- objective function,
- random seed,
- number of simulation evaluations,
- any local data preprocessing choices.

## Contributing

Contributions are welcome. Useful contributions include:

- adding or improving calibration methods,
- improving documentation and setup scripts,
- adding tests for data processing and objective functions,
- making notebooks easier to reproduce,
- adding new benchmark scenarios,
- improving support for external datasets,
- cleaning path assumptions and configuration handling.

Please keep contributions transparent and reproducible: prefer explicit configuration files, document required data, and avoid committing generated artifacts or large local data files.

## Citation

If you use AutoTune in academic work, please cite the associated paper:

```text
Hickert, C., Wang, A., Samaei, M., Zhang, C., Sun, L., Wang, Y.,
Ameli, M., and Wu, C. "AutoTune: A Unified Benchmark for Highway
Traffic Microsimulation Calibration."
```

A formal BibTeX entry can be added here once publication metadata is finalized.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).
