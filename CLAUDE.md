# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational neuroscience project implementing a basal ganglia (BG) network model with genetic algorithm optimization. The project simulates neural networks using Izhikevich neuron models and evolves connection weights through genetic algorithms to match specific behavioral criteria.

## Current Workflow

The project has been streamlined to focus on two main workflows:

### Primary Analysis Tool
- **`analyze_multiple_ga_results.ipynb`** - Main analysis notebook with advanced visualization features:
  - Interactive DNA browsing with voltage traces and network topology
  - **Missed scoring visualization**: Orange highlights show exactly where points were lost
  - Detailed annotations showing expected vs actual neuron behavior
  - Weight pruning analysis and optimization
  - Comparative analysis across multiple GA runs

### Genetic Algorithm Runner  
- **`run_multiple_ga.py`** - Parallel GA execution with multiprocessing optimization
- **`adaptive_tmax_fully_optimized.py`** - Fully optimized GA implementation with memory management

## Core Architecture

### `/src/` - Core Implementation
- **`constants.py`** - Network configuration, neuron parameters, and connectivity definitions
- **`neuron.py`** - Izhikevich neuron model implementation with Numba optimization
- **`network.py`** - Network simulation engine with sparse matrix optimizations
- **`genetic_algorithm.py`** - GA operators (crossover, mutation, selection)
- **`ga_optimization.py`** - High-level GA orchestration and population management
- **`workbench.py`** - Simulation utilities and experiment setup
- **`validation.py`** - Scoring functions with diagnostic capabilities (`diagnose_conditions`)
- **`analysis/`** - Analysis and visualization tools for results

### Supporting Tools
- **`weight_pruning.py`** - Network weight optimization and pruning algorithms
- **`cleanup_unused_files.py`** - Project maintenance tool for removing outdated files

### Key Network Components
- **14 neuron types**: Somatosensory, MSN (Medium Spiny Neurons), SNR (Substantia Nigra), ALM (Anterior Lateral Motor), VM (Ventromedial), PPN, THAL
- **Active synapses**: Pre-defined connectivity matrix between neuron types
- **Inhibitory neurons**: MSNs, SNRs, and ALMinter provide inhibition
- **Tonically active**: SNRs, PPN, and THALgo fire continuously

### Genetic Algorithm Design
- **DNA encoding**: Connection weights as integer arrays, converted to float32 for simulation
- **Operators**: Uniform crossover, self-adaptive Gaussian mutation with niching
- **Selection**: Tournament selection with elite preservation
- **Fitness**: Based on matching experimental vs control firing patterns

## Development Commands

### Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test modules
pytest tests/test_network.py
pytest tests/test_genetic_algorithm.py
```

### Code Quality
```bash
# Format code
black .

# Lint code
flake8 .
```

### Running Simulations
```bash
# Main GA runner - optimized parallel execution
python run_multiple_ga.py

# Direct GA optimization with memory management
python adaptive_tmax_fully_optimized.py

# Clean up old files and environments
python cleanup_unused_files.py
```

### Interactive Development
```bash
# Start Jupyter Lab for analysis
jupyter lab

# Main analysis notebook (with missed scoring visualization)
analyze_multiple_ga_results.ipynb
```

## Missed Scoring Analysis Features

The analysis notebook now provides detailed scoring diagnostics:

### Visual Indicators
- **Orange highlights**: Time periods where scoring criteria were missed
- **Annotations**: Show expected vs actual behavior (e.g., "Miss: W1 G0")
- **Total counts**: Missed points breakdown in plot titles
- **Gold borders**: Highlight neurons used for fitness evaluation

### Scoring Diagnostics
- **Time-bin analysis**: 100ms bins evaluated against behavioral criteria
- **Neuron-specific failures**: Identify which neurons are failing when
- **Condition comparison**: Experimental vs control missed points
- **Pattern identification**: Find systematic scoring failures

### Usage
The `diagnose_conditions()` function from `validation.py` identifies mismatches between expected and actual neuron behavior, providing detailed feedback for network optimization.

## Important Implementation Details

### Performance Optimizations
- **Numba JIT compilation**: Core simulation loops are compiled with `@njit` for ~10x speedup
- **Sparse matrix operations**: Network connectivity uses sparse matrices for 75% sparse networks
- **Vectorized operations**: Neuron updates processed in batches

### Data Storage
- **Results**: Stored in timestamped directories under `/results/`
- **Aggregated results**: Each run produces `aggregated_results.pkl` with all tested DNAs
- **Historical data**: Long-term experiment data in `/data/` directory
- **Cleanup archives**: Old files safely stored in `cleanup_YYYYMMDD_HHMMSS/` folders

### Project Management
- **Environment**: Uses `myenv_3.12/` (Python 3.12) for all dependencies
- **File cleanup**: Automated archiving of outdated experimental files
- **Memory management**: Optimized for long-running GA experiments
- **Multiprocessing**: Parallel GA execution across multiple cores

## Key Configuration Parameters

### Network Timing
- `TMAX = 5000ms` - Total simulation time
- `BIN_SIZE = 100ms` - Time bins for spike counting
- Stimulus timing: Cue at 1000-1200ms, Go signal at 3000-3100ms

### Genetic Algorithm
- Default population size: 100-500 individuals
- Elite preservation: Top 10-20% carried forward
- Mutation rate: ~0.1-0.3 probability per individual
- Crossover: Uniform with 0.5 swap probability

### Neuron Parameters
- Two neuron types: Regular Spiking (RS) and Medium Spiny (MSN)
- Parameters: a, b, k, vr, vt, vpeak, vreset, d, C (Izhikevich model)
- Connection weights: Integer DNA bounds typically (0, 400)

## Validation Criteria

The network is evaluated on specific behavioral requirements:
- **Somatosensory neurons**: Activate only during stimulus period (1000-2000ms)
- **ALM neurons**: Show sustained activity during instruction/delay periods
- **SNR neurons**: Provide appropriate inhibition patterns (tonically active)
- **VM neurons**: Respond to behavioral cues during preparation and response
- **PPN neurons**: Drive movement initiation during response period

### Scoring System
- **100ms time bins**: Network activity evaluated in discrete time windows
- **Experimental vs Control**: Different criteria for each condition
- **Criteria neurons**: Only specific neurons (in `CRITERIA_NAMES`) contribute to fitness
- **Binary scoring**: 1 point per bin where neuron meets expected on/off state
- **Diagnostic feedback**: `diagnose_conditions()` identifies specific failures

### Missed Scoring Analysis
The analysis notebook now shows exactly where points are lost:
- **Orange regions**: Time bins where criteria were not met
- **Detailed annotations**: Expected vs actual spike counts
- **Visual feedback**: Makes it easy to identify problematic time periods and neurons