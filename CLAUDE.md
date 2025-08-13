# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computational neuroscience project implementing a basal ganglia (BG) network model with genetic algorithm optimization. The project simulates neural networks using Izhikevich neuron models and evolves connection weights through genetic algorithms to match specific behavioral criteria.

## Core Architecture

The project is organized into several key components:

### `/src/` - Core Implementation
- **`constants.py`** - Network configuration, neuron parameters, and connectivity definitions
- **`neuron.py`** - Izhikevich neuron model implementation with Numba optimization
- **`network.py`** - Network simulation engine with sparse matrix optimizations
- **`genetic_algorithm.py`** - GA operators (crossover, mutation, selection)
- **`ga_optimization.py`** - High-level GA orchestration and population management
- **`workbench.py`** - Simulation utilities and experiment setup
- **`validation.py`** - Scoring functions that evaluate network behavior against criteria
- **`analysis/`** - Analysis and visualization tools for results

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
# Run main demo with random chromosome
python main.py

# Run genetic algorithm optimization
python ga_runner.py

# Run parallel genetic algorithm
python parallel_ga.py

# Performance testing
python numba_performance_test.py
```

### Interactive Development
```bash
# Start Jupyter Lab for notebooks
jupyter lab

# Key notebooks for exploration:
# - current_workbook.ipynb: Main analysis notebook
# - test_network.ipynb: Network testing and debugging
# - directed_graph.ipynb: Network topology visualization
```

## Important Implementation Details

### Performance Optimizations
- **Numba JIT compilation**: Core simulation loops are compiled with `@njit` for ~10x speedup
- **Sparse matrix operations**: Network connectivity uses sparse matrices for 75% sparse networks
- **Vectorized operations**: Neuron updates processed in batches

### Data Storage
- **Results**: Stored in timestamped directories under `/results/`
- **High scores**: Best fitness values tracked in `high_score.json`
- **Experiment data**: Pickled results in `/data/` directory with metadata

### Testing Strategy
- **Unit tests**: Individual components tested in isolation
- **Integration tests**: Full network simulation validation
- **Performance tests**: Numba compilation and execution speed verification
- **Data validation**: DNA encoding/decoding and matrix operations

### Debugging Tools
- **VS Code debugging**: Configured in `.vscode/launch.json`
- **Line profiler**: Performance bottleneck identification
- **IPython debugger**: Interactive debugging with `ipdb`

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
- Somatosensory neurons activate only during stimulus period
- ALM neurons show sustained activity during instruction/delay
- SNR neurons provide appropriate inhibition patterns
- VM neurons respond to behavioral cues
- Subject chooses behavior corresponding to instruction cue

These criteria are encoded in validation matrices that score experimental vs control conditions.