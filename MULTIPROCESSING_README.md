# Multiprocessing Genetic Algorithm Implementation

This implementation extends the genetic algorithm from `v2.ipynb` to run multiple independent GA populations on separate CPU cores in parallel. Each process evolves its own population and all DNA vectors and scores are saved for analysis.

## Key Features

- **True Parallelism**: Each CPU core runs its own independent GA population
- **Complete Data Tracking**: All DNA vectors tested and their scores are saved
- **Performance Optimization**: Uses Numba JIT compilation for fast simulation
- **Flexible Configuration**: Support for different GA configurations and parameters
- **Comprehensive Results**: Detailed statistics, timing, and analysis capabilities

## Files

- `v2_multiprocessing.py` - Main multiprocessing implementation
- `v2_multiprocessing.ipynb` - Jupyter notebook interface
- `test_multiprocessing.py` - Unit tests and validation
- `multiprocessing_usage_example.py` - Usage examples and analysis
- `MULTIPROCESSING_README.md` - This documentation

## Quick Start

### Command Line Usage

```bash
# Basic usage with default settings (medium config, all cores)
python v2_multiprocessing.py

# Specify configuration and number of processes
python v2_multiprocessing.py --config small --processes 4 --generations 5

# Specify custom results directory
python v2_multiprocessing.py --config large --results-dir my_experiment
```

### Python API Usage

```python
from v2_multiprocessing import run_multiprocess_ga

# Run with default settings
results = run_multiprocess_ga()

# Run with custom parameters
results = run_multiprocess_ga(
    config_name="small",
    num_processes=4,
    num_generations=10,
    results_dir="my_experiment"
)

# Access results
print(f"Best score: {results['summary']['best_overall_score']}")
print(f"Total individuals tested: {results['summary']['total_individuals_tested']}")
```

### Jupyter Notebook Usage

Open `v2_multiprocessing.ipynb` for an interactive interface with visualization and analysis tools.

## Configuration Options

Available configurations from `GA_CONFIG`:

| Config | Generations | Population | Mutation Rate | Bounds | Description |
|--------|-------------|------------|---------------|--------|-------------|
| `single` | 1 | 1 | 0.3 | [0,500] | Single individual test |
| `small` | 10 | 100 | 0.3 | [0,500] | Quick testing |
| `medium` | 20 | 100 | 0.3 | [0,500] | Balanced run |
| `large` | 100 | 500 | 0.4 | [0,500] | Long optimization |
| `E` | 300 | 300 | 0.45 | [0,500] | Extended search |
| `F` | 120 | 150 | 0.5 | [0,500] | High mutation |

## Results Structure

### Output Directory Structure

```
results/multiprocess_ga_<config>_<timestamp>/
├── aggregated_results.pkl          # Complete results data
├── summary.txt                     # Human-readable summary
├── process_0_results.pkl           # Process 0 detailed results
├── process_1_results.pkl           # Process 1 detailed results
└── ...                            # Additional process files
```

### Results Data Format

The main results dictionary contains:

```python
{
    'summary': {
        'config_name': str,
        'num_processes': int,
        'total_runtime': float,
        'total_individuals_tested': int,
        'best_overall_score': int,
        'individuals_per_second': float,
        'results_directory': str
    },
    'all_dna_tested': [
        {
            'process_id': int,
            'generation': int,
            'individual_id': int,
            'dna': np.ndarray,           # DNA vector
            'exp_score': int,            # Experimental condition score
            'cont_score': int,           # Control condition score
            'total_score': int,          # Combined score
            'timestamp': float
        },
        ...
    ],
    'generation_stats': [
        {
            'generation': int,
            'process_id': int,
            'best_score': int,
            'mean_score': float,
            'std_score': float,
            'time_taken': float,
            'population_size': int
        },
        ...
    ]
}
```

## Performance

### Typical Performance Metrics

- **Speed**: 7-15 individuals/second per core (depends on system)
- **Scalability**: Near-linear scaling with number of cores
- **Memory**: ~500MB per process for typical configurations

### Performance Tips

1. **Use appropriate number of processes**: Start with `num_processes = cpu_count()` 
2. **Balance population vs generations**: More generations often better than larger populations
3. **Monitor system resources**: Large populations can use significant memory
4. **Use SSD storage**: Results I/O can be bottleneck with many processes

## Analysis and Visualization

### Loading Results

```python
import pickle
from pathlib import Path

# Load aggregated results
results_dir = "results/multiprocess_ga_small_20250101_120000"
with open(Path(results_dir) / "aggregated_results.pkl", 'rb') as f:
    results = pickle.load(f)

# Get all DNA tested
all_dna = results['all_dna_tested']

# Find best performers
best_dna = sorted(all_dna, key=lambda x: x['total_score'], reverse=True)[:10]
```

### Common Analysis Tasks

```python
import numpy as np
import matplotlib.pyplot as plt

# Score distribution
all_scores = [dna['total_score'] for dna in all_dna]
plt.hist(all_scores, bins=50)
plt.xlabel('Fitness Score')
plt.ylabel('Frequency')
plt.show()

# Evolution over time
gen_stats = results['generation_stats']
for process_id in range(num_processes):
    process_stats = [s for s in gen_stats if s['process_id'] == process_id]
    generations = [s['generation'] for s in process_stats]
    best_scores = [s['best_score'] for s in process_stats]
    plt.plot(generations, best_scores, label=f'Process {process_id}')

plt.xlabel('Generation')
plt.ylabel('Best Score')
plt.legend()
plt.show()
```

## Implementation Details

### Architecture

1. **Main Process**: Orchestrates worker processes and aggregates results
2. **Worker Processes**: Each runs independent GA with own random seed
3. **Shared Counter**: Thread-safe progress tracking across processes
4. **Result Storage**: Individual process results + aggregated summary

### Key Functions

- `run_multiprocess_ga()`: Main entry point
- `worker_process()`: Individual process GA execution
- `evaluate_population()`: Numba-optimized fitness evaluation
- `spawn_next_population()`: Tournament selection, crossover, mutation

### Genetic Algorithm Details

- **Selection**: Tournament selection (k=3) with elitism
- **Crossover**: Uniform crossover (50% swap probability)
- **Mutation**: Gaussian mutation with self-adaptive sigma
- **Niching**: Hamming distance-based diversity maintenance
- **Fitness**: Experimental + control condition scores

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies installed (`pip install -r requirements.txt`)
2. **Memory Issues**: Reduce population size or number of processes
3. **Slow Performance**: Check system load, reduce number of processes
4. **Results Not Saved**: Check disk space and directory permissions

### Environment Setup

```bash
# Using the project's virtual environment
source myenv_3.12/bin/activate  # or appropriate environment
python v2_multiprocessing.py

# Or run directly
myenv_3.12/bin/python v2_multiprocessing.py
```

### Debugging

```python
# Run with single process for debugging
results = run_multiprocess_ga(
    config_name="single",
    num_processes=1,
    num_generations=1
)
```

## Extending the Implementation

### Adding New GA Configurations

Edit `src/constants.py`:

```python
GA_CONFIG["my_config"] = {
    "NUM_GENERATIONS": 50,
    "POP_SIZE": 200,
    "MUT_RATE": 0.35,
    "MUT_SIGMA": 0.6,
    "RANK_DEPTH": 100,
    "ELITE_SIZE": 10,
    "CROSSOVER_POINT": None,
    "DNA_BOUNDS": [0, 600]
}
```

### Custom Analysis Functions

```python
def analyze_convergence(results):
    """Custom analysis of convergence patterns."""
    gen_stats = results['generation_stats']
    # Your analysis code here
    return analysis_results

# Use with results
analysis = analyze_convergence(results)
```

## License and Citation

This implementation is part of the basal ganglia network optimization project. 
See main project documentation for license and citation information.

## Contact

For questions or issues with the multiprocessing implementation, 
see the project's main documentation or issue tracker.