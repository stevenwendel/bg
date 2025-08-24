# Memory Management for Multiple GA Runs

This document explains the memory management features added to prevent out-of-memory crashes during multiple GA runs.

## Overview

The `run_multiple_ga.py` script has been enhanced with memory management capabilities to:
- Monitor memory usage between runs
- Clear memory caches automatically  
- Force memory cleanup when limits are exceeded
- Prevent accumulation of memory across multiple runs

## New Command Line Options

### `--no-clear-memory`
Disables automatic memory clearing between runs.

**Usage:**
```bash
python run_multiple_ga.py --runs 5 --config medium --no-clear-memory
```

**When to use:** 
- When you want maximum performance and aren't concerned about memory usage
- For debugging memory-related issues
- When runs are short and memory accumulation is minimal

### `--memory-limit <MB>`
Sets a memory limit in megabytes. If exceeded before starting a run, forces memory cleanup.

**Usage:**
```bash
python run_multiple_ga.py --runs 10 --config large --memory-limit 8000
```

**When to use:**
- For long-running sessions with many runs
- On systems with limited RAM
- To prevent out-of-memory crashes
- Recommended: 75-80% of available system RAM

### `--clear-cache` (in adaptive_tmax_fully_optimized.py)
Forces memory cleanup at the end of each GA run.

**Usage:**
```bash
python adaptive_tmax_fully_optimized.py --config medium --clear-cache
```

## Memory Management Features

### Automatic Memory Clearing (Default: Enabled)
- Runs garbage collection between GA runs
- Clears NumPy/Numba caches
- Attempts to release memory back to the OS
- Shows before/after memory usage

### Memory Monitoring
- Tracks memory usage before and after each run
- Shows memory increase per run
- Displays total memory usage in run logs
- Requires `psutil` package (optional but recommended)

### Memory Limit Enforcement
- Checks memory before starting each run
- Forces cleanup if limit exceeded
- Prevents runaway memory growth
- Helps maintain system stability

## Installation Requirements

For full memory monitoring functionality:
```bash
pip install psutil
```

Without `psutil`, the script will still work but with limited memory monitoring.

## Usage Examples

### Basic usage with memory management (default):
```bash
python run_multiple_ga.py --runs 5 --config medium --opt-level 3
```
- Memory clearing: **enabled**
- Memory monitoring: **enabled** (if psutil available)
- Memory limit: **none**

### Conservative memory usage:
```bash
python run_multiple_ga.py --runs 10 --config large --memory-limit 6000
```
- Forces cleanup if memory exceeds 6GB
- Good for systems with 8GB+ RAM

### Maximum performance (no memory management):
```bash
python run_multiple_ga.py --runs 3 --config medium --no-clear-memory
```
- Disables memory clearing for maximum speed
- Use only for short runs or systems with abundant RAM

### Heavy-duty runs with aggressive memory management:
```bash
python run_multiple_ga.py --runs 20 --config large --memory-limit 4000 --opt-level 4
```
- Suitable for long overnight runs
- Aggressive memory limit prevents crashes

## Memory Usage Guidelines

### Recommended Memory Limits by System RAM:
- **8GB system:** `--memory-limit 6000`
- **16GB system:** `--memory-limit 12000` 
- **32GB system:** `--memory-limit 24000`
- **64GB+ system:** `--memory-limit 48000`

### Configuration Size vs Memory Usage:
- **single/small:** ~100-500MB per run
- **medium:** ~500-2000MB per run  
- **large:** ~2000-8000MB per run
- **H/J/K configs:** ~8000-20000MB per run

## Troubleshooting

### "Out of Memory" errors:
1. Add `--memory-limit <MB>` with conservative limit
2. Reduce configuration size (e.g., medium → small)
3. Reduce `--processes` count
4. Ensure memory clearing is enabled (default)

### "psutil not available" warning:
```bash
pip install psutil
```
Or continue without detailed memory monitoring.

### Slow performance with memory clearing:
- Use `--no-clear-memory` for short runs
- Memory clearing adds ~1-5 seconds per run
- Trade-off between speed and memory safety

### Memory still growing despite clearing:
- Some memory may not be returnable to OS
- This is normal Python behavior
- Memory limit enforcement still prevents crashes
- Consider restarting between batches of runs

## Implementation Details

### Memory Clearing Process:
1. Force Python garbage collection (3 generations)
2. Clear NumPy/Numba caches
3. Attempt malloc_trim() on Linux systems
4. Clear array pools in adaptive_tmax_fully_optimized.py

### Memory Monitoring:
- Uses `psutil` to get process RSS memory
- Tracks before/after memory for each run
- Calculates memory increase per run
- Logs memory usage in run output

### Cache Clearing in GA Script:
- Clears internal array pools
- Forces garbage collection
- Runs at end of each GA execution
- Triggered by `--clear-cache` flag

## Best Practices

1. **Always use memory limits for production runs:**
   ```bash
   --memory-limit <75% of system RAM>
   ```

2. **Monitor memory usage in initial runs:**
   - Check console output for memory growth patterns
   - Adjust limits based on observed usage

3. **For overnight/long runs:**
   - Use conservative memory limits
   - Enable memory clearing (default)
   - Consider smaller batch sizes

4. **For development/testing:**
   - `--no-clear-memory` for faster iteration
   - Use small configurations
   - Monitor memory manually

5. **System resource planning:**
   - Reserve 20-25% RAM for OS and other processes
   - Consider swap space availability
   - Monitor system memory during runs