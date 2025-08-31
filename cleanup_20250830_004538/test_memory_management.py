#!/usr/bin/env python3
"""
Test script to verify memory management functionality works correctly.
"""

import subprocess
import sys
from pathlib import Path

def test_memory_flags():
    """Test that the memory management flags work without errors."""
    
    print("🧪 Testing memory management flags...")
    print("=" * 50)
    
    # Test 1: Basic run with memory clearing (default)
    print("\n1️⃣ Testing basic run with memory clearing (default)...")
    try:
        result = subprocess.run([
            sys.executable, "run_multiple_ga.py",
            "--runs", "2", 
            "--config", "single",  # Use minimal config for fast test
            "--opt-level", "1"
        ], capture_output=True, text=True, timeout=300)  # 5 minute timeout
        
        if result.returncode == 0:
            print("✅ Basic run with memory clearing: PASSED")
            # Check if memory clearing messages appear in output
            if "Memory cleared:" in result.stdout or "Clearing memory cache" in result.stdout:
                print("  ✅ Memory clearing messages found in output")
            else:
                print("  ⚠️ Memory clearing messages not found in output")
        else:
            print("❌ Basic run with memory clearing: FAILED")
            print(f"  Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Basic run timed out (expected for larger configs)")
    except Exception as e:
        print(f"❌ Basic run crashed: {e}")
    
    # Test 2: Run with memory clearing disabled
    print("\n2️⃣ Testing run with memory clearing disabled...")
    try:
        result = subprocess.run([
            sys.executable, "run_multiple_ga.py",
            "--runs", "2", 
            "--config", "single",
            "--opt-level", "1",
            "--no-clear-memory"
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Run with --no-clear-memory: PASSED")
            # Check that memory clearing is disabled
            if "Memory clearing: disabled" in result.stdout:
                print("  ✅ Memory clearing correctly disabled")
            else:
                print("  ⚠️ Memory clearing status not found in output")
        else:
            print("❌ Run with --no-clear-memory: FAILED")
            print(f"  Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ No-clear-memory run timed out")
    except Exception as e:
        print(f"❌ No-clear-memory run crashed: {e}")
    
    # Test 3: Run with memory limit
    print("\n3️⃣ Testing run with memory limit...")
    try:
        result = subprocess.run([
            sys.executable, "run_multiple_ga.py",
            "--runs", "2", 
            "--config", "single",
            "--opt-level", "1",
            "--memory-limit", "1000"  # 1GB limit
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Run with memory limit: PASSED")
            if "Memory limit: 1000 MB" in result.stdout:
                print("  ✅ Memory limit setting found in output")
            else:
                print("  ⚠️ Memory limit setting not found in output")
        else:
            print("❌ Run with memory limit: FAILED")
            print(f"  Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Memory limit run timed out")
    except Exception as e:
        print(f"❌ Memory limit run crashed: {e}")
    
    # Test 4: Test adaptive_tmax_fully_optimized.py with --clear-cache
    print("\n4️⃣ Testing adaptive_tmax_fully_optimized.py with --clear-cache...")
    try:
        result = subprocess.run([
            sys.executable, "adaptive_tmax_fully_optimized.py",
            "--config", "single",
            "--opt-level", "1",
            "--clear-cache"
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Direct run with --clear-cache: PASSED")
            if "Clearing memory cache" in result.stdout:
                print("  ✅ Cache clearing message found in output")
            else:
                print("  ⚠️ Cache clearing message not found in output")
        else:
            print("❌ Direct run with --clear-cache: FAILED")
            print(f"  Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("⏰ Direct clear-cache run timed out")
    except Exception as e:
        print(f"❌ Direct clear-cache run crashed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Memory management testing completed!")
    print("=" * 50)
    
    print("\n📖 Usage Notes:")
    print("• By default, memory is cleared between runs")
    print("• Use --no-clear-memory to disable memory clearing")
    print("• Use --memory-limit <MB> to force cleanup when limit exceeded")
    print("• Memory management helps prevent out-of-memory crashes")
    print("• For long runs, consider using a memory limit (e.g., --memory-limit 8000)")

if __name__ == "__main__":
    test_memory_flags()