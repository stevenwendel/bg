#!/usr/bin/env python3
"""
Clean up unused files in the BG project, keeping only essential files.

This script identifies files that are not used by the main workflows:
- analyze_multiple_ga_results.ipynb
- run_multiple_ga.py
- Their dependencies in /src/
- weight_pruning.py (used by analysis notebook)

Everything else gets moved to a cleanup/ folder for safe keeping.
"""

import os
import shutil
from pathlib import Path
from datetime import datetime

# Define the essential files that should be kept
ESSENTIAL_FILES = {
    # Main workflow files
    'analyze_multiple_ga_results.ipynb',
    'run_multiple_ga.py',
    
    # Core dependencies
    'weight_pruning.py',
    'adaptive_tmax_fully_optimized.py',  # Used by run_multiple_ga.py
    
    # Documentation and config
    'README.md',
    'CLAUDE.md', 
    'MEMORY_MANAGEMENT.md',
    'MULTIPROCESSING_README.md',
    'requirements.txt',
    
    # Git and project files
    '.gitignore',
    '.git',
}

# Essential directories that should be kept
ESSENTIAL_DIRS = {
    'src',           # Core source code
    'results',       # All results data
    'data',         # Historical data
    'myenv_3.12',   # Keep only the active Python 3.12 environment
    '.git',         # Git repository
}

def get_file_dependencies():
    """
    Analyze which files are actually imported/used by essential files.
    Returns set of files that are dependencies.
    """
    dependencies = set()
    
    # All files in /src/ are dependencies
    src_path = Path('src')
    if src_path.exists():
        for file in src_path.rglob('*.py'):
            dependencies.add(str(file))
    
    return dependencies

def create_cleanup_folder():
    """Create a cleanup folder with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    cleanup_dir = Path(f'cleanup_{timestamp}')
    cleanup_dir.mkdir(exist_ok=True)
    return cleanup_dir

def identify_unused_files(project_root):
    """
    Identify files and directories that can be cleaned up.
    Returns list of paths to move.
    """
    project_path = Path(project_root)
    unused_items = []
    
    # Get dependencies
    dependencies = get_file_dependencies()
    
    for item in project_path.iterdir():
        item_name = item.name
        
        # Skip if it's an essential file or directory
        if item_name in ESSENTIAL_FILES or item_name in ESSENTIAL_DIRS:
            continue
            
        # Skip if it's a dependency
        if str(item.relative_to(project_path)) in dependencies:
            continue
            
        # Skip if it starts with . (hidden files, except .git which is essential)
        if item_name.startswith('.') and item_name != '.git':
            continue
            
        # Skip cleanup folders from previous runs
        if item_name.startswith('cleanup_'):
            continue
            
        # This item can be cleaned up
        unused_items.append(item)
    
    return unused_items

def main():
    """Main cleanup function."""
    project_root = Path('.')
    
    print("🧹 BG Project File Cleanup")
    print("=" * 50)
    
    # Identify unused files
    unused_items = identify_unused_files(project_root)
    
    if not unused_items:
        print("✅ No unused files found! Project is already clean.")
        return
    
    print(f"📁 Found {len(unused_items)} unused items:")
    for item in unused_items:
        item_type = "📁" if item.is_dir() else "📄"
        print(f"  {item_type} {item.name}")
    
    # Proceed automatically with cleanup
    print(f"\n⚠️  These {len(unused_items)} items will be moved to a cleanup folder.")
    print("   They won't be deleted, just moved for safekeeping.")
    print("   Proceeding with cleanup...")
    
    # Create cleanup folder
    cleanup_dir = create_cleanup_folder()
    print(f"\n📦 Created cleanup folder: {cleanup_dir}")
    
    # Move unused items
    moved_count = 0
    for item in unused_items:
        try:
            target = cleanup_dir / item.name
            shutil.move(str(item), str(target))
            print(f"  ✅ Moved {item.name}")
            moved_count += 1
        except Exception as e:
            print(f"  ❌ Failed to move {item.name}: {e}")
    
    print(f"\n✅ Cleanup complete!")
    print(f"   📦 Moved {moved_count} items to {cleanup_dir}")
    print(f"   🎯 Kept essential files for your main workflows:")
    print(f"      • analyze_multiple_ga_results.ipynb")
    print(f"      • run_multiple_ga.py") 
    print(f"      • All files in /src/, /results/, /data/")
    print(f"      • weight_pruning.py and other dependencies")
    
    # Show what's left
    remaining_files = list(project_root.glob('*.py')) + list(project_root.glob('*.ipynb'))
    if remaining_files:
        print(f"\n📋 Remaining Python files in root:")
        for file in sorted(remaining_files):
            print(f"      • {file.name}")

if __name__ == '__main__':
    main()