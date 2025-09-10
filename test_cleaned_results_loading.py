#!/usr/bin/env python3
"""
Test loading cleaned results files
"""

import sys
sys.path.append('.')

from dna_analyzer import load_single_file, find_all_aggregated_results
from pathlib import Path

def test_cleaned_results_loading():
    """Test loading files from cleaned_results folder."""
    print("🧪 Testing Cleaned Results File Loading...")
    
    # Find all pickle files
    files = find_all_aggregated_results("cleaned_results")
    
    if not files:
        print("❌ No files found to test with")
        return
    
    print(f"\n📁 Testing with {len(files)} files")
    
    # Test loading each file
    for file_path in files:
        print(f"\n🔍 Testing file: {file_path}")
        
        try:
            # Test with a low score threshold to get some results
            high_scoring = load_single_file((file_path, 900))
            
            print(f"   ✅ Loaded successfully")
            print(f"   📊 Found {len(high_scoring)} DNAs with score >= 900")
            
            if high_scoring:
                # Show sample data
                sample = high_scoring[0]
                print(f"   🧬 Sample DNA info:")
                print(f"      Total score: {sample.get('total_score', 'N/A')}")
                print(f"      Exp score: {sample.get('exp_score', 'N/A')}")
                print(f"      Cont score: {sample.get('cont_score', 'N/A')}")
                print(f"      Non-zero weights: {sample.get('non_zero_weights', 'N/A')}")
                print(f"      Source file: {sample.get('source_file', 'N/A')}")
                
                # Show DNA vector (first few values)
                dna = sample.get('dna')
                if dna is not None:
                    dna_preview = dna[:10] if len(dna) > 10 else dna
                    print(f"      DNA vector preview: {dna_preview.tolist()}")
                
        except Exception as e:
            print(f"   ❌ Failed to load: {e}")
            import traceback
            traceback.print_exc()
    
    return files

if __name__ == "__main__":
    print("🔍 Testing Cleaned Results File Loading")
    print("=" * 50)
    
    try:
        files = test_cleaned_results_loading()
        print(f"\n✅ Cleaned results loading test completed.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()