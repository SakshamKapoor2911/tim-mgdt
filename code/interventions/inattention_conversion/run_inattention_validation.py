"""
Quick Runner for Phase 7.1 InAttention Validation

Usage:
    python run_inattention_validation.py [--num-samples 500] [--device cuda]
"""

import subprocess
import sys
from pathlib import Path

def run_validation():
    """Run the InAttention validation script."""
    script_path = Path(__file__).parent / "code" / "validate_single_layer.py"
    
    print("="*80)
    print("PHASE 7.1: INATTENTION CONVERSION - QUICK START")
    print("="*80)
    print(f"\nRunning: {script_path}")
    print(f"Working directory: {Path(__file__).parent}")
    
    # Run Phase 7.1 validation
    result = subprocess.run(
        [sys.executable, str(script_path)] + sys.argv[1:],
        cwd=str(Path(__file__).parent / "code")
    )
    
    sys.exit(result.returncode)

if __name__ == "__main__":
    run_validation()
