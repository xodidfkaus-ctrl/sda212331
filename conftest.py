import sys
from pathlib import Path

# Add repo root to sys.path so `nope_analysis` is importable without installation
sys.path.insert(0, str(Path(__file__).parent))
