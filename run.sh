#!/bin/bash
set -e

echo "=== EXAONE 4.5 Weight Analysis ==="
echo "Step 1: Install dependencies"
pip install -r requirements.txt -q

echo ""
echo "Step 2: Run analysis"
# Remove --no-spectral to enable SVD (slower but more complete)
python analyze.py \
    --repo-id LGAI-EXAONE/EXAONE-4.5-33B \
    --cache-dir ./model_cache \
    --output-dir ./outputs \
    --spectral-max-dim 2048

echo ""
echo "Step 3: Save results to GitHub"
git add outputs/
git commit -m "analysis results $(date +%Y%m%d_%H%M)"
git push

echo "Done."
