#!/bin/bash
set -e
echo "=== EXAONE 4.5 NoPE Analysis ==="

echo "[Exp 1] Attention Entropy: SWA vs Global(NoPE)"
python3 nope_analysis/experiments/exp1_attention_entropy.py

echo ""
echo "[Exp 2] Positional Probe: NoPE 레이어의 암묵적 위치 인코딩"
pip install scikit-learn -q
python3 nope_analysis/experiments/exp2_positional_probe.py

echo ""
echo "결과 저장..."
git add outputs/ nope_analysis/
git commit -m "NoPE analysis results $(date +%Y%m%d_%H%M)"
git push

echo "완료. outputs/ 폴더 확인하세요."
