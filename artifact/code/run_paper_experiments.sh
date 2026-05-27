#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

python3 artifact/code/proposed_performance_suite.py
python3 artifact/code/selection_control_ablation.py --review-ratio 0.3327381546720546
python3 artifact/code/yearly_detection_performance.py
python3 artifact/code/yearly_ablation.py
python3 artifact/code/drebin_evidence_inspection.py
python3 artifact/code/marvin_evidence_inspection.py
python3 artifact/code/baseline_table_export.py
