ACSAC Artifact: Semantic Cluster-Guided Android Malware Drift Detection

Overview
This artifact contains the code, data, and expected CSV outputs for the experiments reported in the submitted ACSAC paper only. It is organized according to the ACSAC Artifact Evaluation layout:

artifact/          main code, data, and result CSVs
infrastructure/    public infrastructure and resource notes
claims/            one folder per paper claim, each with claim.txt, run.sh, expected/
install.sh         one-command Python environment setup
license.txt        license and data-use notes
use.txt            intended use and limitations

Inventory
artifact/code/ contains the experiment code used for the paper-facing results:
common.py, validation_hyperparameter_tuning.py, baseline_static.py, baseline_full_update.py, proposed_performance_suite.py, selection_control_ablation.py, yearly_detection_performance.py, yearly_ablation.py, drebin_evidence_inspection.py, marvin_evidence_inspection.py, cluster_review_behavior.py, data_distribution_export.py, baseline_table_export.py.
paired_ttest_analysis.py regenerates the paper-style paired t-test and 95% CI CSVs for Tables VI-VII from archived period-level F1 logs.

artifact/data/ contains the included evaluation data:
extended-features/ monthly Android feature JSON files.
AndroZoo-Year/ yearly AndroZoo-derived CSV.
features/ DREBIN and Marvin feature pickles used for external cluster-level evidence inspection.

artifact/results/full_csv/ contains archived full CSV logs using the same relative paths produced by a full rerun under ./results/.
artifact/results/paper_tables/ contains compact CSVs matching the paper tables and appendix claims.

Quick Start
1. Run setup:
   bash install.sh

2. Validate the paper claims quickly:
   bash claims/claim1_detection_performance/run.sh
   bash claims/claim2_selection_controls/run.sh
   bash claims/claim3_ablation/run.sh
   bash claims/claim4_cluster_evidence/run.sh
   bash claims/claim5_protocol_data_params/run.sh

The quick claim scripts print the expected CSV tables and should complete in under one minute after installation.

To regenerate the paired t-test CSVs reported alongside Tables VI-VII:
   python artifact/code/paired_ttest_analysis.py

Optional Full Rerun
To rerun the paper experiment scripts from the included data:
   . .venv/bin/activate
   bash artifact/code/run_paper_experiments.sh

Expected full rerun time on the authors' workstation was roughly 3 hours for the full suite, dominated by the proposed monthly suite. The quick claim validators are the recommended first "kick the tires" path.
The full rerun writes fresh outputs under ./results/ in the artifact root. The archived paper-facing expected outputs remain under artifact/results/.
The artifact provides monthly and yearly hyperparameter JSON files under artifact/results/full_csv/hyperparameters/. Downstream experiments load those JSON files instead of retuning on test data. Shared representation/protocol/reporting settings such as D=2^18, d=64, K0=20, T_ttl=3, rho_t, TopK, and classifier alpha are fixed before validation; validation selection is limited to the four paper-tuned Proposed parameters. validation_hyperparameter_tuning.py is included for optional audit/regeneration, but the default full rerun uses the provided JSON to avoid spending evaluator time on parameter search.

Claim Mapping
claim1_detection_performance: Table VI, monthly/yearly Static, FullUpdate, Proposed detection performance.
claim2_selection_controls: Table VII, same-budget random, uncertainty, distance-to-centroid, and Proposed selection controls.
claim3_ablation: Table VIII, monthly module ablation.
claim4_cluster_evidence: Tables IX and X plus Section VI-E evidence summary.
claim5_protocol_data_params: Appendix A dataset distributions and Appendix B hyperparameters.

Notes
The paper-facing CSVs intentionally exclude exploratory rows that are not reported in the submitted paper. Full reruns write new outputs under ./results/ in the artifact root; the archived expected outputs remain under artifact/results/.
