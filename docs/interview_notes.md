# Interview Notes

Key talking points for technical interviews. Reference these when discussing the project.

---

## Why Macro F1 instead of Accuracy?

WM-811K has severe class imbalance — the "none" class is ~79% of samples.
A model that predicts "none" for everything achieves 79% accuracy but is useless.
Macro F1 treats all 9 classes equally, exposing poor performance on rare defect types.

---

## Why ResNet-18 and not a bigger model?

Three reasons:
1. Wafer maps are 64×64 — simple spatial structure doesn't benefit from deeper networks.
2. Training on a personal machine (no multi-GPU cluster) requires a model that fits in memory.
3. The goal is to demonstrate the full engineering pipeline, not to publish SOTA results.
   Fine-tuned ResNet-18 is the industry-standard starting point for this class of image classification.

---

## What is Western Electric Rules and why does it matter?

WE Rules are a set of 8 statistical tests defined in the Western Electric Handbook (1956) and
adopted by SEMI/ISO standards for semiconductor SPC. Every process engineer in a fab knows them.
Showing that I can implement and explain WE Rules demonstrates domain familiarity beyond just
knowing scikit-learn.

---

## Why simulated data for Module B?

Honest answer: I don't have access to real fab data (NDA, IP).
Simulated data lets me:
- Control anomaly injection precisely (so I have ground truth for evaluation)
- Demonstrate the data engineering pipeline without IP concerns
- Show I understand what real fab data looks like (tool-to-tool variation, drift, etc.)

This is a valid approach for portfolio projects. I always disclose "simulated data" clearly.

---

## What is CPK and why is it on the dashboard?

CPK (Process Capability Index) = min((USL - μ), (μ - LSL)) / (3σ)
It measures how well a process fits within spec limits.
CPK < 1.0 means the process is producing out-of-spec parts.
FAB engineers look at CPK daily — including it shows process knowledge, not just ML knowledge.

---

## How would you handle tool-to-tool variation in production?

Answer structure:
1. Identify systematic offset between tools using mixed-effects models or ANOVA
2. Apply per-tool normalization before running SPC / ML models
3. Maintain separate control charts per tool, then aggregate at fleet level
4. Use domain adaptation techniques if retraining per tool is not feasible

---

## How would you handle recipe drift (baseline shift)?

Answer structure:
1. Adaptive control limits: recalculate UCL/LCL on a rolling window rather than a fixed baseline
2. CUSUM charts are more sensitive to small gradual drifts than Shewhart charts
3. For ML models: implement data drift detection (e.g., PSI, KS test on input features)
4. Retrain trigger: when drift metric exceeds threshold, flag for engineer review before auto-retrain

---

## False alarm cost in fab

A false alarm in fab means:
- Stopping a tool (costs ~$1k-$10k/hour downtime)
- Engineering investigation time (~2-4 hours)
- Potential yield loss from under-processing subsequent wafers

So high precision is often more important than high recall in production SPC.
The ensemble strategy and threshold tuning in this project can be configured to prioritize
precision ("all" strategy) or recall ("any" strategy).

---

## Limitations to acknowledge proactively

1. All results are on simulated or public dataset data — not validated in a real fab environment.
2. The CNN was trained without wafer-lot-level stratification (ideal split would be by lot to
   avoid data leakage between wafers from the same lot).
3. SECOM has only 1,567 samples — too few for robust deep learning; used for pipeline demo only.
4. The Autoencoder threshold is set on training data — in production, this would need
   continuous recalibration as the process evolves.
