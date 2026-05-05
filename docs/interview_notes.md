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

## What is SPC and why is it essential in semiconductor manufacturing?

Statistical Process Control (SPC) is a real-time monitoring methodology that applies statistical
methods to process data to detect when a process has changed — before defective product is made.
In semiconductor fabs, hundreds of parameters (temperature, pressure, gas flow, etch rate, …) are
measured for every wafer. SPC provides the first line of defense.

Why it is mandatory in fabs:
- SEMI standards (E10, E116) and ISO 7870 require statistical process monitoring at all critical steps.
- Detecting an excursion one lot early can save 25 wafers × up to ~$10k/wafer at advanced nodes.
- The spatial pattern of a violation (gradual drift vs. sudden shift vs. spike) guides root-cause
  analysis toward the right equipment or consumable — before running a full DOE.

SPC output is an *anomaly signal*, not a confirmed root cause. A violation flags that something
changed statistically; determining what and why requires engineer investigation.

---

## Western Electric Rules: what they are and why they matter

WE Rules are a set of 8 statistical tests defined in the Western Electric Handbook (1956) and
adopted by SEMI/ISO standards. Every process engineer in a fab knows them. This project
implements 4 of the most widely used rules:

| Rule | Pattern detected | Severity | What it may suggest |
|------|-----------------|----------|---------------------|
| Rule 1 | 1 point beyond 3σ | HIGH | Sudden large excursion — equipment fault, contamination, measurement error |
| Rule 2 | 2 of 3 consecutive beyond 2σ, same side | MEDIUM | Process operating near limit — gradual parameter shift |
| Rule 3 | 4 of 5 consecutive beyond 1σ, same side | LOW | Sustained bias — consumable wear, recipe drift |
| Rule 4 | 8 consecutive on same side of centerline | LOW | Process mean has shifted — recalibration or component swap |

Severity reflects urgency of engineer response, not certainty that a defect has occurred.

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

## How this project avoids treating ML as a black box

Several design choices make model behavior interpretable and auditable:

1. **SPC layer runs independently first.** Rule-based violations are computed without any neural
   network. An engineer can read the violation table, understand the exact rule that triggered,
   and act — without ever touching a model.

2. **Violations carry rule ID + description + severity**, not just a score. "Rule 1 — HIGH" tells
   the engineer "one measurement exceeded 3σ" — a concrete, actionable statement.

3. **Grad-CAM for wafer maps.** CNN predictions are accompanied by gradient-weighted activation
   maps that highlight which region of the wafer drove the classification. Engineers can compare
   the heatmap with known defect shapes (ring, scratch, center spot) and validate or override.

4. **Ensemble transparency.** The Module B ensemble reports *which sub-model* (SPC, Isolation
   Forest, Autoencoder) flagged each sample. "SPC flagged, Isolation Forest did not" provides
   a richer signal than a single binary prediction.

5. **Ground-truth labels in simulated data.** The synthetic dataset includes `anomaly_type` and
   `suspected_root_cause` columns, enabling per-type precision/recall evaluation. This makes it
   possible to say "SPC catches 95% of step-shifts but misses 40% of slow drifts" — an honest,
   specific claim rather than an opaque accuracy number.

---

## Limitations to acknowledge proactively

1. All results are on simulated or public dataset data — not validated in a real fab environment.
2. The CNN was trained without wafer-lot-level stratification (ideal split would be by lot to
   avoid data leakage between wafers from the same lot).
3. SECOM has only 1,567 samples — too few for robust deep learning; used for pipeline demo only.
4. The Autoencoder threshold is set on training data — in production, this would need
   continuous recalibration as the process evolves.
