"""RCA-style markdown report generator for Module B process anomaly detection.

Produces a human-readable Markdown report summarising root cause candidates.
The report is explicitly labelled as a candidate list — not a confirmed diagnosis.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Literal

from semiconductor_yield.process.rca import RCACandidate

_CONFIDENCE_LABEL: dict[str, str] = {
    "high":   "[HIGH]",
    "medium": "[MEDIUM]",
    "low":    "[LOW]",
}

_REPORT_HEADER = """\
# Process Anomaly Root Cause Candidate Report

> **DISCLAIMER:** This report lists statistical ROOT CAUSE CANDIDATES only.
> It is NOT a confirmed diagnosis. All findings must be reviewed by a process
> engineer before any tool hold or lot disposition decision is made.
> Real fab root cause analysis additionally requires recipe files, equipment
> tool logs, consumable change records, and metrology raw data.

"""


def generate_markdown_report(
    candidates: list[RCACandidate],
    data_source: Literal["synthetic", "secom"],
    output_path: Path,
    meta: dict | None = None,
) -> str:
    """Generate an RCA candidate Markdown report and write it to disk.

    Args:
        candidates: Ranked candidate list from :func:`rca.analyze`.
        data_source: ``"synthetic"`` or ``"secom"`` — shown in report header.
        output_path: Destination file path (e.g. ``reports/process/rca_report.md``).
            Parent directories are created automatically.
        meta: Optional metadata dict. Recognised keys: ``n_samples``,
            ``n_anomalies``, ``anomaly_rate``, ``analysis_period``,
            ``feature_set``, ``split``.

    Returns:
        The full report text (also written to ``output_path``).
    """
    meta = meta or {}
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines: list[str] = [_REPORT_HEADER]

    # ── Metadata block ─────────────────────────────────────────────────────────
    lines.append("## Report Metadata\n")
    lines.append(f"| Key | Value |")
    lines.append(f"|-----|-------|")
    lines.append(f"| Generated | {now} |")
    lines.append(f"| Data source | {data_source} |")
    if "feature_set" in meta:
        lines.append(f"| Feature set | `{meta['feature_set']}` |")
    if "split" in meta:
        lines.append(f"| Evaluation split | {meta['split']} |")
    if "n_samples" in meta:
        lines.append(f"| Samples analyzed | {meta['n_samples']:,} |")
    if "n_anomalies" in meta:
        rate = meta.get("anomaly_rate", "")
        rate_str = f" ({rate:.1%})" if isinstance(rate, float) else ""
        lines.append(f"| Ground-truth anomalies | {meta['n_anomalies']}{rate_str} |")
    if "analysis_period" in meta:
        lines.append(f"| Analysis period | {meta['analysis_period']} |")
    lines.append("")

    # ── Candidate count summary ────────────────────────────────────────────────
    if not candidates:
        lines.append("## Result\n")
        lines.append(
            "_No root cause candidates could be generated from the available evidence. "
            "Verify that SPC violations and/or anomaly scores are available and non-empty._\n"
        )
    else:
        lines.append(f"## Top {len(candidates)} Root Cause Candidate(s)\n")
        lines.append(
            f"_Ranked by suspicion score (highest first). "
            f"Confidence reflects breadth of evidence, not probability of being the true root cause._\n"
        )

        for rank, candidate in enumerate(candidates, start=1):
            label = _CONFIDENCE_LABEL.get(candidate.confidence_level, "")
            lines.append(f"---\n")
            lines.append(
                f"### Candidate {rank} — {candidate.suspected_process_step} "
                f"(Confidence: {label} **{candidate.confidence_level}**)\n"
            )

            # Summary table
            lines.append("| Field | Detail |")
            lines.append("|-------|--------|")
            lines.append(f"| Suspected step | `{candidate.suspected_process_step}` |")
            lines.append(
                f"| Suspicious features | "
                + (", ".join(f"`{f}`" for f in candidate.suspicious_features) or "_none identified_")
                + " |"
            )
            lines.append(f"| Confidence level | **{candidate.confidence_level}** |")
            lines.append("")

            # Evidence
            lines.append("**Evidence:**\n")
            for ev in candidate.evidence:
                lines.append(f"- {ev}")
            lines.append("")

            # Recommended checks
            lines.append("**Recommended engineer checks:**\n")
            for chk in candidate.recommended_checks:
                lines.append(f"- {chk}")
            lines.append("")

            # Limitation note
            lines.append(
                f"> **Limitation:** {candidate.limitation_note}\n"
            )

    # ── Footer ─────────────────────────────────────────────────────────────────
    lines.append("---\n")
    lines.append("## Important Notes\n")
    lines.append(
        "- All candidates above are statistical suggestions based on SPC rule violations "
        "and anomaly detector outputs.\n"
        "- Confidence levels reflect how many independent evidence sources agree, "
        "not the probability that this is the true root cause.\n"
        "- A **high** confidence candidate with a single data source (e.g., only SPC) "
        "is still unconfirmed.\n"
        "- In a real fab, any candidate requires engineer review and corroboration with "
        "recipe audit, tool log review, and additional metrology before action.\n"
    )

    report_text = "\n".join(lines)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")

    return report_text
