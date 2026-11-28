"""Generates forensic authenticity reports with confidence scores."""

from typing import Dict, Any, List


def generate_forensic_report(sample_id: str, fake_prob: float, findings: List[str]) -> str:
    verdict = 'SYNTHETIC / DEEPFAKE' if fake_prob > 0.5 else 'AUTHENTIC'
    pct = fake_prob * 100
    report = f'# Forensic Verification Report: {sample_id}\n'
    report += f'- **Verdict**: {verdict}\n'
    report += f'- **Confidence**: {pct:.2f}%\n'
    report += '- **Key Artifact Indicators**:\n'
    for finding in findings:
        report += f'  - {finding}\n'
    return report
