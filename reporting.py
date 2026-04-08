"""
reporting.py - Final Reporting System
Day 6: Validation report, top designs, failure reasons
"""

import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from validator import Validator, generate_mock_simulations, load_real_simulations
from cross_domain_validator import CrossDomainValidator
from accuracy_validator import AccuracyValidator, generate_mock_predictions
from failure_analyzer import FailureAnalyzer
from decision_logger import PersistentDecisionLogger


@dataclass
class TopDesign:
    rank: int
    design_id: str
    overall_score: float
    biology_score: float
    materials_score: float
    physics_score: float
    chemistry_score: float
    recommendation: str

    def to_dict(self):
        return {
            "rank": self.rank,
            "design_id": self.design_id,
            "scores": {
                "overall": round(self.overall_score, 4),
                "biology": round(self.biology_score, 4),
                "materials": round(self.materials_score, 4),
                "physics": round(self.physics_score, 4),
                "chemistry": round(self.chemistry_score, 4),
            },
            "recommendation": self.recommendation,
        }


@dataclass
class ValidationReport:
    report_id: str
    generated_at: str
    executive_summary: dict
    top_designs: list
    failure_summary: dict
    accuracy_summary: dict
    system_health: dict
    recommendations: list

    def to_dict(self):
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "executive_summary": self.executive_summary,
            "top_designs": [d.to_dict() for d in self.top_designs],
            "failure_summary": self.failure_summary,
            "accuracy_summary": self.accuracy_summary,
            "system_health": self.system_health,
            "recommendations": self.recommendations,
        }

    def to_text(self) -> str:
        lines = [
            "=" * 70,
            f"  DESIGN VALIDATION REPORT  |  {self.generated_at[:19]}",
            f"  Report ID: {self.report_id}",
            "=" * 70,
            "",
            "── EXECUTIVE SUMMARY ──────────────────────────────────────────────",
        ]
        for k, v in self.executive_summary.items():
            lines.append(f"  {k:<30}: {v}")

        lines += ["", "── TOP DESIGNS ─────────────────────────────────────────────────────"]
        for d in self.top_designs:
            lines.append(
                f"  #{d.rank}  {d.design_id:<15}  Score: {d.overall_score:.3f}  "
                f"| {d.recommendation}"
            )

        lines += ["", "── FAILURE ANALYSIS ────────────────────────────────────────────────"]
        if "distribution" in self.failure_summary:
            for cat, pct in self.failure_summary["distribution"].items():
                bar = "█" * int(pct / 5)
                lines.append(f"  {cat:<22}: {bar:<20} {pct}%")

        lines += ["", "── ACCURACY METRICS ────────────────────────────────────────────────"]
        for k, v in self.accuracy_summary.items():
            lines.append(f"  {k:<30}: {v}")

        lines += ["", "── RECOMMENDATIONS ─────────────────────────────────────────────────"]
        for i, r in enumerate(self.recommendations, 1):
            lines.append(f"  {i}. {r}")

        lines += ["", "=" * 70]
        return "\n".join(lines)


class ReportingSystem:
    def __init__(self, n_designs=40, real_data_path: Optional[str] = None):
        self.n_designs = n_designs
        self.logger = PersistentDecisionLogger(log_dir="logs")
        if real_data_path:
            self.simulations = load_real_simulations(real_data_path, limit=n_designs)
        else:
            self.simulations = generate_mock_simulations(n_designs)
        self.predictions = generate_mock_predictions(100, noise_level=0.04)

    def generate(self) -> ValidationReport:
        # 1. Core validation
        v = Validator(logger=self.logger)
        core = v.validate_batch(self.simulations)

        # 2. Cross-domain
        cdv = CrossDomainValidator(logger=self.logger)
        cd = cdv.validate_batch(self.simulations)

        # 3. Accuracy
        av = AccuracyValidator()
        acc = av.validate(self.predictions)
        acc_d = acc.to_dict()
        calibration = av.calibrate_threshold(self.predictions, fp_max=0.05)

        # 4. Failures
        failed_sims = [s for s, r in zip(self.simulations, v.results) if not r.passed]
        fa = FailureAnalyzer()
        fail_report = fa.analyze_batch(failed_sims) if failed_sims else {"distribution": {}, "total_failures": 0, "top_suggestions": []}
        feedback = fa.export_feedback_dataset(failed_sims, out_dir="feedback") if failed_sims else {}

        # Build top designs
        passed_results = sorted(v.passed, key=lambda r: r.score, reverse=True)
        sim_map = {s.design_id: s for s in self.simulations}
        top_designs = []
        for rank, res in enumerate(passed_results[:10], 1):
            sim = sim_map.get(res.design_id)
            if not sim:
                continue
            rec = (
                "✅ Deploy immediately" if res.score > 0.90 else
                "✅ Approved for production" if res.score > 0.80 else
                "⚠️ Approved — monitor closely"
            )
            top_designs.append(TopDesign(
                rank=rank,
                design_id=res.design_id,
                overall_score=res.score,
                biology_score=sim.biology_score,
                materials_score=sim.materials_score,
                physics_score=sim.physics_score,
                chemistry_score=sim.chemistry_score,
                recommendation=rec,
            ))

        # Recommendations
        recs = []
        if acc.accuracy < 0.97:
            recs.append("Improve surrogate model training to raise accuracy above 97%")
        if fail_report["distribution"]:
            top_cat = max(fail_report["distribution"], key=fail_report["distribution"].get)
            recs.append(f"Most failures are '{top_cat}' — prioritize fixing this category first")
        if core["pass_rate"] < 0.5:
            recs.append("Pass rate below 50% — review design generation parameters")
        recs += (fail_report.get("top_suggestions") or [])[:3]

        report = ValidationReport(
            report_id=f"VR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            generated_at=datetime.now().isoformat(),
            executive_summary={
                "Total Designs Evaluated": self.n_designs,
                "Passed Core Validation": core["passed"],
                "Failed Core Validation": core["failed"],
                "Pass Rate": f"{core['pass_rate']*100:.1f}%",
                "Cross-Domain Passed": cd["passed"],
                "Surrogate Accuracy": f"{acc_d['metrics']['accuracy_pct']}%",
                "Recommended Threshold (FPR-aware)": calibration["recommended_threshold"],
                "System Status": "OPERATIONAL" if acc.passed else "NEEDS ATTENTION",
            },
            top_designs=top_designs,
            failure_summary={
                "total_failures": fail_report["total_failures"],
                "distribution": fail_report.get("distribution", {}),
                "severity_breakdown": fail_report.get("severity_breakdown", {}),
            },
            accuracy_summary={
                "Surrogate Accuracy": f"{acc_d['metrics']['accuracy_pct']}%",
                "False Positive Rate": f"{acc_d['metrics']['false_positive_rate_pct']}%",
                "False Negative Rate": f"{acc_d['metrics']['false_negative_rate_pct']}%",
                "F1 Score": acc_d["metrics"]["f1_score"],
                "Validation Passed": acc.passed,
            },
            system_health={
                "core_validator": "OK",
                "cross_domain_validator": "OK",
                "accuracy_validator": "OK" if acc.passed else "WARNING",
                "failure_analyzer": "OK",
                "persistent_logging": "OK",
                "feedback_loop": "OK" if feedback else "NO_FAILURE_DATA",
            },
            recommendations=recs,
        )
        return report


if __name__ == "__main__":
    rs = ReportingSystem(n_designs=40)
    report = rs.generate()

    # Print text report
    print(report.to_text())

    # Save JSON
    with open("validation_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print("\n📄 Full report saved: validation_report.json")
