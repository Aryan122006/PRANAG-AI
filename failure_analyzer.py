"""
failure_analyzer.py - Design Failure Root Cause Analysis
Day 4: Identify WHY designs fail — Physics / Boundary / Data issues
"""

import csv
import json
import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from validator import SimulationResult


FAILURE_CATEGORIES = {
    "physics_issue": {
        "label": "Physics Issue",
        "description": "Score violates physical feasibility constraints",
        "suggestions": [
            "Reduce operating temperature range",
            "Recalibrate force boundary conditions",
            "Apply finite element mesh refinement",
            "Review Reynolds/Navier-Stokes parameters",
        ]
    },
    "boundary_issue": {
        "label": "Boundary Condition Issue",
        "description": "Design exceeds geometric or spatial constraints",
        "suggestions": [
            "Adjust geometric boundary parameters",
            "Tighten spatial constraint tolerances",
            "Revisit domain boundary assumptions",
            "Validate edge-case conditions",
        ]
    },
    "data_issue": {
        "label": "Data Quality Issue",
        "description": "Training or simulation data is noisy/insufficient",
        "suggestions": [
            "Augment training dataset with edge cases",
            "Apply data smoothing/denoising pipeline",
            "Increase simulation resolution",
            "Cross-validate with alternative data source",
        ]
    },
    "domain_mismatch": {
        "label": "Cross-Domain Mismatch",
        "description": "Conflict between domain-specific requirements",
        "suggestions": [
            "Re-run cross-domain optimization",
            "Relax constraints in conflicting domains",
            "Apply multi-objective Pareto optimization",
        ]
    },
    "threshold_breach": {
        "label": "Threshold Breach",
        "description": "Score marginally below viability threshold",
        "suggestions": [
            "Tune hyperparameters near decision boundary",
            "Apply ensemble scoring to reduce variance",
            "Investigate near-pass designs for quick wins",
        ]
    }
}


@dataclass
class FailureAnalysis:
    design_id: str
    score: float
    failure_category: str
    failure_label: str
    failure_description: str
    severity: str  # low / medium / high / critical
    sub_scores: dict = field(default_factory=dict)
    suggestions: list = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self):
        return {
            "design_id": self.design_id,
            "score": round(self.score, 4),
            "failure": {
                "category": self.failure_category,
                "label": self.failure_label,
                "description": self.failure_description,
                "severity": self.severity,
                "confidence": round(self.confidence, 3),
            },
            "sub_scores": {k: round(v, 4) for k, v in self.sub_scores.items()},
            "suggestions": self.suggestions,
        }


class FailureAnalyzer:
    """
    Identifies root cause of design failure.
    Returns % distribution across failure categories + suggestions.
    """
    PASS_THRESHOLD = 0.70
    BOUNDARY_THRESHOLD = 0.60
    DATA_NOISE_THRESHOLD = 0.15  # std deviation in sub-scores

    def _classify_severity(self, score: float) -> str:
        if score >= 0.65:
            return "low"
        elif score >= 0.55:
            return "medium"
        elif score >= 0.40:
            return "high"
        return "critical"

    def _get_sub_scores(self, sim: SimulationResult) -> dict:
        return {
            "biology": sim.biology_score,
            "materials": sim.materials_score,
            "physics": sim.physics_score,
            "chemistry": sim.chemistry_score,
        }

    def _compute_std(self, values: list) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5

    def analyze(self, sim: SimulationResult) -> Optional[FailureAnalysis]:
        if sim.score > self.PASS_THRESHOLD:
            return None  # Not a failure

        sub = self._get_sub_scores(sim)
        sub_values = list(sub.values())
        noise = self._compute_std(sub_values)
        min_score = min(sub_values)
        worst_domain = min(sub, key=sub.get)

        # Classification logic
        if min_score < self.BOUNDARY_THRESHOLD and worst_domain == "physics":
            category = "physics_issue"
            confidence = 0.85
        elif min_score < self.BOUNDARY_THRESHOLD:
            category = "boundary_issue"
            confidence = 0.75
        elif noise > self.DATA_NOISE_THRESHOLD:
            category = "data_issue"
            confidence = 0.70
        elif len([v for v in sub_values if v < 0.65]) >= 2:
            category = "domain_mismatch"
            confidence = 0.80
        else:
            category = "threshold_breach"
            confidence = 0.65

        cat_info = FAILURE_CATEGORIES[category]
        severity = self._classify_severity(sim.score)

        return FailureAnalysis(
            design_id=sim.design_id,
            score=sim.score,
            failure_category=category,
            failure_label=cat_info["label"],
            failure_description=cat_info["description"],
            severity=severity,
            sub_scores=sub,
            suggestions=cat_info["suggestions"],
            confidence=confidence,
        )

    def analyze_batch(self, simulations: list) -> dict:
        analyses = []
        for sim in simulations:
            a = self.analyze(sim)
            if a:
                analyses.append(a)

        # Distribution
        dist = {k: 0 for k in FAILURE_CATEGORIES}
        for a in analyses:
            dist[a.failure_category] += 1

        total = len(analyses)
        dist_pct = {k: round(v / total * 100, 1) if total else 0 for k, v in dist.items()}

        # Aggregate suggestions
        seen = set()
        top_suggestions = []
        for a in analyses:
            for s in a.suggestions:
                if s not in seen:
                    seen.add(s)
                    top_suggestions.append(s)
            if len(top_suggestions) >= 8:
                break

        return {
            "total_failures": total,
            "distribution": dist_pct,
            "severity_breakdown": {
                "critical": sum(1 for a in analyses if a.severity == "critical"),
                "high": sum(1 for a in analyses if a.severity == "high"),
                "medium": sum(1 for a in analyses if a.severity == "medium"),
                "low": sum(1 for a in analyses if a.severity == "low"),
            },
            "top_suggestions": top_suggestions,
            "analyses": [a.to_dict() for a in analyses],
        }

    def export_feedback_dataset(self, simulations: list, out_dir: str = "feedback") -> dict:
        """
        Build feedback-loop artifacts so failed designs inform next training cycle.
        """
        report = self.analyze_batch(simulations)
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        json_path = out_path / "failed_designs_feedback.json"
        csv_path = out_path / "hard_negatives.csv"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=True)

        rows = []
        for a in report["analyses"]:
            rows.append(
                {
                    "design_id": a["design_id"],
                    "score": a["score"],
                    "failure_category": a["failure"]["category"],
                    "severity": a["failure"]["severity"],
                    "label": 0,  # hard negative for retraining
                }
            )
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["design_id", "score", "failure_category", "severity", "label"],
            )
            writer.writeheader()
            writer.writerows(rows)

        return {
            "feedback_json": str(json_path),
            "hard_negatives_csv": str(csv_path),
            "failed_samples": report["total_failures"],
        }


if __name__ == "__main__":
    import json
    from validator import generate_mock_simulations, Validator

    sims = generate_mock_simulations(30)
    v = Validator()
    v.validate_batch(sims)
    failed_sims = [s for s, r in zip(sims, v.results) if not r.passed]

    fa = FailureAnalyzer()
    report = fa.analyze_batch(failed_sims)

    print(json.dumps(report, indent=2))
    print(f"\n🔍 Analyzed {report['total_failures']} failures")
    print(f"📊 Distribution: {report['distribution']}")
