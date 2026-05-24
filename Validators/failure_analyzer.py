"""
failure_analyzer.py  —  DIVYANSHU  (Task 4)
Identifies WHY a design failed: Physics / Boundary / Data issue.
Output: % distribution + suggestions.
"""

from dataclasses import dataclass, field
from typing import Optional
from validator import SimulationResult


FAILURE_CATEGORIES = {
    "physics_issue":    {
        "label": "Physics Issue",
        "desc":  "Violates physical feasibility constraints",
        "suggestions": [
            "Reduce operating temperature range",
            "Recalibrate force boundary conditions",
            "Apply finite element mesh refinement",
        ]
    },
    "boundary_issue":   {
        "label": "Boundary Condition Issue",
        "desc":  "Exceeds geometric or spatial constraints",
        "suggestions": [
            "Adjust geometric boundary parameters",
            "Tighten spatial constraint tolerances",
            "Validate edge-case conditions",
        ]
    },
    "data_issue":       {
        "label": "Data Quality Issue",
        "desc":  "Training or simulation data is noisy/insufficient",
        "suggestions": [
            "Augment training dataset with edge cases",
            "Apply data smoothing pipeline",
            "Increase simulation resolution",
        ]
    },
    "domain_mismatch":  {
        "label": "Cross-Domain Mismatch",
        "desc":  "Conflict between domain-specific requirements",
        "suggestions": [
            "Re-run cross-domain optimisation",
            "Apply multi-objective Pareto optimisation",
        ]
    },
    "threshold_breach": {
        "label": "Threshold Breach",
        "desc":  "Score marginally below viability threshold",
        "suggestions": [
            "Tune hyperparameters near decision boundary",
            "Apply ensemble scoring to reduce variance",
        ]
    },
}


@dataclass
class FailureAnalysis:
    design_id:           str
    score:               float
    failure_category:    str
    failure_label:       str
    failure_description: str
    severity:            str
    sub_scores:          dict = field(default_factory=dict)
    suggestions:         list = field(default_factory=list)
    confidence:          float = 0.0

    def to_dict(self):
        return {
            "design_id": self.design_id,
            "score":     round(self.score, 4),
            "failure":   {
                "category":    self.failure_category,
                "label":       self.failure_label,
                "description": self.failure_description,
                "severity":    self.severity,
                "confidence":  round(self.confidence, 3),
            },
            "sub_scores":  {k: round(v,4) for k,v in self.sub_scores.items()},
            "suggestions": self.suggestions,
        }


class FailureAnalyzer:
    PASS_THRESHOLD      = 0.70  # Spec: viability < 0.7 → KILL
    BOUNDARY_THRESHOLD  = 0.50
    NOISE_THRESHOLD     = 0.15

    def _severity(self, score: float) -> str:
        if score >= 0.65: return "low"
        if score >= 0.55: return "medium"
        if score >= 0.40: return "high"
        return "critical"

    def _std(self, values: list) -> float:
        if not values: return 0.0
        mean = sum(values) / len(values)
        return (sum((v-mean)**2 for v in values) / len(values)) ** 0.5

    def _classify(self, sub: dict, overall: float):
        """
        Classify the root cause of failure using actual sub-scores.
        Returns (category, confidence).
        """
        worst      = min(sub, key=sub.get)
        worst_sc   = sub[worst]
        scores     = list(sub.values())
        std        = self._std(scores)
        mean_sc    = sum(scores) / len(scores)

        # 1. Threshold breach — overall score is close to the passing bar (≥ 0.60)
        #    Marginal failure: one small tweak might flip it.
        if overall >= 0.62:
            return "threshold_breach", 0.88

        # 2. Domain mismatch — high cross-domain variance means one domain is
        #    actively conflicting with the others.
        if std >= 0.12:
            return "domain_mismatch", 0.82

        # 3. Physics issue — physics is the weakest domain
        if worst == "physics":
            return "physics_issue", 0.85

        # 4. Boundary issue — materials score is the weakest
        #    (geometric / structural constraints not met)
        if worst == "materials":
            return "boundary_issue", 0.80

        # 5. Data issue — scores are uniformly low and tightly clustered
        #    (suggests noisy / insufficient training data across the board)
        if std < 0.05 and mean_sc < 0.55:
            return "data_issue", 0.75

        # 6. Data issue — chemistry or biology anomaly
        if worst in ("chemistry", "biology"):
            return "data_issue", 0.72

        # Fallback — physics is the most common failure root cause
        return "physics_issue", 0.65

    def analyze(self, sim: SimulationResult) -> Optional[FailureAnalysis]:
        if sim.score >= self.PASS_THRESHOLD:
            return None

        sub = {
            "biology":   sim.biology_score,
            "materials": sim.materials_score,
            "physics":   sim.physics_score,
            "chemistry": sim.chemistry_score,
        }
        worst     = min(sub, key=sub.get)
        min_score = sub[worst]

        cat, conf = self._classify(sub, sim.score)

        # Build domain-aware suggestions
        cat_suggestions = list(FAILURE_CATEGORIES[cat]["suggestions"])
        domain_tip = f"Worst domain: {worst} (score: {min_score:.3f}) — focus improvements here"
        suggestions = cat_suggestions + [domain_tip]
        # Add a score-specific nudge for threshold breaches
        if cat == "threshold_breach":
            gap = self.PASS_THRESHOLD - sim.score
            suggestions.insert(0, f"Overall score {sim.score:.3f} is only {gap:.3f} below threshold — small gain needed")

        return FailureAnalysis(
            design_id           = sim.design_id,
            score               = sim.score,
            failure_category    = cat,
            failure_label       = FAILURE_CATEGORIES[cat]["label"],
            failure_description = FAILURE_CATEGORIES[cat]["desc"],
            severity            = self._severity(sim.score),
            sub_scores          = sub,
            suggestions         = suggestions,
            confidence          = conf,
        )

    def analyze_batch(self, simulations: list) -> dict:
        analyses = [a for s in simulations if (a := self.analyze(s))]
        total    = len(analyses)
        dist     = {k: 0 for k in FAILURE_CATEGORIES}
        for a in analyses:
            dist[a.failure_category] += 1
        dist_pct = {k: round(v/total*100, 1) if total else 0 for k, v in dist.items()}

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
            "distribution":   dist_pct,
            "severity_breakdown": {
                "critical": sum(1 for a in analyses if a.severity == "critical"),
                "high":     sum(1 for a in analyses if a.severity == "high"),
                "medium":   sum(1 for a in analyses if a.severity == "medium"),
                "low":      sum(1 for a in analyses if a.severity == "low"),
            },
            "top_suggestions": top_suggestions,
            "analyses":        [a.to_dict() for a in analyses],
        }


if __name__ == "__main__":
    import json
    from validator import load_mock_simulations, Validator
    sims = load_mock_simulations(40)
    v    = Validator()
    v.validate_batch(sims)
    failed = [s for s, r in zip(sims, v.results) if not r.passed]
    fa     = FailureAnalyzer()
    report = fa.analyze_batch(failed)
    print(f"Failures: {report['total_failures']}")
    print(f"Distribution: {report['distribution']}")
    print(json.dumps(report['severity_breakdown'], indent=2))
