"""
cross_domain_validator.py - Sequential Cross-Domain Validation
Day 2: Biology → Materials → Physics → Chemistry
If any domain fails → STOP and return failure reason
"""

from dataclasses import dataclass, field
from typing import Optional
from validator import SimulationResult
from decision_logger import PersistentDecisionLogger


DOMAIN_THRESHOLDS = {
    "biology":   0.65,
    "materials": 0.68,
    "physics":   0.70,
    "chemistry": 0.72,
}

DOMAIN_ORDER = ["biology", "materials", "physics", "chemistry"]


@dataclass
class DomainCheckResult:
    domain: str
    score: float
    threshold: float
    passed: bool
    reason: Optional[str] = None


@dataclass
class CrossDomainResult:
    design_id: str
    overall_passed: bool
    checks: list = field(default_factory=list)
    failure_domain: Optional[str] = None
    failure_reason: Optional[str] = None
    domains_checked: int = 0

    def to_dict(self):
        return {
            "design_id": self.design_id,
            "overall_passed": self.overall_passed,
            "failure_domain": self.failure_domain,
            "failure_reason": self.failure_reason,
            "domains_checked": self.domains_checked,
            "checks": [
                {
                    "domain": c.domain,
                    "score": round(c.score, 4),
                    "threshold": c.threshold,
                    "passed": c.passed,
                    "reason": c.reason
                } for c in self.checks
            ]
        }


class CrossDomainValidator:
    """
    Sequential validation: Biology → Materials → Physics → Chemistry
    Stops immediately on first failure.
    """

    def __init__(self, thresholds: dict = None, logger: Optional[PersistentDecisionLogger] = None):
        self.thresholds = thresholds or DOMAIN_THRESHOLDS
        self.domain_order = DOMAIN_ORDER
        self.logger = logger

    def _get_domain_score(self, sim: SimulationResult, domain: str) -> float:
        return getattr(sim, f"{domain}_score", 0.0)

    def _check_domain(self, sim: SimulationResult, domain: str) -> DomainCheckResult:
        score = self._get_domain_score(sim, domain)
        threshold = self.thresholds[domain]
        passed = score >= threshold
        reason = None if passed else (
            f"{domain.capitalize()} score {score:.3f} < threshold {threshold}"
        )
        return DomainCheckResult(domain=domain, score=score,
                                 threshold=threshold, passed=passed, reason=reason)

    def validate(self, sim: SimulationResult) -> CrossDomainResult:
        result = CrossDomainResult(design_id=sim.design_id, overall_passed=False)

        for domain in self.domain_order:
            check = self._check_domain(sim, domain)
            result.checks.append(check)
            result.domains_checked += 1

            if not check.passed:
                result.failure_domain = domain
                result.failure_reason = check.reason
                if self.logger:
                    self.logger.log_decision(
                        {
                            "design_id": sim.design_id,
                            "component": "cross_domain_validator",
                            "passed": False,
                            "score": sim.score,
                            "failure_domain": result.failure_domain,
                            "failure_reason": result.failure_reason,
                            "domains_checked": result.domains_checked,
                            "checks": [c.__dict__ for c in result.checks],
                        }
                    )
                return result  # ← STOP on first failure

        result.overall_passed = True
        if self.logger:
            self.logger.log_decision(
                {
                    "design_id": sim.design_id,
                    "component": "cross_domain_validator",
                    "passed": True,
                    "score": sim.score,
                    "domains_checked": result.domains_checked,
                    "checks": [c.__dict__ for c in result.checks],
                }
            )
        return result

    def validate_batch(self, simulations: list) -> dict:
        results = [self.validate(s) for s in simulations]
        passed = [r for r in results if r.overall_passed]
        failed = [r for r in results if not r.overall_passed]

        # Failure distribution by domain
        failure_dist = {d: 0 for d in self.domain_order}
        for r in failed:
            if r.failure_domain:
                failure_dist[r.failure_domain] += 1

        return {
            "total": len(results),
            "passed": len(passed),
            "failed": len(failed),
            "failure_distribution": failure_dist,
            "results": [r.to_dict() for r in results]
        }


if __name__ == "__main__":
    import json
    from validator import generate_mock_simulations

    sims = generate_mock_simulations(20)
    cdv = CrossDomainValidator()
    report = cdv.validate_batch(sims)

    print(json.dumps(report, indent=2))
    print(f"\n✅ Cross-Domain PASSED: {report['passed']}/{report['total']}")
    print(f"❌ Failure Distribution: {report['failure_distribution']}")
