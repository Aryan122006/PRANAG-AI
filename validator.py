"""
validator.py - Core Design Validation System
Now supports:
- Real-data ingestion from CSV/Parquet
- Persistent decision logging (JSONL + SQLite)
"""

import csv
import json
import random
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

from decision_logger import PersistentDecisionLogger


@dataclass
class SimulationResult:
    design_id: str
    score: float
    biology_score: float
    materials_score: float
    physics_score: float
    chemistry_score: float
    traits: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    design_id: str
    passed: bool
    score: float
    total_traits: int
    passed_traits: int
    failed_traits: int
    top_winners: list
    rejection_reason: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class Validator:
    """
    Core validator: score > 0.7 → PASS
    """
    PASS_THRESHOLD = 0.7

    def __init__(self, pass_threshold: float = 0.7, logger: Optional[PersistentDecisionLogger] = None):
        self.PASS_THRESHOLD = pass_threshold
        self.results = []
        self.passed = []
        self.failed = []
        self.logger = logger

    def validate(self, sim: SimulationResult) -> ValidationResult:
        traits = sim.traits or {}
        total = len(traits) if traits else 10
        passed_t = sum(1 for v in traits.values() if v >= self.PASS_THRESHOLD) if traits else int(sim.score * 10)
        failed_t = total - passed_t

        passed = sim.score > self.PASS_THRESHOLD
        reason = None if passed else f"Score {sim.score:.3f} below threshold {self.PASS_THRESHOLD}"

        result = ValidationResult(
            design_id=sim.design_id,
            passed=passed,
            score=sim.score,
            total_traits=total,
            passed_traits=passed_t,
            failed_traits=failed_t,
            top_winners=[],
            rejection_reason=reason
        )
        self.results.append(result)
        if passed:
            self.passed.append(result)
        else:
            self.failed.append(result)

        if self.logger:
            self.logger.log_decision(
                {
                    "design_id": sim.design_id,
                    "component": "core_validator",
                    "timestamp": result.timestamp,
                    "score": sim.score,
                    "passed": passed,
                    "pass_threshold": self.PASS_THRESHOLD,
                    "rejection_reason": reason,
                    "domain_scores": {
                        "biology": sim.biology_score,
                        "materials": sim.materials_score,
                        "physics": sim.physics_score,
                        "chemistry": sim.chemistry_score,
                    },
                    "metadata": sim.metadata,
                }
            )
        return result

    def validate_batch(self, simulations: list) -> dict:
        self.results = []
        self.passed = []
        self.failed = []
        for sim in simulations:
            self.validate(sim)

        sorted_passed = sorted(self.passed, key=lambda r: r.score, reverse=True)
        top_winners = [r.design_id for r in sorted_passed[:5]]
        for r in self.results:
            r.top_winners = top_winners

        return {
            "total": len(self.results),
            "passed": len(self.passed),
            "failed": len(self.failed),
            "pass_rate": len(self.passed) / len(self.results) if self.results else 0,
            "top_winners": top_winners,
            "results": [asdict(r) for r in self.results]
        }


def generate_mock_simulations(n=20) -> list:
    random.seed(42)
    sims = []
    for i in range(n):
        score = random.uniform(0.3, 1.0)
        sims.append(SimulationResult(
            design_id=f"DESIGN_{i+1:03d}",
            score=score,
            biology_score=random.uniform(0.4, 1.0),
            materials_score=random.uniform(0.4, 1.0),
            physics_score=random.uniform(0.4, 1.0),
            chemistry_score=random.uniform(0.4, 1.0),
            traits={f"trait_{j}": random.uniform(0.3, 1.0) for j in range(10)},
            metadata={"iteration": i, "temperature": random.uniform(20, 80)}
        ))
    return sims


def load_real_simulations(input_path: str, limit: Optional[int] = None) -> list:
    """
    Load real simulation rows from CSV/Parquet and map to SimulationResult objects.
    Expected aliases are auto-mapped where possible.
    """
    path_l = input_path.lower()
    rows = []

    if path_l.endswith(".csv"):
        with open(input_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                if limit and len(rows) >= limit:
                    break
    elif path_l.endswith(".parquet"):
        try:
            import pandas as pd  # type: ignore
        except Exception as e:
            raise RuntimeError("Parquet loading requires pandas + pyarrow.") from e
        df = pd.read_parquet(input_path)
        if limit:
            df = df.head(limit)
        rows = df.to_dict(orient="records")
    else:
        raise ValueError(f"Unsupported file type for input_path: {input_path}")

    sims = []
    for i, row in enumerate(rows):
        norm = {str(k).strip().lower(): v for k, v in row.items()}

        def pick(*keys, default=0.0):
            for k in keys:
                if k in norm and norm[k] not in (None, ""):
                    return norm[k]
            return default

        design_id = str(pick("design_id", "trait_id", "id", default=f"DESIGN_REAL_{i+1:06d}"))
        biology = float(pick("biology_score", "biology", "gene_expression", default=0.0))
        materials = float(pick("materials_score", "materials", "strength", default=0.0))
        physics = float(pick("physics_score", "physics", "temperature", default=0.0))
        chemistry = float(pick("chemistry_score", "chemistry", "conductivity", "chemistry_signal", default=0.0))

        # Normalize obvious large-scale fields into 0..1-ish range.
        if physics > 1.0:
            physics = max(0.0, min(1.0, physics / 100.0))
        if materials > 1.0:
            materials = max(0.0, min(1.0, materials / 100.0))
        if chemistry > 1.0:
            chemistry = max(0.0, min(1.0, chemistry / 100.0))

        score = float(pick("score", "viability_score", "overall_score", default=0.0))
        if score == 0.0:
            score = round(0.25 * biology + 0.25 * materials + 0.30 * physics + 0.20 * chemistry, 6)

        sims.append(
            SimulationResult(
                design_id=design_id,
                score=max(0.0, min(1.0, score)),
                biology_score=max(0.0, min(1.0, biology)),
                materials_score=max(0.0, min(1.0, materials)),
                physics_score=max(0.0, min(1.0, physics)),
                chemistry_score=max(0.0, min(1.0, chemistry)),
                traits={},
                metadata={"data_source": input_path, "row_index": i},
            )
        )

    return sims


if __name__ == "__main__":
    validator = Validator()
    simulations = generate_mock_simulations(20)
    report = validator.validate_batch(simulations)
    print(json.dumps(report, indent=2))
    print(f"\n✅ PASSED: {report['passed']}/{report['total']} ({report['pass_rate']*100:.1f}%)")
    print(f"🏆 TOP WINNERS: {report['top_winners']}")
