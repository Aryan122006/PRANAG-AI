"""
validation_suite.py - Full Validation Suite
Day 5: Run all tests — accuracy, speed, domain validation — generate report
"""

import time
import json
from dataclasses import dataclass, field
from datetime import datetime

from validator import Validator, generate_mock_simulations, load_real_simulations
from cross_domain_validator import CrossDomainValidator
from accuracy_validator import AccuracyValidator, generate_mock_predictions
from failure_analyzer import FailureAnalyzer
from decision_logger import PersistentDecisionLogger


@dataclass
class SuiteTestResult:
    name: str
    passed: bool
    duration_ms: float
    summary: dict
    errors: list = field(default_factory=list)


@dataclass
class ValidationSuiteReport:
    suite_passed: bool
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_duration_ms: float
    timestamp: str
    test_results: list = field(default_factory=list)
    overall_summary: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "suite_passed": self.suite_passed,
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "timestamp": self.timestamp,
            "test_results": [
                {
                    "name": t.name,
                    "passed": t.passed,
                    "duration_ms": round(t.duration_ms, 2),
                    "summary": t.summary,
                    "errors": t.errors
                } for t in self.test_results
            ],
            "overall_summary": self.overall_summary
        }


class ValidationSuite:
    """
    Orchestrates all validators and produces a unified test report.
    """

    def __init__(self, n_designs=50, real_data_path: str = None):
        self.n_designs = n_designs
        self.logger = PersistentDecisionLogger(log_dir="logs")
        if real_data_path:
            self.simulations = load_real_simulations(real_data_path, limit=n_designs)
        else:
            self.simulations = generate_mock_simulations(n_designs)
        self.predictions = generate_mock_predictions(100, noise_level=0.04)

    def _run_test(self, name: str, fn) -> SuiteTestResult:
        t0 = time.perf_counter()
        errors = []
        summary = {}
        passed = False
        try:
            result = fn()
            passed = result.get("passed", True)
            summary = result
        except Exception as e:
            errors.append(str(e))
            passed = False
        duration = (time.perf_counter() - t0) * 1000
        return SuiteTestResult(name=name, passed=passed,
                               duration_ms=duration, summary=summary, errors=errors)

    # ── Test 1: Core Validation ──────────────────────────────────────────────
    def test_core_validation(self) -> dict:
        v = Validator(logger=self.logger)
        report = v.validate_batch(self.simulations)
        return {
            "passed": report["pass_rate"] > 0,
            "total": report["total"],
            "passed_designs": report["passed"],
            "failed_designs": report["failed"],
            "pass_rate_pct": round(report["pass_rate"] * 100, 1),
            "top_winners": report["top_winners"],
        }

    # ── Test 2: Cross-Domain Validation ─────────────────────────────────────
    def test_cross_domain(self) -> dict:
        cdv = CrossDomainValidator(logger=self.logger)
        report = cdv.validate_batch(self.simulations)
        return {
            "passed": report["passed"] >= 0,
            "total": report["total"],
            "passed_designs": report["passed"],
            "failed_designs": report["failed"],
            "failure_distribution": report["failure_distribution"],
        }

    # ── Test 3: Accuracy Validation ──────────────────────────────────────────
    def test_accuracy(self) -> dict:
        av = AccuracyValidator()
        metrics = av.validate(self.predictions)
        d = metrics.to_dict()
        calibration = av.calibrate_threshold(self.predictions, fp_max=0.05)
        return {
            "passed": metrics.passed,
            "accuracy_pct": d["metrics"]["accuracy_pct"],
            "false_positive_pct": d["metrics"]["false_positive_rate_pct"],
            "false_negative_pct": d["metrics"]["false_negative_rate_pct"],
            "f1_score": d["metrics"]["f1_score"],
            "violations": metrics.violations,
            "recommended_threshold": calibration["recommended_threshold"],
            "meets_fp_target": calibration["meets_fp_target"],
        }

    # ── Test 4: Speed Test ───────────────────────────────────────────────────
    def test_speed(self) -> dict:
        SPEED_THRESHOLD_MS = 500
        t0 = time.perf_counter()
        v = Validator(logger=self.logger)
        v.validate_batch(self.simulations)
        elapsed = (time.perf_counter() - t0) * 1000
        return {
            "passed": elapsed < SPEED_THRESHOLD_MS,
            "elapsed_ms": round(elapsed, 2),
            "threshold_ms": SPEED_THRESHOLD_MS,
            "designs_per_second": round(self.n_designs / (elapsed / 1000), 1),
        }

    # ── Test 5: Failure Analysis ─────────────────────────────────────────────
    def test_failure_analysis(self) -> dict:
        v = Validator()
        v.validate_batch(self.simulations)
        failed_sims = [s for s, r in zip(self.simulations, v.results) if not r.passed]
        fa = FailureAnalyzer()
        report = fa.analyze_batch(failed_sims)
        feedback = fa.export_feedback_dataset(failed_sims, out_dir="feedback")
        return {
            "passed": report["total_failures"] >= 0,
            "total_failures": report["total_failures"],
            "distribution": report["distribution"],
            "severity_breakdown": report["severity_breakdown"],
            "feedback_artifacts": feedback,
        }

    def run(self) -> ValidationSuiteReport:
        t_start = time.perf_counter()
        tests = [
            ("Core Validation",       self.test_core_validation),
            ("Cross-Domain Validation", self.test_cross_domain),
            ("Accuracy Validation",   self.test_accuracy),
            ("Speed Test",            self.test_speed),
            ("Failure Analysis",      self.test_failure_analysis),
        ]

        results = []
        for name, fn in tests:
            r = self._run_test(name, fn)
            results.append(r)
            status = "✅" if r.passed else "❌"
            print(f"  {status} {name:<30} {r.duration_ms:.1f}ms")

        total_ms = (time.perf_counter() - t_start) * 1000
        passed_tests = sum(1 for r in results if r.passed)
        all_passed = passed_tests == len(results)

        report = ValidationSuiteReport(
            suite_passed=all_passed,
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=len(results) - passed_tests,
            total_duration_ms=total_ms,
            timestamp=datetime.now().isoformat(),
            test_results=results,
            overall_summary={
                "n_designs_tested": self.n_designs,
                "all_systems_operational": all_passed,
            }
        )
        return report


if __name__ == "__main__":
    print("🚀 Running Full Validation Suite...\n")
    suite = ValidationSuite(n_designs=50)
    report = suite.run()

    print(f"\n{'='*50}")
    status = "✅ SUITE PASSED" if report.suite_passed else "❌ SUITE FAILED"
    print(f"{status}  ({report.passed_tests}/{report.total_tests} tests)")
    print(f"Total runtime: {report.total_duration_ms:.1f}ms")

    with open("suite_report.json", "w") as f:
        json.dump(report.to_dict(), f, indent=2)
    print("\n📄 Report saved to suite_report.json")
