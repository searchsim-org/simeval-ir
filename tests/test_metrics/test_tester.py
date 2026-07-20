"""Tests for tester reliability metrics."""

import pytest

from simeval_ir.metrics.tester.correlation import (
    KendallTau,
    SpearmanRho,
    PearsonCorr,
)
from simeval_ir.metrics.tester.rate import RATE


class TestKendallTau:
    """Tests for Kendall's Tau correlation."""

    def test_perfect_correlation(self):
        """Test perfect correlation returns 1."""
        scores_1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        scores_2 = {"a": 1.0, "b": 2.0, "c": 3.0}

        metric = KendallTau()
        result = metric.compute(scores_1=scores_1, scores_2=scores_2)

        assert result.name == "kendall_tau"
        assert result.value == 1.0

    def test_inverse_correlation(self):
        """Test inverse correlation returns -1."""
        scores_1 = {"a": 1.0, "b": 2.0, "c": 3.0}
        scores_2 = {"a": 3.0, "b": 2.0, "c": 1.0}

        metric = KendallTau()
        result = metric.compute(scores_1=scores_1, scores_2=scores_2)

        assert result.value == -1.0


class TestSpearmanRho:
    """Tests for Spearman's Rho correlation."""

    def test_perfect_correlation(self):
        """Test perfect correlation returns 1."""
        scores_1 = {"a": 0.5, "b": 0.7, "c": 0.9}
        scores_2 = {"a": 0.1, "b": 0.2, "c": 0.3}

        metric = SpearmanRho()
        result = metric.compute(scores_1=scores_1, scores_2=scores_2)

        assert result.name == "spearman_rho"
        assert abs(result.value - 1.0) < 0.01


class TestPearsonCorr:
    """Tests for Pearson correlation."""

    def test_compute(self):
        """Test computing Pearson correlation."""
        scores_1 = {"a": 0.5, "b": 0.7, "c": 0.9, "d": 0.4}
        scores_2 = {"a": 0.4, "b": 0.8, "c": 0.85, "d": 0.5}

        metric = PearsonCorr()
        result = metric.compute(scores_1=scores_1, scores_2=scores_2)

        assert result.name == "pearson_corr"
        assert -1 <= result.value <= 1


class TestRATE:
    """Tests for RATE tester reliability."""

    def test_compute(self):
        """Test computing RATE reliability estimates."""
        tester_scores = {
            "tester1": {"sys_a": 0.5, "sys_b": 0.7, "sys_c": 0.6},
            "tester2": {"sys_a": 0.55, "sys_b": 0.65, "sys_c": 0.62},
            "tester3": {"sys_a": 0.48, "sys_b": 0.72, "sys_c": 0.58},
        }

        metric = RATE()
        result = metric.compute(tester_scores=tester_scores)

        assert result.name == "rate"
        assert "n_testers" in result.meta
        assert result.per_item is not None
        assert len(result.per_item) == 3
