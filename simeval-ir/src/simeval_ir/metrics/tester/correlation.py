"""Correlation-based tester agreement metrics."""

from typing import Literal

import numpy as np
from scipy import stats

from simeval_ir.core.types import MetricResult
from simeval_ir.metrics.base import BaseMetric, metric


@metric("kendall_tau")
class KendallTau(BaseMetric):
    """Kendall's Tau rank correlation between testers.

    Measures the agreement between two testers' system rankings.
    """

    name = "kendall_tau"
    objective: Literal["tester"] = "tester"
    granularity: Literal["system"] = "system"
    scenarios = {"T", "C"}
    description = "Kendall's Tau rank correlation between tester rankings"

    def compute(
        self,
        scores_1: dict[str, float],
        scores_2: dict[str, float],
        **kwargs,
    ) -> MetricResult:
        """Compute Kendall's Tau between two testers.

        Args:
            scores_1: System scores from tester 1 {system_id: score}.
            scores_2: System scores from tester 2 {system_id: score}.

        Returns:
            MetricResult with Tau value (-1 to 1).
        """
        # Get common systems
        common = set(scores_1.keys()) & set(scores_2.keys())
        if len(common) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "Not enough common systems"},
            )

        x = [scores_1[s] for s in sorted(common)]
        y = [scores_2[s] for s in sorted(common)]

        tau, p_value = stats.kendalltau(x, y)

        return MetricResult(
            name=self.name,
            value=float(tau),
            meta={
                "p_value": float(p_value),
                "n_systems": len(common),
            },
        )


@metric("spearman_rho")
class SpearmanRho(BaseMetric):
    """Spearman's Rho rank correlation between testers."""

    name = "spearman_rho"
    objective: Literal["tester"] = "tester"
    granularity: Literal["system"] = "system"
    scenarios = {"T", "C"}
    description = "Spearman's Rho rank correlation between tester rankings"

    def compute(
        self,
        scores_1: dict[str, float],
        scores_2: dict[str, float],
        **kwargs,
    ) -> MetricResult:
        """Compute Spearman's Rho between two testers.

        Args:
            scores_1: System scores from tester 1.
            scores_2: System scores from tester 2.

        Returns:
            MetricResult with Rho value (-1 to 1).
        """
        common = set(scores_1.keys()) & set(scores_2.keys())
        if len(common) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "Not enough common systems"},
            )

        x = [scores_1[s] for s in sorted(common)]
        y = [scores_2[s] for s in sorted(common)]

        rho, p_value = stats.spearmanr(x, y)

        return MetricResult(
            name=self.name,
            value=float(rho),
            meta={
                "p_value": float(p_value),
                "n_systems": len(common),
            },
        )


@metric("pearson_corr")
class PearsonCorr(BaseMetric):
    """Pearson correlation between tester scores."""

    name = "pearson_corr"
    objective: Literal["tester"] = "tester"
    granularity: Literal["system"] = "system"
    scenarios = {"T", "C"}
    description = "Pearson correlation between tester scores"

    def compute(
        self,
        scores_1: dict[str, float],
        scores_2: dict[str, float],
        **kwargs,
    ) -> MetricResult:
        """Compute Pearson correlation between two testers.

        Args:
            scores_1: System scores from tester 1.
            scores_2: System scores from tester 2.

        Returns:
            MetricResult with correlation value (-1 to 1).
        """
        common = set(scores_1.keys()) & set(scores_2.keys())
        if len(common) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "Not enough common systems"},
            )

        x = [scores_1[s] for s in sorted(common)]
        y = [scores_2[s] for s in sorted(common)]

        r, p_value = stats.pearsonr(x, y)

        return MetricResult(
            name=self.name,
            value=float(r),
            meta={
                "p_value": float(p_value),
                "n_systems": len(common),
            },
        )
