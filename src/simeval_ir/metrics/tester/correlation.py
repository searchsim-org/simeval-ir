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
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = 42,
        **kwargs,
    ) -> MetricResult:
        """Compute Kendall's Tau between two testers.

        Args:
            scores_1: System scores from tester 1 {system_id: score}.
            scores_2: System scores from tester 2 {system_id: score}.
            n_bootstrap: Number of bootstrap resamples for CI (0 to disable).
            confidence_level: Confidence level for bootstrap CI.
            random_state: Random seed for reproducibility.

        Returns:
            MetricResult with Tau value (-1 to 1) and optional bootstrap CI.
        """
        # Get common systems
        common = set(scores_1.keys()) & set(scores_2.keys())
        if len(common) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "Not enough common systems"},
            )

        systems = sorted(common)
        x = [scores_1[s] for s in systems]
        y = [scores_2[s] for s in systems]

        tau, p_value = stats.kendalltau(x, y)

        # Bootstrap confidence interval
        ci_lower, ci_upper = None, None
        if n_bootstrap > 0 and len(systems) >= 3:
            rng = np.random.default_rng(random_state)
            n = len(systems)
            boot_stats = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                bx = [x[i] for i in idx]
                by = [y[i] for i in idx]
                bt, _ = stats.kendalltau(bx, by)
                if not np.isnan(bt):
                    boot_stats.append(bt)
            if boot_stats:
                alpha = 1 - confidence_level
                ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        return MetricResult(
            name=self.name,
            value=float(tau),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            meta={
                "p_value": float(p_value),
                "n_systems": len(common),
                "n_bootstrap": n_bootstrap,
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
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = 42,
        **kwargs,
    ) -> MetricResult:
        """Compute Spearman's Rho between two testers.

        Args:
            scores_1: System scores from tester 1.
            scores_2: System scores from tester 2.
            n_bootstrap: Number of bootstrap resamples for CI (0 to disable).
            confidence_level: Confidence level for bootstrap CI.
            random_state: Random seed for reproducibility.

        Returns:
            MetricResult with Rho value (-1 to 1) and optional bootstrap CI.
        """
        common = set(scores_1.keys()) & set(scores_2.keys())
        if len(common) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "Not enough common systems"},
            )

        systems = sorted(common)
        x = [scores_1[s] for s in systems]
        y = [scores_2[s] for s in systems]

        rho, p_value = stats.spearmanr(x, y)

        # Bootstrap confidence interval
        ci_lower, ci_upper = None, None
        if n_bootstrap > 0 and len(systems) >= 3:
            rng = np.random.default_rng(random_state)
            n = len(systems)
            boot_stats = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                bx = [x[i] for i in idx]
                by = [y[i] for i in idx]
                br, _ = stats.spearmanr(bx, by)
                if not np.isnan(br):
                    boot_stats.append(br)
            if boot_stats:
                alpha = 1 - confidence_level
                ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        return MetricResult(
            name=self.name,
            value=float(rho),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            meta={
                "p_value": float(p_value),
                "n_systems": len(common),
                "n_bootstrap": n_bootstrap,
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
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
        random_state: int | None = 42,
        **kwargs,
    ) -> MetricResult:
        """Compute Pearson correlation between two testers.

        Args:
            scores_1: System scores from tester 1.
            scores_2: System scores from tester 2.
            n_bootstrap: Number of bootstrap resamples for CI (0 to disable).
            confidence_level: Confidence level for bootstrap CI.
            random_state: Random seed for reproducibility.

        Returns:
            MetricResult with correlation value (-1 to 1) and optional bootstrap CI.
        """
        common = set(scores_1.keys()) & set(scores_2.keys())
        if len(common) < 2:
            return MetricResult(
                name=self.name,
                value=0.0,
                meta={"warning": "Not enough common systems"},
            )

        systems = sorted(common)
        x = [scores_1[s] for s in systems]
        y = [scores_2[s] for s in systems]

        r, p_value = stats.pearsonr(x, y)

        # Bootstrap confidence interval
        ci_lower, ci_upper = None, None
        if n_bootstrap > 0 and len(systems) >= 3:
            rng = np.random.default_rng(random_state)
            n = len(systems)
            boot_stats = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                bx = [x[i] for i in idx]
                by = [y[i] for i in idx]
                br, _ = stats.pearsonr(bx, by)
                if not np.isnan(br):
                    boot_stats.append(br)
            if boot_stats:
                alpha = 1 - confidence_level
                ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
                ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

        return MetricResult(
            name=self.name,
            value=float(r),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            meta={
                "p_value": float(p_value),
                "n_systems": len(common),
                "n_bootstrap": n_bootstrap,
            },
        )


def _ranks_descending(scores: dict[str, float]) -> dict[str, int]:
    """Rank assignment with average-rank tie handling, 1 = best."""
    ordered = sorted(scores.items(), key=lambda kv: -kv[1])
    ranks: dict[str, int] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = avg
        i = j + 1
    return ranks


@metric("tau_ap")
class TauAP(BaseMetric):
    """Yilmaz-Aslam-Robertson rank-biased $\\tau_{AP}$ correlation.

    Top-of-list disagreements count more than tail disagreements, which
    matches the way IR practitioners actually use system rankings.
    Defined for two rankings of $N$ systems by Yilmaz et al. (SIGIR
    2008). Reported value is in $[-1, 1]$.
    """

    name = "tau_ap"
    objective: Literal["tester"] = "tester"
    granularity: Literal["system"] = "system"
    scenarios = {"T", "C"}
    description = "Top-weighted rank correlation (Yilmaz et al., 2008)"

    def compute(
        self,
        scores_1: dict[str, float],
        scores_2: dict[str, float],
        **kwargs,
    ) -> MetricResult:
        common = set(scores_1) & set(scores_2)
        if len(common) < 3:
            return MetricResult(name=self.name, value=0.0,
                                meta={"warning": "fewer than 3 systems"})
        r1 = _ranks_descending({s: scores_1[s] for s in common})
        r2 = _ranks_descending({s: scores_2[s] for s in common})
        # Order systems by reference (scores_1) ranking
        order = sorted(common, key=lambda s: r1[s])
        n = len(order)
        total = 0.0
        for i in range(1, n):  # i is 0-indexed position; AP-tau loops i=2..N
            sys_i = order[i]
            concord = 0
            for j in range(i):
                sys_j = order[j]
                # Concordant if scores_2 ranks sys_j above sys_i too
                if r2[sys_j] < r2[sys_i]:
                    concord += 1
            p_i = concord / i
            total += 2 * p_i - 1
        tau_ap = total / (n - 1)
        return MetricResult(name=self.name, value=float(tau_ap),
                            meta={"n_systems": n})


@metric("pairwise_concordance")
class PairwiseConcordance(BaseMetric):
    """Fraction of system pairs that agree in sign between two rankings.

    Equivalent to $(\\tau + 1) / 2$ in the no-ties case; reported
    separately because IR papers often quote it directly when
    comparing system orderings (e.g., as a sanity metric next to RATE).
    """

    name = "pairwise_concordance"
    objective: Literal["tester"] = "tester"
    granularity: Literal["system"] = "system"
    scenarios = {"T", "C"}
    description = "Fraction of system pairs that agree in sign across two testers"

    def compute(
        self,
        scores_1: dict[str, float],
        scores_2: dict[str, float],
        **kwargs,
    ) -> MetricResult:
        common = sorted(set(scores_1) & set(scores_2))
        if len(common) < 2:
            return MetricResult(name=self.name, value=0.0,
                                meta={"warning": "fewer than 2 systems"})
        agree = 0
        total = 0
        for i in range(len(common)):
            for j in range(i + 1, len(common)):
                a, b = common[i], common[j]
                d1 = scores_1[a] - scores_1[b]
                d2 = scores_2[a] - scores_2[b]
                if d1 == 0 or d2 == 0:
                    continue
                if (d1 > 0) == (d2 > 0):
                    agree += 1
                total += 1
        value = agree / total if total else 0.0
        return MetricResult(name=self.name, value=float(value),
                            meta={"n_pairs": total, "n_agree": agree})
