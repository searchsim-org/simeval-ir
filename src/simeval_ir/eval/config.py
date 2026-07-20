"""Benchmark configuration system.

Enables reproducible benchmark runs via YAML/JSON configuration files.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from simeval_ir.core.session import InteractionSession
from simeval_ir.datasets import get_adapter, load_sessions_from_json, load_sessions_from_jsonl
from simeval_ir.eval.protocols import run_protocol, PROTOCOLS


@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    Attributes:
        name: Dataset identifier.
        path: Path to the dataset files.
        adapter: Adapter name (e.g., "trec-session", "synthetic").
        split: Data split to load ("train", "dev", "test", "all").
        max_sessions: Maximum sessions to load (None for all).
    """

    name: str
    path: str | Path
    adapter: str | None = None
    split: str = "all"
    max_sessions: int | None = None

    def load(self) -> list[InteractionSession]:
        """Load sessions from this dataset configuration.

        Returns:
            List of InteractionSession objects.
        """
        path = Path(self.path)

        # If adapter specified, use it
        if self.adapter:
            adapter_cls = get_adapter(self.adapter)
            adapter_instance = adapter_cls()
            sessions = list(adapter_instance.load_sessions(path, self.split))
        # Otherwise, infer from file extension
        elif path.suffix == ".json":
            sessions = load_sessions_from_json(path)
        elif path.suffix == ".jsonl":
            sessions = list(load_sessions_from_jsonl(path))
        else:
            raise ValueError(
                f"Cannot infer format for {path}. Specify an adapter."
            )

        # Apply max_sessions limit if specified
        if self.max_sessions is not None:
            sessions = sessions[:self.max_sessions]

        return sessions


@dataclass
class MetricConfig:
    """Configuration for metric computation.

    Attributes:
        name: Metric name.
        params: Additional parameters for the metric.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run.

    Attributes:
        name: Benchmark name.
        description: Description of the benchmark.
        protocol: Named protocol to run (e.g., "realism-benchmark-v1").
        real_data: Configuration for real sessions (for behavior metrics).
        sim_data: Configuration for simulated sessions (for behavior metrics).
        metrics: List of metric configurations (alternative to protocol).
        output_dir: Directory for output files.
        seed: Random seed for reproducibility.
        bootstrap_samples: Number of bootstrap samples for confidence intervals.
        n_folds: Number of CV folds for classifier metrics.
    """

    name: str
    description: str = ""
    protocol: str | None = None
    real_data: DatasetConfig | None = None
    sim_data: DatasetConfig | None = None
    metrics: list[MetricConfig] = field(default_factory=list)
    output_dir: str | Path = "./results"
    seed: int = 42
    bootstrap_samples: int = 1000
    n_folds: int = 5

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkConfig":
        """Create config from dictionary.

        Args:
            data: Configuration dictionary.

        Returns:
            BenchmarkConfig instance.
        """
        real_data = None
        if "real_data" in data:
            real_data = DatasetConfig(**data["real_data"])

        sim_data = None
        if "sim_data" in data:
            sim_data = DatasetConfig(**data["sim_data"])

        metrics = []
        if "metrics" in data:
            for m in data["metrics"]:
                if isinstance(m, str):
                    metrics.append(MetricConfig(name=m))
                else:
                    metrics.append(MetricConfig(**m))

        return cls(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            protocol=data.get("protocol"),
            real_data=real_data,
            sim_data=sim_data,
            metrics=metrics,
            output_dir=data.get("output_dir", "./results"),
            seed=data.get("seed", 42),
            bootstrap_samples=data.get("bootstrap_samples", 1000),
            n_folds=data.get("n_folds", 5),
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BenchmarkConfig":
        """Load config from YAML file.

        Args:
            path: Path to YAML file.

        Returns:
            BenchmarkConfig instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: Path | str) -> "BenchmarkConfig":
        """Load config from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            BenchmarkConfig instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def load(cls, path: Path | str) -> "BenchmarkConfig":
        """Load config from file (auto-detect format).

        Args:
            path: Path to config file (.yaml, .yml, or .json).

        Returns:
            BenchmarkConfig instance.
        """
        path = Path(path)
        if path.suffix in {".yaml", ".yml"}:
            return cls.from_yaml(path)
        elif path.suffix == ".json":
            return cls.from_json(path)
        else:
            raise ValueError(
                f"Unknown config format: {path.suffix}. Use .yaml, .yml, or .json"
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary."""
        result = {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "bootstrap_samples": self.bootstrap_samples,
            "n_folds": self.n_folds,
            "output_dir": str(self.output_dir),
        }

        if self.protocol:
            result["protocol"] = self.protocol

        if self.real_data:
            result["real_data"] = {
                "name": self.real_data.name,
                "path": str(self.real_data.path),
                "adapter": self.real_data.adapter,
                "split": self.real_data.split,
                "max_sessions": self.real_data.max_sessions,
            }

        if self.sim_data:
            result["sim_data"] = {
                "name": self.sim_data.name,
                "path": str(self.sim_data.path),
                "adapter": self.sim_data.adapter,
                "split": self.sim_data.split,
                "max_sessions": self.sim_data.max_sessions,
            }

        if self.metrics:
            result["metrics"] = [
                {"name": m.name, "params": m.params} for m in self.metrics
            ]

        return result

    def save(self, path: Path | str) -> None:
        """Save config to file.

        Args:
            path: Output path (.yaml, .yml, or .json).
        """
        path = Path(path)
        data = self.to_dict()

        path.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix in {".yaml", ".yml"}:
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, default_flow_style=False)
        else:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)


@dataclass
class BenchmarkResult:
    """Result of a benchmark run.

    Attributes:
        config: The configuration used.
        protocol_result: Result from protocol run (if using protocol).
        metric_results: Individual metric results.
        meta: Additional metadata.
    """

    config: BenchmarkConfig
    protocol_result: Any = None
    metric_results: list[Any] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "config": self.config.to_dict(),
            "metric_results": [
                {
                    "name": r.name,
                    "value": r.value,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                    "meta": r.meta,
                }
                for r in self.metric_results
            ],
            "meta": self.meta,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def run_benchmark(config: BenchmarkConfig) -> BenchmarkResult:
    """Run a benchmark from configuration.

    Args:
        config: Benchmark configuration.

    Returns:
        BenchmarkResult with all metric results.
    """
    import numpy as np
    np.random.seed(config.seed)

    # Load data if specified
    real_sessions = None
    sim_sessions = None

    if config.real_data:
        real_sessions = config.real_data.load()

    if config.sim_data:
        sim_sessions = config.sim_data.load()

    # Run protocol if specified
    if config.protocol:
        protocol_result = run_protocol(
            protocol=config.protocol,
            real=real_sessions,
            sim=sim_sessions,
            n_folds=config.n_folds,
        )
        return BenchmarkResult(
            config=config,
            protocol_result=protocol_result,
            metric_results=protocol_result.metric_results,
            meta={
                "protocol_name": protocol_result.protocol_name,
                "n_real_sessions": len(real_sessions) if real_sessions else 0,
                "n_sim_sessions": len(sim_sessions) if sim_sessions else 0,
            },
        )

    # Otherwise run individual metrics
    from simeval_ir.metrics import get_metric

    results = []
    for metric_config in config.metrics:
        metric_cls = get_metric(metric_config.name)
        metric = metric_cls()

        # Build kwargs from metric config params
        kwargs = {
            **metric_config.params,
            "n_folds": config.n_folds,
        }

        if metric.objective == "behavior":
            result = metric.compute(
                real=real_sessions,
                sim=sim_sessions,
                **kwargs,
            )
        else:
            result = metric.compute(
                sessions=real_sessions or sim_sessions,
                **kwargs,
            )

        results.append(result)

    return BenchmarkResult(
        config=config,
        metric_results=results,
        meta={
            "n_metrics": len(results),
            "n_real_sessions": len(real_sessions) if real_sessions else 0,
            "n_sim_sessions": len(sim_sessions) if sim_sessions else 0,
        },
    )
