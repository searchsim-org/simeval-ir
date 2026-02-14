# SimEval-IR

**SimEval-IR** is a unified framework for evaluating simulated search and conversational sessions. It provides standardized data models, behavioral realism metrics, system effectiveness metrics, and tester reliability estimation.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Dataset-Agnostic Core**: Works with any data matching the standard schema - no specific dataset dependencies
- **Unified Data Model**: `InteractionSession` handles both traditional search and conversational interactions
- **Comprehensive Metrics**:
  - Behavioral realism (JSD, Fréchet Distance, classifier-based)
  - System effectiveness (session nDCG, EGU)
  - Tester reliability (Kendall τ, RATE)
- **Integrations**: SimIIR and PyTerrier support
- **CLI & API**: Both programmatic and command-line interfaces

## Installation

```bash
pip install simeval-ir
```

**Optional dependencies:**

```bash
# ML features (embeddings with transformers)
pip install simeval-ir[ml]

# PyTerrier integration
pip install simeval-ir[pyterrier]

# Development
pip install simeval-ir[dev]
```

## Quick Start

### Creating Sessions Programmatically

```python
from simeval_ir import InteractionSession, Event, EventType, SessionType

# Create a search session
session = InteractionSession(
    session_id="my-session",
    dataset_id="custom",
    session_type=SessionType.SEARCH,
    events=[
        Event(
            event_id="e1",
            type=EventType.QUERY_ISSUED,
            query="python tutorial"
        ),
        Event(
            event_id="e2",
            type=EventType.CLICK,
            clicked_items=[Click(doc_id="doc1", rank=1)]
        )
    ]
)
```

### Loading Sessions from JSON/JSONL

```python
from simeval_ir.datasets import load_sessions_from_jsonl

sessions = list(load_sessions_from_jsonl("my_sessions.jsonl"))
```

### Computing Metrics

```python
from simeval_ir.metrics import get_metric

# Compare real vs simulated sessions
jsd_metric = get_metric("jsd_action_types")()
result = jsd_metric.compute(real=real_sessions, sim=simulated_sessions)

print(f"{result.name}: {result.value:.4f}")
```

### Running Evaluation Protocols

```python
from simeval_ir.eval import run_protocol

result = run_protocol(
    "realism-benchmark-v1",
    real=real_sessions,
    sim=simulated_sessions
)

for metric_result in result.metric_results:
    print(f"{metric_result.name}: {metric_result.value:.4f}")
```

### Command-Line Interface

```bash
# List available metrics
simeval list-metrics

# List evaluation protocols
simeval list-protocols

# Evaluate behavioral realism
simeval eval-behavior --real real.jsonl --sim sim.jsonl --metrics jsd_action_types,session_length_distribution

# Run a protocol
simeval run-protocol --protocol realism-benchmark-v1 --real real.jsonl --sim sim.jsonl
```

## Data Schema

SimEval-IR uses a standard JSON schema for sessions:

```json
{
  "session_id": "session_1",
  "dataset_id": "my_dataset",
  "session_type": "search",
  "events": [
    {
      "event_id": "e1",
      "type": "query_issued",
      "query": "example query",
      "timestamp": 0.0
    },
    {
      "event_id": "e2", 
      "type": "click",
      "clicked_items": [{"doc_id": "doc1", "rank": 1}],
      "timestamp": 1.5
    }
  ]
}
```

## Creating Custom Dataset Adapters

```python
from simeval_ir.datasets import DatasetAdapter, register_adapter

class MyDatasetAdapter(DatasetAdapter):
    name = "my-dataset"
    dataset_type = {"T"}  # Traditional search
    domain = "web"
    
    def load_sessions(self, data_path, split="all"):
        # Load and yield InteractionSession objects
        for row in load_my_data(data_path):
            yield InteractionSession(...)
    
    def validate_data_path(self, data_path):
        return (data_path / "data.csv").exists()

# Register the adapter
register_adapter("my-dataset", MyDatasetAdapter)
```

## SimIIR Integration

```python
from simeval_ir.integrations.simiir import from_simiir_logs

# Load SimIIR log files
sessions = from_simiir_logs("path/to/simiir/logs/")
```

## PyTerrier Integration

```python
from simeval_ir.integrations.pyterrier import (
    from_pyterrier_experiment,
    SimEvalEvaluator
)

# Convert PyTerrier results
sessions = from_pyterrier_experiment(topics, qrels, {"bm25": run_df})

# Use in evaluation
evaluator = SimEvalEvaluator(metrics=["session_ndcg"])
results = evaluator.evaluate(run_df, qrels, topics)
```

## Available Metrics

### Behavioral Realism

| Metric | Description |
|--------|-------------|
| `jsd_action_types` | JSD between action type distributions |
| `session_fd` | Fréchet Distance on session embeddings |
| `realism_classifier` | Classifier-based realism score |
| `session_length_distribution` | KS test on session lengths |
| `timing_distribution` | KS test on inter-event times |

### System Effectiveness

| Metric | Description |
|--------|-------------|
| `session_ndcg` | Session-level nDCG |
| `egu` | Expected Global Utility |

### Tester Reliability

| Metric | Description |
|--------|-------------|
| `kendall_tau` | Kendall's τ rank correlation |
| `spearman_rho` | Spearman's ρ rank correlation |
| `pearson_corr` | Pearson correlation |
| `rate` | RATE reliability estimation |


## License

MIT License - see [LICENSE](LICENSE) for details.
