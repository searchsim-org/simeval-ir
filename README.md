# SimEval-IR

A unified framework for evaluating simulated search and conversational sessions.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

SimEval-IR provides standardized tools for evaluating user simulators in Information Retrieval research:

- **Behavioral Realism Metrics**: JSD, Fréchet Distance, classifier-based scoring
- **System Effectiveness Metrics**: Session nDCG, Expected Global Utility (EGU)
- **Tester Reliability Metrics**: Kendall τ, Spearman ρ, RATE

## Repository Structure

```
simeval-ir/
├── simeval-ir/          # Python package
│   ├── src/             # Source code
│   ├── eval/            # Evaluation scripts and configs
│   ├── data/            # Sample datasets
│   └── tests/           # Unit tests
└── website/             # Documentation website (Next.js)
```

## Quick Start

```bash
# Install the package
pip install simeval-ir

# Or install from source
cd simeval-ir
pip install -e .
```

```python
from simeval_ir import InteractionSession
from simeval_ir.metrics import get_metric

# Compare real vs simulated sessions
jsd = get_metric("jsd_action_types")()
result = jsd.compute(real=real_sessions, sim=simulated_sessions)
print(f"JSD: {result.value:.4f}")
```

## Documentation

See the [simeval-ir package README](simeval-ir/README.md) for detailed documentation including:

- Full installation options
- Data schema specification
- Available metrics reference
- Creating custom dataset adapters
- SimIIR and PyTerrier integrations
- CLI usage

## License

MIT License - see [LICENSE](simeval-ir/LICENSE) for details.
