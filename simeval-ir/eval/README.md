# SimEval-IR Experiments

This directory contains scripts and configurations to reproduce the experiments from the paper.

## Directory Structure

```
eval/
├── README.md           # This file
├── run_experiments.py  # Main experiment runner
├── configs/            # YAML configuration files
│   ├── b1_tripclick.yaml
│   ├── b1_trec_session.yaml
│   ├── b2_tester_reliability.yaml
│   └── b3_realism_reliability.yaml
└── scripts/
    ├── run_b1.py       # B1: Behavioral Realism
    ├── run_b2.py       # B2: Tester Reliability
    ├── run_b3.py       # B3: Realism-Reliability Analysis
    └── generate_simulated.py  # Generate simulated sessions
```

## Prerequisites

1. **Install SimEval-IR**:
   ```bash
   pip install -e .
   ```

2. **Download datasets** (obtain from original sources):
   - TripClick: https://tripdatabase.github.io/tripclick/
   - TREC Session: https://trec.nist.gov/data/session.html
   - TREC CAsT: https://www.treccast.ai/

3. **Generate simulated sessions** (or use your own):
   ```bash
   python eval/scripts/generate_simulated.py --dataset tripclick --output data/simulated/
   ```

## Quick Start

### Run all experiments:
```bash
python eval/run_experiments.py --config eval/configs/b1_tripclick.yaml
```

### Run individual benchmarks:

**B1: Behavioral Realism**
```bash
python eval/scripts/run_b1.py \
    --real-data data/tripclick/ \
    --sim-data data/simulated/simiir_pbm.jsonl \
    --output results/b1/
```

**B2: Tester Reliability**
```bash
python eval/scripts/run_b2.py \
    --runs data/system_runs/ \
    --qrels data/qrels.txt \
    --output results/b2/
```

**B3: Realism-Reliability Analysis**
```bash
python eval/scripts/run_b3.py \
    --b1-results results/b1/ \
    --b2-results results/b2/ \
    --output results/b3/
```

## Configuration Files

Experiments can be configured via YAML files. Example:

```yaml
name: "B1-TripClick-v1"
protocol: "realism-benchmark-v1"

real_data:
  path: "./data/tripclick/"
  adapter: "tripclick"

simulators:
  - name: "simiir-pbm"
    path: "./data/simulated/simiir_pbm.jsonl"
  - name: "simiir-dbn"
    path: "./data/simulated/simiir_dbn.jsonl"
  - name: "llm-sim"
    path: "./data/simulated/llm_sim.jsonl"

output_dir: "./results/"
seed: 42
```

## Expected Output

Results are saved as JSON files with the following structure:

```json
{
  "config": { ... },
  "simulators": {
    "simiir-pbm": {
      "b1_metrics": {
        "jsd_action_types": 0.05,
        "session_fd": 18.2,
        "wasserstein_session_length": 0.12,
        "mmd": 0.03,
        "realism_classifier": 0.21
      },
      "leakage_audit": {
        "main_auc": 0.79,
        "metadata_only_auc": 0.52,
        "permutation_auc": 0.50,
        "leakage_detected": false
      }
    }
  }
}
```

## Reproducing Paper Results

To reproduce Table 2 (B1 results) from the paper:
```bash
python eval/run_experiments.py --experiment b1 --all-datasets
```

To reproduce Table 3 (B2 results):
```bash
python eval/run_experiments.py --experiment b2 --all-datasets
```

To reproduce Table 4 (B3 correlations):
```bash
python eval/run_experiments.py --experiment b3
```
