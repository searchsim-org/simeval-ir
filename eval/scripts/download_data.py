#!/usr/bin/env python
"""Download sample datasets for SimEval-IR testing.

This script downloads publicly available sample data for testing the toolkit.
Some datasets require manual download due to licensing restrictions.

Usage:
    python download_data.py --output data/ --datasets all
    python download_data.py --output data/ --datasets persona-chat,trec-cast
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

# Dataset download information
DATASETS = {
    "persona-chat": {
        "description": "Persona-Chat dialogue dataset (ParlAI format)",
        "url": "https://raw.githubusercontent.com/facebookresearch/ParlAI/main/parlai/tasks/personachat/personachat_self_original_train.json",
        "type": "json",
        "auto": True,
    },
    "trec-cast-2019": {
        "description": "TREC CAsT 2019 topics",
        "url": "https://raw.githubusercontent.com/daltonj/treccastweb/master/2019/data/evaluation/evaluation_topics_v1.0.json",
        "type": "json",
        "auto": True,
    },
    "trec-cast-2020": {
        "description": "TREC CAsT 2020 topics",
        "url": "https://raw.githubusercontent.com/daltonj/treccastweb/master/2020/2020_automatic_evaluation_topics_v1.0.json",
        "type": "json",
        "auto": True,
    },
    "aol-sample": {
        "description": "AOL Query Log (sample - full requires manual download)",
        "url": None,  # Requires manual download
        "instructions": "Download from: https://jeffhuang.com/search_query_logs/\nExtract to data/aol/",
        "auto": False,
    },
    "tripclick": {
        "description": "TripClick health search dataset",
        "url": None,
        "instructions": "Download from: https://tripdatabase.github.io/tripclick/\nRequires registration. Extract to data/tripclick/",
        "auto": False,
    },
    "tiangong-st": {
        "description": "TianGong-ST Chinese session search",
        "url": None,
        "instructions": "Download from: http://www.thuir.cn/tiangong-st/\nRequires registration. Extract to data/tiangong-st/",
        "auto": False,
    },
    "yandex": {
        "description": "Yandex Relevance Prediction Challenge",
        "url": None,
        "instructions": "Download from: https://www.kaggle.com/c/yandex-personalized-web-search-challenge/data\nRequires Kaggle account. Extract to data/yandex/",
        "auto": False,
    },
    "trec-session": {
        "description": "TREC Session Track 2014",
        "url": None,
        "instructions": "Download from: https://trec.nist.gov/data/session.html\nExtract to data/trec-session/",
        "auto": False,
    },
}


def download_file(url: str, output_path: Path) -> bool:
    """Download a file from URL."""
    print(f"  Downloading from {url}...")
    try:
        urllib.request.urlretrieve(url, output_path)
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_persona_chat(output_dir: Path) -> bool:
    """Download Persona-Chat dataset."""
    print("\nDownloading Persona-Chat...")

    pc_dir = output_dir / "persona-chat"
    pc_dir.mkdir(parents=True, exist_ok=True)

    # Download train file
    url = DATASETS["persona-chat"]["url"]
    output_file = pc_dir / "train.json"

    if download_file(url, output_file):
        print(f"  Saved to {output_file}")

        # Also try to get validation set
        val_url = url.replace("train", "valid")
        val_file = pc_dir / "valid.json"
        download_file(val_url, val_file)

        return True
    return False


def download_trec_cast(output_dir: Path) -> bool:
    """Download TREC CAsT topics."""
    print("\nDownloading TREC CAsT topics...")

    cast_dir = output_dir / "trec-cast"
    cast_dir.mkdir(parents=True, exist_ok=True)

    success = True

    # Download 2019 topics
    url_2019 = DATASETS["trec-cast-2019"]["url"]
    if download_file(url_2019, cast_dir / "topics_2019.json"):
        print(f"  Saved 2019 topics")
    else:
        success = False

    # Download 2020 topics
    url_2020 = DATASETS["trec-cast-2020"]["url"]
    if download_file(url_2020, cast_dir / "topics_2020.json"):
        print(f"  Saved 2020 topics")
    else:
        success = False

    return success


def try_download_lmsys_sample(output_dir: Path) -> bool:
    """Try to download LMSYS-Chat sample via Hugging Face."""
    print("\nDownloading LMSYS-Chat sample...")

    lmsys_dir = output_dir / "lmsys-chat"
    lmsys_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try using huggingface_hub if available
        from huggingface_hub import hf_hub_download

        print("  Using huggingface_hub to download sample...")
        # Download a small sample file
        file_path = hf_hub_download(
            repo_id="lmsys/lmsys-chat-1m",
            filename="data/train-00000-of-00006-4feeb3f83346a0e9.parquet",
            repo_type="dataset",
            local_dir=lmsys_dir,
        )
        print(f"  Downloaded to {file_path}")
        return True

    except ImportError:
        print("  huggingface_hub not installed. Creating sample data instead.")
        return create_lmsys_sample(lmsys_dir)
    except Exception as e:
        print(f"  Could not download: {e}")
        print("  Creating sample data instead.")
        return create_lmsys_sample(lmsys_dir)


def create_lmsys_sample(output_dir: Path) -> bool:
    """Create sample LMSYS-style data for testing."""
    sample_conversations = [
        {
            "conversation_id": "sample_1",
            "model": "gpt-4",
            "conversation": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of artificial intelligence..."},
                {"role": "user", "content": "Can you give me an example?"},
                {"role": "assistant", "content": "Sure! A common example is email spam filtering..."},
            ],
            "language": "en",
        },
        {
            "conversation_id": "sample_2",
            "model": "claude-2",
            "conversation": [
                {"role": "user", "content": "Help me write a Python function"},
                {"role": "assistant", "content": "I'd be happy to help! What should the function do?"},
                {"role": "user", "content": "It should calculate factorial"},
                {"role": "assistant", "content": "Here's a simple factorial function:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"},
            ],
            "language": "en",
        },
    ]

    # Create more samples
    for i in range(3, 51):
        sample_conversations.append({
            "conversation_id": f"sample_{i}",
            "model": ["gpt-4", "claude-2", "llama-2"][i % 3],
            "conversation": [
                {"role": "user", "content": f"Question {i} about topic {i % 10}"},
                {"role": "assistant", "content": f"Response to question {i}"},
            ],
            "language": "en",
        })

    output_file = output_dir / "sample.jsonl"
    with open(output_file, "w") as f:
        for conv in sample_conversations:
            f.write(json.dumps(conv) + "\n")

    print(f"  Created sample with {len(sample_conversations)} conversations at {output_file}")
    return True


def create_aol_sample(output_dir: Path) -> bool:
    """Create sample AOL-style data for testing."""
    print("\nCreating AOL sample data...")

    aol_dir = output_dir / "aol"
    aol_dir.mkdir(parents=True, exist_ok=True)

    # Create sample in AOL format
    # Format: AnonID  Query  QueryTime  ItemRank  ClickURL
    sample_data = [
        "AnonID\tQuery\tQueryTime\tItemRank\tClickURL",
    ]

    import random
    random.seed(42)

    queries = [
        "python tutorial", "machine learning", "best laptop 2024",
        "weather forecast", "recipe ideas", "travel deals",
        "how to code", "data science jobs", "fitness tips",
    ]

    for user_id in range(1, 21):
        base_time = f"2006-03-{10 + user_id % 20:02d}"
        for q_idx in range(random.randint(1, 5)):
            query = random.choice(queries)
            hour = random.randint(8, 22)
            minute = random.randint(0, 59)
            time = f"{base_time} {hour:02d}:{minute:02d}:00"

            # Some queries have clicks, some don't
            if random.random() < 0.7:
                rank = random.randint(1, 10)
                url = f"http://example.com/page{random.randint(1, 1000)}"
                sample_data.append(f"{user_id}\t{query}\t{time}\t{rank}\t{url}")
            else:
                sample_data.append(f"{user_id}\t{query}\t{time}\t\t")

    output_file = aol_dir / "sample.txt"
    with open(output_file, "w") as f:
        f.write("\n".join(sample_data))

    print(f"  Created sample at {output_file}")
    return True


def create_tripclick_sample(output_dir: Path) -> bool:
    """Create sample TripClick-style data for testing."""
    print("\nCreating TripClick sample data...")

    tc_dir = output_dir / "tripclick"
    tc_dir.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(42)

    health_queries = [
        "diabetes symptoms", "heart disease prevention", "covid vaccine",
        "back pain treatment", "migraine relief", "sleep apnea",
        "anxiety medication", "blood pressure", "cholesterol diet",
    ]

    sessions = []
    for i in range(50):
        query = random.choice(health_queries)
        session = {
            "session_id": f"tc_{i}",
            "query_id": f"q_{i}",
            "query": query,
            "impressions": [f"doc_{random.randint(1000, 9999)}" for _ in range(10)],
            "clicks": [
                {"doc_id": f"doc_{random.randint(1000, 9999)}", "rank": random.randint(1, 5)}
                for _ in range(random.randint(0, 3))
            ],
        }
        sessions.append(session)

    output_file = tc_dir / "sample.jsonl"
    with open(output_file, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")

    print(f"  Created sample at {output_file}")
    return True


def create_tiangong_sample(output_dir: Path) -> bool:
    """Create sample TianGong-ST-style data for testing."""
    print("\nCreating TianGong-ST sample data...")

    tg_dir = output_dir / "tiangong-st"
    tg_dir.mkdir(parents=True, exist_ok=True)

    import random
    random.seed(42)

    # Chinese-style queries (using pinyin for simplicity)
    queries = [
        "shouji pingjia", "diannao xuangou", "lvyou gonglue",
        "meishi tuijian", "jiankang zhishi", "xuexai fangfa",
    ]

    sessions = []
    for i in range(50):
        n_queries = random.randint(1, 4)
        session_queries = []

        for q_idx in range(n_queries):
            query_data = {
                "query": random.choice(queries),
                "query_id": f"tg_{i}_{q_idx}",
                "timestamp": q_idx * 60.0,
                "serp": [
                    {"doc_id": f"doc_{random.randint(1, 9999)}", "score": 1.0 / (r + 1)}
                    for r in range(10)
                ],
                "clicks": [
                    {"doc_id": f"doc_{random.randint(1, 9999)}", "rank": random.randint(1, 5), "dwell": random.uniform(10, 60)}
                    for _ in range(random.randint(0, 3))
                ],
            }
            session_queries.append(query_data)

        sessions.append({
            "session_id": f"tg_session_{i}",
            "user_id": f"user_{random.randint(1, 20)}",
            "queries": session_queries,
            "satisfaction": random.randint(1, 5),
        })

    output_file = tg_dir / "sample.jsonl"
    with open(output_file, "w") as f:
        for s in sessions:
            f.write(json.dumps(s) + "\n")

    print(f"  Created sample at {output_file}")
    return True


def print_manual_instructions():
    """Print instructions for manual downloads."""
    print("\n" + "=" * 60)
    print("MANUAL DOWNLOAD REQUIRED")
    print("=" * 60)
    print("\nSome datasets require manual download due to licensing:")

    for name, info in DATASETS.items():
        if not info.get("auto", False) and "instructions" in info:
            print(f"\n{name}:")
            print(f"  {info['description']}")
            for line in info["instructions"].split("\n"):
                print(f"  {line}")


def main():
    parser = argparse.ArgumentParser(
        description="Download sample datasets for SimEval-IR"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data"),
        help="Output directory for datasets",
    )
    parser.add_argument(
        "--datasets", "-d",
        type=str,
        default="all",
        help="Comma-separated list of datasets to download, or 'all'",
    )
    parser.add_argument(
        "--create-samples",
        action="store_true",
        help="Create sample data for datasets that require manual download",
    )

    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("SimEval-IR Dataset Downloader")
    print("=" * 60)
    print(f"Output directory: {args.output}")

    datasets_to_download = args.datasets.split(",") if args.datasets != "all" else list(DATASETS.keys())

    results = {}

    # Download auto-downloadable datasets
    if "persona-chat" in datasets_to_download:
        results["persona-chat"] = download_persona_chat(args.output)

    if "trec-cast-2019" in datasets_to_download or "trec-cast-2020" in datasets_to_download or "trec-cast" in datasets_to_download:
        results["trec-cast"] = download_trec_cast(args.output)

    if "lmsys-chat" in datasets_to_download:
        results["lmsys-chat"] = try_download_lmsys_sample(args.output)

    # Create samples for manual-download datasets if requested
    if args.create_samples:
        if "aol" in datasets_to_download or "aol-sample" in datasets_to_download:
            results["aol"] = create_aol_sample(args.output)

        if "tripclick" in datasets_to_download:
            results["tripclick"] = create_tripclick_sample(args.output)

        if "tiangong-st" in datasets_to_download:
            results["tiangong-st"] = create_tiangong_sample(args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)

    for name, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {name}: {status}")

    # Print manual instructions
    print_manual_instructions()

    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Run experiments with downloaded data:")
    print("   python eval/scripts/run_b1.py --real-path data/persona-chat --real-adapter persona-chat ...")
    print("\n2. Generate simulated sessions:")
    print("   python eval/scripts/generate_simulated.py --output data/simulated")


if __name__ == "__main__":
    main()
