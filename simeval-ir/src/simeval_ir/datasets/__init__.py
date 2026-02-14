"""Dataset adapters for SimEval-IR."""

from simeval_ir.datasets.base import (
    ADAPTER_REGISTRY,
    DatasetAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
    adapter,
)
from simeval_ir.datasets.io import (
    load_sessions_from_json,
    load_sessions_from_jsonl,
    save_sessions_to_json,
    save_sessions_to_jsonl,
)
from simeval_ir.datasets.adapters import (
    # Session search adapters
    TRECSessionAdapter,
    AOLAdapter,
    TripClickAdapter,
    TianGongSTAdapter,
    YandexAdapter,
    SogouAdapter,
    # Conversational adapters
    TRECCaSTAdapter,
    PersonaChatAdapter,
    LMSYSChatAdapter,
    # Utilities
    SyntheticSessionGenerator,
)
from simeval_ir.datasets.adapters.trec_session import LossAccountingManifest

__all__ = [
    # Base classes and utilities
    "DatasetAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    "adapter",
    "ADAPTER_REGISTRY",
    # I/O functions
    "load_sessions_from_json",
    "load_sessions_from_jsonl",
    "save_sessions_to_json",
    "save_sessions_to_jsonl",
    # Session search adapters
    "TRECSessionAdapter",
    "AOLAdapter",
    "TripClickAdapter",
    "TianGongSTAdapter",
    "YandexAdapter",
    "SogouAdapter",
    # Conversational adapters
    "TRECCaSTAdapter",
    "PersonaChatAdapter",
    "LMSYSChatAdapter",
    # Utilities
    "SyntheticSessionGenerator",
    "LossAccountingManifest",
]
