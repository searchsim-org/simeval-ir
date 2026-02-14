"""Dataset adapters for specific data formats.

Adapters convert external dataset formats to the SimEval-IR canonical schema.
"""

from simeval_ir.datasets.adapters.trec_session import TRECSessionAdapter
from simeval_ir.datasets.adapters.synthetic import SyntheticSessionGenerator
from simeval_ir.datasets.adapters.aol import AOLAdapter
from simeval_ir.datasets.adapters.tripclick import TripClickAdapter
from simeval_ir.datasets.adapters.trec_cast import TRECCaSTAdapter
from simeval_ir.datasets.adapters.tiangong import TianGongSTAdapter
from simeval_ir.datasets.adapters.yandex import YandexAdapter
from simeval_ir.datasets.adapters.sogou import SogouAdapter
from simeval_ir.datasets.adapters.persona_chat import PersonaChatAdapter
from simeval_ir.datasets.adapters.lmsys_chat import LMSYSChatAdapter

__all__ = [
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
]
