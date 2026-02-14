"""Base adapter class for dataset implementations.

This module provides the abstract base class that all dataset adapters
must implement. The core package works without any specific dataset -
adapters are optional and external.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Literal

from simeval_ir.core.session import InteractionSession
from simeval_ir.core.types import DatasetMetadata


class DatasetAdapter(ABC):
    """Abstract base class for dataset adapters.

    Implement this class to create a custom adapter for a new dataset.
    The adapter handles loading data from disk and converting it to
    the standard InteractionSession format.

    Example:
        >>> class MyDatasetAdapter(DatasetAdapter):
        ...     name = "my-dataset"
        ...     dataset_type = {"T"}  # Traditional search
        ...     domain = "web"
        ...
        ...     def load_sessions(self, data_path, split="all"):
        ...         # Load and yield InteractionSession objects
        ...         pass
        ...
        ...     def validate_data_path(self, data_path):
        ...         return data_path.exists()
    """

    # Class attributes to be set by subclasses
    name: str
    dataset_type: set[Literal["T", "C", "A"]]
    domain: str
    description: str = ""
    url: str | None = None
    languages: list[str] = ["en"]

    @abstractmethod
    def load_sessions(
        self,
        data_path: Path,
        split: str = "all",
    ) -> Iterator[InteractionSession]:
        """Load sessions from the dataset.

        Args:
            data_path: Path to the dataset directory or file.
            split: Which split to load ("train", "dev", "test", "all").

        Yields:
            InteractionSession objects.
        """
        ...

    @abstractmethod
    def validate_data_path(self, data_path: Path) -> bool:
        """Validate that the data path contains the expected files.

        Args:
            data_path: Path to validate.

        Returns:
            True if the path is valid for this adapter.
        """
        ...

    def get_metadata(self) -> DatasetMetadata:
        """Get metadata about this dataset.

        Returns:
            DatasetMetadata with information about the dataset.
        """
        return DatasetMetadata(
            name=self.name,
            dataset_type=self.dataset_type,
            domain=self.domain,
            description=self.description,
            languages=self.languages,
            url=self.url,
        )

    def count_sessions(self, data_path: Path, split: str = "all") -> int:
        """Count the number of sessions in the dataset.

        Default implementation iterates through all sessions.
        Override for more efficient counting.

        Args:
            data_path: Path to the dataset.
            split: Which split to count.

        Returns:
            Number of sessions.
        """
        return sum(1 for _ in self.load_sessions(data_path, split))


# Global adapter registry
ADAPTER_REGISTRY: dict[str, type[DatasetAdapter]] = {}


def register_adapter(name: str, adapter_cls: type[DatasetAdapter]) -> None:
    """Register a dataset adapter.

    Args:
        name: Name to register the adapter under.
        adapter_cls: The adapter class.
    """
    ADAPTER_REGISTRY[name] = adapter_cls


def get_adapter(name: str) -> type[DatasetAdapter]:
    """Get a registered adapter by name.

    Args:
        name: Name of the adapter.

    Returns:
        The adapter class.

    Raises:
        KeyError: If the adapter is not registered.
    """
    if name not in ADAPTER_REGISTRY:
        available = ", ".join(ADAPTER_REGISTRY.keys()) or "none"
        raise KeyError(
            f"Adapter '{name}' not found. Available adapters: {available}. "
            f"Adapters must be registered with register_adapter()."
        )
    return ADAPTER_REGISTRY[name]


def list_adapters() -> list[str]:
    """List all registered adapter names.

    Returns:
        List of adapter names.
    """
    return list(ADAPTER_REGISTRY.keys())


# Decorator for convenient registration
def adapter(name: str):
    """Decorator to register a dataset adapter.

    Example:
        >>> @adapter("my-dataset")
        ... class MyAdapter(DatasetAdapter):
        ...     pass
    """

    def decorator(cls: type[DatasetAdapter]) -> type[DatasetAdapter]:
        register_adapter(name, cls)
        return cls

    return decorator
