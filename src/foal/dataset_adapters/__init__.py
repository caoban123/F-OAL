"""Dataset adapters package for cross-dataset continual learning benchmarks."""
from src.foal.dataset_adapters.base import (
    BaseDatasetAdapter,
    DatasetSplitMetadata,
)
from src.foal.dataset_adapters.dtd import DTDAdapter, DTD_ALL_CLASSES
from src.foal.dataset_adapters.fgvc_aircraft import FGVCAircraftAdapter

__all__ = [
    "BaseDatasetAdapter",
    "DatasetSplitMetadata",
    "DTDAdapter",
    "DTD_ALL_CLASSES",
    "FGVCAircraftAdapter",
]
