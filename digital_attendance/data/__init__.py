"""Data loading, processing, and collage generation package."""

from digital_attendance.data.collage import (
    CollageGridCell,
    CollageResult,
    compose_grid_collage,
    save_collage,
)
from digital_attendance.data.loader import (
    FaceSample,
    image_to_vector,
    load_dataset,
    preprocess_image,
    resolve_data_path,
    split_dataset,
    vector_to_image,
)

__all__ = [
    "FaceSample",
    "preprocess_image",
    "image_to_vector",
    "vector_to_image",
    "load_dataset",
    "split_dataset",
    "resolve_data_path",
    "CollageGridCell",
    "CollageResult",
    "compose_grid_collage",
    "save_collage",
]
