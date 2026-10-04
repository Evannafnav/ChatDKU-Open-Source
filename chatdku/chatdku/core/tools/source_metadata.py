"""Keep retrieved source metadata consistent across retrieval backends."""

from typing import Any, Mapping


def normalize_source_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Expose a common file_name field without discarding backend-specific keys.

    Vector retrieval supplies ``file_name`` and keyword retrieval supplies
    ``filename``. The returned copy leaves the source node unchanged.
    """
    result = dict(metadata)
    file_name = result.get("file_name") or result.get("filename")
    if file_name:
        result["file_name"] = file_name
    return result
