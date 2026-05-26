"""LanceDB vector store writer with JSON fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_vector_store(records: list[dict[str, Any]], uri: str = "artifacts/lancedb") -> dict[str, Any]:
    """Write records to LanceDB when installed, otherwise persist JSON fallback."""

    try:
        import lancedb
        import pyarrow as pa
    except Exception:
        fallback = Path("artifacts/lancedb_fallback/facility_capabilities.json")
        fallback.parent.mkdir(parents=True, exist_ok=True)
        fallback.write_text(json.dumps(records, indent=2), encoding="utf-8")
        return {"backend": "json-fallback", "path": str(fallback), "records": len(records)}

    db = lancedb.connect(uri)
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32())),
            pa.field("metadata", pa.string()),
        ]
    )
    normalized = [
        {"id": row["id"], "text": row["text"], "vector": row["vector"], "metadata": json.dumps(row.get("metadata", {}))}
        for row in records
    ]
    db.create_table("facility_capabilities", data=normalized, schema=schema, mode="overwrite")
    return {"backend": "lancedb", "path": uri, "records": len(records), "table": "facility_capabilities"}

