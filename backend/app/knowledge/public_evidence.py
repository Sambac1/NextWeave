"""One selected public source packet shared by forecasting and simulation.

Selection happens once, upstream. These projections never select a second,
smaller history or turn an excerpt into an observed event or actor authority.
"""
from hashlib import sha256
import json

from app.knowledge.table_evidence import parse_table_evidence

VERSION = "shared-public-evidence-1"


def labeled_tables(text):
    """Retain column hierarchy once per table, rather than repeating it per cell."""
    tables = []
    for table in parse_table_evidence(text):
        columns = list(dict.fromkeys(tuple(field["column_path"])
                                    for row in table["rows"] for field in row["fields"]))
        tables.append({"table_id": table["table_id"], "columns": [list(path) for path in columns],
                       "ambiguity": table["ambiguity"],
                       "rows": [{"row_id": row["row_id"],
                                 "values": [next((field["value"] for field in row["fields"]
                                                  if tuple(field["column_path"]) == path), None)
                                            for path in columns],
                                 "ambiguity": row["ambiguity"]} for row in table["rows"]]})
    return tables


def public_sources(sources, *, text_key="content"):
    rows = []
    for source in sources:
        text = str(source.get("content") or "")
        metadata = {key: source.get(key) for key in
                    ("url", "title", "revision_id", "revision_at", "retrieved_at") if source.get(key) is not None}
        identity = json.dumps(metadata | {"content": text}, sort_keys=True, ensure_ascii=False)
        row = metadata | {"evidence_id": "public_" + sha256(identity.encode()).hexdigest()[:24],
                          text_key: text, "snapshot_sha256": sha256(text.encode()).hexdigest(),
                          "public_packet_version": VERSION}
        if tables := labeled_tables(text):
            row["table_schema"] = tables
        rows.append(row)
    return rows


def packet_manifest(rows, *, text_key="content"):
    identities = [{"evidence_id": row["evidence_id"], "snapshot_sha256": row["snapshot_sha256"]}
                  for row in rows]
    return {"version": VERSION,
            "sha256": sha256(json.dumps(identities, sort_keys=True).encode()).hexdigest(),
            "source_count": len(rows), "source_chars": sum(len(row[text_key]) for row in rows)}
