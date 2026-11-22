"""Synthetic NF-v3 fixtures for data-stage tests.

The tests must run on CPU without the 2.3 GB dataset, so this module builds
tiny CSV files with the *real* 55-column NF-v3 schema. The column order below
matches the Kaggle mirrors exactly (verified 2026-11-01). A consistency test in
``src/tests/data/test_registry.py`` asserts that the YAML feature spec covers every
column here, so the fixture cannot silently drift from the real dataset.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

#: Exact column order of the four NF-v3 CSV mirrors (55 columns).
NF_V3_COLUMNS: tuple[str, ...] = (
    "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "L7_PROTO",
    "IN_BYTES",
    "IN_PKTS",
    "OUT_BYTES",
    "OUT_PKTS",
    "TCP_FLAGS",
    "CLIENT_TCP_FLAGS",
    "SERVER_TCP_FLAGS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "MIN_TTL",
    "MAX_TTL",
    "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES",
    "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES",
    "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT",
    "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES",
    "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES",
    "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "ICMP_TYPE",
    "ICMP_IPV4_TYPE",
    "DNS_QUERY_ID",
    "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER",
    "FTP_COMMAND_RET_CODE",
    "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX",
    "SRC_TO_DST_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN",
    "DST_TO_SRC_IAT_MAX",
    "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV",
    "Label",
    "Attack",
)

#: Columns that carry strings in the mirrors; everything else is numeric.
STRING_COLUMNS: frozenset[str] = frozenset(
    {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"}
)

#: Supported multiclass labels for the synthetic primary dataset.
CLASSES: tuple[str, ...] = (
    "Benign",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
)


def make_nf_row(**overrides: Any) -> dict[str, str]:
    """Return one synthetic NF-v3 row as strings, with optional overrides.

    Defaults are deliberately benign-looking values that parse as numbers. The
    caller overrides only the columns a test cares about, which keeps each test
    focused on one behaviour.
    """
    row: dict[str, str] = {column: "0" for column in NF_V3_COLUMNS}
    row.update(
        {
            "FLOW_START_MILLISECONDS": "1424242193000",
            "FLOW_END_MILLISECONDS": "1424242193010",
            "IPV4_SRC_ADDR": "10.0.0.1",
            "L4_SRC_PORT": "50000",
            "IPV4_DST_ADDR": "10.0.0.2",
            "L4_DST_PORT": "80",
            "PROTOCOL": "6",
            "L7_PROTO": "0",
            "IN_BYTES": "100",
            "IN_PKTS": "2",
            "OUT_BYTES": "200",
            "OUT_PKTS": "3",
            "MIN_TTL": "31",
            "MAX_TTL": "32",
            "FLOW_DURATION_MILLISECONDS": "10",
            "Label": "0",
            "Attack": "Benign",
        }
    )
    row.update({key: str(value) for key, value in overrides.items()})
    return row


def write_nf_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    columns: Sequence[str] = NF_V3_COLUMNS,
    extra_columns: Sequence[str] = (),
    encoding: str = "utf-8",
) -> Path:
    """Write a synthetic NF-v3 CSV at *path* and return it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [*columns, *extra_columns]
    with target.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            full = {column: "" for column in fieldnames}
            full.update({key: str(value) for key, value in dict(row).items()})
            writer.writerow(full)
    return target


def benign_then_attacks(n_rows: int = 8) -> list[dict[str, str]]:
    """Deterministic row mix: 5 Benign, 2 Exploits, 1 Worms at n_rows=8.

    Timestamps increase by one second per row, hosts alternate between two
    source and two destination addresses so host cardinality is predictable.
    """
    if n_rows != 8:
        raise ValueError("benign_then_attacks is defined for exactly 8 rows")
    rows: list[dict[str, str]] = []
    labels = ["Benign"] * 5 + ["Exploits"] * 2 + ["Worms"]
    for index, label in enumerate(labels):
        binary = "0" if label == "Benign" else "1"
        rows.append(
            make_nf_row(
                FLOW_START_MILLISECONDS=str(1424242193000 + index * 1000),
                FLOW_END_MILLISECONDS=str(1424242193000 + index * 1000 + 10),
                IPV4_SRC_ADDR=f"10.0.0.{1 + index % 2}",
                IPV4_DST_ADDR=f"10.0.0.{3 + index % 2}",
                Label=binary,
                Attack=label,
                IN_BYTES=str(100 + index),
            )
        )
    return rows
