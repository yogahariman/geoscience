from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

import segyio


def _to_text_lines(existing_header: bytes | str | None) -> list[str]:
    if existing_header is None:
        return []
    text = existing_header.decode("ascii", errors="ignore") if isinstance(existing_header, bytes) else str(existing_header)
    chunks = [text[i : i + 80] for i in range(0, len(text), 80)]
    return [c.rstrip() for c in chunks if c.strip()]


def _normalize_line(line: str) -> str:
    line = str(line).replace("\n", " ").strip()
    return line[:76]


def append_process_text_header(
    segy_path: str,
    process_name: str,
    details: Iterable[str] | None = None,
) -> None:
    """
    Append process metadata into SEG-Y textual header (block 0).

    Writes lines in standard C## format and keeps at most 40 lines.
    """

    details = list(details or [])
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    new_lines = [f"{process_name} @ {stamp}"] + [d for d in details if str(d).strip()]

    with segyio.open(segy_path, "r+", ignore_geometry=True) as f:
        existing = _to_text_lines(f.text[0] if len(f.text) > 0 else None)
        merged = existing + [f"PROC: {_normalize_line(line)}" for line in new_lines]
        merged = merged[-40:]

        header_map = {i + 1: f"C{i+1:02d} {line}"[:80] for i, line in enumerate(merged)}
        f.text[0] = segyio.tools.create_text_header(header_map)


def read_text_header(
    segy_path: str,
    block: int = 0,
    drop_empty: bool = True,
) -> list[str]:
    """
    Read SEG-Y textual header block and return lines (80-char cards).

    Parameters
    ----------
    segy_path : str
        SEG-Y path.
    block : int, default=0
        Text header block index (0-based).
    drop_empty : bool, default=True
        If True, remove blank lines from output.
    """
    if block < 0:
        raise ValueError("block must be >= 0.")

    with segyio.open(segy_path, "r", ignore_geometry=True) as f:
        if block >= len(f.text):
            raise ValueError(f"Text header block {block} not found.")
        text = f.text[block]

    lines = _to_text_lines(text)
    if drop_empty:
        return [line for line in lines if line.strip()]
    return lines
