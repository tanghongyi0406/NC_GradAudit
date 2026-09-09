"""Verify and list the frozen, directly usable datasets shipped in this release."""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""): h.update(block)
    return h.hexdigest()

def main() -> None:
    expected = {}
    for line in (ROOT / "data_archives/SHA256SUMS").read_text().splitlines():
        value, name = line.split(maxsplit=1); expected[Path(name).name] = value
    for archive in sorted((ROOT / "data_archives").glob("*.tar.gz")):
        actual = digest(archive); wanted = expected[archive.name]
        if actual != wanted: raise SystemExit(f"Checksum mismatch: {archive.name}")
        print(f"OK {archive.name} {actual}")

if __name__ == "__main__": main()
